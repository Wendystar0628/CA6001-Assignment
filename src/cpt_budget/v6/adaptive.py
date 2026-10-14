"""Observed-only profile stacking selected by whole-location leave-one-out.

The stack sees only the currently revealed profiles. A frozen source FPCA
basis is reused, while its target coefficients, profile registration and IDW
are refitted inside every leave-one-location-out split. Evaluation labels never
enter the fitting interface. Residual variants retain a fixed half share of
the previously frozen best residual predictor; that member is not LOO-fitted.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.special import expit

from cpt_budget.config import PROJECT_ROOT, write_json
from cpt_budget.diagnostics.common import diagnostic_config, iter_validation_states, STATE_KEYS
from cpt_budget.models.base import validate_observed, validate_query_site
from cpt_budget.models.idw import IDWPredictor
from cpt_budget.types import QUERY_COLUMNS
from cpt_budget.v5.functional import FunctionalProfilePredictor, ProfileBasis
from cpt_budget.v5.layer_alignment import LayerAlignmentPredictor
from cpt_budget.v6.common import BEST, KEYS, RUN, emit_candidate, load_frozen


MEMBERS = ("idw", "functional_fpca_krr", "registered_profile_shape", "spatial_weighted_median")
MODELS = ("support_stack_l1", "support_stack_structural",
          "support_residual_stack_l1", "support_residual_stack_structural")
SETTINGS = {
    "members": MEMBERS, "weight_grid_step": 0.25,
    "weight_regularization": 0.05, "weight_prior": "uniform",
    "loo_unit": "whole_revealed_location plus same near_group and colocated profiles excluded",
    "loo_no_independent_support_fallback": "all members predict fixed log_qc=0; no held profile labels",
    "loo_small_support": "all members use available 1 or 2 independent profiles",
    "structural_loss": "L1 + 0.5*soft_thickness_absolute_error_m + 0.5*weighted_event_Brier",
    "soft_event_temperature_log_qc": 0.25, "event_false_negative_weight": 2.0,
    "threshold_qc_mpa": 1.0, "depth_step_m": 0.25,
    "weighted_median_neighbors": 3, "distance_power": 2.0,
    "residual_blend_share": 0.5,
    "selection_data": "current revealed support only; no evaluation/query measurements",
    "residual_loo_limitation": "BEST has fixed 0.5 weight and is absent from LOO selection",
    "native_probability": "none; sigmoid is only a training loss surrogate",
}


def weighted_median_predict(observed: pd.DataFrame, query: pd.DataFrame) -> np.ndarray:
    """Geometry-weighted L1 interpolation; values remain support order statistics."""
    observed = validate_observed(observed)
    validate_query_site(query, observed)
    support = observed.sort_values(["location_id", "depth_index"])
    profiles = support.pivot(index="location_id", columns="depth_index", values="log_qc")
    if not np.isfinite(profiles.to_numpy(float)).all():
        raise ValueError("Weighted-median support profiles must be complete")
    geometry = support.drop_duplicates("location_id").set_index("location_id")
    xy = geometry.loc[profiles.index, ["x_m", "y_m"]].to_numpy(float)
    output = np.empty(len(query))
    for _, rows in query.reset_index(drop=True).groupby("location_id", sort=False):
        qxy = rows[["x_m", "y_m"]].drop_duplicates().to_numpy(float)
        if len(qxy) != 1 or not np.isfinite(qxy).all():
            raise ValueError("A query profile requires one finite coordinate")
        distances = np.linalg.norm(xy - qxy[0], axis=1)
        nearest = np.argsort(distances, kind="stable")[:3]
        values = profiles.to_numpy(float)[nearest]
        if distances[nearest[0]] < 1e-8:
            predicted = values[0]
        else:
            weights = distances[nearest] ** -2
            weights /= weights.sum()
            order = np.argsort(values, axis=0, kind="stable")
            sorted_values = np.take_along_axis(values, order, axis=0)
            sorted_weights = weights[order]
            idx = (np.cumsum(sorted_weights, axis=0) >= 0.5).argmax(axis=0)
            predicted = sorted_values[idx, np.arange(values.shape[1])]
        depth = profiles.columns.get_indexer(rows.depth_index)
        if (depth < 0).any():
            raise ValueError("Query depth is outside support profile grid")
        output[rows.index] = predicted[depth]
    return output


def member_predictions(observed, query, basis, config=None):
    """Only geometry from query is forwarded to the actual members."""
    query = query[[c for c in QUERY_COLUMNS if c in query]].copy()
    predictions = []
    for predictor in (IDWPredictor(config), FunctionalProfilePredictor(basis, method="krr"),
                      LayerAlignmentPredictor(config, mode="shape")):
        predictor.condition(observed)
        predictions.append(predictor.predict(query).frame.pred_log_qc.to_numpy(float))
    predictions.append(weighted_median_predict(observed, query))
    return np.column_stack(predictions)


def leave_one_location_out(observed, basis, config=None):
    """Predictions for each revealed profile while withholding that whole profile."""
    observed = validate_observed(observed).reset_index(drop=True)
    if observed.location_id.nunique() < 3:
        raise ValueError("Adaptive stacking requires at least three revealed profiles")
    predictions = np.empty((len(observed), len(MEMBERS)))
    exclusions = []
    for location, rows in observed.groupby("location_id", sort=False):
        xy = rows[["x_m", "y_m"]].iloc[0].to_numpy(float)
        excluded = observed.location_id.eq(location) | (
            np.linalg.norm(observed[["x_m", "y_m"]].to_numpy(float) - xy, axis=1) < 1e-8)
        if "near_group_id" in observed:
            excluded |= observed.near_group_id.isin(rows.near_group_id.dropna().unique())
        train = observed.loc[~excluded]
        query = rows[[c for c in QUERY_COLUMNS if c in rows]]
        predictions[rows.index] = member_predictions(train, query, basis, config) if len(train) else 0.0
        exclusions.append({"held_location_id": location,
                           "excluded_location_ids": observed.loc[excluded, "location_id"].unique().tolist(),
                           "n_remaining_locations": train.location_id.nunique(),
                           "fallback": "fixed_log_qc_zero" if train.empty else None})
    if not np.isfinite(predictions).all():
        raise ValueError("Nonfinite leave-one-location-out prediction")
    observed.attrs["loo_exclusions"] = exclusions
    return observed, predictions


def simplex_candidates():
    return np.array([w for w in itertools.product(range(5), repeat=4) if sum(w) == 4], float) / 4


def fit_weights(observed, loo_predictions, *, objective="l1"):
    """Choose a coarse regularized stack using revealed labels only."""
    if objective not in {"l1", "structural"}:
        raise ValueError("Unknown stacking objective")
    observed = observed.reset_index(drop=True)
    matrix = np.asarray(loo_predictions, float)
    if matrix.shape != (len(observed), len(MEMBERS)) or not np.isfinite(matrix).all():
        raise ValueError("Invalid LOO member matrix")
    weights = simplex_candidates()
    prediction = matrix @ weights.T
    truth = observed.log_qc.to_numpy(float)
    l1 = np.abs(prediction - truth[:, None]).mean(axis=0)
    regularization = 0.05 * np.square(weights - 0.25).sum(axis=1)
    probability = expit(-prediction / 0.25)
    labels = truth < 0
    event = np.mean(np.where(labels[:, None], 2 * (1 - probability) ** 2, probability ** 2), axis=0)
    thickness = []
    for _, rows in observed.groupby("location_id", sort=False):
        thickness.append(np.abs(0.25 * probability[rows.index].sum(axis=0)
                                - 0.25 * labels[rows.index].sum()))
    thickness = np.mean(thickness, axis=0)
    loss = l1 + regularization
    if objective == "structural":
        loss = loss + 0.5 * thickness + 0.5 * event
    best = int(np.argmin(loss))
    return weights[best], {"objective": objective, "loss": float(loss[best]),
                           "loo_log_mae": float(l1[best]),
                           "loo_soft_thickness_mae_m": float(thickness[best]),
                           "loo_weighted_event_brier": float(event[best]),
                           "regularization": float(regularization[best]),
                           "weights": dict(zip(MEMBERS, weights[best].tolist()))}


def load_basis(fold_id):
    path = PROJECT_ROOT / "runs/v5-exploration-20261014/functional/bases" / f"fold_{fold_id}"
    manifest = json.loads((path / "basis_manifest.json").read_text(encoding="utf-8"))
    with np.load(path / "basis.npz") as arrays:
        return ProfileBasis(arrays["depth_indices"].copy(), arrays["components"].copy(),
                            arrays["eigenvalues"].copy(), manifest["retained_variance"],
                            tuple(manifest["source_site_ids"]), manifest["source_profile_count"])


def run(output_root=None):
    output = Path(output_root or RUN / "adaptive")
    output.mkdir(parents=True, exist_ok=True)
    # Written before executing any new candidate, without consulting validation scores.
    write_json(output / "method_protocol.json", {"models": MODELS, **SETTINGS,
               "precommit_scope": "parameters frozen before this family run; exploratory repeated validation"})
    wide = load_frozen()
    config = diagnostic_config()
    bases = {fold: load_basis(fold) for fold in (0, 1, 2)}
    output_parts, logs = [], []
    run_started = time.perf_counter()
    for state in iter_validation_states(config):
        state_started = time.perf_counter()
        fold = state.metadata["fold_id"]
        if state.observed.site_id.isin(bases[fold].source_site_ids).any():
            raise ValueError("Source FPCA basis contains validation target site")
        observed, loo = leave_one_location_out(state.observed, bases[fold], config)
        query_members = member_predictions(state.observed, state.query, bases[fold], config)
        frame = state.query[["location_id", "depth_index"]].copy()
        for name in STATE_KEYS:
            frame[name] = state.metadata[name]
        for objective in ("l1", "structural"):
            weights, record = fit_weights(observed, loo, objective=objective)
            frame[f"support_stack_{objective}"] = query_members @ weights
            logs.append({**{k: state.metadata[k] for k in STATE_KEYS},
                         "observed_set_id": state.metadata["observed_set_id"],
                         "n_loo_locations": observed.location_id.nunique(), **record})
        output_parts.append(frame)
        write_json(output / "contexts" / f"{state.key}.json", {
            "observed_location_ids": observed.location_id.drop_duplicates().tolist(),
            "basis_source_sites": bases[fold].source_site_ids,
            "selections": logs[-2:], "query_inputs": "geometry only",
            "loo_exclusions": observed.attrs["loo_exclusions"],
            "elapsed_seconds": time.perf_counter() - state_started})
        print(f"adaptive: {state.key}", flush=True)
    calculated = pd.concat(output_parts, ignore_index=True)
    aligned = wide[KEYS].merge(calculated, on=KEYS, how="left", validate="one_to_one")
    for objective in ("l1", "structural"):
        name = f"support_stack_{objective}"
        values = aligned[name].to_numpy(float)
        emit_candidate(name, values, wide=wide, output_root=output, metadata=SETTINGS)
        residual_name = f"support_residual_stack_{objective}"
        emit_candidate(residual_name, 0.5 * wide[BEST].to_numpy(float) + 0.5 * values,
                       wide=wide, output_root=output, metadata=SETTINGS)
    records = pd.json_normalize(logs)
    records.to_csv(output / "support_selections.csv", index=False)
    write_json(output / "completion.json", {"states": len(output_parts), "rows_per_model": len(wide),
               "models": MODELS, "loo_selections": len(logs), "query_truth_used": False,
               "elapsed_seconds": time.perf_counter() - run_started})
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.output)
