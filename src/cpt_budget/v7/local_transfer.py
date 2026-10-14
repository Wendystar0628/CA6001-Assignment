"""Adapt frozen source count/event heads using only revealed target profiles.

Every calibration target is predicted while its entire near/co-location
component is excluded from the context. Evaluation profiles never fit either
the local count bias or the regularized monotone probability map.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from cpt_budget.config import PROJECT_ROOT, write_json
from cpt_budget.diagnostics.common import STATE_KEYS, iter_validation_states
from cpt_budget.v6.count_projection import project_count
from cpt_budget.v6.indicator import structured_event_profile
from cpt_budget.v6.thickness_head import profile_features, event_probabilities
from .common import RUN, BEST, KEYS, load_frozen, emit_candidate

SOURCE_RUN = PROJECT_ROOT / "runs/v6-combinations-20261014/thickness_head"
BASES = {"residual": BEST, "stack": "support_stack_structural"}
MODELS = {f"v7_local_{mode}_{base}": (mode, base)
          for mode in ("count", "half_count", "event_dp") for base in BASES}
QUERY_COLUMNS = ["site_id", "location_id", "depth_index", "x_m", "y_m"]


def protocol():
    return {
        "family": "revealed_target_adaptation_of_source_count_event_heads",
        "candidates": MODELS,
        "source_heads": str(SOURCE_RUN),
        "parameters_agreed_before_new_results": True,
        "target_context": "Current revealed complete profiles only; never unmeasured query labels.",
        "local_crossfit": "Leave each full connected component of near-group equality or co-location <=1e-8 m out.",
        "count_bias": "Median of within-component median true_count-minus-source_count residuals; multiply G/(G+10).",
        "count_projection": "Round corrected count to nearest integer; choose ascending base logqc and only flip disagreements to +/- .05.",
        "half_count": "Average local-corrected source count with the actual base point count before rounding.",
        "probability_map": "sigmoid(a*logit(p)+b), a>=0; independent of query/base amplitudes.",
        "map_objective": "sum(component-mean binary cross entropy)+0.5*((a-1)^2+.25*b^2)",
        "map_bounds": {"a": [0., 4.], "b": [-6., 6.]},
        "map_fallback": "Identity if no independent component remains or numerical optimizer fails.",
        "decoder": {"tau": .5, "potts": .25, "cardinality": .25, "margin": .05,
                    "target_count": "local-corrected source count"},
        "native_probability": "Count projections use local sigmoid-calibrated source event p. Event DP uses exact Gibbs marginals.",
        "thickness_semantics": "Threshold actual final point predictions; no substitution of expected/predicted count for thickness.",
        "outer_test_used": False, "query_truth_used": False, "old_oof_risk_used": False,
        "sources": ["https://scikit-learn.org/stable/modules/calibration.html",
                     "https://proceedings.mlr.press/v54/kull17a.html"],
        "source_scope": "Regularized logistic probability calibration adapted to target support; not a reproduction of beta calibration.",
    }


def independent_components(observed):
    """Connected near/co-location components, invariant to support row order."""
    required = {"site_id", "location_id", "near_group_id", "x_m", "y_m"}
    if not required.issubset(observed):
        raise ValueError("Revealed support needs explicit site and near-group geometry")
    if observed.site_id.nunique() != 1 or observed.near_group_id.isna().any():
        raise ValueError("One site and known near groups are required")
    locations = observed[list(required)].drop_duplicates().sort_values("location_id")
    if locations.location_id.duplicated().any():
        raise ValueError("Each profile needs one fixed geometry and near group")
    ids = locations.location_id.tolist()
    xy = locations[["x_m", "y_m"]].to_numpy(float)
    near = locations.near_group_id.to_numpy()
    if not np.isfinite(xy).all():
        raise ValueError("Finite support geometry required")
    parent = np.arange(len(ids))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(ids)):
        for j in range(i):
            if near[i] == near[j] or np.linalg.norm(xy[i] - xy[j]) <= 1e-8:
                parent[root(i)] = root(j)
    groups = {}
    for i, location in enumerate(ids):
        groups.setdefault(root(i), []).append(location)
    return sorted(groups.values(), key=lambda values: values[0])


def fit_monotone_platt(probability, labels, group_ids):
    """Group-balanced likelihood with a finite identity prior, including one-class contexts."""
    probability, labels = np.asarray(probability, float), np.asarray(labels, float)
    groups = np.asarray(group_ids)
    if (probability.ndim != 2 or probability.shape != labels.shape
            or len(groups) != len(probability) or not np.isfinite(probability).all()
            or ((probability < 0) | (probability > 1)).any()
            or not np.isin(labels, [0., 1.]).all()):
        raise ValueError("Expected finite profile probabilities, binary labels and one group per profile")
    if not len(groups):
        return {"a": 1., "b": 0., "success": True, "status": "no_independent_context", "groups": 0}
    values, counts = np.unique(groups, return_counts=True)
    n_by_group = dict(zip(values, counts))
    weight = np.array([1. / (n_by_group[group] * probability.shape[1]) for group in groups])[:, None]
    x = logit(np.clip(probability, 1e-6, 1 - 1e-6))

    def objective(theta):
        a, b = theta
        z = a*x + b
        loss = np.sum(weight * (np.logaddexp(0., z) - labels*z))
        loss += .5*((a - 1.)**2 + .25*b*b)
        delta = weight*(expit(z)-labels)
        gradient = np.array([np.sum(delta*x) + a - 1., np.sum(delta) + .25*b])
        return float(loss), gradient

    result = minimize(objective, [1., 0.], jac=True, bounds=[(0., 4.), (-6., 6.)],
                      method="L-BFGS-B", options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-9})
    valid = bool(result.success and np.isfinite(result.x).all())
    a, b = result.x if valid else (1., 0.)
    return {"a": float(a), "b": float(b), "success": valid,
            "status": str(result.message), "groups": len(values),
            "objective_identity": objective(np.array([1., 0.]))[0],
            "objective_fitted": objective(np.array([a, b]))[0]}


def apply_platt(probability, calibration):
    probability = np.asarray(probability, float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Valid source probabilities required")
    return expit(calibration["a"]*logit(np.clip(probability, 1e-6, 1 - 1e-6)) + calibration["b"])


def local_adjustment(observed, heads):
    """Return adaptations fit only to revealed whole-profile cross predictions."""
    components = independent_components(observed)
    records, probabilities, labels, group_ids, count_errors = [], [], [], [], []
    for component_id, held_ids in enumerate(components):
        support = observed[~observed.location_id.isin(held_ids)].copy()
        if support.empty:
            continue
        held = observed[observed.location_id.isin(held_ids)].copy()
        query = held[QUERY_COLUMNS].copy()
        locations, xp, xe, _ = profile_features(support, query)
        count = np.clip(heads["count"].predict(xp), 0., 24.)
        p = event_probabilities(heads["event"], xe)
        truth = held.pivot(index="location_id", columns="depth_index", values="log_qc").loc[locations, np.arange(24)].to_numpy(float)
        event = truth < 0.
        group_error = float(np.median(event.sum(axis=1) - count))
        count_errors.append(group_error)
        probabilities.append(p)
        labels.append(event)
        group_ids.extend([component_id]*len(locations))
        records.append({"component_id": component_id, "held_locations": locations,
            "support_locations": sorted(support.location_id.unique()),
            "source_count": count.tolist(), "revealed_true_count": event.sum(axis=1).tolist(),
            "component_median_count_residual": group_error})
    groups = len(records)
    bias = float(np.median(count_errors)) if groups else 0.
    shrink = groups / (groups + 10.)
    p = np.concatenate(probabilities) if probabilities else np.empty((0, 24))
    y = np.concatenate(labels) if labels else np.empty((0, 24))
    calibration = fit_monotone_platt(p, y, group_ids)
    return {"components": components, "loo_groups": groups,
            "raw_count_bias": bias, "count_shrinkage": shrink,
            "count_adjustment": bias*shrink, "calibration": calibration,
            "loo_records": records}


def run(output=None):
    output = Path(output or RUN / "local_transfer")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "method_protocol.json", protocol())
    started = perf_counter()
    wide = load_frozen()
    lookup = wide.reset_index().set_index(KEYS)["index"]
    values = {name: np.full(len(wide), np.nan) for name in MODELS}
    native = {name: np.full(len(wide), np.nan) for name in MODELS}
    heads, records, contexts = {}, [], []
    with threadpool_limits(limits=1):
        for state in iter_validation_states():
            fid = int(state.metadata["fold_id"])
            if fid not in heads:
                heads[fid] = joblib.load(SOURCE_RUN / f"fold_{fid}/heads.joblib")
                for head in (heads[fid]["count"], heads[fid]["event"]):
                    head.n_jobs = 1
            adjustment = local_adjustment(state.observed, heads[fid])
            locations, xp, xe, _ = profile_features(state.observed, state.query[QUERY_COLUMNS])
            source_count = np.clip(heads[fid]["count"].predict(xp), 0., 24.)
            count = np.clip(source_count + adjustment["count_adjustment"], 0., 24.)
            source_p = event_probabilities(heads[fid]["event"], xe)
            p = apply_platt(source_p, adjustment["calibration"])
            keys = [(*[state.metadata[k] for k in STATE_KEYS], location, depth)
                    for location in locations for depth in range(24)]
            indices = lookup.loc[pd.MultiIndex.from_tuples(keys, names=KEYS)].to_numpy().reshape(-1, 24)
            for name, (mode, base_short) in MODELS.items():
                base = wide[BASES[base_short]].to_numpy()[indices]
                for i, location in enumerate(locations):
                    target = .5*(count[i] + np.sum(base[i] < 0.)) if mode == "half_count" else count[i]
                    if mode == "event_dp":
                        prediction, probability, _ = structured_event_profile(base[i], p[i], target,
                            tau=.5, potts=.25, cardinality=.25, margin=.05)
                    else:
                        prediction = project_count(base[i], int(np.clip(np.floor(target+.5), 0, 24)))
                        probability = p[i]
                    values[name][indices[i]] = prediction
                    native[name][indices[i]] = probability
                    records.append({**{k: state.metadata[k] for k in STATE_KEYS},
                        "model_id": name, "location_id": location,
                        "source_count": float(source_count[i]), "local_corrected_count": float(count[i]),
                        "target_count": float(target), "actual_point_count": int(np.sum(prediction < 0.)),
                        "base_count": int(np.sum(base[i] < 0.)),
                        "flipped_cells": int(np.sum((prediction < 0.) != (base[i] < 0.))),
                        "native_probability_mean": float(probability.mean())})
            context = {**{k: state.metadata[k] for k in STATE_KEYS}, **adjustment}
            write_json(output / "contexts" / f"{state.key}.json", context)
            contexts.append({**{k: state.metadata[k] for k in STATE_KEYS},
                **{k: adjustment[k] for k in ("loo_groups", "raw_count_bias", "count_shrinkage", "count_adjustment")},
                **{f"calibration_{k}": v for k, v in adjustment["calibration"].items()}})
            print(f"local transfer: {state.key}, bias={adjustment['count_adjustment']:.3f}, "
                  f"a={adjustment['calibration']['a']:.3f}, b={adjustment['calibration']['b']:.3f}", flush=True)
    for name in MODELS:
        emit_candidate(name, values[name], wide=wide, output_root=output, probability=native[name],
                       metadata={**protocol(), "mode_and_base": MODELS[name]})
    pd.DataFrame(records).to_csv(output / "profile_decisions.csv", index=False)
    pd.DataFrame(contexts).to_csv(output / "adaptation_summary.csv", index=False)
    write_json(output / "execution_summary.json", {"models": list(MODELS), "states_per_model": len(contexts),
        "rows_per_model": len(wide), "seconds": perf_counter()-started,
        "all_local_calibrations_successful": all(r["calibration_success"] for r in contexts),
        "new_source_training": False, "query_truth_used": False, "old_oof_risk_used": False})
    return values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.output)
