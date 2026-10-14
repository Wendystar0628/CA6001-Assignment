"""Source-only profile count and depth-event heads with point-label constraints.

The regression label is the full-profile low-qc cell count. The classification
label is the physical event qc<1 MPa at a depth. Both heads see only geometry
and revealed-support summaries; query measurements are training targets only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from threadpoolctl import threadpool_limits

from cpt_budget.config import PROJECT_ROOT, write_json
from cpt_budget.diagnostics.common import diagnostic_config, iter_validation_states, source_folds, STATE_KEYS
from cpt_budget.types import load_dataset
from .common import RUN, BEST, KEYS, load_frozen, emit_candidate

N_DEPTHS = 24
MODEL_IDS = (
    "v6_count_head_residual",
    "v6_count_head_fpca50",
    "v6_count_head_half_residual",
    "v6_event_head_residual",
)
SETTINGS = dict(n_estimators=256, max_depth=12, max_features=.75,
                random_state=20261014, n_jobs=1)
MARGIN = .05
PROFILE_NAMES = ([f"idw_d{i}" for i in range(N_DEPTHS)]
    + [f"weighted_low_fraction_d{i}" for i in range(N_DEPTHS)]
    + [f"support_depth_median_d{i}" for i in range(N_DEPTHS)]
    + ["support_n", "log_nearest_distance", "log_median_distance", "log_max_distance",
       "idw_low_count_fraction", "support_count_mean", "support_count_median",
       "support_count_std", "weighted_support_count", "nearest_support_count",
       "support_count_min", "support_count_max", "log_support_spread"])
EVENT_NAMES = [*PROFILE_NAMES, "depth_fraction", "depth_idw", "depth_weighted_low_fraction",
               "depth_support_median", "idw_gradient", "idw_local_three_mean"]


def protocol():
    return {
        "round": 3, "trigger": "Round 1 combinations did not establish a clear comprehensive winner.",
        "freeze": "Four candidates, features, tree settings and .05 crossing margin agreed before training.",
        "source_episodes": "Frozen v4 fold-specific train and validation trajectories.",
        "profile_feature_names": PROFILE_NAMES, "event_feature_names": EVENT_NAMES,
        "training": {**SETTINGS, "count_min_samples_leaf": 8,
                     "event_min_samples_leaf": 24, "count_loss": "squared_error",
                     "event_criterion": "gini", "sample_weight": "equal total weight per source training site",
                     "class_weight": None},
        "selection": "No hyperparameter selection or early stopping. Source validation is report-only.",
        "point_base": "Frozen best residual or its 50% FPCA mixture.",
        "candidates": {
            MODEL_IDS[0]: "Round predicted count to nearest integer, event probability top-k; residual base.",
            MODEL_IDS[1]: "Same learned count and event ranking; 50% residual/FPCA base.",
            MODEL_IDS[2]: "Round mean of learned count and residual's existing low count; event ranking.",
            MODEL_IDS[3]: "Event probability >= .5; residual base; no count constraint.",
        },
        "point_correction": "Preserve values whose label agrees; flip disagreements to -.05 or +.05 logqc.",
        "thickness": "Always recomputed from final point values below zero, not substituted by head output.",
        "native_probability": "Source-trained classifier event probability, not a logqc CDF or calibrated risk.",
        "outer_test_used": False, "query_labels_in_features": False,
        "references": [
            "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html",
            "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html",
        ],
    }


def profile_features(observed, query):
    """Complete support-only curve summaries; return profiles in sorted ID order."""
    required = {"site_id", "location_id", "depth_index", "x_m", "y_m"}
    if not required.issubset(query) or not (required | {"log_qc"}).issubset(observed):
        raise ValueError("Missing geometry or revealed measurement columns")
    if (observed.site_id.nunique() != 1 or query.site_id.nunique() != 1
            or set(observed.site_id) != set(query.site_id)):
        raise ValueError("Feature construction must remain within a single site")
    if set(observed.location_id) & set(query.location_id):
        raise ValueError("Query locations must not be in revealed support")
    if (observed.duplicated(["location_id", "depth_index"]).any()
            or query.duplicated(["location_id", "depth_index"]).any()):
        raise ValueError("Duplicate profile depths")
    depths = np.arange(N_DEPTHS)
    support = observed.pivot(index="location_id", columns="depth_index", values="log_qc").reindex(columns=depths).sort_index()
    if support.empty or not np.isfinite(support.to_numpy(float)).all():
        raise ValueError("Support requires complete finite 24-cell profiles")
    locations = sorted(query.location_id.unique())
    for _, block in query.groupby("location_id"):
        if len(block) != N_DEPTHS or set(block.depth_index) != set(depths):
            raise ValueError("Query requires complete 24-cell profiles")
    for frame in (observed, query):
        if (frame.groupby("location_id")[["x_m", "y_m"]].nunique() > 1).any().any():
            raise ValueError("Profile geometry must be constant across depth")
    support_xy = observed.drop_duplicates("location_id").set_index("location_id").loc[support.index, ["x_m", "y_m"]].to_numpy(float)
    query_xy = query.drop_duplicates("location_id").set_index("location_id").loc[locations, ["x_m", "y_m"]].to_numpy(float)
    if not np.isfinite(np.r_[support_xy, query_xy]).all():
        raise ValueError("Nonfinite geometry")
    values = support.to_numpy(float)
    counts = (values < 0.).sum(axis=1)
    median = np.median(values, axis=0)
    spread = np.sqrt(((support_xy - support_xy.mean(axis=0)) ** 2).sum(axis=1)).mean()
    matrix, events, idws = [], [], []
    for xy in query_xy:
        distances = np.linalg.norm(support_xy - xy, axis=1)
        nearest = np.argsort(distances, kind="stable")[:3]
        d = distances[nearest]
        if np.any(d < 1e-9):
            weight = (d < 1e-9).astype(float)
        else:
            weight = (d.min() / d) ** 2
        weight /= weight.sum()
        idw = weight @ values[nearest]
        low_fraction = weight @ (values[nearest] < 0.)
        summary = np.array([len(support), *np.log1p([d.min(), np.median(distances), distances.max()]),
            np.mean(idw < 0.), counts.mean()/24., np.median(counts)/24., counts.std()/24.,
            float(weight @ counts[nearest])/24., counts[nearest[0]]/24.,
            counts.min()/24., counts.max()/24., np.log1p(spread)])
        feature = np.r_[idw, low_fraction, median, summary]
        local = np.array([idw[max(0, i-1):min(24, i+2)].mean() for i in range(24)])
        depth_features = np.c_[depths/23., idw, low_fraction, median, np.gradient(idw), local]
        matrix.append(feature)
        events.append(np.c_[np.repeat(feature[None, :], 24, axis=0), depth_features])
        idws.append(idw)
    return locations, np.asarray(matrix), np.asarray(events), np.asarray(idws)


def constrained_labels(probability, counts, base):
    probability, base = np.asarray(probability, float), np.asarray(base, float)
    counts = np.asarray(counts, float).reshape(-1)
    if (probability.shape != base.shape or probability.ndim != 2
            or len(counts) != len(base) or not np.isfinite(probability).all()
            or not np.isfinite(counts).all() or not np.isfinite(base).all()):
        raise ValueError("Invalid event/count/base arrays")
    rounded = np.clip(np.floor(counts + .5), 0, probability.shape[1]).astype(int)
    labels = np.zeros_like(probability, dtype=bool)
    for i, k in enumerate(rounded):
        # Probability first; deterministic base and depth tie-breaking only.
        order = np.lexsort((np.arange(base.shape[1]), base[i], -probability[i]))
        labels[i, order[:k]] = True
    return labels, rounded


def minimally_relabel(base, desired_low, margin=MARGIN):
    base, desired_low = np.asarray(base, float), np.asarray(desired_low, bool)
    if base.shape != desired_low.shape or not np.isfinite(base).all() or margin <= 0:
        raise ValueError("Invalid point relabeling input")
    changed = (base < 0.) != desired_low
    return np.where(changed, np.where(desired_low, -float(margin), float(margin)), base)


def build_source(dataset, plan, fold, role):
    if role not in ("train", "validation"):
        raise ValueError("Only source train/validation roles are allowed")
    if (not plan.role.eq(role).all() or not plan.fold_id.eq(int(fold["fold_id"])).all()
            or not set(plan.site_id).issubset(set(fold[f"{role}_sites"]))):
        raise ValueError("Source episode fold/site role violation")
    features, event_features, labels, sites, idws = [], [], [], [], []
    for ep in plan.itertuples(index=False):
        support_ids, query_ids = list(ep.support_location_ids), list(ep.query_location_ids)
        support, query = dataset.observed(support_ids), dataset.query(query_ids)
        if set(support.near_group_id) & set(query.near_group_id):
            raise ValueError("Source support/query near groups overlap")
        if set(support.site_id) != {ep.site_id} or set(query.site_id) != {ep.site_id}:
            raise ValueError("Source episode crosses site boundary")
        locations, xp, xe, idw = profile_features(support, query)
        truth = dataset.truth(query_ids).pivot(index="location_id", columns="depth_index", values="log_qc").loc[locations, np.arange(24)].to_numpy(float)
        if not np.isfinite(truth).all():
            raise ValueError("Source training labels must be finite")
        features.append(xp); event_features.append(xe); labels.append(truth < 0.)
        idws.append(idw); sites.extend([ep.site_id] * len(locations))
    return (np.concatenate(features), np.concatenate(event_features), np.concatenate(labels),
            np.asarray(sites), np.concatenate(idws))


def event_probabilities(classifier, matrix):
    raw = classifier.predict_proba(matrix.reshape(-1, matrix.shape[-1]))
    positives = np.flatnonzero(classifier.classes_ == 1)
    probability = raw[:, positives[0]] if len(positives) else np.zeros(len(raw))
    return probability.reshape(matrix.shape[:-1])


def evaluate_heads(heads, source):
    xp, xe, labels, sites, idw = source
    count, event = heads
    predicted_count = np.clip(count.predict(xp), 0., 24.)
    probability = event_probabilities(event, xe)
    rows = pd.DataFrame({"site_id": sites,
        "count_mae_m": np.abs(predicted_count-labels.sum(axis=1))*.25,
        "rounded_count_mae_m": np.abs(np.floor(predicted_count+.5)-labels.sum(axis=1))*.25,
        "idw_thickness_mae_m": np.abs((idw < 0.).sum(axis=1)-labels.sum(axis=1))*.25,
        "event_brier": ((probability-labels)**2).mean(axis=1)})
    return {"profiles": len(xp), "sites": len(set(sites)),
            "macro": rows.groupby("site_id").mean(numeric_only=True).mean().to_dict(),
            "by_site": rows.groupby("site_id").mean(numeric_only=True).reset_index().to_dict("records")}


def train_heads(dataset, fold, output):
    fid = int(fold["fold_id"])
    target = output / f"fold_{fid}"
    target.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    episode_root = PROJECT_ROOT / "runs/v4-residual-20261014/episodes"
    train_plan = pd.read_parquet(episode_root / f"fold_{fid}_train.parquet")
    validation_plan = pd.read_parquet(episode_root / f"fold_{fid}_validation.parquet")
    print(f"thickness head fold {fid}: build {len(train_plan)} train episodes", flush=True)
    train = build_source(dataset, train_plan, fold, "train")
    validation = build_source(dataset, validation_plan, fold, "validation")
    feature_seconds = perf_counter() - started
    xp, xe, labels, sites, _ = train
    totals = pd.Series(sites).value_counts()
    weight = np.array([1./totals[site] for site in sites])
    weight *= len(weight)/weight.sum()
    count = ExtraTreesRegressor(**SETTINGS, min_samples_leaf=8, criterion="squared_error")
    event = ExtraTreesClassifier(**SETTINGS, min_samples_leaf=24, criterion="gini", class_weight=None)
    started = perf_counter()
    count.fit(xp, labels.sum(axis=1), sample_weight=weight)
    event.fit(xe.reshape(-1, xe.shape[-1]), labels.reshape(-1).astype(int),
              sample_weight=np.repeat(weight / 24., 24))
    training_seconds = perf_counter() - started
    joblib.dump({"count": count, "event": event, "protocol": protocol()}, target / "heads.joblib", compress=3)
    summary = {"fold_id": fid, "train_episodes": len(train_plan), "validation_episodes": len(validation_plan),
        "train_profile_rows": len(xp), "train_event_rows": int(labels.size),
        "unique_train_query_cpts": len({loc for ids in train_plan.query_location_ids for loc in ids}),
        "unique_validation_query_cpts": len({loc for ids in validation_plan.query_location_ids for loc in ids}),
        "dependence_note": "Profile and event rows repeat CPTs across budgets/trajectories; not independent samples.",
        "source_train_sites": sorted(set(sites)), "source_validation_sites": sorted(set(validation[3])),
        "profile_feature_count": xp.shape[1], "event_feature_count": xe.shape[-1],
        "feature_seconds": feature_seconds, "training_seconds": training_seconds,
        "train": evaluate_heads((count, event), train), "validation": evaluate_heads((count, event), validation),
        "validation_usage": "Reporting only; no selection or updates.",
        "effective_trees": {"count": len(count.estimators_), "event": len(event.estimators_)}}
    write_json(target / "training_summary.json", summary)
    print(f"thickness head fold {fid}: trained {len(xp)} profiles / {labels.size} events in {training_seconds:.1f}s", flush=True)
    return (count, event), summary


def run(output=None):
    output = Path(output or RUN / "thickness_head")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "method_protocol.json", protocol())
    config = diagnostic_config()
    dataset = load_dataset(config)
    heads, training = {}, []
    with threadpool_limits(limits=1):
        for fold in source_folds():
            pair, summary = train_heads(dataset, fold, output)
            heads[int(fold["fold_id"])] = pair
            training.append(summary)
        wide = load_frozen()
        predictions = {name: np.full(len(wide), np.nan) for name in MODEL_IDS}
        probabilities = np.full(len(wide), np.nan)
        row_lookup = wide.reset_index().set_index(KEYS)["index"]
        records = []
        for state in iter_validation_states(config):
            fid = int(state.metadata["fold_id"])
            locations, xp, xe, _ = profile_features(state.observed, state.query)
            count_head, event_head = heads[fid]
            counts = np.clip(count_head.predict(xp), 0., 24.)
            p = event_probabilities(event_head, xe)
            keys = [(*[state.metadata[k] for k in STATE_KEYS], location, depth)
                    for location in locations for depth in range(24)]
            positions = row_lookup.loc[pd.MultiIndex.from_tuples(keys, names=KEYS)].to_numpy().reshape(-1, 24)
            residual = wide[BEST].to_numpy()[positions]
            fpca = wide.functional_fpca_krr.to_numpy()[positions]
            mixture = .5*(residual+fpca)
            full_labels, full_count = constrained_labels(p, counts, residual)
            half_labels, half_count = constrained_labels(p, .5*(counts+(residual < 0.).sum(axis=1)), residual)
            candidates = {MODEL_IDS[0]: minimally_relabel(residual, full_labels),
                          MODEL_IDS[1]: minimally_relabel(mixture, full_labels),
                          MODEL_IDS[2]: minimally_relabel(residual, half_labels),
                          MODEL_IDS[3]: minimally_relabel(residual, p >= .5)}
            for name, values in candidates.items():
                predictions[name][positions.reshape(-1)] = values.reshape(-1)
            probabilities[positions.reshape(-1)] = p.reshape(-1)
            for i, location in enumerate(locations):
                records.append({**state.metadata, "location_id": location,
                    "predicted_continuous_count": float(counts[i]),
                    "count_decision": int(full_count[i]), "half_count_decision": int(half_count[i]),
                    "residual_point_count": int((residual[i] < 0.).sum()),
                    "event_p05_count": int((p[i] >= .5).sum()),
                    "count_point_count_actual": int((candidates[MODEL_IDS[0]][i] < 0.).sum())})
        for name, values in predictions.items():
            emit_candidate(name, values, wide=wide, output_root=output, probability=probabilities,
                metadata={"family": "source_thickness_event_heads", "round": 3,
                    "formula": protocol()["candidates"][name], "source_trained": True,
                    "native_probability_source": "Source-only ExtraTrees event classifier; same head for all four candidates.",
                    "point_margin": MARGIN})
        pd.DataFrame(records).to_csv(output / "profile_decisions.csv", index=False)
        write_json(output / "training_summary.json", {
            "folds": training, "models": list(MODEL_IDS), "rows_per_model": len(wide),
            "states_per_model": 60, "source_validation_was_report_only": True,
            "query_labels_in_features": False, "outer_test_used": False})
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.output)
