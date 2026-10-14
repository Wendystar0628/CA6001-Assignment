"""Source-trained multi-output forests predict whole CPT curves together.

The 24 output depths share each tree's split decisions.  This is deliberately
different from a depth-row learner or a classifier/count-head postprocessor.
Only revealed-support summaries and query geometry enter the feature matrix.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits

from cpt_budget.config import PROJECT_ROOT, write_json
from cpt_budget.diagnostics.common import diagnostic_config, iter_validation_states, source_folds, STATE_KEYS
from cpt_budget.types import load_dataset
from cpt_budget.v6.thickness_head import profile_features, PROFILE_NAMES
from .common import RUN, BEST, KEYS, load_frozen, emit_candidate

N_DEPTHS = 24
MODEL_IDS = (
    "v7_profile_forest_direct",
    "v7_profile_forest_residual",
    "v7_profile_forest_direct_best50",
    "v7_profile_forest_residual_best50",
)
SETTINGS = dict(n_estimators=256, max_depth=12, min_samples_leaf=8,
                max_features=.75, random_state=20261014, n_jobs=1,
                criterion="squared_error", bootstrap=False)


def protocol():
    return {
        "family": "source_multioutput_profile_forest",
        "hypothesis": "Shared tree partitions across all depths may preserve profile context better than independent depth-row regression.",
        "freeze": "Two 256-tree models and four point candidates specified in agent proposal before training.",
        "feature_names": PROFILE_NAMES,
        "labels": ["full 24-cell logqc curve", "full 24-cell logqc minus support-only IDW curve"],
        "training": {**SETTINGS, "sample_weight": "equal total weight per source training site"},
        "source_episodes": "Frozen v4 fold-specific train/validation trajectories; repeated CPTs remain correlated.",
        "validation_usage": "Source validation reported only; no selection, parameter update or early stopping.",
        "candidates": {
            MODEL_IDS[0]: "Direct full-profile forest output.",
            MODEL_IDS[1]: "Current support-only IDW plus full-profile residual forest output.",
            MODEL_IDS[2]: "50% direct profile forest + 50% frozen best residual prediction.",
            MODEL_IDS[3]: "50% IDW-plus-profile-residual forest + 50% frozen best residual prediction.",
        },
        "risk": "Point-only exports; common downstream crossfit calibration. Tree dispersion is not asserted to be predictive uncertainty.",
        "outer_test_used": False, "query_labels_in_features": False,
        "reference": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html",
    }


def build_source(dataset, plan, fold, role):
    """Read labels only for approved source episodes, as complete profiles."""
    if role not in ("train", "validation"):
        raise ValueError("Only source train/validation roles are allowed")
    if (plan.empty or not plan.role.eq(role).all()
            or not plan.fold_id.eq(int(fold["fold_id"])).all()
            or not set(plan.site_id).issubset(set(fold[f"{role}_sites"]))):
        raise ValueError("Source episode fold/site role violation")
    for other in ("train", "validation", "test"):
        if other != role and set(fold[f"{role}_sites"]) & set(fold.get(f"{other}_sites", [])):
            raise ValueError("Source fold site roles overlap")
    features, targets, idws, sites, identifiers = [], [], [], [], []
    for ep in plan.itertuples(index=False):
        support_ids, query_ids = list(ep.support_location_ids), list(ep.query_location_ids)
        support, query = dataset.observed(support_ids), dataset.query(query_ids)
        if set(support.near_group_id) & set(query.near_group_id):
            raise ValueError("Source support/query near groups overlap")
        if set(support.site_id) != {ep.site_id} or set(query.site_id) != {ep.site_id}:
            raise ValueError("Source episode crosses site boundary")
        locations, x, _, idw = profile_features(support, query)
        truth = dataset.truth(query_ids).pivot(index="location_id", columns="depth_index", values="log_qc").loc[locations, np.arange(N_DEPTHS)].to_numpy(float)
        if not np.isfinite(truth).all():
            raise ValueError("Source training targets must be complete finite curves")
        features.append(x); targets.append(truth); idws.append(idw)
        sites.extend([ep.site_id] * len(locations)); identifiers.extend(locations)
    return (np.concatenate(features), np.concatenate(targets), np.concatenate(idws),
            np.asarray(sites), np.asarray(identifiers))


def site_weights(sites):
    sites = np.asarray(sites)
    if not len(sites):
        raise ValueError("Need nonempty training sites")
    counts = pd.Series(sites).value_counts()
    weights = np.asarray([1. / counts[site] for site in sites])
    return weights * len(weights) / weights.sum()


def fit_forests(source, settings=None):
    x, targets, idw, sites, _ = source
    if targets.shape != idw.shape or targets.shape != (len(x), N_DEPTHS):
        raise ValueError("Training requires one complete 24-cell target and IDW per profile")
    weights = site_weights(sites)
    models = {}
    for mode, y in (("direct", targets), ("residual", targets-idw)):
        model = ExtraTreesRegressor(**(SETTINGS if settings is None else settings))
        model.fit(x, y, sample_weight=weights)
        models[mode] = model
    return models


def predict_profiles(models, x, idw):
    direct = models["direct"].predict(x)
    residual = idw + models["residual"].predict(x)
    if direct.shape != idw.shape or not np.isfinite(np.r_[direct, residual]).all():
        raise ValueError("Full-profile forest prediction invalid")
    return direct, residual


def evaluate_source(models, source):
    x, truth, idw, sites, identifiers = source
    direct, residual = predict_profiles(models, x, idw)
    summaries = {}
    low = truth < 0.
    for name, prediction in (("direct", direct), ("residual", residual), ("idw", idw)):
        predicted_low = prediction < 0.
        table = pd.DataFrame({"site_id": sites,
            "log_mae": np.abs(prediction-truth).mean(axis=1),
            "h_point_mae_m": np.abs(predicted_low.sum(axis=1)-low.sum(axis=1))*.25,
            "tp": (predicted_low & low).sum(axis=1), "positive": low.sum(axis=1),
            "fp": (predicted_low & ~low).sum(axis=1), "negative": (~low).sum(axis=1)})
        by_site = table.groupby("site_id").agg(
            log_mae=("log_mae", "mean"), h_point_mae_m=("h_point_mae_m", "mean"),
            tp=("tp", "sum"), positive=("positive", "sum"), fp=("fp", "sum"), negative=("negative", "sum"))
        by_site["recall"] = by_site.tp/by_site.positive.replace(0, np.nan)
        by_site["false_positive_rate"] = by_site.fp/by_site.negative.replace(0, np.nan)
        metrics = ["log_mae", "h_point_mae_m", "recall", "false_positive_rate"]
        summaries[name] = {"macro": by_site[metrics].mean().to_dict(),
                           "by_site": by_site.reset_index().replace({np.nan: None}).to_dict("records")}
    return {"profile_episode_rows": len(x), "unique_query_cpts": len(set(identifiers)),
            "sites": len(set(sites)), "models": summaries}


def train_fold(dataset, fold, output):
    fid = int(fold["fold_id"])
    target = output / f"fold_{fid}"
    target.mkdir(parents=True, exist_ok=True)
    episode_root = PROJECT_ROOT / "runs/v4-residual-20261014/episodes"
    train_plan = pd.read_parquet(episode_root / f"fold_{fid}_train.parquet")
    validation_plan = pd.read_parquet(episode_root / f"fold_{fid}_validation.parquet")
    print(f"profile forest fold {fid}: building {len(train_plan)} source train episodes", flush=True)
    started = perf_counter()
    train = build_source(dataset, train_plan, fold, "train")
    validation = build_source(dataset, validation_plan, fold, "validation")
    feature_seconds = perf_counter()-started
    started = perf_counter()
    models = fit_forests(train)
    training_seconds = perf_counter()-started
    joblib.dump({"models": models, "protocol": protocol()}, target / "models.joblib", compress=3)
    model_storage = {}
    for name, model in models.items():
        tree_states = [tree.tree_.__getstate__() for tree in model.estimators_]
        model_storage[name] = {
            "node_count": sum(tree.tree_.node_count for tree in model.estimators_),
            "tree_array_bytes": sum(state["nodes"].nbytes + state["values"].nbytes for state in tree_states),
            "scope": "Internal node and output arrays only; not total process RSS.",
        }
    summary = {"fold_id": fid, "train_episodes": len(train_plan), "validation_episodes": len(validation_plan),
        "source_train_sites": sorted(set(train[3])), "source_validation_sites": sorted(set(validation[3])),
        "feature_count": train[0].shape[1], "output_depths": N_DEPTHS,
        "effective_trees": {name: len(model.estimators_) for name, model in models.items()},
        "model_storage": model_storage, "compressed_checkpoint_bytes": (target / "models.joblib").stat().st_size,
        "feature_seconds": feature_seconds, "training_seconds": training_seconds,
        "train": evaluate_source(models, train), "validation": evaluate_source(models, validation),
        "validation_usage": "Reporting only; no selection or updates.",
        "dependence_note": "Episode rows repeat query CPTs across budgets and trajectories; they are not independent CPTs."}
    write_json(target / "training_summary.json", summary)
    print(f"profile forest fold {fid}: trained 512 trees on {len(train[0])} profile-episode rows in {training_seconds:.2f}s", flush=True)
    return models, summary


def run(output=None):
    output = Path(output or RUN / "profile_forest")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "method_protocol.json", protocol())
    config = diagnostic_config()
    dataset = load_dataset(config)
    models, training = {}, []
    with threadpool_limits(limits=1):
        for fold in source_folds():
            fitted, summary = train_fold(dataset, fold, output)
            models[int(fold["fold_id"])] = fitted
            training.append(summary)
        wide = load_frozen()
        predictions = {name: np.full(len(wide), np.nan) for name in MODEL_IDS}
        row_lookup = wide.reset_index().set_index(KEYS)["index"]
        inference_started = perf_counter()
        for state in iter_validation_states(config):
            locations, x, _, idw = profile_features(state.observed, state.query)
            direct, residual = predict_profiles(models[int(state.metadata["fold_id"])], x, idw)
            keys = [(*[state.metadata[k] for k in STATE_KEYS], loc, depth)
                    for loc in locations for depth in range(N_DEPTHS)]
            positions = row_lookup.loc[pd.MultiIndex.from_tuples(keys, names=KEYS)].to_numpy().reshape(-1, N_DEPTHS)
            best = wide[BEST].to_numpy()[positions]
            outputs = (direct, residual, .5*(direct+best), .5*(residual+best))
            for name, values in zip(MODEL_IDS, outputs):
                predictions[name][positions.reshape(-1)] = values.reshape(-1)
        inference_seconds = perf_counter()-inference_started
        for name, values in predictions.items():
            emit_candidate(name, values, wide=wide, output_root=output,
                metadata={"family": "source_multioutput_profile_forest", "source_trained": True,
                          "formula": protocol()["candidates"][name], "native_event_probability": False})
    write_json(output / "training_summary.json", {"folds": training, "models": list(MODEL_IDS),
        "effective_forest_count": 6, "effective_total_trees": sum(sum(f["effective_trees"].values()) for f in training),
        "inference_seconds": inference_seconds, "rows_per_model": len(wide), "states_per_model": 60,
        "source_validation_was_report_only": True, "query_labels_in_features": False, "outer_test_used": False})
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    run(parser.parse_args().output)
