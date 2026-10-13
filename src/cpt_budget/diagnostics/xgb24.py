"""CPU control: direct log-qc XGBoost with the same 24 revealed-only features.

This diagnostic deliberately leaves the v0.1/v0.2 candidates unchanged. It
isolates whether the spatial24 representation benefits an ordinary supervised
learner. Source-validation episodes choose only the tree count; no outer-test
labels or replay outcomes participate in fitting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ..config import load_config, write_json
from ..models.base import validate_observed, validate_query_site
from ..models.distribution import point_prediction_frame
from ..models.tabpfn_train import _check_episode
from ..types import PredictionBundle, load_dataset
from ..v2.features import FEATURE_COLUMNS_V2, build_features_v2


def episode_data(dataset, episodes):
    """Return direct targets and site/episode/location/depth-balanced weights."""
    episodes = pd.DataFrame(episodes)
    if episodes.empty or episodes.duplicated(["site_id", "episode_id"]).any():
        raise ValueError("Nonempty, unique source episodes are required")
    counts = episodes.groupby("site_id").size()
    features, targets, weights, metadata = [], [], [], []
    for record in episodes.to_dict("records"):
        support, queries = _check_episode(dataset, record)
        observed, query = dataset.observed(support), dataset.query(queries)
        keys = ["site_id", "location_id", "depth_index"]
        truth = query[keys].merge(dataset.truth(queries)[keys + ["log_qc"]],
                                  on=keys, how="left", validate="one_to_one", sort=False)
        if not np.isfinite(truth.log_qc).all():
            raise ValueError("Every training query needs a finite direct log-qc target")
        depth_count = query.groupby("location_id").depth_index.transform("size").to_numpy()
        weight = 1. / (counts[record["site_id"]] * query.location_id.nunique() * depth_count)
        meta = truth.rename(columns={"log_qc": "truth_log_qc"})
        meta["episode_id"] = str(record["episode_id"])
        features.append(build_features_v2(observed, query))
        targets.append(truth.log_qc.to_numpy(float))
        weights.append(weight)
        metadata.append(meta)
    weight = np.concatenate(weights)
    weight *= len(weight) / weight.sum()
    return (pd.concat(features, ignore_index=True)[FEATURE_COLUMNS_V2],
            np.concatenate(targets), weight, pd.concat(metadata, ignore_index=True))


def fit_xgb24(dataset, train_episodes, validation_episodes, config, output_dir):
    """Fit one fold at the registered tree budget, with validation early stop."""
    train, validation = pd.DataFrame(train_episodes), pd.DataFrame(validation_episodes)
    for role, episodes in (("train", train), ("validation", validation)):
        if episodes.empty or "role" not in episodes or not episodes.role.eq(role).all():
            raise ValueError(f"Only explicit {role}-role source episodes are allowed")
    if set(train.site_id) & set(validation.site_id):
        raise ValueError("Training and validation sites must be disjoint")
    combined = pd.concat([train, validation], ignore_index=True)
    if combined.fold_id.nunique() != 1:
        raise ValueError("Fit exactly one outer fold")
    if "investigation_group" in dataset.sites:
        groups = dataset.sites.set_index("site_id").investigation_group
        if set(groups.loc[train.site_id.unique()]) & set(groups.loc[validation.site_id.unique()]):
            raise ValueError("Training and validation investigation groups overlap")
    started = perf_counter()
    x_train, y_train, w_train, train_meta = episode_data(dataset, train)
    x_valid, y_valid, w_valid, valid_meta = episode_data(dataset, validation)
    # Match the existing direct-XGB control. No hidden hyperparameter search.
    model = XGBRegressor(
        objective="reg:squarederror", tree_method="hist", device="cpu",
        max_depth=4, learning_rate=0.05, n_estimators=500,
        early_stopping_rounds=30, eval_metric="mae", n_jobs=4,
        random_state=int(config.get("split", {}).get("seed", 20261013)),
        reg_lambda=1.0, min_child_weight=1.0, verbosity=0,
    )
    model.fit(x_train, y_train, sample_weight=w_train,
              eval_set=[(x_valid, y_valid)], sample_weight_eval_set=[w_valid], verbose=False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "xgb24_model.ubj"
    artifact = output_dir / "xgb24_training.json"
    model.save_model(model_path)
    valid_meta["prediction"] = model.predict(x_valid)
    valid_meta["absolute_error"] = np.abs(valid_meta.prediction - y_valid)
    macro = (valid_meta.groupby(["site_id", "episode_id", "location_id"]).absolute_error.mean()
             .groupby(["site_id", "episode_id"]).mean().groupby("site_id").mean())
    train_weights = train_meta.assign(weight=w_train).groupby("site_id").weight.sum()
    validation_weights = valid_meta.assign(weight=w_valid).groupby("site_id").weight.sum()
    summary = {
        "model_id": "xgb24", "artifact_path": str(artifact.resolve()),
        "model_filename": model_path.name, "fold_id": int(combined.fold_id.iloc[0]),
        "feature_columns": FEATURE_COLUMNS_V2, "target": "log(qc_mpa)",
        "train_sites": sorted(train.site_id.unique()),
        "validation_sites": sorted(validation.site_id.unique()),
        "train_rows": len(train_meta), "validation_rows": len(valid_meta),
        "weighting": "equal_site_then_episode_then_location_then_depth; mean_weight=1",
        "train_site_total_weights": train_weights.to_dict(),
        "validation_site_total_weights": validation_weights.to_dict(),
        "best_iteration": int(model.best_iteration),
        "best_validation_mae": float(model.best_score),
        "validation_macro_mae": float(macro.mean()),
        "validation_site_mae": macro.to_dict(),
        "training_parameters": model.get_params(), "evaluation_history": model.evals_result(),
        "seconds": perf_counter() - started, "source_commit": config.get("_source_commit"),
        "scope": "source_train_fit_source_validation_early_stop_only",
        "predictive_distribution": None,
    }
    write_json(artifact, summary)
    valid_meta.to_parquet(output_dir / "source_validation_predictions.parquet", index=False)
    return summary


class XGB24Predictor:
    model_id = "xgb24"

    def __init__(self, config=None, artifact=None):
        if artifact is None:
            raise ValueError("XGB24Predictor requires a source-trained artifact")
        self.config = config or {}
        artifact = Path(artifact)
        metadata = json.loads(artifact.read_text(encoding="utf-8"))
        if metadata["feature_columns"] != FEATURE_COLUMNS_V2:
            raise ValueError("XGB24 artifact feature schema does not match")
        self.regressor = XGBRegressor(n_jobs=4, device="cpu")
        self.regressor.load_model(artifact.parent / metadata["model_filename"])
        if self.regressor.get_booster().feature_names != FEATURE_COLUMNS_V2:
            raise ValueError("XGB24 model feature names/order do not match")
        self.observed = None

    def condition(self, observed):
        self.observed = validate_observed(observed)

    def predict(self, query_grid):
        validate_query_site(query_grid, self.observed)
        values = (self.regressor.predict(build_features_v2(self.observed, query_grid))
                  if len(query_grid) else np.array([]))
        return PredictionBundle(point_prediction_frame(query_grid, values), self.model_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v2.yaml")
    parser.add_argument("--output", default="runs/diagnostics-20261014/xgb24")
    parser.add_argument("--fold", type=int, choices=[0, 1, 2])
    args = parser.parse_args()
    config = load_config(args.config)
    from ..pipeline import _git
    config["_source_commit"] = _git(config["_root"], "rev-parse", "HEAD")
    dataset = load_dataset(config)
    episodes = pd.read_parquet(Path(config["_root"]) / config["project"]["prepared_dir"] / "episodes.parquet")
    summaries = []
    for fold in [args.fold] if args.fold is not None else [0, 1, 2]:
        subset = episodes.loc[episodes.fold_id.eq(fold)]
        summary = fit_xgb24(dataset, subset.loc[subset.role.eq("train")],
                            subset.loc[subset.role.eq("validation")], config,
                            Path(args.output) / f"fold_{fold}")
        summaries.append(summary)
        print(json.dumps({key: summary[key] for key in
                          ("fold_id", "best_iteration", "validation_macro_mae", "seconds")}), flush=True)
    write_json(Path(args.output) / "training_summary.json", summaries)


if __name__ == "__main__":
    main()
