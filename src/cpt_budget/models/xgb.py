"""Source-episode boosted trees with equal total weight per training site."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ..features import FEATURE_COLUMNS, build_features
from ..types import PredictionBundle
from .base import validate_observed, validate_query_site
from .distribution import point_prediction_frame


def _ids(value) -> list[str]:
    if isinstance(value, str):
        value = json.loads(value)
    return list(value)


def episode_training_data(dataset, episodes: pd.DataFrame):
    """Build query-only supervision; context labels enter only permitted features.

    A site's weight is split equally over episodes, then query locations, then
    depth cells. Returned weights have mean 1 to preserve regularization scale.
    """
    if episodes.empty:
        raise ValueError("Source training/validation requires at least one episode.")
    if episodes.duplicated(["site_id", "episode_id"]).any():
        raise ValueError("Duplicate source episodes would distort site weights.")
    episode_counts = episodes.groupby("site_id").size()
    features, labels, weights, metadata = [], [], [], []
    for episode in episodes.itertuples(index=False):
        support_ids = _ids(episode.support_location_ids)
        query_ids = _ids(episode.query_location_ids)
        if not support_ids or not query_ids or set(support_ids).intersection(query_ids):
            raise ValueError("Each source episode needs disjoint, nonempty support and query locations.")
        observed = dataset.observed(support_ids)
        query = dataset.query(query_ids)
        truth = dataset.truth(query_ids)
        if set(observed["site_id"]) != {episode.site_id} or set(query["site_id"]) != {episode.site_id}:
            raise ValueError("An episode cannot mix sites.")
        if "near_group_id" in observed and "near_group_id" in query:
            if set(observed["near_group_id"]).intersection(query["near_group_id"]):
                raise ValueError("Source support/query boundary splits a near group.")
        keys = ["site_id", "location_id", "depth_index"]
        target = query[keys].merge(truth[keys + ["log_qc"]], on=keys, how="left", validate="one_to_one", sort=False)
        if len(target) != len(query) or not np.isfinite(target["log_qc"]).all():
            raise ValueError("Every query row needs one finite training label.")
        n_locations = query["location_id"].nunique()
        depth_count = query.groupby("location_id")["depth_index"].transform("size").to_numpy()
        weight = 1.0 / (episode_counts[episode.site_id] * n_locations * depth_count)
        features.append(build_features(observed, query))
        labels.append(target["log_qc"].to_numpy(dtype=float))
        weights.append(weight)
        meta = query[keys].copy()
        meta["episode_id"] = episode.episode_id
        metadata.append(meta)
    weight = np.concatenate(weights)
    weight *= len(weight) / weight.sum()
    return (
        pd.concat(features, ignore_index=True)[FEATURE_COLUMNS],
        np.concatenate(labels), weight, pd.concat(metadata, ignore_index=True),
    )


def fit_xgb(dataset, episodes, config: dict, output_dir) -> Path:
    """Fit one outer fold, early-stop only on its held-out validation sites."""
    episodes = pd.DataFrame(episodes)
    if "role" not in episodes:
        raise ValueError("Episodes must identify train versus validation roles.")
    if "fold_id" in episodes and episodes["fold_id"].nunique() != 1:
        raise ValueError("Fit one outer fold at a time; never combine fold training roles.")
    train = episodes.loc[episodes["role"] == "train"]
    validation = episodes.loc[episodes["role"] == "validation"]
    if train.empty or validation.empty:
        raise ValueError("XGBoost requires both training and validation source episodes.")
    if set(train["site_id"]).intersection(validation["site_id"]):
        raise ValueError("Training and validation sites must be disjoint.")
    x_train, y_train, w_train, train_meta = episode_training_data(dataset, train)
    x_valid, y_valid, w_valid, valid_meta = episode_training_data(dataset, validation)
    settings = config.get("xgb", {})
    model = XGBRegressor(
        objective="reg:squarederror", tree_method="hist",
        max_depth=int(settings.get("max_depth", 4)),
        learning_rate=float(settings.get("learning_rate", 0.05)),
        n_estimators=min(500, max(1, int(settings.get("n_estimators", 500)))),
        early_stopping_rounds=max(1, int(settings.get("early_stopping_rounds", 30))),
        eval_metric="mae", n_jobs=int(settings.get("n_jobs", 4)),
        random_state=int(config.get("split", {}).get("seed", 20261013)),
        reg_lambda=float(settings.get("reg_lambda", 1.0)),
        min_child_weight=float(settings.get("min_child_weight", 1.0)),
        verbosity=0,
    )
    model.fit(
        x_train, y_train, sample_weight=w_train,
        eval_set=[(x_valid, y_valid)], sample_weight_eval_set=[w_valid], verbose=False,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "xgb_model.json"
    model.save_model(artifact)
    valid_meta["absolute_error"] = np.abs(model.predict(x_valid) - y_valid)
    # Additional macro statistic is explicit; early stopping uses the same
    # site/episode/location/depth-weighted absolute-error objective.
    valid_site_mae = valid_meta.groupby(["site_id", "episode_id", "location_id"])["absolute_error"].mean()
    valid_site_mae = valid_site_mae.groupby(["site_id", "episode_id"]).mean().groupby("site_id").mean()
    summary = {
        "model_id": "xgb", "artifact": str(artifact.resolve()),
        "fold_id": int(episodes["fold_id"].iloc[0]) if "fold_id" in episodes else None,
        "feature_columns": FEATURE_COLUMNS,
        "train_sites": sorted(train["site_id"].unique().tolist()),
        "validation_sites": sorted(validation["site_id"].unique().tolist()),
        "train_rows": len(train_meta), "validation_rows": len(valid_meta),
        "weighting": "equal_site_then_episode_then_location_then_depth; mean_weight=1",
        "best_iteration": int(model.best_iteration), "best_validation_mae": float(model.best_score),
        "validation_site_mae": {str(k): float(v) for k, v in valid_site_mae.items()},
        "validation_macro_mae": float(valid_site_mae.mean()),
        "training_parameters": model.get_params(),
        "evaluation_history": model.evals_result(),
    }
    (output_dir / "xgb_training.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return artifact


class XGBPredictor:
    model_id = "xgb"

    def __init__(self, config: dict | None = None, artifact=None):
        self.config = config or {}
        if artifact is None:
            raise ValueError("XGBPredictor requires a source-trained fold artifact.")
        self.regressor = XGBRegressor(n_jobs=int(self.config.get("xgb", {}).get("n_jobs", 4)))
        self.regressor.load_model(artifact)
        if self.regressor.get_booster().feature_names != FEATURE_COLUMNS:
            raise ValueError("XGBoost artifact has incompatible neighbourhood feature names/order.")
        self.observed = None

    def condition(self, observed: pd.DataFrame) -> None:
        self.observed = validate_observed(observed)

    def predict(self, query_grid: pd.DataFrame) -> PredictionBundle:
        validate_query_site(query_grid, self.observed)
        prediction = self.regressor.predict(build_features(self.observed, query_grid)) if len(query_grid) else np.array([])
        return PredictionBundle(point_prediction_frame(query_grid, prediction), self.model_id)
