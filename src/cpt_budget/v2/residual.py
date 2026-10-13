"""Source-trained IDW correction with validation-selected shrinkage and fallback."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ..config import write_json
from ..evaluate import evaluate
from ..models.base import validate_observed, validate_query_site
from ..models.distribution import point_prediction_frame
from ..models.idw import IDWPredictor
from ..models.tabpfn_train import _check_episode
from ..types import PredictionBundle
from .features import FEATURE_COLUMNS_V2, build_features_v2


ALPHA_CANDIDATES = (0.0, 0.25, 0.5, 1.0)
GUARDRAILS = {"h_point_mae_m": 0.05, "recall": 0.02, "false_positive_rate": 0.02}
_METRICS = ["log_mae", "h_point_mae_m", "recall", "false_positive_rate"]


def _baseline(features, observed, query, config):
    settings = config.get("idw", {})
    if int(settings.get("nearest_neighbors", 3)) == 3 and float(settings.get("power", 2.0)) == 2.0:
        return features.idw_log_qc.to_numpy(float)
    predictor = IDWPredictor(config)
    predictor.condition(observed)
    return predictor.predict(query).frame.pred_log_qc.to_numpy(float)


def _episode_data(dataset, episodes, config):
    counts = episodes.groupby("site_id").size()
    features, labels, weights, baselines, metadata = [], [], [], [], []
    for record in episodes.to_dict("records"):
        support_ids, query_ids = _check_episode(dataset, record)
        observed, query = dataset.observed(support_ids), dataset.query(query_ids)
        truth = dataset.truth(query_ids)
        keys = ["site_id", "location_id", "depth_index"]
        aligned = query[keys].merge(truth[keys + ["log_qc", "qc_mpa"]], on=keys,
                                    how="left", validate="one_to_one", sort=False)
        if not np.isfinite(aligned[["log_qc", "qc_mpa"]].to_numpy(float)).all():
            raise ValueError("Every residual query row needs finite training/evaluation truth.")
        x = build_features_v2(observed, query)
        base = _baseline(x, observed, query, config)
        if not np.isfinite(base).all():
            raise ValueError("IDW must have an observed same-depth neighbour for every query row.")
        depth_count = query.groupby("location_id").depth_index.transform("size").to_numpy()
        weight = 1.0 / (counts[record["site_id"]] * query.location_id.nunique() * depth_count)
        meta = aligned.rename(columns={"log_qc": "truth_log_qc", "qc_mpa": "truth_qc_mpa"})
        meta["episode_id"] = str(record["episode_id"])
        metadata.append(meta)
        features.append(x)
        labels.append(aligned.log_qc.to_numpy(float) - base)
        baselines.append(base)
        weights.append(weight)
    weight = np.concatenate(weights)
    weight *= len(weight) / weight.sum()
    return (pd.concat(features, ignore_index=True), np.concatenate(labels), weight,
            np.concatenate(baselines), pd.concat(metadata, ignore_index=True))


def _candidate_metrics(meta, baseline, correction, alpha, config):
    rows = []
    predictions = point_prediction_frame(meta, baseline + alpha * correction)
    for (site, episode), indices in meta.groupby(["site_id", "episode_id"], sort=False).groups.items():
        metrics = evaluate(predictions.loc[indices], meta.loc[indices], config)
        rows.append({"site_id": str(site), "episode_id": str(episode), "alpha": float(alpha),
                     **{key: metrics[key] for key in _METRICS}})
    frame = pd.DataFrame(rows)
    sites = frame.groupby("site_id")[_METRICS].mean()
    means = sites.mean()
    macro = {key: float(means[key]) if pd.notna(means[key]) else None for key in _METRICS}
    return {"alpha": float(alpha), **macro}, rows


def _choose_alpha(candidates):
    """Apply frozen macro-site guardrails; undefined class metrics are inapplicable."""
    baseline = next(row for row in candidates if row["alpha"] == 0.0)
    for candidate in candidates:
        failures = []
        if not np.isfinite(candidate["log_mae"]):
            failures.append("nonfinite_log_mae")
        for metric, tolerance in GUARDRAILS.items():
            reference, value = baseline[metric], candidate[metric]
            if reference is None:
                continue
            if value is None or not np.isfinite(value):
                failures.append(f"{metric}_unavailable")
            elif metric == "recall" and value < reference - tolerance - 1e-12:
                failures.append("recall_drop_gt_0.02")
            elif metric != "recall" and value > reference + tolerance + 1e-12:
                failures.append(f"{metric}_increase_gt_{tolerance:g}")
        candidate["guardrails_passed"] = not failures
        candidate["guardrail_failures"] = failures
    eligible = [row for row in candidates if row["guardrails_passed"]]
    if not eligible:
        raise RuntimeError("The zero-correction baseline must be a valid fallback.")
    selected = min(eligible, key=lambda row: (row["log_mae"], row["alpha"]))
    fallback = selected["alpha"] == 0.0
    if fallback and not any(row["alpha"] > 0 and row["guardrails_passed"] for row in candidates):
        reason = "all_nonzero_candidates_failed_guardrails"
    elif fallback:
        reason = "no_eligible_nonzero_candidate_improved_log_mae"
    else:
        reason = None
    return selected, reason


def fit_residual(dataset, train_episodes, validation_episodes, config, output_dir) -> dict:
    """Train on whole-CPT held-out residuals; select alpha using validation only.

    Early stopping chooses a residual tree count by weighted validation MAE;
    alpha is then chosen with H/Recall/FPR guardrails at the frozen 1 MPa event.
    This source-episode selection does not constitute the later replay promotion
    gate. The returned model emits point predictions only, even when alpha=0.
    """
    train, validation = pd.DataFrame(train_episodes), pd.DataFrame(validation_episodes)
    for role, episodes in (("train", train), ("validation", validation)):
        if episodes.empty or "role" not in episodes or not episodes.role.eq(role).all():
            raise ValueError(f"Residual fitting requires explicit, nonempty {role}-role episodes.")
        if episodes.duplicated(["site_id", "episode_id"]).any():
            raise ValueError("Duplicate episodes would distort source-site weighting.")
    train_sites, validation_sites = set(train.site_id), set(validation.site_id)
    if train_sites & validation_sites:
        raise ValueError("Training and validation sites must be disjoint.")
    if "investigation_group" in dataset.sites:
        site_groups = dataset.sites.set_index("site_id").investigation_group
        if set(site_groups.loc[list(train_sites)]) & set(site_groups.loc[list(validation_sites)]):
            raise ValueError("Training and validation investigation groups must be disjoint.")
    combined = pd.concat([train, validation], ignore_index=True)
    if "fold_id" in combined and combined.fold_id.nunique() != 1:
        raise ValueError("Fit one residual artifact per outer fold.")
    if float(config.get("data", {}).get("tau_mpa", 1.0)) != 1.0:
        raise ValueError("v0.2 residual selection is registered at the 1 MPa event.")
    x_train, y_train, w_train, _, train_meta = _episode_data(dataset, train, config)
    x_valid, y_valid, w_valid, valid_base, valid_meta = _episode_data(dataset, validation, config)
    settings = config.get("v2", {}).get("residual", {})
    model = XGBRegressor(
        objective="reg:squarederror", tree_method="hist", device="cpu",
        max_depth=int(settings.get("max_depth", 3)),
        learning_rate=float(settings.get("learning_rate", 0.03)),
        n_estimators=max(1, min(500, int(settings.get("n_estimators", 300)))),
        early_stopping_rounds=max(1, int(settings.get("early_stopping_rounds", 30))),
        eval_metric="mae", n_jobs=max(1, int(settings.get("n_jobs", 4))),
        random_state=int(config.get("split", {}).get("seed", 20261013)),
        reg_lambda=float(settings.get("reg_lambda", 10.0)),
        min_child_weight=float(settings.get("min_child_weight", 5.0)), verbosity=0,
    )
    model.fit(x_train, y_train, sample_weight=w_train, eval_set=[(x_valid, y_valid)],
              sample_weight_eval_set=[w_valid], verbose=False)
    correction = model.predict(x_valid)
    candidates, episode_metrics = [], []
    for alpha in ALPHA_CANDIDATES:
        candidate, detail = _candidate_metrics(valid_meta, valid_base, correction, alpha, config)
        candidates.append(candidate)
        episode_metrics.extend(detail)
    selected, reason = _choose_alpha(candidates)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path, artifact_path = output_dir / "residual_trees.ubj", output_dir / "idw_residual.json"
    model.save_model(model_path)
    detail_path = output_dir / "residual_validation_candidates.csv"
    pd.DataFrame(episode_metrics).to_csv(detail_path, index=False)
    summary = {
        "model_id": "idw_residual", "schema_version": "v0.2", "artifact_path": str(artifact_path.resolve()),
        "model_filename": model_path.name, "feature_columns": FEATURE_COLUMNS_V2,
        "alpha": selected["alpha"], "fallback_to_idw": selected["alpha"] == 0.0,
        "fallback_reason": reason, "selection_candidates": candidates,
        "selection_metric": "mean_site(mean_episode(mean_location(mean_depth_absolute_log_error)))",
        "guardrails": GUARDRAILS, "guardrail_scope": "site_macro_source_validation_episodes",
        "class_metric_missingness": "class_absent_is_inapplicable; average_applicable_episodes_then_sites",
        "selection_evidence_path": str(detail_path.resolve()), "selected_validation_metrics": selected,
        "fold_id": int(combined.fold_id.iloc[0]) if "fold_id" in combined else None,
        "train_sites": sorted(str(site) for site in train_sites),
        "validation_sites": sorted(str(site) for site in validation_sites),
        "train_rows": len(train_meta), "validation_rows": len(valid_meta),
        "weighting": "equal_site_then_episode_then_location_then_depth; mean_weight=1",
        "best_iteration": int(model.best_iteration), "best_residual_validation_mae": float(model.best_score),
        "training_parameters": model.get_params(), "evaluation_history": model.evals_result(),
        "idw_settings": config.get("idw", {}), "predictive_distribution": None,
        "uncertainty_note": "Point model only. Alpha zero restores IDW, never zero observation variance.",
    }
    write_json(artifact_path, summary)
    return summary


class ResidualPredictor:
    model_id = "idw_residual"

    def __init__(self, config: dict | None = None, artifact: Path | None = None):
        self.config = config or {}
        if artifact is None:
            raise ValueError("ResidualPredictor requires a source-trained fold artifact.")
        artifact = Path(artifact)
        self.metadata = json.loads(artifact.read_text(encoding="utf-8"))
        if self.metadata.get("feature_columns") != FEATURE_COLUMNS_V2:
            raise ValueError("Residual artifact has incompatible feature names/order.")
        self.alpha = float(self.metadata["alpha"])
        if self.alpha not in ALPHA_CANDIDATES:
            raise ValueError("Residual alpha is not in the registered candidate set.")
        stored = self.metadata.get("idw_settings", {})
        current = self.config.get("idw", {})
        for name, default in (("nearest_neighbors", 3), ("power", 2.0)):
            if stored.get(name, default) != current.get(name, default):
                raise ValueError("Residual artifact and inference IDW settings must match.")
        self.regressor = None
        if self.alpha:
            self.regressor = XGBRegressor(n_jobs=int(self.config.get("v2", {}).get("residual", {}).get("n_jobs", 4)), device="cpu")
            self.regressor.load_model(artifact.parent / self.metadata["model_filename"])
            if self.regressor.get_booster().feature_names != FEATURE_COLUMNS_V2:
                raise ValueError("Residual trees have incompatible feature names/order.")
        self.idw = IDWPredictor(self.config)
        self.observed = None

    def condition(self, observed: pd.DataFrame) -> None:
        self.observed = validate_observed(observed)
        self.idw.condition(observed)

    def predict(self, query_grid: pd.DataFrame) -> PredictionBundle:
        validate_query_site(query_grid, self.observed)
        if self.alpha == 0 or query_grid.empty:
            frame = self.idw.predict(query_grid).frame
        else:
            features = build_features_v2(self.observed, query_grid)
            baseline = _baseline(features, self.observed, query_grid, self.config)
            frame = point_prediction_frame(query_grid, baseline + self.alpha * self.regressor.predict(features))
        return PredictionBundle(frame, self.model_id)
