"""Official TabPFN v2 applied to leave-location-out IDW residuals.

The weight architecture is unchanged. The in-context label is
``ln(qc) - IDW_leave_location_out``. At query time the native residual
distribution is mapped to ``IDW_revealed + alpha * residual``. This is an
affine distribution construction, not a claim that shrinkage is calibrated.
Alpha selection belongs to the external, registered validation procedure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cpt_budget.features import idw_value
from cpt_budget.models.base import validate_observed, validate_query_site
from cpt_budget.models.distribution import (
    QUANTILES, THRESHOLDS, distribution_prediction_frame, point_prediction_frame,
)
from cpt_budget.models.idw import IDWPredictor
from cpt_budget.models.tabpfn_adapter import geometry_only, make_estimator
from cpt_budget.types import PredictionBundle
from cpt_budget.v2.tabpfn import feature_array_v2


REGISTERED_ALPHAS = (0.0, 0.25, 0.5, 1.0)
GUARDED_ALPHAS = (0.25, 0.5)


def _validate_alpha(alpha: float) -> float:
    value = float(alpha)
    if value not in REGISTERED_ALPHAS:
        raise ValueError(f"Residual alpha must be pre-registered in {REGISTERED_ALPHAS}")
    return value


def leave_location_out_idw(observed: pd.DataFrame, config: dict | None = None) -> np.ndarray:
    """Predict each context row from other complete, noncoincident locations.

    Exclusion is by location and physical coordinate, never by depth row. All
    same-coordinate CPTs are excluded too, matching the spatial feature policy.
    A missing independent reference at any depth is an error, not a zero target.
    """
    revealed = validate_observed(observed)
    settings = (config or {}).get("idw", {})
    neighbors = int(settings.get("nearest_neighbors", 3))
    power = float(settings.get("power", 2.0))
    if neighbors < 1 or not np.isfinite(power) or power <= 0:
        raise ValueError("IDW needs positive neighbor count and finite distance power")
    groups = {
        key: rows.sort_values("location_id", kind="stable")
        for key, rows in revealed.groupby(["site_id", "depth_index"], sort=False)
    }
    values = []
    for row in geometry_only(revealed).itertuples(index=False):
        candidates = groups[(row.site_id, row.depth_index)]
        distance = np.linalg.norm(
            candidates[["x_m", "y_m"]].to_numpy(float) - [row.x_m, row.y_m], axis=1)
        mask = ((candidates.location_id.to_numpy() != row.location_id)
                & (distance > 1e-9))
        distance = distance[mask]
        targets = candidates.log_qc.to_numpy(float)[mask]
        nearest = np.argsort(distance, kind="stable")[:neighbors]
        values.append(idw_value(targets[nearest], distance[nearest], power=power))
    predictions = np.asarray(values, dtype=float)
    if not np.isfinite(predictions).all():
        raise ValueError("Residual context requires another noncoincident CPT at every depth")
    return predictions


def residual_context(observed: pd.DataFrame, config: dict | None = None) -> dict[str, np.ndarray]:
    """Expose deterministic, label-safe context construction for diagnostics."""
    revealed = validate_observed(observed)
    reference = leave_location_out_idw(revealed, config)
    features = feature_array_v2(revealed, geometry_only(revealed), leave_location_out=True)
    targets = revealed.log_qc.to_numpy(float) - reference
    return {"X_support": features, "y_residual": targets, "loo_baseline_log_qc": reference}


def reconstruct_residual_output(
    full: dict[str, Any] | None,
    query: pd.DataFrame,
    baseline_log_qc: np.ndarray,
    *,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Reconstruct query-specific physical quantiles/CDF from native residuals.

    For positive alpha, F_Y(t)=F_R((t-baseline)/alpha). The official returned
    criterion already works in input target space (here residual log units),
    so no second target normalization may be applied. Alpha zero deliberately
    returns only point columns, because IDW has no native uncertainty model.
    """
    alpha = _validate_alpha(alpha)
    query = geometry_only(query).reset_index(drop=True)
    baseline = np.asarray(baseline_log_qc, dtype=float).reshape(-1)
    if len(baseline) != len(query) or not np.isfinite(baseline).all():
        raise ValueError("One finite baseline value is required per query row")
    if alpha == 0:
        frame = point_prediction_frame(query, baseline)
    else:
        if full is None:
            raise ValueError("A positive alpha requires the native residual distribution")
        import torch
        logits = full["logits"]
        if not torch.is_tensor(logits):
            raise TypeError("Native residual logits must be a tensor")
        residual_quantiles = np.column_stack(full["quantiles"])
        if residual_quantiles.shape != (len(query), len(QUANTILES)):
            raise ValueError("Native residual quantiles have an unexpected shape")
        criterion = full["criterion"].to(device=logits.device, dtype=logits.dtype)
        residual_thresholds = (np.log(THRESHOLDS)[None, :] - baseline[:, None]) / alpha
        thresholds = torch.as_tensor(residual_thresholds, device=logits.device, dtype=logits.dtype)
        with torch.no_grad():
            probabilities = criterion.cdf(logits, thresholds).detach().cpu().numpy()
        quantiles = baseline[:, None] + alpha * residual_quantiles
        frame = distribution_prediction_frame(query, quantiles, probabilities)
        frame["residual_pred_log_qc"] = residual_quantiles[:, 3]
    frame["baseline_log_qc"] = baseline
    frame["residual_alpha"] = alpha
    return frame


class BaselineResidualTabPFNPredictor:
    """Inference-only TabPFN residual adaptation with the normal predictor API.

    ``artifact=None`` loads the official pretrained v2 weights. An explicit
    artifact is passed to the existing official-model loader; it must be
    documented separately by the experiment that supplies it. ``condition``
    never changes weights. ``predict_variants`` reuses each full native forward
    pass across all requested, pre-registered alphas.
    """
    def __init__(self, config=None, *, baseline="idw", artifact=None,
                 model_id="tabpfn_idw_residual", alpha=1.0):
        if baseline != "idw":
            raise ValueError("This registered diagnostic supports only the IDW baseline")
        self.config = config or {}
        self.artifact = Path(artifact) if artifact is not None else None
        self.model_id = model_id
        self.alpha = _validate_alpha(alpha)
        self._baseline = IDWPredictor(self.config)
        self._estimator = None
        self._observed = None
        self._context = None
        self._needs_fit = True
        self.query_profiles_per_batch = min(8, int(
            self.config.get("episodes", {}).get("query_profiles_per_batch", 8)))
        if self.query_profiles_per_batch < 1:
            raise ValueError("query_profiles_per_batch must be positive")
        self.metadata_ = {
            "target": "ln(qc_mpa) minus leave-location-out IDW ln(qc_mpa)",
            "context_exclusion": "whole location and all coincident locations",
            "features": "revealed-only original-qc spatial features, 24 dimensions",
            "query_distribution": "revealed IDW + alpha * native residual distribution",
            "alpha_zero": "IDW point prediction only; no native probabilities or intervals",
            "calibration_claim": False,
            "source_weight_artifact": str(self.artifact) if self.artifact else "official TabPFN v2",
        }

    def condition(self, observed: pd.DataFrame) -> None:
        self._observed = validate_observed(observed)
        self._baseline.condition(self._observed)
        self._context = residual_context(self._observed, self.config)
        self._needs_fit = True

    def _fit_if_needed(self) -> None:
        if not self._needs_fit:
            return
        if self._estimator is None:
            self._estimator = make_estimator(self.config, self.artifact)
        self._estimator.fit(self._context["X_support"], self._context["y_residual"])
        for model in self._estimator.models_:
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        self._needs_fit = False

    def predict(self, query_grid: pd.DataFrame) -> PredictionBundle:
        return self.predict_variants(query_grid, alphas=(self.alpha,))[self.alpha]

    def predict_variants(self, query_grid: pd.DataFrame,
                         alphas=REGISTERED_ALPHAS) -> dict[float, PredictionBundle]:
        """Return all affine variants from one residual inference per query batch."""
        alphas = tuple(dict.fromkeys(_validate_alpha(a) for a in alphas))
        if not alphas:
            raise ValueError("At least one registered alpha is required")
        validate_query_site(query_grid, self._observed)
        query = geometry_only(query_grid).reset_index(drop=True)
        baseline = self._baseline.predict(query).frame.pred_log_qc.to_numpy(float)
        results = {}
        if 0.0 in alphas:
            frame = reconstruct_residual_output(None, query, baseline, alpha=0.0)
            results[0.0] = PredictionBundle(frame, self.model_id)
        positive = [a for a in alphas if a > 0]
        if not positive:
            return results
        if query.empty:
            for alpha in positive:
                frame = distribution_prediction_frame(query, np.empty((0, 7)), np.empty((0, 3)))
                frame["baseline_log_qc"] = baseline
                frame["residual_pred_log_qc"] = np.array([], dtype=float)
                frame["residual_alpha"] = alpha
                results[alpha] = PredictionBundle(frame, self.model_id)
            return results
        self._fit_if_needed()
        features = feature_array_v2(self._observed, query)
        ids = list(query.location_id.drop_duplicates())
        pieces = {a: [] for a in positive}
        position, batch_size = 0, self.query_profiles_per_batch
        import torch
        while position < len(ids):
            selected = ids[position:position + batch_size]
            indices = np.flatnonzero(query.location_id.isin(selected).to_numpy())
            try:
                with torch.no_grad():
                    full = self._estimator.predict(
                        features[indices], output_type="full", quantiles=list(QUANTILES))
                batch_pieces = {}
                for alpha in positive:
                    piece = reconstruct_residual_output(full, query.iloc[indices], baseline[indices], alpha=alpha)
                    piece["_row_order"] = indices
                    batch_pieces[alpha] = piece
            except torch.OutOfMemoryError:
                if batch_size <= 1:
                    raise
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
                continue
            for alpha, piece in batch_pieces.items():
                pieces[alpha].append(piece)
            position += len(selected)
        for alpha, frames in pieces.items():
            frame = pd.concat(frames).sort_values("_row_order").drop(columns="_row_order").reset_index(drop=True)
            results[alpha] = PredictionBundle(frame, self.model_id)
        return results


def guarded_residual_frame(
    query: pd.DataFrame,
    baseline: np.ndarray,
    residual_median: np.ndarray,
    alpha: float,
) -> pd.DataFrame:
    """Protect IDW's existing 1 MPa point labels while correcting magnitudes.

    This is an explicitly exploratory point-only adaptation. It cannot improve
    the baseline's point low-value labels: each output remains on the same side
    of ln(1 MPa)=0 as the baseline. No query truth is read, and native residual
    CDF/quantile columns are not retained after the nonlinear clipping.
    """
    alpha = float(alpha)
    if alpha not in GUARDED_ALPHAS:
        raise ValueError(f"Point-guarded residual alpha must be pre-registered in {GUARDED_ALPHAS}")
    geometry = geometry_only(query).reset_index(drop=True)
    base = np.asarray(baseline, dtype=float).reshape(-1)
    residual = np.asarray(residual_median, dtype=float).reshape(-1)
    if (len(base) != len(geometry) or len(residual) != len(geometry)
            or not np.isfinite(base).all() or not np.isfinite(residual).all()):
        raise ValueError("One finite baseline and residual median are required per query row")
    corrected = base + alpha * residual
    if not np.isfinite(corrected).all():
        raise ValueError("Residual correction must be finite")
    prediction = np.where(base < 0.0, np.minimum(corrected, -1e-9), np.maximum(corrected, 0.0))
    frame = point_prediction_frame(geometry, prediction)
    frame["baseline_log_qc"] = base
    frame["residual_pred_log_qc"] = residual
    frame["residual_alpha"] = alpha
    frame["unguarded_pred_log_qc"] = corrected
    frame["point_guard_applied"] = prediction != corrected
    return frame


class PointGuardedResidualPredictor:
    """Deployable wrapper preserving IDW point labels, with no native risk CDF.

    Risk probabilities, if required by the experiment, must be supplied by its
    common external calibration procedure. Protecting the existing point label
    does not imply that an externally calibrated risk label is also unchanged.
    """
    def __init__(self, config=None, *, baseline="idw", artifact=None,
                 model_id="tabpfn_idw_residual_point_guarded", alpha=0.25):
        alpha = float(alpha)
        if alpha not in GUARDED_ALPHAS:
            raise ValueError(f"Point-guarded residual alpha must be pre-registered in {GUARDED_ALPHAS}")
        self.model_id = model_id
        self.alpha = alpha
        self._residual = BaselineResidualTabPFNPredictor(
            config, baseline=baseline, artifact=artifact, alpha=1.0)
        self.metadata_ = {
            **self._residual.metadata_,
            "adaptation_scope": "exploratory candidate added after inspecting unguarded residual diagnostics",
            "output": "point prediction only; no native CDF or intervals after clipping",
            "point_rule": "IDW + alpha * residual median, clipped to the IDW side of ln(1 MPa)=0",
            "negative_side_maximum": -1e-9,
            "registered_alphas": list(GUARDED_ALPHAS),
            "low_value_improvement_claim": False,
            "protected_quantity": "existing IDW point label only, not an externally calibrated risk decision",
        }

    def condition(self, observed: pd.DataFrame) -> None:
        self._residual.condition(observed)

    def predict(self, query_grid: pd.DataFrame) -> PredictionBundle:
        return self.predict_variants(query_grid, alphas=(self.alpha,))[self.alpha]

    def predict_variants(self, query_grid: pd.DataFrame,
                         alphas=GUARDED_ALPHAS) -> dict[float, PredictionBundle]:
        alphas = tuple(dict.fromkeys(float(a) for a in alphas))
        if not alphas or any(a not in GUARDED_ALPHAS for a in alphas):
            raise ValueError(f"Point-guarded residual alphas must be pre-registered in {GUARDED_ALPHAS}")
        native = self._residual.predict(query_grid).frame
        return {
            alpha: PredictionBundle(guarded_residual_frame(
                native, native.baseline_log_qc.to_numpy(float),
                native.residual_pred_log_qc.to_numpy(float), alpha), self.model_id)
            for alpha in alphas
        }
