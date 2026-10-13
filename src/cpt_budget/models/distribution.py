"""Decode log(qc) predictions without confusing medians with expectations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
QUANTILE_COLUMNS = ("q05_log", "q10_log", "q25_log", "q50_log", "q75_log", "q90_log", "q95_log")
THRESHOLDS = (0.5, 1.0, 2.0)
PROBABILITY_COLUMNS = ("p_low_0_5", "p_low_1", "p_low_2")
QUERY_COLUMNS = (
    "site_id", "location_id", "profile_id", "near_group_id", "depth_index",
    "z_center_m", "depth_m", "x_m", "y_m", "geometry_scale_m",
)


def point_prediction_frame(query_grid: pd.DataFrame, pred_log_qc: np.ndarray) -> pd.DataFrame:
    values = np.asarray(pred_log_qc, dtype=float).reshape(-1)
    if len(values) != len(query_grid) or not np.isfinite(values).all():
        raise ValueError("One finite log prediction is required per query row.")
    frame = query_grid[[c for c in QUERY_COLUMNS if c in query_grid]].copy()
    frame["pred_log_qc"] = values
    with np.errstate(over="raise"):
        frame["pred_qc_mpa"] = np.exp(values)
    return frame


def distribution_prediction_frame(
    query_grid: pd.DataFrame, quantiles: np.ndarray, probabilities: np.ndarray
) -> pd.DataFrame:
    """Store native distribution quantiles and native CDF values at 3 thresholds."""
    quantiles = np.asarray(quantiles, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if quantiles.shape != (len(query_grid), 7) or probabilities.shape != (len(query_grid), 3):
        raise ValueError("Expected quantiles [n_query,7] and probabilities [n_query,3].")
    if not np.isfinite(quantiles).all() or not np.isfinite(probabilities).all():
        raise ValueError("Distribution outputs must be finite.")
    if (np.diff(quantiles, axis=1) < -1e-7).any():
        raise ValueError("Native prediction quantiles must be ordered.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("CDF probabilities must lie in [0,1].")
    if (np.diff(probabilities, axis=1) < -1e-7).any():
        raise ValueError("Low-resistance probability must increase with its threshold.")
    frame = point_prediction_frame(query_grid, quantiles[:, 3])
    frame[list(QUANTILE_COLUMNS)] = quantiles
    frame[list(PROBABILITY_COLUMNS)] = probabilities
    return frame


def gaussian_prediction_frame(
    query_grid: pd.DataFrame, mean: np.ndarray, std: np.ndarray
) -> pd.DataFrame:
    mean = np.asarray(mean, dtype=float).reshape(-1)
    std = np.asarray(std, dtype=float).reshape(-1)
    if len(mean) != len(query_grid) or len(std) != len(query_grid):
        raise ValueError("Mean and standard deviation must match query rows.")
    if not np.isfinite(std).all() or (std < 0).any():
        raise ValueError("Standard deviations must be finite and nonnegative.")
    quantiles = mean[:, None] + std[:, None] * ndtri(QUANTILES)
    delta = np.log(THRESHOLDS)[None, :] - mean[:, None]
    # At zero variance, P(Y < threshold) is the strict indicator used by H_point.
    probabilities = np.empty_like(delta)
    nonzero = std > 0
    probabilities[nonzero] = ndtr(delta[nonzero] / std[nonzero, None])
    probabilities[~nonzero] = (delta[~nonzero] > 0).astype(float)
    return distribution_prediction_frame(query_grid, quantiles, probabilities)


def thickness_summary(frame: pd.DataFrame, *, threshold_mpa: float = 1.0, step_m: float = 0.25) -> pd.DataFrame:
    """Per-location point and expected thickness; no joint interval is inferred."""
    if threshold_mpa <= 0 or step_m <= 0:
        raise ValueError("Threshold and depth step must be positive.")
    keys = [c for c in ("site_id", "location_id") if c in frame]
    work = frame[keys].copy()
    work["h_point_m"] = step_m * (frame["pred_qc_mpa"] < threshold_mpa)
    probability = dict(zip(THRESHOLDS, PROBABILITY_COLUMNS)).get(float(threshold_mpa))
    if probability is not None and probability in frame:
        work["h_expected_m"] = step_m * frame[probability]
    return work.groupby(keys, sort=False, as_index=False).sum()
