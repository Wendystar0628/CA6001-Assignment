"""Retained inference component, consolidated from cpt_budget.diagnostics.residual_adapter (v0.8-engineering)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from cpt_budget.models.distribution import point_prediction_frame
from cpt_budget.models.tabpfn_adapter import geometry_only
GUARDED_ALPHAS = (0.25, 0.5)

def guarded_residual_frame(query: pd.DataFrame, baseline: np.ndarray, residual_median: np.ndarray, alpha: float) -> pd.DataFrame:
    """Protect IDW's existing 1 MPa point labels while correcting magnitudes.

    This is an explicitly exploratory point-only adaptation. It cannot improve
    the baseline's point low-value labels: each output remains on the same side
    of ln(1 MPa)=0 as the baseline. No query truth is read, and native residual
    CDF/quantile columns are not retained after the nonlinear clipping.
    """
    alpha = float(alpha)
    if alpha not in GUARDED_ALPHAS:
        raise ValueError(f'Point-guarded residual alpha must be pre-registered in {GUARDED_ALPHAS}')
    geometry = geometry_only(query).reset_index(drop=True)
    base = np.asarray(baseline, dtype=float).reshape(-1)
    residual = np.asarray(residual_median, dtype=float).reshape(-1)
    if len(base) != len(geometry) or len(residual) != len(geometry) or (not np.isfinite(base).all()) or (not np.isfinite(residual).all()):
        raise ValueError('One finite baseline and residual median are required per query row')
    corrected = base + alpha * residual
    if not np.isfinite(corrected).all():
        raise ValueError('Residual correction must be finite')
    prediction = np.where(base < 0.0, np.minimum(corrected, -1e-09), np.maximum(corrected, 0.0))
    frame = point_prediction_frame(geometry, prediction)
    frame['baseline_log_qc'] = base
    frame['residual_pred_log_qc'] = residual
    frame['residual_alpha'] = alpha
    frame['unguarded_pred_log_qc'] = corrected
    frame['point_guard_applied'] = prediction != corrected
    return frame
