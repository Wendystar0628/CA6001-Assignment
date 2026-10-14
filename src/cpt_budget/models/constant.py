"""Retained inference component, consolidated from cpt_budget.v5.weak_baselines (v0.8-engineering)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from cpt_budget.models.base import validate_observed, validate_query_site
from cpt_budget.models.distribution import point_prediction_frame
from cpt_budget.types import PredictionBundle

class TargetConstantMedianPredictor:
    """One target-context median for every query location and depth."""
    model_id = 'target_constant_median'

    def __init__(self, config: dict | None=None, artifact=None):
        self.observed = None
        self.median_log_qc = None

    def condition(self, observed: pd.DataFrame) -> None:
        self.observed = validate_observed(observed)
        self.median_log_qc = float(self.observed.log_qc.median())

    def predict(self, query_grid: pd.DataFrame) -> PredictionBundle:
        validate_query_site(query_grid, self.observed)
        values = np.full(len(query_grid), self.median_log_qc, dtype=float)
        return PredictionBundle(point_prediction_frame(query_grid, values), self.model_id)
