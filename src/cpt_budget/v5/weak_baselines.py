"""Honest low-information references, fitted only on revealed target CPTs.

These models remove information; they do not deliberately corrupt predictions.
The constant median is the empirical L1-optimal intercept in log(qc), matching
the standard DummyRegressor(strategy="median") rule. The depth median adds a
vertical profile while remaining invariant to horizontal coordinates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cpt_budget.models.base import validate_observed, validate_query_site
from cpt_budget.models.distribution import point_prediction_frame
from cpt_budget.types import PredictionBundle


class TargetConstantMedianPredictor:
    """One target-context median for every query location and depth."""

    model_id = "target_constant_median"

    def __init__(self, config: dict | None = None, artifact=None):
        self.observed = None
        self.median_log_qc = None

    def condition(self, observed: pd.DataFrame) -> None:
        self.observed = validate_observed(observed)
        self.median_log_qc = float(self.observed.log_qc.median())

    def predict(self, query_grid: pd.DataFrame) -> PredictionBundle:
        validate_query_site(query_grid, self.observed)
        values = np.full(len(query_grid), self.median_log_qc, dtype=float)
        return PredictionBundle(point_prediction_frame(query_grid, values), self.model_id)


class TargetDepthMedianPredictor:
    """A depth-wise median profile shared by all query locations."""

    model_id = "target_depth_median"

    def __init__(self, config: dict | None = None, artifact=None):
        self.observed = None
        self.depth_medians = None

    def condition(self, observed: pd.DataFrame) -> None:
        self.observed = validate_observed(observed)
        self.depth_medians = self.observed.groupby("depth_index").log_qc.median()

    def predict(self, query_grid: pd.DataFrame) -> PredictionBundle:
        validate_query_site(query_grid, self.observed)
        values = query_grid.depth_index.map(self.depth_medians)
        if values.isna().any():
            missing = query_grid.loc[values.isna(), "depth_index"].unique().tolist()
            raise ValueError(f"No revealed resistance at query depths: {missing}")
        return PredictionBundle(
            point_prediction_frame(query_grid, values.to_numpy(float)), self.model_id
        )


def run(output_dir=None, *, resume=True):
    """Evaluate both references on exactly the common 60 frozen states."""
    from pathlib import Path
    import json

    from cpt_budget.config import PROJECT_ROOT, write_json
    from cpt_budget.diagnostics.common import run_predictor

    output = Path(output_dir) if output_dir is not None else (
        PROJECT_ROOT / "runs/v5-exploration-20261014/weak_baselines"
    )
    results = {}
    for predictor_type in (TargetConstantMedianPredictor, TargetDepthMedianPredictor):
        destination = output / predictor_type.model_id
        result = run_predictor(lambda fold_id: predictor_type(), destination, resume=resume)
        # The shared evaluator retains its original v3 labels; record this run's
        # provenance without changing its frozen states or any old artifacts.
        for frame in result.values():
            frame["study_version"] = "0.5-exploration"
            frame["run_id"] = "v5-exploration-20261014"
        result["predictions"].to_parquet(destination / "predictions.parquet", index=False)
        result["metrics"].to_csv(destination / "metrics.csv", index=False)
        for state_path in (destination / "states").glob("*.parquet"):
            state = pd.read_parquet(state_path)
            state["study_version"] = "0.5-exploration"
            state["run_id"] = "v5-exploration-20261014"
            state.to_parquet(state_path, index=False)
        protocol = json.loads((destination / "evaluation_protocol.json").read_text(encoding="utf-8"))
        protocol.update(
            study_version="0.5-exploration", run_id="v5-exploration-20261014",
            model_id=predictor_type.model_id, reference_role="low_information_reference",
            model_selection="none", uncertainty="point_only",
        )
        write_json(destination / "evaluation_protocol.json", protocol)
        results[predictor_type.model_id] = result
    return results


if __name__ == "__main__":
    run()
