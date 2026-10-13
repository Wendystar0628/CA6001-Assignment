"""Common predictor protocol and validation of the revealed observation set."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from ..types import PredictionBundle


@runtime_checkable
class Predictor(Protocol):
    model_id: str

    def condition(self, observed: pd.DataFrame) -> None: ...

    def predict(self, query_grid: pd.DataFrame) -> PredictionBundle: ...


def validate_observed(observed: pd.DataFrame) -> pd.DataFrame:
    required = {"site_id", "location_id", "depth_index", "x_m", "y_m", "log_qc"}
    missing = required.difference(observed.columns)
    if missing:
        raise ValueError(f"Observation set is missing {sorted(missing)}")
    if observed.empty:
        raise ValueError("At least one revealed CPT is required.")
    if observed["site_id"].nunique() != 1:
        raise ValueError("A target context must belong to exactly one site.")
    if observed.duplicated(["location_id", "depth_index"]).any():
        raise ValueError("Duplicate location/depth rows are not independent observations.")
    if not np.isfinite(observed[["x_m", "y_m", "log_qc"]].to_numpy(dtype=float)).all():
        raise ValueError("Observed coordinates and log_qc must be finite.")
    return observed.copy(deep=True)


def validate_query_site(query_grid: pd.DataFrame, observed: pd.DataFrame | None) -> None:
    if observed is None:
        raise RuntimeError("Call condition(observed) before predict().")
    if not query_grid["site_id"].isin(observed["site_id"].unique()).all():
        raise ValueError("Query sites must match the conditioned target site.")
