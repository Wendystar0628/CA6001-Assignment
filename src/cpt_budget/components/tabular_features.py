"""Retained inference component, consolidated from cpt_budget.v2.tabpfn (v0.8-engineering)."""
from __future__ import annotations
import numpy as np
from cpt_budget.components.spatial_features import FEATURE_COLUMNS_V2, build_features_v2

def feature_array_v2(observed, query, *, leave_location_out=False):
    frame = build_features_v2(observed, query, leave_location_out=leave_location_out)
    if list(frame.columns) != list(FEATURE_COLUMNS_V2):
        raise ValueError('Feature order differs from the spatial v2 schema')
    values = frame.to_numpy(dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError('Features may contain NaN but cannot contain infinity')
    return values
