"""Shared neighbourhood features, built only from the currently revealed CPTs."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "depth_m",
    *[f"neighbor_{i}_log_qc" for i in range(1, 4)],
    *[f"neighbor_{i}_log_distance" for i in range(1, 4)],
    *[f"neighbor_{i}_valid" for i in range(1, 4)],
    "neighbor_mean_log_qc", "neighbor_std_log_qc", "neighbor_median_log_qc",
    "idw_log_qc", "support_count", "log_geometry_scale",
]


def depth_values(frame: pd.DataFrame) -> np.ndarray:
    column = "z_center_m" if "z_center_m" in frame else "depth_m"
    if column not in frame:
        raise ValueError("The grid needs z_center_m (or depth_m).")
    values = frame[column].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Depth coordinates must be finite.")
    return values


def _check(frame: pd.DataFrame, observed: bool = False) -> None:
    required = {"site_id", "location_id", "depth_index", "x_m", "y_m"}
    required.add("log_qc" if observed else "geometry_scale_m")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing feature input columns: {sorted(missing)}")
    if frame.duplicated(["site_id", "location_id", "depth_index"]).any():
        raise ValueError("Each observed/query location must have one row per depth index.")
    numeric = ["x_m", "y_m"] + (["log_qc"] if observed else ["geometry_scale_m"])
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError(f"Feature input must be finite: {numeric}")
    if not observed and (frame["geometry_scale_m"] <= 0).any():
        raise ValueError("The frozen geometry scale must be strictly positive.")


def idw_value(values: np.ndarray, distances: np.ndarray, *, power: float = 2.0) -> float:
    """Interpolate log resistance; coincident supports receive all the weight."""
    if len(values) == 0:
        return float("nan")
    coincident = distances <= 1e-9
    if coincident.any():
        return float(values[coincident].mean())
    # Relative distances give the same normalized weights without huge powers.
    weights = (distances.min() / distances) ** power
    return float(np.dot(weights, values) / weights.sum())


def build_features(
    observed: pd.DataFrame, query_grid: pd.DataFrame, *, leave_location_out: bool = False
) -> pd.DataFrame:
    """Return the blueprint's 16 features in the original query row order.

    Context features exclude the whole query location and any coincident location.
    The near-group boundary is a splitting constraint, not a reason to remove
    physically distinct revealed neighbours. Query labels are never read.
    """
    _check(observed, observed=True)
    _check(query_grid)
    output = np.full((len(query_grid), len(FEATURE_COLUMNS)), np.nan, dtype=float)
    output[:, 0] = depth_values(query_grid)
    output[:, 7:10] = 0.0
    output[:, 14] = 0.0
    output[:, 15] = np.log1p(query_grid["geometry_scale_m"].to_numpy(dtype=float))
    groups = {
        key: value.sort_values("location_id", kind="stable")
        for key, value in observed.groupby(["site_id", "depth_index"], sort=False)
    }
    for i, row in enumerate(query_grid.itertuples(index=False)):
        support = groups.get((row.site_id, row.depth_index))
        if support is None or support.empty:
            continue
        coords = support[["x_m", "y_m"]].to_numpy(dtype=float)
        distances = np.linalg.norm(coords - [row.x_m, row.y_m], axis=1)
        allowed = np.ones(len(support), dtype=bool)
        if leave_location_out:
            allowed &= support["location_id"].to_numpy() != row.location_id
            allowed &= distances > 1e-9
        distances = distances[allowed]
        values = support["log_qc"].to_numpy(dtype=float)[allowed]
        if len(values) == 0:
            continue
        nearest = np.argsort(distances, kind="stable")[:3]
        n = len(nearest)
        output[i, 1:1+n] = values[nearest]
        output[i, 4:4+n] = np.log1p(distances[nearest] / row.geometry_scale_m)
        output[i, 7:7+n] = 1.0
        output[i, 10:13] = values.mean(), values.std(ddof=0), np.median(values)
        output[i, 13] = idw_value(values[nearest], distances[nearest])
        output[i, 14] = len(values)
    return pd.DataFrame(output, columns=FEATURE_COLUMNS, index=query_grid.index)
