"""Revealed-only spatial features whose fixed metre scales affect measurements.

The first sixteen features retain the v0.1 definition. Fixed-length kernels mix
observed values before TabPFN preprocessing, so absolute distance is not carried
solely by a constant site column or by a rescalable distance feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import FEATURE_COLUMNS, build_features, depth_values, idw_value


FEATURE_COLUMNS_V2 = [
    *FEATURE_COLUMNS,
    "kernel_10m_log_qc", "kernel_50m_log_qc",
    "kernel_10m_low_fraction", "kernel_50m_low_fraction",
    "idw_above_log_qc", "idw_below_log_qc",
    "idw_vertical_gradient", "idw_vertical_curvature",
]


def build_features_v2(
    observed: pd.DataFrame, query_grid: pd.DataFrame, *, leave_location_out: bool = False,
) -> pd.DataFrame:
    """Return 24 features in query order without reading any query measurement.

    Above/below refer to the adjacent *available depth grid* within the site.
    Their values are interpolated from revealed profiles only. At the upper or
    lower window boundary, the gradient is one-sided and curvature is missing;
    a missing neighbour never invents a zero-valued geological observation.
    Leave-location-out excludes the entire query CPT and coincident CPTs at all
    depths, including the vertical and physical-kernel summaries.
    """
    base = build_features(observed, query_grid, leave_location_out=leave_location_out)
    extra = np.full((len(query_grid), 8), np.nan, dtype=float)
    if query_grid.empty:
        return pd.concat([base, pd.DataFrame(extra, columns=FEATURE_COLUMNS_V2[16:], index=query_grid.index)], axis=1)
    observed_depth = depth_values(observed)
    query_depth = depth_values(query_grid)
    work = observed.assign(_depth=observed_depth)
    groups = {}
    adjacent = {}
    for site, site_rows in work.groupby("site_id", sort=False):
        depth_table = site_rows.groupby("depth_index", sort=False)._depth.agg(["min", "max"])
        if not np.allclose(depth_table["min"], depth_table["max"], atol=1e-9, rtol=0):
            raise ValueError("A depth index must identify the same depth across revealed profiles.")
        ordered = depth_table.sort_values("min").index.tolist()
        adjacent.update({(site, depth): (ordered[i - 1] if i else None,
                                        ordered[i + 1] if i + 1 < len(ordered) else None)
                         for i, depth in enumerate(ordered)})
    for key, rows in work.groupby(["site_id", "depth_index"], sort=False):
        rows = rows.sort_values("location_id", kind="stable")
        groups[key] = (rows[["x_m", "y_m"]].to_numpy(float),
                       rows.location_id.to_numpy(), rows.log_qc.to_numpy(float),
                       float(rows._depth.iloc[0]))

    def available(row, depth):
        group = groups.get((row.site_id, depth))
        if group is None:
            return np.array([]), np.array([]), np.nan
        coordinates, ids, values, z = group
        distances = np.linalg.norm(coordinates - [row.x_m, row.y_m], axis=1)
        allowed = (ids != row.location_id) & (distances > 1e-9) if leave_location_out else np.ones(len(ids), bool)
        return values[allowed], distances[allowed], z

    for i, row in enumerate(query_grid.itertuples(index=False)):
        values, distances, z = available(row, row.depth_index)
        if not len(values):
            continue
        if not np.isclose(z, query_depth[i], atol=1e-9, rtol=0):
            raise ValueError("Query and observed rows must share a common depth grid.")
        # Subtracting the minimum avoids underflow and leaves normalized
        # exp(-distance / length) weights unchanged, even at kilometre scales.
        for j, length_m in enumerate((10.0, 50.0)):
            weight = np.exp(-(distances - distances.min()) / length_m)
            weight /= weight.sum()
            extra[i, j] = weight @ values
            extra[i, j + 2] = weight @ (values < 0.0)  # physical 1 MPa event
        vertical = []
        for j, depth in enumerate(adjacent.get((row.site_id, row.depth_index), (None, None))):
            if depth is None:
                vertical.append((np.nan, np.nan))
                continue
            neighbour_values, neighbour_distances, neighbour_z = available(row, depth)
            nearest = np.argsort(neighbour_distances, kind="stable")[:3]
            value = idw_value(neighbour_values[nearest], neighbour_distances[nearest])
            extra[i, j + 4] = value
            vertical.append((value, neighbour_z))
        above, below = vertical
        center = float(base.iloc[i].idw_log_qc)
        if np.isfinite(above[0]) and np.isfinite(below[0]):
            dz_up, dz_down = z - above[1], below[1] - z
            extra[i, 6] = (below[0] - above[0]) / (dz_up + dz_down)
            extra[i, 7] = 2 * ((below[0] - center) / dz_down - (center - above[0]) / dz_up) / (dz_up + dz_down)
        elif np.isfinite(above[0]):
            extra[i, 6] = (center - above[0]) / (z - above[1])
        elif np.isfinite(below[0]):
            extra[i, 6] = (below[0] - center) / (below[1] - z)
    return pd.concat([base, pd.DataFrame(extra, columns=FEATURE_COLUMNS_V2[16:], index=query_grid.index)], axis=1)
