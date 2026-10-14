"""Small-support geometric regressors, distinct from existing GP/IDW tuning.

Three fixed constructions use revealed profiles and horizontal coordinates only.
The driver exports all frozen states; promotion and risk calibration are external.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, linprog
from scipy.spatial import ConvexHull, Delaunay
from threadpoolctl import threadpool_limits

from cpt_budget.config import PROJECT_ROOT, write_json
from cpt_budget.models.base import validate_observed, validate_query_site
from cpt_budget.models.distribution import point_prediction_frame
from cpt_budget.types import PredictionBundle

VARIANTS = ("mls_affine", "lad_affine", "simplex_linear")
SETTINGS = {
    "scale": "distance to min(4,n)-th nearest support location, at least 1 metre",
    "geometric_weights": "normalized exp(-0.5*(distance/scale)^2)",
    "mls_slope_l2": 0.1,
    "lad_slope_l1": 0.05,
    "regression_bounds": "clip each depth to revealed min/max at that depth",
    "simplex_outside_hull": "nearest convex-hull edge projection with linear interpolation",
    "collinear_simplex": "principal-axis piecewise linear interpolation, endpoint hold",
    "coincident_support": "average full profiles at exactly identical xy",
    "tuning": "none; recorded before running frozen validation",
    "probability": "point predictions only; common external calibration",
    "sources": [
        "https://www.ams.org/mcom/1998-67-224/S0025-5718-98-00974-0/S0025-5718-98-00974-0.pdf",
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html",
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html",
    ],
}


def local_design(xy, query):
    """Dimensionless centered affine design and normalized locality weights."""
    delta = np.asarray(xy, float) - np.asarray(query, float)
    distance = np.linalg.norm(delta, axis=1)
    scale = max(float(np.sort(distance)[min(4, len(distance)) - 1]), 1.0)
    weight = np.exp(-0.5 * (distance / scale) ** 2)
    weight /= weight.sum()
    return np.column_stack([np.ones(len(xy)), delta / scale]), weight


def affine_profiles(xy, profiles, query, *, loss="l2", penalty=None):
    """Fit every depth independently; query is the local intercept."""
    xy, profiles = np.asarray(xy, float), np.asarray(profiles, float)
    if profiles.ndim != 2 or len(xy) != len(profiles) or not len(xy):
        raise ValueError("Expected nonempty support-by-depth profiles")
    design, weight = local_design(xy, query)
    if loss == "l2":
        penalty = 0.1 if penalty is None else penalty
        gram = design.T @ (weight[:, None] * design) + np.diag([0., penalty, penalty])
        coefficient = np.linalg.lstsq(gram, design.T @ (weight[:, None] * profiles), rcond=None)[0]
        result = coefficient[0]
    elif loss == "l1":
        penalty = 0.05 if penalty is None else penalty
        n = len(xy)
        # Variables: [intercept, slope_x, slope_y, residual_abs[n], slope_abs[2]].
        matrix = np.zeros((2 * n + 4, n + 5))
        matrix[:n, :3], matrix[n:2*n, :3] = design, -design
        matrix[:n, 3:3+n] = -np.eye(n)
        matrix[n:2*n, 3:3+n] = -np.eye(n)
        matrix[2*n:, 1:3] = [[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]
        matrix[2*n:2*n+2, -2], matrix[2*n+2:, -1] = -1., -1.
        objective = np.r_[np.zeros(3), weight, penalty, penalty]
        bounds = [(None, None)] * 3 + [(0., None)] * (n + 2)
        result = []
        for y in profiles.T:
            with warnings.catch_warnings():
                # scipy forwards this valid HiGHS option but emits an interface warning.
                warnings.filterwarnings("ignore", message="Unrecognized options detected.*", category=OptimizeWarning)
                fit = linprog(objective, A_ub=matrix, b_ub=np.r_[y, -y, np.zeros(4)],
                              bounds=bounds, method="highs-ds", options={"threads": 1})
            if not fit.success:
                raise RuntimeError(f"LAD solve failed: {fit.message}")
            result.append(fit.x[0])
        result = np.asarray(result)
    else:
        raise ValueError("Unknown affine loss")
    return np.clip(result, profiles.min(axis=0), profiles.max(axis=0))


def simplex_weights(xy, query):
    """Nonnegative partition-of-unity weights, with explicit outside-hull rules."""
    xy, query = np.asarray(xy, float), np.asarray(query, float)
    n = len(xy)
    if n == 0 or xy.shape != (n, 2) or query.shape != (2,):
        raise ValueError("Expected nonempty 2-D support and one query")
    distance = np.linalg.norm(xy - query, axis=1)
    if distance.min() < 1e-10:
        weights = (distance < 1e-10).astype(float)
        return weights / weights.sum(), "coincident"
    centered = xy - xy.mean(axis=0)
    if n < 3 or np.linalg.matrix_rank(centered, tol=1e-8) < 2:
        if n == 1 or np.max(np.linalg.norm(centered, axis=1)) < 1e-10:
            return np.full(n, 1. / n), "constant_degenerate"
        axis = np.linalg.svd(centered, full_matrices=False)[2][0]
        coordinate, q = centered @ axis, (query - xy.mean(axis=0)) @ axis
        order = np.argsort(coordinate, kind="stable")
        coordinate = coordinate[order]
        weights = np.zeros(n)
        if q <= coordinate[0]:
            weights[order[0]] = 1.
        elif q >= coordinate[-1]:
            weights[order[-1]] = 1.
        else:
            high = np.searchsorted(coordinate, q, side="right")
            t = (q - coordinate[high - 1]) / (coordinate[high] - coordinate[high - 1])
            weights[order[high - 1]], weights[order[high]] = 1. - t, t
        return weights, "collinear_segment"
    triangulation = Delaunay(xy)
    simplex = int(triangulation.find_simplex(query, tol=1e-10))
    weights = np.zeros(n)
    if simplex >= 0:
        transform = triangulation.transform[simplex]
        first = transform[:2] @ (query - transform[2])
        barycentric = np.maximum(np.r_[first, 1. - first.sum()], 0.)
        barycentric /= barycentric.sum()
        weights[triangulation.simplices[simplex]] = barycentric
        return weights, "inside_simplex"
    best_distance, best = np.inf, None
    for a, b in ConvexHull(xy).simplices:
        direction = xy[b] - xy[a]
        t = np.clip(np.dot(query - xy[a], direction) / np.dot(direction, direction), 0., 1.)
        squared = float(np.sum((query - (xy[a] + t * direction)) ** 2))
        if squared < best_distance:
            best_distance, best = squared, (a, b, t)
    a, b, t = best
    weights[a], weights[b] = 1. - t, t
    return weights, "projected_hull_edge"


class LocalGeometryPredictor:
    def __init__(self, variant="mls_affine"):
        if variant not in VARIANTS:
            raise ValueError("Unknown local geometry variant")
        self.variant, self.model_id = variant, f"v7_{variant}"
        self.observed = None

    def condition(self, observed):
        self.observed = validate_observed(observed)
        locations = observed[["location_id", "x_m", "y_m"]].drop_duplicates().sort_values("location_id")
        if locations.location_id.duplicated().any():
            raise ValueError("Location coordinates must be unique")
        profiles = observed.pivot(index="location_id", columns="depth_index", values="log_qc")
        self.depth_ids = np.sort(profiles.columns.to_numpy())
        profile_values = profiles.reindex(index=locations.location_id, columns=self.depth_ids).to_numpy(float)
        if not np.isfinite(profile_values).all():
            raise ValueError("Full revealed profiles are required")
        self.xy, inverse = np.unique(locations[["x_m", "y_m"]].to_numpy(float), axis=0, return_inverse=True)
        self.profiles = np.asarray([profile_values[inverse == i].mean(axis=0) for i in range(len(self.xy))])
        self.diagnostics = []

    def predict(self, query_grid):
        validate_query_site(query_grid, self.observed)
        work = query_grid.reset_index(drop=True)
        if not np.isfinite(work[["x_m", "y_m"]].to_numpy(float)).all():
            raise ValueError("Finite query coordinates required")
        if not work.depth_index.isin(self.depth_ids).all():
            raise ValueError("Unknown query depth")
        result = np.empty(len(work))
        for location, rows in work.groupby("location_id", sort=False):
            coordinate = rows[["x_m", "y_m"]].drop_duplicates().to_numpy(float)
            if len(coordinate) != 1:
                raise ValueError("A query location must have one coordinate")
            if self.variant == "simplex_linear":
                weights, status = simplex_weights(self.xy, coordinate[0])
                value = weights @ self.profiles
            else:
                loss = "l2" if self.variant == "mls_affine" else "l1"
                value = affine_profiles(self.xy, self.profiles, coordinate[0], loss=loss)
                status = "solved"
            result[rows.index] = value[pd.Index(self.depth_ids).get_indexer(rows.depth_index)]
            self.diagnostics.append({"location_id": location, "status": status})
        return PredictionBundle(point_prediction_frame(query_grid, result), self.model_id)


def main():
    from cpt_budget.diagnostics.common import iter_validation_states, STATE_KEYS
    from cpt_budget.v7.common import KEYS, RUN, emit_candidate, load_frozen
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RUN / "local_geometry")
    args = parser.parse_args()
    write_json(args.output / "frozen_settings.json", SETTINGS)
    wide = load_frozen()
    with threadpool_limits(limits=1):
        for variant in VARIANTS:
            predictor = LocalGeometryPredictor(variant)
            records, diagnostics = [], []
            for state in iter_validation_states():
                predictor.condition(state.observed)
                frame = predictor.predict(state.query).frame
                for key in STATE_KEYS:
                    frame[key] = state.metadata[key]
                records.append(frame[KEYS + ["pred_log_qc"]])
                diagnostics.extend([{**state.metadata, **row} for row in predictor.diagnostics])
            predicted = pd.concat(records, ignore_index=True)
            values = wide[KEYS].merge(predicted, on=KEYS, how="left", validate="one_to_one").pred_log_qc.to_numpy()
            emit_candidate(predictor.model_id, values, wide=wide, output_root=args.output,
                           metadata={"family": "local_geometry", "settings": SETTINGS, "variant": variant})
            pd.DataFrame(diagnostics).to_csv(args.output / predictor.model_id / "geometry_diagnostics.csv", index=False)
            print(f"{predictor.model_id}: 60 states, {len(values)} cells", flush=True)


if __name__ == "__main__":
    main()
