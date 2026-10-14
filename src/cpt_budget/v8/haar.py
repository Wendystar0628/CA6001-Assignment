"""Engineered full-profile Haar count decisions with support-only calibration.

The equivalent version reproduces v7_hcount_simplex_median_a100_gp point
predictions. Native risk heads remain an explicitly separate evaluation choice.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import ConvexHull, Delaunay
from threadpoolctl import threadpool_limits

from ..config import write_json
from ..models.base import validate_observed, validate_query_site
from ..models.distribution import point_prediction_frame
from ..types import PredictionBundle
from ..v7.local_geometry import LocalGeometryPredictor
from ..v7.multiscale import MultiscaleProfilePredictor, haar_forward, local_weights
from ..v7.multiscale_combinations import transfer_low_count

VARIANTS = ("equivalent", "loo_bias", "loo_blend", "loo_localblend")
HAAR_MATRIX = haar_forward(np.eye(24))
HAAR_MATRIX.setflags(write=False)


def project_counts(base, count, margin=.05):
    """Batched minimum-change projection with stable depth tie-breaking."""
    base = np.asarray(base, dtype=float)
    count = np.asarray(count, dtype=float)
    if (base.ndim != 2 or count.shape != (len(base),) or not np.isfinite(base).all()
            or not np.isfinite(count).all() or margin <= 0):
        raise ValueError("Finite profile rows and one finite count per row required")
    count = np.clip(np.floor(count + .5), 0, base.shape[1]).astype(int)
    order = np.argsort(base, axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(base.shape[1]), order.shape), axis=1)
    low = rank < count[:, None]
    return np.where(low & (base >= 0), -margin, np.where((~low) & (base < 0), margin, base))


class _SimplexGeometry:
    """Cache triangulation/hull once per revealed context, preserving v7 rules."""
    def __init__(self, xy, profiles):
        self.xy, inverse = np.unique(xy, axis=0, return_inverse=True)
        self.profiles = np.asarray([profiles[inverse == i].mean(axis=0) for i in range(len(self.xy))])
        self.center = self.xy.mean(axis=0)
        centered = self.xy - self.center
        self.degenerate = len(self.xy) < 3 or np.linalg.matrix_rank(centered, tol=1e-8) < 2
        self.constant = len(self.xy) == 1 or np.max(np.linalg.norm(centered, axis=1)) < 1e-10
        if self.degenerate and not self.constant:
            self.axis = np.linalg.svd(centered, full_matrices=False)[2][0]
            self.coordinate = centered @ self.axis
            self.order = np.argsort(self.coordinate, kind="stable")
            self.coordinate = self.coordinate[self.order]
        elif not self.degenerate:
            self.triangulation = Delaunay(self.xy)
            self.edges = ConvexHull(self.xy).simplices
            self.a = self.xy[self.edges[:, 0]]
            self.delta = self.xy[self.edges[:, 1]] - self.a
            self.norm = np.sum(self.delta ** 2, axis=1)

    def predict(self, query_xy):
        query_xy = np.asarray(query_xy, dtype=float)
        weights = np.zeros((len(query_xy), len(self.xy)))
        distance = np.linalg.norm(query_xy[:, None] - self.xy[None], axis=2)
        zero = distance < 1e-10
        coincident = zero.any(axis=1)
        weights[coincident] = zero[coincident] / zero[coincident].sum(axis=1, keepdims=True)
        active = np.flatnonzero(~coincident)
        if not len(active):
            return weights @ self.profiles
        q = query_xy[active]
        if self.degenerate:
            if self.constant:
                weights[active] = 1 / len(self.xy)
            else:
                coordinate = (q - self.center) @ self.axis
                for i, x in zip(active, coordinate):
                    if x <= self.coordinate[0]:
                        weights[i, self.order[0]] = 1
                    elif x >= self.coordinate[-1]:
                        weights[i, self.order[-1]] = 1
                    else:
                        j = np.searchsorted(self.coordinate, x, side="right")
                        t = (x - self.coordinate[j-1]) / (self.coordinate[j] - self.coordinate[j-1])
                        weights[i, self.order[j-1]], weights[i, self.order[j]] = 1-t, t
        else:
            simplex = self.triangulation.find_simplex(q, tol=1e-10)
            inside = simplex >= 0
            if inside.any():
                transform = self.triangulation.transform[simplex[inside]]
                bary = np.einsum("nij,nj->ni", transform[:, :2], q[inside] - transform[:, 2])
                bary = np.maximum(np.column_stack((bary, 1 - bary.sum(axis=1))), 0)
                bary /= bary.sum(axis=1, keepdims=True)
                weights[active[inside, None], self.triangulation.simplices[simplex[inside]]] = bary
            if (~inside).any():
                outside = q[~inside]
                t = np.clip(np.einsum("qei,ei->qe", outside[:, None] - self.a, self.delta) / self.norm, 0, 1)
                square = np.sum((outside[:, None] - (self.a + t[..., None]*self.delta))**2, axis=2)
                best = np.argmin(square, axis=1)
                selected = t[np.arange(len(outside)), best]
                rows = active[~inside]
                weights[rows, self.edges[best, 0]], weights[rows, self.edges[best, 1]] = 1-selected, selected
        return weights @ self.profiles


class _FastCore:
    def __init__(self, xy, profiles):
        self.xy, self.profiles = np.asarray(xy, float), np.asarray(profiles, float)
        self.simplex = _SimplexGeometry(self.xy, self.profiles)
        coefficients = self.profiles @ HAAR_MATRIX
        self.coarse_mean = coefficients[:, :3].mean(axis=0)
        distance = np.linalg.norm(self.xy[:, None] - self.xy[None], axis=2)
        pair = distance[np.triu_indices(len(xy), k=1)]
        pair = pair[pair > 1e-8]
        self.length_scale = float(np.median(pair)) if len(pair) else 1.
        kernel = np.exp(-.5*(distance/self.length_scale)**2)
        self.coarse_fit = cho_solve(cho_factor(kernel + .1*np.eye(len(xy)), check_finite=False),
                                   coefficients[:, :3]-self.coarse_mean, check_finite=False)
        detail = coefficients[:, 3:]
        self.detail_order = np.argsort(detail, axis=0, kind="stable")
        self.sorted_detail = np.take_along_axis(detail, self.detail_order, axis=0)

    def predict(self, xy):
        base = self.simplex.predict(xy)
        distance = np.linalg.norm(np.asarray(xy)[:, None] - self.xy[None], axis=2)
        coarse = self.coarse_mean + np.exp(-.5*(distance/self.length_scale)**2) @ self.coarse_fit
        weight = local_weights(distance, power=2., neighbours=3)
        cumulative = np.cumsum(weight[:, self.detail_order], axis=1)
        index = np.minimum((cumulative < .5).sum(axis=1), len(self.xy)-1)
        detail = self.sorted_detail[index, np.arange(21)[None]]
        haar = np.column_stack((coarse, detail)) @ HAAR_MATRIX.T
        zero = distance <= 1e-8
        exact = zero.any(axis=1)
        if exact.any():
            haar[exact] = (zero[exact] / zero[exact].sum(axis=1, keepdims=True)) @ self.profiles
        return base, haar


class HaarCountPredictor:
    """condition/predict API for real support-to-query computation and timing."""
    def __init__(self, variant="equivalent", *, optimized=True):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown Haar variant {variant}")
        if not optimized and variant != "equivalent":
            raise ValueError("Reference implementation is for exact-equivalent control only")
        self.variant, self.optimized = variant, optimized
        self.model_id = "v8_haar_" + ("fast" if variant == "equivalent" else variant)
        self.observed = None

    def condition(self, observed):
        self.observed = validate_observed(observed)
        profile = self.observed.pivot(index="location_id", columns="depth_index", values="log_qc").sort_index()
        if profile.shape[1] != 24 or not np.array_equal(profile.columns, np.arange(24)) or not np.isfinite(profile.to_numpy()).all():
            raise ValueError("Complete finite 24-depth profiles are required")
        geometry = self.observed.drop_duplicates("location_id").set_index("location_id").loc[profile.index]
        if self.observed.groupby("location_id")[["x_m", "y_m"]].nunique().to_numpy().max() > 1:
            raise ValueError("Each observed CPT must have one coordinate")
        self.locations, self.xy, self.profiles = profile.index, geometry[["x_m", "y_m"]].to_numpy(float), profile.to_numpy(float)
        self.groups = geometry.near_group_id.astype(str).to_numpy() if "near_group_id" in geometry else self.locations.astype(str).to_numpy()
        self.calibration_records = []
        if self.optimized:
            self.core = _FastCore(self.xy, self.profiles)
        else:
            self.simplex = LocalGeometryPredictor("simplex_linear")
            self.haar = MultiscaleProfilePredictor("haar_krr_median")
            self.simplex.condition(self.observed)
            self.haar.condition(self.observed)
        if self.variant != "equivalent":
            for group in np.unique(self.groups):
                held = self.groups == group
                if held.all():
                    continue
                core = _FastCore(self.xy[~held], self.profiles[~held])
                base, haar = core.predict(self.xy[held])
                for loc, xy, actual, b, h in zip(self.locations[held], self.xy[held], self.profiles[held], base, haar):
                    self.calibration_records.append(dict(location_id=str(loc), held_group=str(group),
                        support_location_ids=list(map(str, self.locations[~held])),
                        support_groups=list(map(str, np.unique(self.groups[~held]))),
                        x_m=float(xy[0]), y_m=float(xy[1]), true_count=int((actual<0).sum()),
                        base_count=int((b<0).sum()), haar_count=int((h<0).sum())))
            self.calibration = pd.DataFrame(self.calibration_records)
        self.context_manifest = {"variant": self.variant, "optimized": self.optimized,
            "observed_locations": list(map(str, self.locations)), "observed_groups": list(map(str, self.groups)),
            "calibration": "leave entire near_group out; observed labels only", "calibration_rows": len(self.calibration_records),
            "coarse_ridge": .1, "detail_neighbours": 3, "bias_shrinkage_pseudocount": 4,
            "bias_max_cells": 4, "blend_ridge_to_haar": 1., "risk_head": "external independent head"}

    def _counts(self, xy, base, haar):
        base_count, haar_count = (base<0).sum(axis=1), (haar<0).sum(axis=1)
        if self.variant == "equivalent" or self.calibration.empty:
            return haar_count.astype(float)
        data = self.calibration
        delta = (data.haar_count-data.base_count).to_numpy(float)
        target = (data.true_count-data.base_count).to_numpy(float)
        if self.variant == "loo_blend":
            alpha = np.clip((np.mean(delta*target)+1.)/(np.mean(delta**2)+1.), 0, 1)
            return base_count + alpha*(haar_count-base_count)
        distance = np.linalg.norm(np.asarray(xy)[:, None] - data[["x_m", "y_m"]].to_numpy()[None], axis=2)
        weights = local_weights(distance, power=2., neighbours=min(5, len(data)))
        if self.variant == "loo_localblend":
            alpha = np.clip((weights@(delta*target)+1.)/(weights@(delta**2)+1.), 0, 1)
            return base_count + alpha*(haar_count-base_count)
        bias = weights @ (data.haar_count-data.true_count).to_numpy(float)
        bias *= len(data)/(len(data)+4.)
        return haar_count - np.clip(bias, -4., 4.)

    def predict(self, query_grid):
        validate_query_site(query_grid, self.observed)
        work = query_grid.reset_index(drop=True)
        if not np.isfinite(work[["x_m", "y_m"]].to_numpy(float)).all() or not work.depth_index.isin(range(24)).all():
            raise ValueError("Finite query coordinates and conditioned depth indices required")
        if work.groupby("location_id")[["x_m", "y_m"]].nunique().to_numpy().max() > 1:
            raise ValueError("Each query CPT must have one coordinate")
        locations = work.location_id.drop_duplicates()
        geometry = work.drop_duplicates("location_id").set_index("location_id").loc[locations]
        xy = geometry[["x_m", "y_m"]].to_numpy(float)
        if self.optimized:
            base, haar = self.core.predict(xy)
            counts = self._counts(xy, base, haar)
            values = project_counts(base, counts)
        else:
            # Reference follows the actual v7 classes and count projection.
            full = pd.DataFrame([dict(site_id=work.site_id.iloc[0], location_id=loc,
                depth_index=d, x_m=point[0], y_m=point[1]) for loc, point in zip(locations, xy) for d in range(24)])
            base = self.simplex.predict(full).frame.pred_log_qc.to_numpy().reshape(-1, 24)
            haar = self.haar.predict(full).frame.pred_log_qc.to_numpy().reshape(-1, 24)
            values, counts = transfer_low_count(base, haar, 1.)
        row_index = pd.Index(locations).get_indexer(work.location_id)
        prediction = values[row_index, work.depth_index.to_numpy(int)]
        self.last_diagnostics = pd.DataFrame({"location_id": locations.to_numpy(),
            "base_count": (base<0).sum(axis=1), "haar_count": (haar<0).sum(axis=1),
            "proposed_count": counts, "actual_point_count": (values<0).sum(axis=1),
            "flipped_cells": ((base<0)!=(values<0)).sum(axis=1)})
        return PredictionBundle(point_prediction_frame(query_grid, prediction), self.model_id)


def make_predictor(variant="equivalent", optimized=True):
    return HaarCountPredictor(variant, optimized=optimized)


def run(output_root=None):
    from ..diagnostics.common import iter_validation_states, STATE_KEYS
    from .common import RUN, KEYS, load_frozen, emit_candidate, load_native
    output = Path(output_root or RUN / "haar")
    write_json(output / "method_protocol.json", {"family": "engineered_haar_count", "variants": VARIANTS,
        "equivalent_reference": "v7_hcount_simplex_median_a100_gp", "optimized": [
            "cached orthonormal Haar matrix and coefficient sort", "multi-output Cholesky solve",
            "cached Delaunay and hull with batched barycentric/edge projection", "batched stable count projection"],
        "quality_variants": {"loo_bias": "local 5-neighbour signed leave-group-out count bias; n/(n+4) shrinkage; +/-4-cell cap",
            "loo_blend": "global support leave-group-out squared-error convex count blend; ridge 1 towards Haar",
            "loo_localblend": "local 5-neighbour support leave-group-out squared-error convex count blend; ridge 1 towards Haar"},
        "truth_boundary": "only current revealed complete profiles; entire near_group held out for adaptation",
        "query_truth_used": False, "source_training": False, "point_threshold": "log qc < 0", "step_m": .25,
        "risk_output": "independent frozen v6_empirical_graph_union_local native event head; not point-model CDF", "all_options_before_scores": True})
    wide = load_frozen()
    risk_probability = load_native("v6_empirical_graph_union_local")
    parts, diagnostics, calibration, timings = {v: [] for v in VARIANTS}, [], [], []
    import time
    with threadpool_limits(limits=1):
        for state in iter_validation_states():
            for variant in VARIANTS:
                model = make_predictor(variant)
                start = time.perf_counter()
                model.condition(state.observed)
                conditioned = time.perf_counter()
                frame = model.predict(state.query).frame
                done = time.perf_counter()
                for key in STATE_KEYS:
                    frame[key] = state.metadata[key]
                parts[variant].append(frame[KEYS+["pred_log_qc"]])
                diagnostics.append(model.last_diagnostics.assign(model_id=model.model_id, **{k:state.metadata[k] for k in STATE_KEYS}))
                calibration.extend([{**row, "model_id":model.model_id, **{k:state.metadata[k] for k in STATE_KEYS}}
                    for row in model.calibration_records])
                timings.append({"model_id":model.model_id, **{k:state.metadata[k] for k in STATE_KEYS},
                    "condition_seconds":conditioned-start, "predict_seconds":done-conditioned, "total_seconds":done-start})
            print(f"haar: {state.key}", flush=True)
    paths = {}
    for variant, frames in parts.items():
        values = wide[KEYS].merge(pd.concat(frames, ignore_index=True), on=KEYS, how="left", validate="one_to_one").pred_log_qc
        model_id = make_predictor(variant).model_id
        paths[model_id] = str(emit_candidate(model_id, values, wide=wide, output_root=output, probability=risk_probability,
            metadata={"family":"haar", "variant":variant, "predictor":"HaarCountPredictor", "native_uncertainty":False,
                "independent_native_risk_head":"v6_empirical_graph_union_local", "risk_is_point_model_cdf":False}))
    pd.concat(diagnostics, ignore_index=True).to_csv(output / "count_diagnostics.csv", index=False)
    pd.DataFrame(timings).to_csv(output / "run_timing.csv", index=False)
    write_json(output / "support_calibration.json", calibration)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    run(parser.parse_args().output)
