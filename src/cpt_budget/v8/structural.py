"""Array-first, support-only engineering of the v7 simplex/structural point head.

``exact`` changes execution, not the statistical rule.  The two quality
variants make simplex an observed-only LOO member.  Neither receives hidden
query measurements.  Factories below execute real conditioning/inference;
they never replay archived predictions.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from cpt_budget.config import PROJECT_ROOT, write_json
from cpt_budget.diagnostics.common import STATE_KEYS, iter_validation_states
from cpt_budget.models.base import validate_observed, validate_query_site
from cpt_budget.models.distribution import point_prediction_frame
from cpt_budget.types import PredictionBundle, QUERY_COLUMNS
from cpt_budget.v6.adaptive import (fit_weights, leave_one_location_out, load_basis,
                                    member_predictions)
from cpt_budget.v7.local_geometry import LocalGeometryPredictor, simplex_weights

VARIANTS = ("exact", "loo_simplex", "reliability")
MEMBERS = ("idw", "functional_fpca_krr", "registered_profile_shape",
           "spatial_weighted_median", "simplex_linear")
SETTINGS = {
    "exact_reference": "v7_simplex_structural_s050_gp point head",
    "exact_formula": "0.5 simplex + 0.5 original four-member structural LOO stack",
    "engineering": ["one profile/geometry extraction", "shared pairwise distances",
                    "batched FPCA linear solves by support size",
                    "directed registration-pair memoization across LOO contexts",
                    "vectorized monotone registration dynamic programming",
                    "one simplex triangulation per support subset"],
    "quality_members": MEMBERS,
    "weight_grid_step": .25, "weight_regularization": .05,
    "weight_prior": "uniform over five members",
    "loss": "L1 + 0.5 soft thickness MAE + 0.5 weighted event Brier + weight L2",
    "event_temperature": .25, "false_negative_weight": 2.,
    "reliability_gate": "clip((fixed-mixture LOO loss - five-member LOO loss)/0.10, 0, 1)",
    "loo_exclusion": "whole location, same near_group and colocated locations",
    "quality_selection": "same revealed labels fit and assess weights; training loss, not independent evidence",
    "native_probability": "none; point-only family, external risk/decision heads remain separate",
    "source_basis": "unchanged fold-specific source-trained v5 FPCA",
    "warmup": "one unmeasured condition + predict before a fresh measured condition",
}


def fast_bounded_warp(reference, source, *, band_bins=2.):
    """Same lattice, costs and tie order as v5 bounded_warp; vectorized states."""
    reference, source = np.asarray(reference, float), np.asarray(source, float)
    if reference.ndim != 1 or reference.shape != source.shape or len(source) < 2:
        raise ValueError("Profiles must have identical one-dimensional shapes")
    if not np.isfinite(reference).all() or not np.isfinite(source).all() or band_bins <= 0:
        raise ValueError("Finite profiles and positive band required")
    n = len(reference)
    nodes = np.arange(2 * n - 1, dtype=float) / 2
    values = np.interp(nodes, np.arange(n), source)
    cost = np.full((n, len(nodes)), np.inf)
    back = np.full((n, len(nodes)), -1, dtype=int)
    cost[0, 0] = (reference[0] - source[0]) ** 2
    deltas = np.array([2, 1, 3])
    penalty = .05 * (deltas / 2 - 1) ** 2
    for j in range(1, n):
        eligible = np.flatnonzero(np.abs(nodes - j) <= band_bins + 1e-12)
        if j == n - 1:
            eligible = np.array([len(nodes) - 1])
        previous = eligible[:, None] - deltas
        local = (reference[j] - values[eligible]) ** 2 + .05 * ((nodes[eligible] - j) / band_bins) ** 2
        # Addition order and step preference intentionally match the scalar path.
        candidates = cost[j - 1, np.maximum(previous, 0)] + local[:, None] + penalty
        candidates[previous < 0] = np.inf
        choice = candidates.argmin(axis=1)
        cost[j, eligible] = candidates[np.arange(len(eligible)), choice]
        back[j, eligible] = previous[np.arange(len(eligible)), choice]
    if not np.isfinite(cost[-1, -1]):
        raise RuntimeError("No admissible registration path")
    mapping, k = np.empty(n), len(nodes) - 1
    for j in range(n - 1, -1, -1):
        mapping[j], k = nodes[k], back[j, k]
    return mapping


class _SimplexContext:
    """Cache geometry but preserve v7 coincident/collinear/outside-hull rules."""
    def __init__(self, xy, values):
        from scipy.spatial import ConvexHull, Delaunay
        self.xy, inverse = np.unique(xy, axis=0, return_inverse=True)
        self.values = np.array([values[inverse == i].mean(axis=0) for i in range(len(self.xy))])
        centered = self.xy - self.xy.mean(axis=0)
        self.tri = self.hull = None
        if len(self.xy) >= 3 and np.linalg.matrix_rank(centered, tol=1e-8) >= 2:
            self.tri, self.hull = Delaunay(self.xy), ConvexHull(self.xy).simplices

    def predict(self, query):
        distance = np.linalg.norm(self.xy - query, axis=1)
        if self.tri is None or distance.min() < 1e-10:
            weights, _ = simplex_weights(self.xy, query)
        else:
            simplex = int(self.tri.find_simplex(query, tol=1e-10))
            weights = np.zeros(len(self.xy))
            if simplex >= 0:
                transform = self.tri.transform[simplex]
                first = transform[:2] @ (query - transform[2])
                barycentric = np.maximum(np.r_[first, 1. - first.sum()], 0.)
                barycentric /= barycentric.sum()
                weights[self.tri.simplices[simplex]] = barycentric
            else:
                best_distance, best = np.inf, None
                for a, b in self.hull:
                    direction = self.xy[b] - self.xy[a]
                    t = np.clip(np.dot(query - self.xy[a], direction) / np.dot(direction, direction), 0., 1.)
                    squared = float(np.sum((query - (self.xy[a] + t * direction)) ** 2))
                    if squared < best_distance:
                        best_distance, best = squared, (a, b, t)
                a, b, t = best
                weights[a], weights[b] = 1. - t, t
        return weights @ self.values


class ProfileEngine:
    """A conditioned support owns caches; no cache survives changed labels."""
    def __init__(self, observed, basis, *, quality_context=True):
        self.observed, self.basis = observed, basis
        self.quality_context = quality_context
        locations = observed[["location_id", "x_m", "y_m"]].drop_duplicates().sort_values("location_id")
        if locations.location_id.duplicated().any():
            raise ValueError("Coordinates must be constant per location")
        self.ids = locations.location_id.to_numpy()
        self.xy = locations[["x_m", "y_m"]].to_numpy(float)
        self.depths = basis.depth_indices
        self.profiles = observed.pivot(index="location_id", columns="depth_index", values="log_qc").reindex(
            index=self.ids, columns=self.depths).to_numpy(float)
        if not np.isfinite(self.profiles).all():
            raise ValueError("Complete finite profiles on the source basis grid required")
        depth_rows = observed[["depth_index", "z_center_m"]].drop_duplicates().sort_values("depth_index")
        z = depth_rows.z_center_m.to_numpy(float)
        if len(z) != len(self.depths) or np.any(np.diff(z) <= 0) or not np.allclose(np.diff(z), np.diff(z)[0]):
            raise ValueError("Common regular depth grid required")
        self.band_bins = .5 / np.diff(z)[0]
        self.distance = np.linalg.norm(self.xy[:, None] - self.xy[None, :], axis=2)
        self.matching = (self.profiles - self.profiles.mean(axis=1, keepdims=True)) / np.maximum(
            self.profiles.std(axis=1, keepdims=True), .25)
        self.match_distance = ((self.matching[:, None] - self.matching[None, :]) ** 2).mean(axis=2)
        self.warps, self.registered, self.contexts = {}, {}, {}
        self.full = tuple(range(len(self.ids)))
        self.subsets, self.exclusions = [], []
        for i, location in enumerate(self.ids):
            excluded = self.distance[i] < 1e-8
            if "near_group_id" in observed:
                groups = observed.loc[observed.location_id.eq(location), "near_group_id"].dropna().unique()
                exclude_ids = observed.loc[observed.near_group_id.isin(groups), "location_id"].unique()
                excluded |= np.isin(self.ids, exclude_ids)
            subset = tuple(np.flatnonzero(~excluded))
            self.subsets.append(subset)
            self.exclusions.append({"held_location_id": location, "excluded_location_ids": self.ids[excluded].tolist(),
                                    "n_remaining_locations": len(subset)})
        self._prepare_contexts(set(self.subsets) | {self.full})

    def _prepare_contexts(self, subsets):
        batches = {}
        for subset in sorted(subsets):
            if not subset:
                continue
            index = list(subset)
            values, xy = self.profiles[index], self.xy[index]
            mean = values.mean(axis=0)
            scores = (values - mean) @ self.basis.components.T
            # Match functional predictor's squared-distance arithmetic exactly.
            squared = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2)
            d = np.sqrt(squared[np.triu_indices(len(xy), k=1)])
            d = d[d > 1e-12]
            length = float(np.median(d)) if len(d) else 1.
            matrix = np.exp(-squared / (2 * length ** 2)) + .1 * np.eye(len(xy))
            self.contexts[subset] = {"mean": mean, "length": length, "xy": xy,
                                     "simplex": _SimplexContext(xy, values) if self.quality_context or subset == self.full else None}
            batches.setdefault(len(subset), []).append((subset, matrix, scores))
        for batch in batches.values():
            solved = np.linalg.solve(np.stack([x[1] for x in batch]), np.stack([x[2] for x in batch]))
            for (subset, _, _), coefficient in zip(batch, solved):
                self.contexts[subset]["coefficient"] = coefficient

    def _registered(self, nearest):
        key = tuple(sorted(map(int, nearest)))
        if key not in self.registered:
            medoid = key[int(np.argmin(self.match_distance[np.ix_(key, key)].sum(axis=1)))]
            identity = np.arange(len(self.depths), dtype=float)
            maps = []
            for index in key:
                pair = medoid, index
                if pair not in self.warps:
                    self.warps[pair] = identity if index == medoid else fast_bounded_warp(
                        self.matching[medoid], self.matching[index], band_bins=self.band_bins)
                maps.append(self.warps[pair])
            maps = np.array(maps)
            amplitude = np.array([np.interp(mapping, identity, self.profiles[index])
                                  for mapping, index in zip(maps, key)])
            self.registered[key] = maps, amplitude
        return key, self.registered[key]

    def predict_one(self, subset, query, *, distance=None):
        if not subset:
            return np.zeros((len(self.depths), len(MEMBERS)))
        index = np.array(subset)
        context = self.contexts[subset]
        distance = np.linalg.norm(self.xy[index] - query, axis=1) if distance is None else distance[index]
        nearest = np.argsort(distance, kind="stable")[:3]
        near_indices = index[nearest]
        d, values = distance[nearest], self.profiles[near_indices]
        coincident = d <= 1e-9
        if coincident.any():
            idw = values[coincident].mean(axis=0)
        else:
            weight = (d.min() / d) ** 2
            idw = weight @ values / weight.sum()
        squared = ((np.asarray(query)[None, :] - context["xy"]) ** 2).sum(axis=1)
        scores = np.exp(-squared / (2 * context["length"] ** 2)) @ context["coefficient"]
        functional = context["mean"] + scores @ self.basis.components
        if d[0] < 1e-8:
            registered = median = values[0]
        else:
            key, (maps, amplitude) = self._registered(near_indices)
            # The registration uses location-sorted weights, the median distance-sorted.
            key_distance = np.linalg.norm(self.xy[list(key)] - query, axis=1)
            weight = key_distance ** -2
            weight /= weight.sum()
            registered = np.interp(np.arange(len(self.depths)), weight @ maps, weight @ amplitude)
            weight = d ** -2
            weight /= weight.sum()
            order = np.argsort(values, axis=0, kind="stable")
            sorted_values = np.take_along_axis(values, order, axis=0)
            selected = (np.cumsum(weight[order], axis=0) >= .5).argmax(axis=0)
            median = sorted_values[selected, np.arange(len(self.depths))]
        simplex = context["simplex"].predict(query) if context["simplex"] is not None else np.zeros(len(self.depths))
        return np.column_stack([idw, functional, registered, median, simplex])

    def loo(self):
        by_location = np.array([self.predict_one(subset, self.xy[i], distance=self.distance[i])
                                for i, subset in enumerate(self.subsets)])
        return self.to_rows(self.observed, by_location, self.ids)

    def query(self, query):
        geometry = query[["location_id", "x_m", "y_m"]].drop_duplicates()
        if geometry.location_id.duplicated().any() or not np.isfinite(geometry[["x_m", "y_m"]].to_numpy(float)).all():
            raise ValueError("Each query needs one finite coordinate")
        predicted = np.array([self.predict_one(self.full, xy) for xy in geometry[["x_m", "y_m"]].to_numpy(float)])
        return self.to_rows(query, predicted, geometry.location_id.to_numpy())

    def to_rows(self, rows, profiles, ids):
        location_index = pd.Index(ids).get_indexer(rows.location_id)
        depth_index = pd.Index(self.depths).get_indexer(rows.depth_index)
        if (location_index < 0).any() or (depth_index < 0).any():
            raise ValueError("Unseen profile or depth index")
        return profiles[location_index, depth_index]


def _loss(observed, predictions):
    values = np.asarray(predictions, float)
    if values.ndim == 1:
        values = values[:, None]
    truth = observed.log_qc.to_numpy(float)
    labels = truth < 0
    probability = expit(-values / .25)
    l1 = np.abs(values - truth[:, None]).mean(axis=0)
    event = np.mean(np.where(labels[:, None], 2 * (1 - probability) ** 2, probability ** 2), axis=0)
    thickness = np.mean([np.abs(.25 * probability[rows.index].sum(axis=0) - .25 * labels[rows.index].sum())
                         for _, rows in observed.groupby("location_id", sort=False)], axis=0)
    return l1 + .5 * thickness + .5 * event


def fit_five_weights(observed, matrix):
    weights = np.array([w for w in itertools.product(range(5), repeat=5) if sum(w) == 4], float) / 4
    loss = _loss(observed, matrix @ weights.T) + .05 * np.square(weights - .2).sum(axis=1)
    best = int(np.argmin(loss))
    return weights[best], float(loss[best])


class StructuralPredictor:
    def __init__(self, fold_id=None, *, variant="exact", basis=None, quality_context=None):
        started = time.perf_counter()
        if variant not in VARIANTS:
            raise ValueError("Unknown structural variant")
        self.basis = basis if basis is not None else load_basis(fold_id)
        self.initialization_seconds = time.perf_counter() - started
        self.variant, self.model_id = variant, f"v8_structural_{variant}"
        self.quality_context = variant != "exact" if quality_context is None else bool(quality_context)
        self.stage_timings = {}

    def condition(self, observed):
        started = time.perf_counter()
        self.observed = validate_observed(observed).reset_index(drop=True)
        if self.observed.site_id.isin(self.basis.source_site_ids).any():
            raise ValueError("Source basis overlaps target site")
        self.engine = ProfileEngine(self.observed, self.basis, quality_context=self.quality_context)
        extracted = time.perf_counter()
        self.loo_predictions = self.engine.loo()
        fitted = time.perf_counter()
        self.weights, four_record = fit_weights(self.observed, self.loo_predictions[:, :4], objective="structural")
        self.five_weights, five_loss, old_loss, new_loss, self.gate = None, None, None, None, 0.
        if self.quality_context:
            self.five_weights, five_loss = fit_five_weights(self.observed, self.loo_predictions)
            old_prediction = .5 * self.loo_predictions[:, 4] + .5 * (self.loo_predictions[:, :4] @ self.weights)
            old_loss, new_loss = float(_loss(self.observed, old_prediction)[0]), float(
                _loss(self.observed, self.loo_predictions @ self.five_weights)[0])
            self.gate = float(np.clip((old_loss - new_loss) / .10, 0., 1.))
        end = time.perf_counter()
        self.stage_timings = {"source_basis_initialization_seconds": self.initialization_seconds,
                              "condition_seconds": end - started, "array_and_fpca_seconds": extracted - started,
                              "loo_members_seconds": fitted - extracted, "weight_fit_seconds": end - fitted}
        self.context_manifest = {"variant": self.variant, "observed_location_ids": self.engine.ids.tolist(),
                                 "weights_four": four_record, "weights_five": None if self.five_weights is None else dict(zip(MEMBERS, self.five_weights.tolist())),
                                 "five_regularized_training_loss": five_loss, "fixed_mix_training_loss": old_loss,
                                 "five_training_loss": new_loss, "reliability_gate": self.gate,
                                 "loo_exclusions": self.engine.exclusions,
                                 "source_basis_sites": self.basis.source_site_ids,
                                 "query_inputs": "geometry only; no query labels"}

    def predict_all(self, query_grid):
        started = time.perf_counter()
        validate_query_site(query_grid, self.observed)
        query = query_grid[[c for c in QUERY_COLUMNS if c in query_grid]].copy()
        matrix = self.engine.query(query)
        exact = .5 * matrix[:, 4] + .5 * (matrix[:, :4] @ self.weights)
        output = {"exact": exact}
        if self.quality_context:
            five = matrix @ self.five_weights
            output.update(loo_simplex=five, reliability=(1 - self.gate) * exact + self.gate * five)
        self.stage_timings["predict_seconds"] = time.perf_counter() - started
        self.stage_timings["registered_pairs"] = len(self.engine.warps)
        self.stage_timings["registered_triplets"] = len(self.engine.registered)
        return output

    def predict(self, query_grid):
        result = self.predict_all(query_grid)[self.variant]
        return PredictionBundle(point_prediction_frame(query_grid, result), self.model_id)


class OriginalStructuralPredictor:
    """Unoptimized live v7 point path, for fair measured before/after inference."""
    model_id = "v7_simplex_structural_s050_gp_point_live"

    def __init__(self, fold_id=None, *, basis=None):
        started = time.perf_counter()
        self.basis = basis if basis is not None else load_basis(fold_id)
        self.initialization_seconds = time.perf_counter() - started
        self.stage_timings = {}

    def condition(self, observed):
        started = time.perf_counter()
        self.observed = validate_observed(observed).reset_index(drop=True)
        ordered, loo = leave_one_location_out(self.observed, self.basis)
        self.weights, _ = fit_weights(ordered, loo, objective="structural")
        self.simplex = LocalGeometryPredictor("simplex_linear")
        self.simplex.condition(self.observed)
        self.stage_timings = {"source_basis_initialization_seconds": self.initialization_seconds,
                              "condition_seconds": time.perf_counter() - started}

    def predict(self, query_grid):
        started = time.perf_counter()
        query = query_grid[[c for c in QUERY_COLUMNS if c in query_grid]].copy()
        stack = member_predictions(self.observed, query, self.basis) @ self.weights
        simplex = self.simplex.predict(query).frame.pred_log_qc.to_numpy(float)
        result = PredictionBundle(point_prediction_frame(query, .5 * simplex + .5 * stack), self.model_id)
        self.stage_timings["predict_seconds"] = time.perf_counter() - started
        return result


def old_factory(fold_id):
    return OriginalStructuralPredictor(fold_id)


def new_factory(fold_id):
    return StructuralPredictor(fold_id, variant="exact")


def run(output=None):
    from cpt_budget.v8.common import KEYS, RUN, emit_candidate, load_frozen
    directory = Path(output or RUN / "structural")
    write_json(directory / "protocol.json", SETTINGS)
    wide = load_frozen()
    records, timings, contexts = [], [], []
    with threadpool_limits(limits=1):
        for state in iter_validation_states():
            predictor = StructuralPredictor(state.metadata["fold_id"], quality_context=True)
            predictor.condition(state.observed)
            values = predictor.predict_all(state.query)
            frame = state.query[["location_id", "depth_index"]].copy()
            for key in STATE_KEYS:
                frame[key] = state.metadata[key]
            for variant, value in values.items():
                frame[f"v8_structural_{variant}"] = value
            records.append(frame)
            timings.append({**{k: state.metadata[k] for k in STATE_KEYS}, **predictor.stage_timings})
            contexts.append({**{k: state.metadata[k] for k in STATE_KEYS}, **predictor.context_manifest})
            print(f"structural {state.key}", flush=True)
    aligned = wide[KEYS].merge(pd.concat(records, ignore_index=True), on=KEYS, how="left", validate="one_to_one")
    for variant in VARIANTS:
        name = f"v8_structural_{variant}"
        emit_candidate(name, aligned[name].to_numpy(float), wide=wide, output_root=directory,
                       metadata={**SETTINGS, "variant": variant})
    old = wide["v7_simplex_structural_s050_gp"].to_numpy(float)
    exact = aligned.v8_structural_exact.to_numpy(float)
    equivalence = {"reference": "v7_simplex_structural_s050_gp", "scope": "point head only",
                   "states": len(records), "cells": len(exact), "max_absolute_difference": float(np.max(np.abs(old - exact))),
                   "threshold_label_differences": int(np.sum((old < 0) != (exact < 0))),
                   "allclose_atol_1e_10": bool(np.allclose(old, exact, rtol=0, atol=1e-10))}
    if not equivalence["allclose_atol_1e_10"] or equivalence["threshold_label_differences"]:
        raise AssertionError(f"Numerical optimization changed predictions: {equivalence}")
    write_json(directory / "equivalence.json", equivalence)
    write_json(directory / "contexts.json", contexts)
    pd.DataFrame(timings).to_csv(directory / "diagnostic_timing.csv", index=False)
    write_json(directory / "completion.json", {"states": len(records), "models": len(VARIANTS), "rows_each": len(exact),
               "timing_warning": "concurrent diagnostic timings; final benchmark must run serially",
               "query_truth_used": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    run(parser.parse_args().output)
