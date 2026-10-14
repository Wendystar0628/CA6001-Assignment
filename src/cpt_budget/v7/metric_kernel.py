"""Pair-trained profile transport and support-supervised spatial metric learning.

No query profile is required: source pair fits use source training sites, and
the target metric uses currently revealed profiles only. Positive weights are
normalised averaging weights, not a claimed positive-definite GP covariance.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from ..config import write_json
from ..diagnostics.common import diagnostic_config, iter_validation_states, source_folds
from ..models.base import validate_observed, validate_query_site
from ..models.distribution import point_prediction_frame
from ..types import PredictionBundle, load_dataset
from ..v5.functional import _location_geometry, _profile_matrix


METHODS = ("source_pair_variance", "support_psd_metric", "source_pair_transport",
           "metric_pair_transport", "euclidean_all_control")


def radial_features(radius):
    radius = np.asarray(radius, dtype=float).reshape(-1)
    return np.column_stack([np.ones(len(radius)), *[
        -np.expm1(-radius / scale) for scale in (0.25, 1.0, 4.0)]])


def geometry_scale(xy):
    distances = np.sqrt(((xy[:, None] - xy[None, :]) ** 2).sum(axis=-1))
    values = distances[np.triu_indices(len(xy), k=1)]
    values = values[values > 1e-8]
    return float(np.median(values)) if len(values) else 1.0


def learn_support_metric(xy, profiles):
    """Fit 3 quadratic coefficients, shrink to Euclidean, project to SPD.

    Only a directional metric is retained (unit determinant); condition number
    is capped at 4. A ridge penalty equal to pair count makes small contexts
    conservative. The metric is not fitted against validation predictions.
    """
    xy, profiles = np.asarray(xy, float), np.asarray(profiles, float)
    if len(xy) < 3:
        return np.eye(2)
    first, second = np.triu_indices(len(xy), k=1)
    delta = (xy[first] - xy[second]) / geometry_scale(xy)
    design = np.column_stack([delta[:, 0] ** 2, 2 * delta[:, 0] * delta[:, 1],
                              delta[:, 1] ** 2])
    differences = np.mean((profiles[first] - profiles[second]) ** 2, axis=1)
    mean_difference = float(differences.mean())
    if mean_difference < 1e-12:
        return np.eye(2)
    target = differences / mean_difference * np.mean(np.sum(delta ** 2, axis=1))
    penalty = float(len(target))
    coefficients = np.linalg.solve(design.T @ design + penalty * np.eye(3),
                                    design.T @ target + penalty * np.array([1., 0., 1.]))
    matrix = np.array([[coefficients[0], coefficients[1]],
                       [coefficients[1], coefficients[2]]])
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    upper = max(float(eigenvalues.max()), 1e-4)
    eigenvalues = np.clip(eigenvalues, upper / 4, upper)
    eigenvalues /= np.sqrt(np.prod(eigenvalues))
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def transport_features(anchor_profiles, radius):
    """One row per anchor/query pair and depth; anchor is observed, query is xy."""
    values = np.asarray(anchor_profiles, float)
    radius = np.asarray(radius, float).reshape(-1)
    if values.ndim != 2 or len(values) != len(radius):
        raise ValueError("One full anchor profile is required for each pair")
    count, depth_count = values.shape
    previous = np.column_stack([values[:, :1], values[:, :-1]])
    following = np.column_stack([values[:, 1:], values[:, -1:]])
    repeated_radius = np.repeat(np.log1p(radius), depth_count)
    return np.column_stack([
        repeated_radius,
        np.tile(np.linspace(0, 1, depth_count), count),
        values.ravel(), (values - previous).ravel(), (following - values).ravel(),
        np.repeat(np.mean(values, axis=1), depth_count),
        np.repeat(np.quantile(values, 0.25, axis=1), depth_count),
        np.repeat(np.min(values, axis=1), depth_count),
    ])


@dataclass
class SourcePairModel:
    depth_indices: np.ndarray
    variogram_coefficients: np.ndarray
    transport: HistGradientBoostingRegressor
    manifest: dict


def fit_source_pairs(source_rows):
    """Train within-site ordered pairs with equal total source-site weight."""
    profiles = _profile_matrix(source_rows)
    sites = profiles.index.get_level_values("site_id").unique()
    source_features, source_targets, source_weights = [], [], []
    radii, difference_sq, pair_weights = [], [], []
    site_counts = {}
    for site in sites:
        site_profiles = profiles.xs(site)
        values = site_profiles.to_numpy(dtype=float)
        if len(values) < 2:
            continue
        geometry = source_rows.loc[source_rows.site_id.eq(site)]
        xy = _location_geometry(geometry, site_profiles.index)
        first, second = np.where(~np.eye(len(values), dtype=bool))
        radius = np.linalg.norm(xy[first] - xy[second], axis=1) / geometry_scale(xy)
        features = transport_features(values[first], radius)
        target = (values[second] - values[first]).ravel()
        # Equal site weight, not equal pair weight across differently sized sites.
        source_features.append(features)
        source_targets.append(target)
        source_weights.append(np.full(len(target), 1 / len(target)))
        radii.append(radius)
        difference_sq.append(0.5 * (values[second] - values[first]) ** 2)
        pair_weights.append(np.full(len(radius), 1 / len(radius)))
        site_counts[str(site)] = {"profiles": len(values), "ordered_pairs": len(radius)}
    if not source_features:
        raise ValueError("At least one source site needs two complete profiles")
    weights = np.concatenate(source_weights)
    weights *= len(weights) / weights.sum()
    model = HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.05,
        max_iter=80, max_leaf_nodes=7, max_depth=3, min_samples_leaf=80,
        l2_regularization=1.0, early_stopping=False, random_state=20261014)
    with threadpool_limits(limits=1):
        model.fit(np.concatenate(source_features), np.concatenate(source_targets), sample_weight=weights)
    radius = np.concatenate(radii)
    differences = np.concatenate(difference_sq)
    radial = radial_features(radius)
    weights_pair = np.concatenate(pair_weights)
    sqrt_weight = np.sqrt(weights_pair / weights_pair.mean())
    coefficients = []
    # Three contiguous depth bands limit the fit to twelve nonnegative coefficients.
    for indices in np.array_split(np.arange(profiles.shape[1]), 3):
        target = np.mean(differences[:, indices], axis=1)
        coefficients.append(nnls(radial * sqrt_weight[:, None], target * sqrt_weight)[0])
    manifest = {
        "source_site_ids": list(map(str, sites)), "source_profile_count": len(profiles),
        "site_counts": site_counts, "training_cell_pairs": len(weights),
        "depth_indices": profiles.columns.to_list(),
        "transport_features": ["log1p(normalised_distance)", "normalised_depth",
            "anchor_log_qc", "anchor_previous_difference", "anchor_next_difference",
            "anchor_profile_mean", "anchor_profile_q25", "anchor_profile_min"],
        "source_weighting": "equal total weight per source training site",
        "transport_model": "HistGradientBoosting absolute error; 80 trees, depth <= 3, <= 7 leaves",
        "variogram_model": "3 depth bands, 4 nonnegative radial terms each",
        "variogram_coefficients": np.asarray(coefficients).tolist(),
        "truth_boundary": "source train sites only; query profile never available",
    }
    return SourcePairModel(profiles.columns.to_numpy(), np.asarray(coefficients), model, manifest)


class MetricKernelPredictor:
    def __init__(self, source_model=None, *, method="support_psd_metric"):
        if method not in METHODS or (method not in {"support_psd_metric", "euclidean_all_control"}
                                     and source_model is None):
            raise ValueError("Unknown method or missing source pair model")
        self.method, self.model_id, self.source_model = method, f"v7_{method}", source_model
        self.observed = None

    def condition(self, observed):
        self.observed = validate_observed(observed)
        depths = None if self.source_model is None else self.source_model.depth_indices
        profile_table = _profile_matrix(self.observed, depths)
        self.depth_indices = profile_table.columns.to_numpy()
        self.values = profile_table.to_numpy(dtype=float)
        self.xy = _location_geometry(self.observed, profile_table.index.get_level_values("location_id"))
        self.scale = geometry_scale(self.xy)
        self.metric = (learn_support_metric(self.xy, self.values)
            if self.method in {"support_psd_metric", "metric_pair_transport"} else np.eye(2))
        self.context_manifest = {
            "observed_location_ids": profile_table.index.get_level_values("location_id").to_list(),
            "metric": self.metric.tolist(), "metric_eigenvalues": np.linalg.eigvalsh(self.metric).tolist(),
            "geometry_scale_m": self.scale, "query_covariates": "geometry and depth only",
        }

    def predict(self, query):
        validate_query_site(query, self.observed)
        locations = query.location_id.drop_duplicates().tolist()
        qxy = _location_geometry(query, locations)
        delta = qxy[:, None] - self.xy[None]
        squared = np.einsum("qni,ij,qnj->qn", delta, self.metric, delta)
        radius = np.linalg.norm(delta, axis=-1) / self.scale
        if self.method == "source_pair_variance":
            bands = np.empty(len(self.depth_indices), dtype=int)
            for band, indices in enumerate(np.array_split(np.arange(len(bands)), 3)):
                bands[indices] = band
            variance = radial_features(radius.ravel()) @ self.source_model.variogram_coefficients.T
            variance = variance[:, bands].reshape(len(locations), len(self.xy), len(bands))
            weights = 1 / np.maximum(variance, 0.02)
        else:
            weights = 1 / np.maximum(squared[:, :, None], 1e-8)
        weights = weights / weights.sum(axis=1, keepdims=True)
        proposals = np.broadcast_to(self.values[None], (len(locations), *self.values.shape)).copy()
        if self.method in {"source_pair_transport", "metric_pair_transport"}:
            anchors = np.tile(self.values, (len(locations), 1))
            features = transport_features(anchors, radius.ravel())
            with threadpool_limits(limits=1):
                correction = self.source_model.transport.predict(features).reshape(proposals.shape)
            # Modest source correction, vanishing at a known coordinate.
            correction = 0.5 * np.clip(correction, -0.75, 0.75) * (1 - np.exp(-radius[:, :, None]))
            proposals += correction
        predicted = np.sum(weights * proposals, axis=1)
        # Exact spatial observations remain exact for every candidate.
        for i in range(len(locations)):
            coincident = np.flatnonzero(np.linalg.norm(delta[i], axis=1) < 1e-8)
            if len(coincident):
                predicted[i] = self.values[coincident].mean(axis=0)
        location_index = {value: i for i, value in enumerate(locations)}
        depth_index = {value: i for i, value in enumerate(self.depth_indices)}
        if not query.depth_index.isin(depth_index).all():
            raise ValueError("Query depth is outside fitted grid")
        output = predicted[query.location_id.map(location_index).to_numpy(),
                           query.depth_index.map(depth_index).to_numpy()]
        return PredictionBundle(point_prediction_frame(query, output), self.model_id)


def run(output_root=None):
    from .common import RUN, KEYS, emit_candidate, load_frozen
    output = Path(output_root or RUN / "metric_kernel")
    config = diagnostic_config()
    dataset = load_dataset(config)
    locations = dataset.locations()
    models = {}
    for fold in source_folds():
        fold_id = int(fold["fold_id"])
        source_ids = locations.loc[locations.site_id.isin(fold["train_sites"]), "location_id"]
        model = fit_source_pairs(dataset.observed(source_ids.tolist()))
        if not set(model.manifest["source_site_ids"]).issubset(set(fold["train_sites"])):
            raise ValueError("Source pair fit crossed fold boundary")
        models[fold_id] = model
        write_json(output / f"source_fold_{fold_id}.json", model.manifest)
    wide = load_frozen()
    parts = {method: [] for method in METHODS}
    contexts = []
    for state in iter_validation_states(config):
        for method in METHODS:
            predictor = MetricKernelPredictor(models[state.metadata["fold_id"]], method=method)
            predictor.condition(state.observed)
            frame = predictor.predict(state.query).frame
            for key in KEYS:
                if key in state.metadata:
                    frame[key] = state.metadata[key]
            parts[method].append(frame[KEYS + ["pred_log_qc"]])
            contexts.append({"state": state.key, "method": method, **predictor.context_manifest})
        print(f"metric_kernel: {state.key}", flush=True)
    protocol = {
        "family": "source pair transport and support supervised PSD geometry",
        "methods": list(METHODS), "fit_scope": "source train + revealed target CPTs only",
        "matched_control": "euclidean_all_control is ordinary all-neighbour IDW, not a new algorithm",
        "query_scope": "location, horizontal coordinates, depth only",
        "source_transport": {"correction_weight": 0.5, "correction_clip_log_qc": 0.75,
            "model": "absolute-error HGB 80 iterations, max depth 3, max 7 leaves, min leaf 80"},
        "support_metric": {"ridge": "pair count", "condition_number_cap": 4,
            "determinant": 1, "target": "mean squared full-profile difference"},
        "limitations": ["Pair observations are dependent, so pair count is not sample size.",
            "Source relative-distance patterns need not transfer to different geology.",
            "Positive interpolation weights are not a GP covariance or uncertainty model."],
        "references": [
            "https://papers.nips.cc/paper_files/paper/2002/hash/c3e4035af2a1cde9f21e1ae1951ac80b-Abstract.html",
            "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html"],
        "reference_scope": "metric learning motivation and official estimator; adaptations, not paper replications",
    }
    for method, frames in parts.items():
        aligned = wide[KEYS].merge(pd.concat(frames), on=KEYS, validate="one_to_one")
        emit_candidate(f"v7_{method}", aligned.pred_log_qc.to_numpy(), wide=wide,
                       output_root=output, metadata=protocol)
    write_json(output / "protocol.json", protocol)
    write_json(output / "contexts.json", contexts)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.output)
