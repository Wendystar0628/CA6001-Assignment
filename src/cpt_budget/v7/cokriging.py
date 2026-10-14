"""Whole-profile multi-output Gaussian conditioning using source covariance.

The 24 depth cells are correlated outputs. Source sites supply the vertical
covariance, while only revealed target profiles supply the target intercept and
conditioning values. The LMC variant assigns different horizontal kernels to
different source-derived vertical modes. Native uncertainties are plug-in
Gaussian observation marginals, not calibrated confidence guarantees.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from ..config import PROJECT_ROOT, write_json
from ..models.base import validate_observed, validate_query_site
from ..models.distribution import gaussian_prediction_frame
from ..types import PredictionBundle, load_dataset
from ..v5.functional import _location_geometry, _profile_matrix


VARIANTS = ("full", "short", "rank6", "lmc", "loo")


@dataclass
class ProfileCovariance:
    depth_indices: np.ndarray
    covariance: np.ndarray
    source_site_ids: tuple[str, ...]
    source_profile_count: int

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "covariance.npz", depth_indices=self.depth_indices,
                 covariance=self.covariance)
        eig = np.linalg.eigvalsh(self.covariance)
        write_json(directory / "manifest.json", {
            "source_site_ids": self.source_site_ids,
            "source_profile_count": self.source_profile_count,
            "depth_indices": self.depth_indices.tolist(),
            "covariance": "within-source-site centered; source sites equally weighted",
            "minimum_eigenvalue": float(eig.min()),
            "trace": float(eig.sum()),
        })


def fit_profile_covariance(source_rows):
    profiles = _profile_matrix(source_rows)
    sites = profiles.index.get_level_values("site_id").unique()
    covariance = np.zeros((profiles.shape[1], profiles.shape[1]), dtype=float)
    for site in sites:
        y = profiles.xs(site).to_numpy(dtype=float)
        centered = y - y.mean(axis=0)
        covariance += centered.T @ centered / len(y) / len(sites)
    covariance = (covariance + covariance.T) * 0.5
    return ProfileCovariance(profiles.columns.to_numpy(), covariance,
                             tuple(map(str, sites)), len(profiles))


def horizontal_kernel(a, b, length):
    distance = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    u = np.sqrt(3.0) * distance / max(float(length), 1e-6)
    return (1 + u) * np.exp(-u)


def median_distance(xy):
    distance = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))
    upper = distance[np.triu_indices(len(xy), k=1)]
    upper = upper[upper > 1e-9]
    return float(np.median(upper)) if len(upper) else 1.0


class CoKrigingPredictor:
    def __init__(self, covariance, *, variant="full", nugget_fraction=0.03):
        if variant not in VARIANTS or nugget_fraction <= 0:
            raise ValueError("Unknown cokriging variant or invalid nugget")
        self.source = covariance
        self.variant = variant
        self.model_id = "v7_cokriging_" + variant
        self.observed = None
        c = covariance.covariance
        diagonal = np.maximum(np.diag(c), 1e-8)
        shrunk = 0.9 * c + 0.1 * np.diag(diagonal)
        eigenvalues, eigenvectors = np.linalg.eigh(c)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]
        if variant == "rank6":
            rank = min(6, len(c))
            low = (eigenvectors[:, :rank] * eigenvalues[:rank]) @ eigenvectors[:, :rank].T
            self.components = [(low + np.diag(np.maximum(np.diag(c - low), 1e-8)), 1.0)]
        elif variant == "lmc":
            rank = min(3, len(c))
            slow = (eigenvectors[:, :rank] * eigenvalues[:rank]) @ eigenvectors[:, :rank].T
            fast = c - slow
            fast = (fast + fast.T) * 0.5
            self.components = [(0.9 * slow, 1.5),
                               (0.9 * fast + 0.1 * np.diag(diagonal), 0.5)]
        else:
            self.components = [(shrunk, 0.5 if variant == "short" else 1.0)]
        self.nugget_variance = float(nugget_fraction * max(float(np.mean(diagonal)), 1e-8))
        self.jitter = max(1e-10, self.nugget_variance * 1e-8)

    def covariance_matrix(self, a, b, length):
        return sum(np.kron(horizontal_kernel(a, b, length * multiplier), c)
                   for c, multiplier in self.components)

    def _fit_arrays(self, xy, values, length):
        intercept = values.mean(axis=0)
        residual = (values - intercept).reshape(-1)
        covariance = self.covariance_matrix(xy, xy, length)
        covariance.flat[::len(covariance) + 1] += self.nugget_variance + self.jitter
        factor = cho_factor(covariance, lower=True, check_finite=False)
        alpha = cho_solve(factor, residual, check_finite=False)
        residual_error = float(np.max(np.abs(covariance @ alpha - residual)))
        return intercept, factor, alpha, residual_error

    def condition(self, observed):
        self.observed = validate_observed(observed)
        if set(map(str, self.observed.site_id)) & set(self.source.source_site_ids):
            raise ValueError("Source training and target conditioning sites must be disjoint")
        profiles = _profile_matrix(self.observed, self.source.depth_indices)
        ids = profiles.index.get_level_values("location_id")
        self.xy = _location_geometry(self.observed, ids)
        geometry = self.observed.drop_duplicates("location_id").set_index("location_id").loc[list(ids)]
        groups = geometry.near_group_id.to_numpy() if "near_group_id" in geometry else np.asarray(ids)
        values = profiles.to_numpy(dtype=float)
        base_length = median_distance(self.xy)
        self.loo_scores = []
        self.length_scale_m = base_length
        if self.variant == "loo" and len(values) > 2:
            # Every held-out whole profile is absent from the fold intercept,
            # conditioning labels and geometry-derived length estimate.
            for multiplier in (0.35, 0.7, 1.4):
                error = []
                for held in range(len(values)):
                    same_coordinate = np.linalg.norm(self.xy - self.xy[held], axis=1) < 1e-9
                    keep = (groups != groups[held]) & ~same_coordinate
                    if not np.any(keep):
                        raise ValueError("Support LOO needs at least two independent coordinate/near groups")
                    length = median_distance(self.xy[keep]) * multiplier
                    mean, _, alpha, _ = self._fit_arrays(self.xy[keep], values[keep], length)
                    cross = self.covariance_matrix(self.xy[[held]], self.xy[keep], length)
                    prediction = mean + (cross @ alpha).reshape(1, -1)[0]
                    error.append(np.mean(np.abs(prediction - values[held])))
                self.loo_scores.append({"length_multiplier": multiplier,
                                        "support_profile_mae": float(np.mean(error))})
            chosen = min(self.loo_scores, key=lambda row: row["support_profile_mae"])
            self.length_scale_m = base_length * chosen["length_multiplier"]
        self.mean_profile, self.factor, self.alpha, solve_error = self._fit_arrays(
            self.xy, values, self.length_scale_m)
        self.context_manifest = {
            "model_id": self.model_id,
            "observed_location_ids": list(map(str, ids)),
            "source_site_ids": self.source.source_site_ids,
            "length_scale_m": self.length_scale_m,
            "base_support_median_distance_m": base_length,
            "component_length_multipliers": [m for _, m in self.components],
            "nugget_variance": self.nugget_variance,
            "linear_solve_max_absolute_residual": solve_error,
            "support_loo_scores": self.loo_scores,
            "loo_exclusion": "whole matching near group and duplicate horizontal coordinates",
            "intercept": "mean currently revealed complete target profile",
            "probability_scope": "plug-in Gaussian observation marginals; intercept and covariance uncertainty omitted",
        }

    def predict(self, query_grid):
        validate_query_site(query_grid, self.observed)
        if not np.isfinite(query_grid[["x_m", "y_m"]].to_numpy(dtype=float)).all():
            raise ValueError("Query coordinates must be finite")
        locations = query_grid.location_id.drop_duplicates().tolist()
        query_xy = _location_geometry(query_grid, locations)
        cross = self.covariance_matrix(query_xy, self.xy, self.length_scale_m)
        predicted = self.mean_profile + (cross @ self.alpha).reshape(len(locations), -1)
        solved = cho_solve(self.factor, cross.T, check_finite=False)
        prior_diagonal = np.tile(sum(np.diag(c) for c, _ in self.components), len(locations))
        variance = prior_diagonal + self.nugget_variance - np.einsum("ij,ji->i", cross, solved)
        if variance.min(initial=0) < -1e-7:
            raise ValueError("Posterior variance is materially negative")
        std = np.sqrt(np.maximum(variance, 0)).reshape(len(locations), -1)
        loc_index = {loc: i for i, loc in enumerate(locations)}
        depth_index = {depth: i for i, depth in enumerate(self.source.depth_indices)}
        if not query_grid.depth_index.isin(depth_index).all():
            raise ValueError("Query depth is outside the fitted covariance grid")
        li = query_grid.location_id.map(loc_index).to_numpy()
        di = query_grid.depth_index.map(depth_index).to_numpy()
        frame = gaussian_prediction_frame(query_grid, predicted[li, di], std[li, di])
        return PredictionBundle(frame, self.model_id)


def run_cokriging(output_root=None, *, variants=VARIANTS):
    from ..diagnostics.common import diagnostic_config, run_predictor, source_folds
    config = diagnostic_config()
    dataset = load_dataset(config)
    output = Path(output_root or PROJECT_ROOT / "runs/v7-discovery-20261014/cokriging")
    covariances = {}
    for fold in source_folds():
        fid = int(fold["fold_id"])
        locations = dataset.locations()
        ids = locations.loc[locations.site_id.isin(fold["train_sites"]), "location_id"].tolist()
        rows = dataset.observed(ids)
        if not set(rows.site_id).issubset(set(fold["train_sites"])):
            raise ValueError("Source covariance crossed its training fold boundary")
        covariance = fit_profile_covariance(rows)
        covariance.save(output / "covariances" / f"fold_{fid}")
        covariances[fid] = covariance
    write_json(output / "method_protocol.json", {
        "model_ids": ["v7_cokriging_" + v for v in variants],
        "scope": "exploratory validation; no independent test claim",
        "source": "https://arxiv.org/abs/1106.6251",
        "gaussian_conditioning_reference": "https://gaussianprocess.org/gpml/chapters/RW2.pdf",
        "distinction": "correlated full-profile outputs; source-learned vertical covariance; LMC mode-dependent horizontal scales",
        "parameters": {"shrinkage_to_diagonal": 0.1, "nugget_fraction": 0.03,
                       "horizontal_kernel": "Matern 3/2", "length": "median revealed pair distance",
                       "short_multiplier": 0.5, "rank_cap": 6,
                       "lmc_long_rank": 3, "lmc_multipliers": [1.5, 0.5],
                       "whole_profile_support_loo_multipliers": [0.35, 0.7, 1.4]},
        "native_distribution": "Gaussian observation marginals conditional on plug-in intercept/covariance; no joint thickness interval",
        "failures": [],
    })
    all_metrics = []
    with threadpool_limits(limits=1):
        for variant in variants:
            directory = output / ("v7_cokriging_" + variant)
            result = run_predictor(lambda fid, v=variant: CoKrigingPredictor(covariances[fid], variant=v),
                                   directory, config=config)
            for frame in result.values():
                frame["run_id"] = "v7-discovery-20261014"
                frame["study_version"] = "0.7-discovery"
            result["predictions"].to_parquet(directory / "predictions.parquet", index=False)
            result["metrics"].to_csv(directory / "metrics.csv", index=False)
            for path in (directory / "states").glob("*.parquet"):
                frame = pd.read_parquet(path)
                frame["run_id"], frame["study_version"] = "v7-discovery-20261014", "0.7-discovery"
                frame.to_parquet(path, index=False)
            protocol_path = directory / "evaluation_protocol.json"
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            protocol.update(run_id="v7-discovery-20261014", study_version="0.7-discovery")
            write_json(protocol_path, protocol)
            all_metrics.append(result["metrics"])
    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(output / "all_metrics.csv", index=False)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    run_cokriging(args.output, variants=args.variants)
