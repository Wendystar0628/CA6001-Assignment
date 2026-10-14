"""Source-trained functional profile bases with spatial coefficient regression.

Each CPT is a complete curve, not 24 independent targets. The source sites
contribute equally to a within-site covariance; the frozen eigenvectors then
regularize new-site profile reconstruction. This is a discrete FPCA adaptation,
not a replication of spatial-PACE or a Gaussian-process baseline adjustment.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT, write_json
from ..models.base import validate_observed, validate_query_site
from ..models.distribution import point_prediction_frame
from ..types import PredictionBundle, load_dataset


def _profile_matrix(rows: pd.DataFrame, depth_indices=None):
    keys = ["site_id", "location_id"]
    if rows.duplicated([*keys, "depth_index"]).any():
        raise ValueError("Duplicate profile/depth observations")
    matrix = rows.pivot(index=keys, columns="depth_index", values="log_qc").sort_index()
    if depth_indices is not None:
        matrix = matrix.reindex(columns=depth_indices)
    if matrix.empty or not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError("Functional models require complete finite profiles on the fitted depth grid")
    return matrix


def _location_geometry(rows: pd.DataFrame, locations):
    for column in ("x_m", "y_m"):
        if rows.groupby("location_id")[column].nunique().gt(1).any():
            raise ValueError("Coordinates must be constant within each CPT")
    geometry = rows.drop_duplicates("location_id").set_index("location_id")
    return geometry.loc[list(locations), ["x_m", "y_m"]].to_numpy(dtype=float)


@dataclass
class ProfileBasis:
    depth_indices: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray
    retained_variance: float
    source_site_ids: tuple[str, ...]
    source_profile_count: int

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "basis.npz", depth_indices=self.depth_indices,
                 components=self.components, eigenvalues=self.eigenvalues)
        write_json(directory / "basis_manifest.json", {
            "source_site_ids": self.source_site_ids,
            "source_profile_count": self.source_profile_count,
            "depth_indices": self.depth_indices.tolist(),
            "n_components": len(self.components),
            "retained_variance": self.retained_variance,
            "covariance": "within-site centered; each source site has equal total weight",
            "rank_rule": "90 percent source variance, maximum 8 components",
            "target_intercept": "mean complete profile of revealed target CPTs",
            "probability_output": "none; point predictor only",
        })


def fit_profile_basis(source_rows: pd.DataFrame, *, variance_fraction=0.90, max_components=8):
    """Fit only on current-fold source training profiles, never target labels."""
    if not (0 < variance_fraction <= 1) or max_components < 1:
        raise ValueError("Invalid FPCA rank rule")
    profiles = _profile_matrix(source_rows)
    sites = profiles.index.get_level_values("site_id").unique()
    covariance = np.zeros((profiles.shape[1], profiles.shape[1]), dtype=float)
    for site_id in sites:
        values = profiles.xs(site_id).to_numpy(dtype=float)
        centered = values - values.mean(axis=0)
        covariance += centered.T @ centered / len(values) / len(sites)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    total = float(eigenvalues.sum())
    rank = (int(np.searchsorted(np.cumsum(eigenvalues) / total, variance_fraction)) + 1
            if total > 1e-14 else 1)
    rank = min(rank, max_components, profiles.shape[1])
    components = eigenvectors[:, order[:rank]].T
    # Fix eigenvector signs for readable/reproducible manifests.
    signs = np.sign(components[np.arange(rank), np.abs(components).argmax(axis=1)])
    components *= np.where(signs == 0, 1, signs)[:, None]
    return ProfileBasis(profiles.columns.to_numpy(), components, eigenvalues,
                        float(eigenvalues[:rank].sum() / total) if total > 0 else 1.0,
                        tuple(map(str, sites)), len(profiles))


class FunctionalProfilePredictor:
    def __init__(self, basis: ProfileBasis, *, method="krr"):
        if method not in {"krr", "linear"}:
            raise ValueError("Functional method must be krr or linear")
        self.basis = basis
        self.method = method
        self.model_id = f"functional_fpca_{method}"
        self.observed = None

    def condition(self, observed: pd.DataFrame):
        self.observed = validate_observed(observed)
        profiles = _profile_matrix(self.observed, self.basis.depth_indices)
        locations = profiles.index.get_level_values("location_id")
        self.xy = _location_geometry(self.observed, locations)
        values = profiles.to_numpy(dtype=float)
        self.mean_profile = values.mean(axis=0)
        scores = (values - self.mean_profile) @ self.basis.components.T
        if self.method == "krr":
            squared = ((self.xy[:, None, :] - self.xy[None, :, :]) ** 2).sum(axis=2)
            distances = np.sqrt(squared[np.triu_indices(len(self.xy), k=1)])
            distances = distances[distances > 1e-12]
            self.length_scale_m = float(np.median(distances)) if len(distances) else 1.0
            kernel = np.exp(-squared / (2 * self.length_scale_m ** 2))
            self.coefficients = np.linalg.solve(kernel + 0.1 * np.eye(len(kernel)), scores)
        else:
            self.xy_center = self.xy.mean(axis=0)
            self.xy_scale = np.maximum(self.xy.std(axis=0), 1.0)
            standardized = (self.xy - self.xy_center) / self.xy_scale
            self.coefficients = np.linalg.solve(standardized.T @ standardized + np.eye(2),
                                                 standardized.T @ scores)
        self.context_manifest = {
            "model_id": self.model_id,
            "observed_location_ids": list(map(str, locations)),
            "basis_source_sites": self.basis.source_site_ids,
            "basis_rank": len(self.basis.components),
            "basis_retained_source_variance": self.basis.retained_variance,
            "fit_target_labels": "revealed CPTs only",
            "ridge": 0.1 if self.method == "krr" else 1.0,
            "length_scale_m": getattr(self, "length_scale_m", None),
            "length_rule": "median nonzero pairwise revealed distance" if self.method == "krr" else None,
        }

    def predict(self, query_grid: pd.DataFrame):
        validate_query_site(query_grid, self.observed)
        if not np.isfinite(query_grid[["x_m", "y_m"]].to_numpy(dtype=float)).all():
            raise ValueError("Query coordinates must be finite")
        locations = query_grid.location_id.drop_duplicates().tolist()
        xy = _location_geometry(query_grid, locations)
        if self.method == "krr":
            squared = ((xy[:, None, :] - self.xy[None, :, :]) ** 2).sum(axis=2)
            scores = np.exp(-squared / (2 * self.length_scale_m ** 2)) @ self.coefficients
        else:
            scores = ((xy - self.xy_center) / self.xy_scale) @ self.coefficients
        profiles = self.mean_profile + scores @ self.basis.components
        location_index = {location: i for i, location in enumerate(locations)}
        depth_index = {depth: i for i, depth in enumerate(self.basis.depth_indices)}
        if not query_grid.depth_index.isin(depth_index).all():
            raise ValueError("Query depth is outside the trained functional grid")
        predicted = profiles[query_grid.location_id.map(location_index).to_numpy(),
                             query_grid.depth_index.map(depth_index).to_numpy()]
        return PredictionBundle(point_prediction_frame(query_grid, predicted), self.model_id)


def run_functional(output_root=None, *, methods=("krr", "linear")):
    """Run both predeclared methods on the common 60 frozen validation states."""
    from ..diagnostics.common import diagnostic_config, run_predictor, source_folds
    config = diagnostic_config()
    dataset = load_dataset(config)
    folds = {int(fold["fold_id"]): fold for fold in source_folds()}
    output = Path(output_root or PROJECT_ROOT / "runs/v5-exploration-20261014/functional")
    bases = {}
    for fold_id, fold in folds.items():
        locations = dataset.locations()
        ids = locations.loc[locations.site_id.isin(fold["train_sites"]), "location_id"].tolist()
        rows = dataset.observed(ids)
        if not set(rows.site_id).issubset(set(fold["train_sites"])):
            raise ValueError("FPCA training source crossed fold boundary")
        basis = fit_profile_basis(rows)
        basis.save(output / "bases" / f"fold_{fold_id}")
        bases[fold_id] = basis
    write_json(output / "method_protocol.json", {
        "run_id": "v5-exploration-20261014", "study_version": "0.5-exploration",
        "frozen_candidates": [f"functional_fpca_{method}" for method in methods],
        "new_direction": "shared full-depth profile covariance and low-rank spatial score prediction",
        "source": "https://link.springer.com/article/10.1007/s11222-016-9708-4",
        "source_scope": "motivates spatial prediction through functional principal-component scores; not exact replication",
        "kernel_ridge_reference": "https://scikit-learn.org/1.5/modules/generated/sklearn.kernel_ridge.KernelRidge.html",
        "parameters": {"variance_fraction": 0.90, "rank_cap": 8, "krr_ridge": 0.1,
                       "linear_ridge": 1.0, "krr_length": "median revealed distance"},
        "limitations": ["Low rank can smooth thin layers and unfamiliar target patterns.",
                        "Linear scores can extrapolate outside observed geometry.",
                        "Point predictions; no native probability or interval claims."],
    })
    results = {}
    for method in methods:
        directory = output / f"functional_fpca_{method}"
        result = run_predictor(lambda fold_id, method=method: FunctionalProfilePredictor(
            bases[fold_id], method=method), directory, config=config)
        for frame in result.values():
            frame["run_id"] = "v5-exploration-20261014"
            frame["study_version"] = "0.5-exploration"
        protocol_path = directory / "evaluation_protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol.update(run_id="v5-exploration-20261014", study_version="0.5-exploration")
        write_json(protocol_path, protocol)
        result["predictions"].to_parquet(directory / "predictions.parquet", index=False)
        result["metrics"].to_csv(directory / "metrics.csv", index=False)
        for path in (directory / "states").glob("*.parquet"):
            frame = pd.read_parquet(path)
            frame["run_id"] = "v5-exploration-20261014"
            frame["study_version"] = "0.5-exploration"
            frame.to_parquet(path, index=False)
        results[method] = result
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_functional(args.output)
