"""Retained inference component, consolidated from cpt_budget.v5.functional (v0.8-engineering)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from cpt_budget.config import write_json
from cpt_budget.models.base import validate_observed, validate_query_site
from cpt_budget.models.distribution import point_prediction_frame
from cpt_budget.types import PredictionBundle

def _profile_matrix(rows: pd.DataFrame, depth_indices=None):
    keys = ['site_id', 'location_id']
    if rows.duplicated([*keys, 'depth_index']).any():
        raise ValueError('Duplicate profile/depth observations')
    matrix = rows.pivot(index=keys, columns='depth_index', values='log_qc').sort_index()
    if depth_indices is not None:
        matrix = matrix.reindex(columns=depth_indices)
    if matrix.empty or not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError('Functional models require complete finite profiles on the fitted depth grid')
    return matrix

def _location_geometry(rows: pd.DataFrame, locations):
    for column in ('x_m', 'y_m'):
        if rows.groupby('location_id')[column].nunique().gt(1).any():
            raise ValueError('Coordinates must be constant within each CPT')
    geometry = rows.drop_duplicates('location_id').set_index('location_id')
    return geometry.loc[list(locations), ['x_m', 'y_m']].to_numpy(dtype=float)

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
        np.savez(directory / 'basis.npz', depth_indices=self.depth_indices, components=self.components, eigenvalues=self.eigenvalues)
        write_json(directory / 'basis_manifest.json', {'source_site_ids': self.source_site_ids, 'source_profile_count': self.source_profile_count, 'depth_indices': self.depth_indices.tolist(), 'n_components': len(self.components), 'retained_variance': self.retained_variance, 'covariance': 'within-site centered; each source site has equal total weight', 'rank_rule': '90 percent source variance, maximum 8 components', 'target_intercept': 'mean complete profile of revealed target CPTs', 'probability_output': 'none; point predictor only'})

class FunctionalProfilePredictor:

    def __init__(self, basis: ProfileBasis, *, method='krr'):
        if method not in {'krr', 'linear'}:
            raise ValueError('Functional method must be krr or linear')
        self.basis = basis
        self.method = method
        self.model_id = f'functional_fpca_{method}'
        self.observed = None

    def condition(self, observed: pd.DataFrame):
        self.observed = validate_observed(observed)
        profiles = _profile_matrix(self.observed, self.basis.depth_indices)
        locations = profiles.index.get_level_values('location_id')
        self.xy = _location_geometry(self.observed, locations)
        values = profiles.to_numpy(dtype=float)
        self.mean_profile = values.mean(axis=0)
        scores = (values - self.mean_profile) @ self.basis.components.T
        if self.method == 'krr':
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
            self.coefficients = np.linalg.solve(standardized.T @ standardized + np.eye(2), standardized.T @ scores)
        self.context_manifest = {'model_id': self.model_id, 'observed_location_ids': list(map(str, locations)), 'basis_source_sites': self.basis.source_site_ids, 'basis_rank': len(self.basis.components), 'basis_retained_source_variance': self.basis.retained_variance, 'fit_target_labels': 'revealed CPTs only', 'ridge': 0.1 if self.method == 'krr' else 1.0, 'length_scale_m': getattr(self, 'length_scale_m', None), 'length_rule': 'median nonzero pairwise revealed distance' if self.method == 'krr' else None}

    def predict(self, query_grid: pd.DataFrame):
        validate_query_site(query_grid, self.observed)
        if not np.isfinite(query_grid[['x_m', 'y_m']].to_numpy(dtype=float)).all():
            raise ValueError('Query coordinates must be finite')
        locations = query_grid.location_id.drop_duplicates().tolist()
        xy = _location_geometry(query_grid, locations)
        if self.method == 'krr':
            squared = ((xy[:, None, :] - self.xy[None, :, :]) ** 2).sum(axis=2)
            scores = np.exp(-squared / (2 * self.length_scale_m ** 2)) @ self.coefficients
        else:
            scores = (xy - self.xy_center) / self.xy_scale @ self.coefficients
        profiles = self.mean_profile + scores @ self.basis.components
        location_index = {location: i for i, location in enumerate(locations)}
        depth_index = {depth: i for i, depth in enumerate(self.basis.depth_indices)}
        if not query_grid.depth_index.isin(depth_index).all():
            raise ValueError('Query depth is outside the trained functional grid')
        predicted = profiles[query_grid.location_id.map(location_index).to_numpy(), query_grid.depth_index.map(depth_index).to_numpy()]
        return PredictionBundle(point_prediction_frame(query_grid, predicted), self.model_id)
