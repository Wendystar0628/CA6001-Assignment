"""Retained inference component, consolidated from cpt_budget.v8.baselines (v0.8-engineering)."""
from __future__ import annotations
from time import perf_counter
import numpy as np
import pandas as pd
from cpt_budget.features import _check, depth_values, idw_value
from cpt_budget.models.base import validate_observed, validate_query_site
from cpt_budget.models.distribution import point_prediction_frame
from cpt_budget.types import PredictionBundle

class OptimizedIDWPredictor:
    """Real live point inference, with arrays rebuilt when support changes."""
    model_id = 'v8_idw_optimized'

    def __init__(self, config=None):
        settings = (config or {}).get('idw', {})
        if int(settings.get('nearest_neighbors', 3)) != 3 or float(settings.get('power', 2.0)) != 2.0:
            raise ValueError('This equivalent engineering control is fixed to original 3NN/power-2 IDW')
        self.observed = None
        self.stage_timings = {}

    def condition(self, observed):
        started = perf_counter()
        self.observed = validate_observed(observed)
        _check(self.observed, observed=True)
        geometry = self.observed[['location_id', 'x_m', 'y_m']].drop_duplicates().sort_values('location_id', kind='stable')
        matrix = self.observed.pivot(index='location_id', columns='depth_index', values='log_qc')
        self.complete_profiles = not geometry.location_id.duplicated().any() and np.isfinite(matrix.to_numpy(float)).all()
        self.depth_index = pd.Index(matrix.columns)
        if self.complete_profiles:
            self.xy = geometry[['x_m', 'y_m']].to_numpy(float)
            self.profiles = matrix.reindex(index=geometry.location_id).to_numpy(float)
            self.depth_groups = None
        else:
            self.xy = self.profiles = None
            self.depth_groups = {}
            for depth, rows in self.observed.groupby('depth_index', sort=False):
                rows = rows.sort_values('location_id', kind='stable')
                self.depth_groups[depth] = (rows[['x_m', 'y_m']].to_numpy(float), rows.log_qc.to_numpy(float))
        self.stage_timings = {'condition_seconds': perf_counter() - started}
        return self

    def predict(self, query_grid):
        started = perf_counter()
        validate_query_site(query_grid, self.observed)
        _check(query_grid)
        depth_values(query_grid)
        qxy, inverse = np.unique(query_grid[['x_m', 'y_m']].to_numpy(float), axis=0, return_inverse=True)
        depth = query_grid.depth_index.to_numpy()
        result = np.empty(len(query_grid), dtype=float)
        if self.complete_profiles:
            depth_column = self.depth_index.get_indexer(depth)
            if (depth_column < 0).any():
                raise ValueError('No revealed support at a requested query depth')
            distances = np.linalg.norm(self.xy[None, :, :] - qxy[:, None, :], axis=2)
            nearest = np.argsort(distances, axis=1, kind='stable')[:, :3]
            for qindex in range(len(qxy)):
                rows = np.flatnonzero(inverse == qindex)
                neighbors = nearest[qindex]
                selected_distances = distances[qindex, neighbors]
                for column in np.unique(depth_column[rows]):
                    value = idw_value(self.profiles[neighbors, column], selected_distances)
                    result[rows[depth_column[rows] == column]] = value
        else:
            for level in pd.unique(depth):
                if level not in self.depth_groups:
                    raise ValueError('No revealed support at a requested query depth')
                xy, values = self.depth_groups[level]
                rows = np.flatnonzero(depth == level)
                for qindex in np.unique(inverse[rows]):
                    distances = np.linalg.norm(xy - qxy[qindex], axis=1)
                    nearest = np.argsort(distances, kind='stable')[:3]
                    result[rows[inverse[rows] == qindex]] = idw_value(values[nearest], distances[nearest])
        self.stage_timings['predict_seconds'] = perf_counter() - started
        return PredictionBundle(point_prediction_frame(query_grid, result), self.model_id)
