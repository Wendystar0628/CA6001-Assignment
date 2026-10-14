"""Retained optimized TabPFN residual component (v0.8-engineering).

Keep the original one-dimensional reduction order: tiny feature rounding
differences can change rank preprocessing. The retained factories use the
fold-specific trained checkpoint, not a silent pretrained replacement.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from cpt_budget.experiment import diagnostic_config
from cpt_budget.components.residual_point import guarded_residual_frame
from cpt_budget.features import _check, depth_values, idw_value
from cpt_budget.models.base import validate_observed, validate_query_site
from cpt_budget.models.tabpfn_adapter import geometry_only, make_estimator
from cpt_budget.types import PredictionBundle
from cpt_budget.components.tabular_features import feature_array_v2
MODEL_ID = 'v8_tabpfn_guarded_fast'

class ProfileFeatureContext:
    """Reuse sorted profile values; vectorize repeated depth computations.

The fast path requires a complete common depth grid and constant profile
coordinates. Unusual incomplete profiles retain the existing implementation.
No query measurements are read. Coincident support exclusion remains strict.
"""

    def __init__(self, observed):
        _check(observed, observed=True)
        self.observed = observed
        self.ids = sorted(observed.location_id.unique())
        self.depths = observed.groupby('depth_index').apply(lambda g: depth_values(g)[0], include_groups=False).sort_values()
        self.fast = observed.site_id.nunique() == 1
        self.fast &= len(observed) == len(self.ids) * len(self.depths)
        self.fast &= observed.groupby('location_id')[['x_m', 'y_m']].nunique().le(1).all().all()
        if not self.fast:
            return
        table = observed.set_index(['location_id', 'depth_index'])
        index = pd.MultiIndex.from_product([self.ids, self.depths.index])
        ordered = table.reindex(index)
        self.values = ordered.log_qc.to_numpy(float).reshape(len(self.ids), len(self.depths))
        if not np.isfinite(self.values).all():
            self.fast = False
            return
        self.xy = observed.groupby('location_id')[['x_m', 'y_m']].first().loc[self.ids].to_numpy(float)
        self.ids_array = np.asarray(self.ids)
        self.z = self.depths.to_numpy(float)
        actual_z = depth_values(ordered.reset_index()).reshape(len(self.ids), len(self.depths))
        if not np.allclose(actual_z, self.z[None, :], atol=1e-09, rtol=0):
            raise ValueError('A depth index must identify the same depth across revealed profiles.')

    def features(self, query, *, leave_location_out=False):
        _check(query)
        if not self.fast or not query.depth_index.isin(self.depths.index).all():
            return feature_array_v2(self.observed, query, leave_location_out=leave_location_out)
        if not query.site_id.isin(self.observed.site_id.unique()).all():
            return feature_array_v2(self.observed, query, leave_location_out=leave_location_out)
        if query.empty:
            return np.empty((0, 24), dtype=float)
        qz = depth_values(query)
        depth_pos = self.depths.index.get_indexer(query.depth_index)
        if not np.allclose(qz, self.z[depth_pos], atol=1e-09, rtol=0):
            raise ValueError('Query and observed rows must share a common depth grid.')
        output = np.full((len(query), 24), np.nan)
        groups = query.groupby(['location_id', 'x_m', 'y_m', 'geometry_scale_m'], sort=False).indices
        for (location_id, x, y, scale), rows in groups.items():
            distance = np.linalg.norm(self.xy - [x, y], axis=1)
            allowed = np.ones(len(distance), dtype=bool)
            if leave_location_out:
                allowed &= (self.ids_array != location_id) & (distance > 1e-09)
            values, distance = (self.values[allowed], distance[allowed])
            all_depth = np.full((len(self.z), 24), np.nan)
            all_depth[:, 0] = self.z
            all_depth[:, 7:10] = 0
            all_depth[:, 14] = len(distance)
            all_depth[:, 15] = np.log1p(scale)
            if len(distance):
                nearest = np.argsort(distance, kind='stable')[:3]
                n = len(nearest)
                near, nd = (values[nearest], distance[nearest])
                all_depth[:, 1:1 + n] = near.T
                all_depth[:, 4:4 + n] = np.log1p(nd / scale)
                all_depth[:, 7:7 + n] = 1
                columns = [values[:, d].copy() for d in range(len(self.z))]
                all_depth[:, 10:13] = [(v.mean(), v.std(ddof=0), np.median(v)) for v in columns]
                idw = np.asarray([idw_value(v[nearest], nd) for v in columns])
                all_depth[:, 13] = idw
                for j, length in enumerate((10.0, 50.0)):
                    w = np.exp(-(distance - distance.min()) / length)
                    w /= w.sum()
                    all_depth[:, 16 + j] = [w @ v for v in columns]
                    all_depth[:, 18 + j] = [w @ (v < 0.0) for v in columns]
                all_depth[1:, 20] = idw[:-1]
                all_depth[:-1, 21] = idw[1:]
                if len(self.z) > 1:
                    dz = np.diff(self.z)
                    slope = np.diff(idw) / dz
                    all_depth[0, 22], all_depth[-1, 22] = (slope[0], slope[-1])
                    all_depth[1:-1, 22] = (idw[2:] - idw[:-2]) / (self.z[2:] - self.z[:-2])
                    all_depth[1:-1, 23] = 2 * np.diff(slope) / (self.z[2:] - self.z[:-2])
            output[rows] = all_depth[depth_pos[rows]]
        return output

class FastGuardedPredictor:

    def __init__(self, config=None, *, artifact=None, model_id=MODEL_ID, alpha=0.5):
        self.config = config or diagnostic_config()
        settings = self.config.get('idw', {})
        if int(settings.get('nearest_neighbors', 3)) != 3 or float(settings.get('power', 2)) != 2:
            raise ValueError('The exact shared-feature path requires the registered 3-neighbour power-2 IDW.')
        if alpha not in (0.25, 0.5):
            raise ValueError('Guarded residual alpha must be .25 or .5.')
        self.artifact, self.alpha, self.model_id = (artifact, alpha, model_id)
        self._estimator = None
        self._observed = None
        self._needs_fit = True
        self.query_profiles_per_batch = min(8, int(self.config.get('episodes', {}).get('query_profiles_per_batch', 8)))
        if self.query_profiles_per_batch < 1:
            raise ValueError('query_profiles_per_batch must be positive')

    def initialize(self):
        if self._estimator is None:
            self._estimator = make_estimator(self.config, self.artifact)
            self._estimator._initialize_model_variables()

    def condition(self, observed):
        self._observed = validate_observed(observed)
        self._features = ProfileFeatureContext(self._observed)
        self._support = self._features.features(geometry_only(self._observed), leave_location_out=True)
        self._targets = self._observed.log_qc.to_numpy(float) - self._support[:, 13]
        if not np.isfinite(self._targets).all():
            raise ValueError('Residual context requires an independent revealed CPT.')
        self._needs_fit = True

    def predict(self, query_grid):
        validate_query_site(query_grid, self._observed)
        query = geometry_only(query_grid).reset_index(drop=True)
        features = self._features.features(query)
        baseline = features[:, 13]
        if query.empty:
            return PredictionBundle(guarded_residual_frame(query, baseline, np.array([]), self.alpha), self.model_id)
        self.initialize()
        if self._needs_fit:
            self._estimator.fit(self._support, self._targets)
            for model in self._estimator.models_:
                model.eval()
                for p in model.parameters():
                    p.requires_grad_(False)
            self._needs_fit = False
        import torch
        ids = list(query.location_id.drop_duplicates())
        residual = np.empty(len(query))
        position, batch_size = (0, self.query_profiles_per_batch)
        while position < len(ids):
            selected = ids[position:position + batch_size]
            indices = np.flatnonzero(query.location_id.isin(selected).to_numpy())
            try:
                with torch.no_grad():
                    residual[indices] = self._estimator.predict(features[indices], output_type='quantiles', quantiles=[0.5])[0]
            except torch.OutOfMemoryError:
                if batch_size <= 1:
                    raise
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
                continue
            position += len(selected)
        return PredictionBundle(guarded_residual_frame(query, baseline, residual, self.alpha), self.model_id)
