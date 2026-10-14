"""Semantics-preserving engineering of the v4 guarded TabPFN residual.

No new weights or predictive hyperparameters are selected. Complete profiles
share neighbour geometry across depth, and the guarded point-only API decodes
only the required median instead of a discarded full distribution. Both paths
use the official estimator, identical preprocessing, checkpoint and precision.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from cpt_budget.config import PROJECT_ROOT, write_json
from cpt_budget.diagnostics.common import diagnostic_config, evaluated_frame, iter_validation_states, metrics_for
from cpt_budget.diagnostics.residual_adapter import PointGuardedResidualPredictor, guarded_residual_frame
from cpt_budget.features import _check, depth_values, idw_value
from cpt_budget.models.base import validate_observed, validate_query_site
from cpt_budget.models.tabpfn_adapter import geometry_only, make_estimator
from cpt_budget.types import PredictionBundle
from cpt_budget.v2.tabpfn import feature_array_v2
from cpt_budget.v4.evaluation import _checkpoint


RUN = PROJECT_ROOT / 'runs/v8-engineering-20261014/tabpfn'
SOURCE = PROJECT_ROOT / 'runs/v4-residual-20261014'
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
        self.depths = observed.groupby('depth_index').apply(
            lambda g: depth_values(g)[0], include_groups=False).sort_values()
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
        if not np.allclose(actual_z, self.z[None, :], atol=1e-9, rtol=0):
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
        if not np.allclose(qz, self.z[depth_pos], atol=1e-9, rtol=0):
            raise ValueError('Query and observed rows must share a common depth grid.')
        output = np.full((len(query), 24), np.nan)
        # Geometry grouping is independent of query labels and row order.
        groups = query.groupby(['location_id', 'x_m', 'y_m', 'geometry_scale_m'], sort=False).indices
        for (location_id, x, y, scale), rows in groups.items():
            distance = np.linalg.norm(self.xy - [x, y], axis=1)
            allowed = np.ones(len(distance), dtype=bool)
            if leave_location_out:
                allowed &= (self.ids_array != location_id) & (distance > 1e-9)
            values, distance = self.values[allowed], distance[allowed]
            all_depth = np.full((len(self.z), 24), np.nan)
            all_depth[:, 0] = self.z
            all_depth[:, 7:10] = 0
            all_depth[:, 14] = len(distance)
            all_depth[:, 15] = np.log1p(scale)
            if len(distance):
                nearest = np.argsort(distance, kind='stable')[:3]
                n = len(nearest)
                near, nd = values[nearest], distance[nearest]
                all_depth[:, 1:1+n] = near.T
                all_depth[:, 4:4+n] = np.log1p(nd / scale)
                all_depth[:, 7:7+n] = 1
                # Preserve the reference's one-dimensional reduction order.
                # GEMM-level rounding at ~1e-16 changes tied kernel fractions
                # under the official quantile/rank preprocessing, so a fully
                # vectorized dot product is not semantically equivalent here.
                columns = [values[:, d].copy() for d in range(len(self.z))]
                all_depth[:, 10:13] = [(v.mean(), v.std(ddof=0), np.median(v)) for v in columns]
                idw = np.asarray([idw_value(v[nearest], nd) for v in columns])
                all_depth[:, 13] = idw
                for j, length in enumerate((10., 50.)):
                    w = np.exp(-(distance - distance.min()) / length)
                    w /= w.sum()
                    all_depth[:, 16+j] = [w @ v for v in columns]
                    all_depth[:, 18+j] = [w @ (v < 0.) for v in columns]
                all_depth[1:, 20] = idw[:-1]
                all_depth[:-1, 21] = idw[1:]
                if len(self.z) > 1:
                    dz = np.diff(self.z)
                    slope = np.diff(idw) / dz
                    all_depth[0, 22], all_depth[-1, 22] = slope[0], slope[-1]
                    all_depth[1:-1, 22] = (idw[2:] - idw[:-2]) / (self.z[2:] - self.z[:-2])
                    all_depth[1:-1, 23] = 2 * np.diff(slope) / (self.z[2:] - self.z[:-2])
            output[rows] = all_depth[depth_pos[rows]]
        return output


class ReferenceGuardedPredictor(PointGuardedResidualPredictor):
    """Unmodified numerical reference with explicit optional weight loading."""
    def initialize(self):
        if self._residual._estimator is None:
            self._residual._estimator = make_estimator(self._residual.config, self._residual.artifact)
            self._residual._estimator._initialize_model_variables()


class FastGuardedPredictor:
    def __init__(self, config=None, *, artifact=None, model_id=MODEL_ID, alpha=.5):
        self.config = config or diagnostic_config()
        settings = self.config.get('idw', {})
        if int(settings.get('nearest_neighbors', 3)) != 3 or float(settings.get('power', 2)) != 2:
            raise ValueError('The exact shared-feature path requires the registered 3-neighbour power-2 IDW.')
        if alpha not in (.25, .5):
            raise ValueError('Guarded residual alpha must be .25 or .5.')
        self.artifact, self.alpha, self.model_id = artifact, alpha, model_id
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
        position, batch_size = 0, self.query_profiles_per_batch
        while position < len(ids):
            selected = ids[position:position+batch_size]
            indices = np.flatnonzero(query.location_id.isin(selected).to_numpy())
            try:
                with torch.no_grad():
                    residual[indices] = self._estimator.predict(features[indices], output_type='quantiles', quantiles=[.5])[0]
            except torch.OutOfMemoryError:
                if batch_size <= 1:
                    raise
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
                continue
            position += len(selected)
        return PredictionBundle(guarded_residual_frame(query, baseline, residual, self.alpha), self.model_id)


def old_factory(fold_id=0, config=None):
    artifact, _ = _checkpoint(SOURCE, int(fold_id), 'crps')
    return ReferenceGuardedPredictor(config or diagnostic_config(), artifact=artifact, alpha=.5,
                                     model_id='residual_ft_crps_guarded_a050')


def new_factory(fold_id=0, config=None):
    artifact, _ = _checkpoint(SOURCE, int(fold_id), 'crps')
    return FastGuardedPredictor(config or diagnostic_config(), artifact=artifact)


def synchronize():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def evaluate(output=RUN):
    """Actual old/new forwards on all states; no final prediction cache reuse."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = diagnostic_config()
    all_frames, records, comparison, timings = [], [], [], []
    last_fold = None
    old = new = None
    for state in iter_validation_states(config):
        fid = state.metadata['fold_id']
        if fid != last_fold:
            old = new = None
            gc.collect()
            old, new = old_factory(fid, config), new_factory(fid, config)
            for name, predictor in [('reference', old), ('optimized', new)]:
                synchronize(); start = perf_counter()
                predictor.initialize(); synchronize()
                timings.append({'fold_id': fid, 'state': None, 'path': name, 'stage': 'init', 'seconds': perf_counter()-start})
            last_fold = fid
        outputs = {}
        # Alternate order to reduce systematic warmup/order effects.
        order = [('reference', old), ('optimized', new)]
        if len(comparison) % 2:
            order.reverse()
        for name, predictor in order:
            synchronize(); start = perf_counter()
            predictor.condition(state.observed.copy()); synchronize()
            conditioned = perf_counter()
            bundle = predictor.predict(state.query.copy()); synchronize()
            ended = perf_counter()
            for stage, value in [('condition', conditioned-start), ('predict', ended-conditioned)]:
                timings.append({'fold_id': fid, 'state': state.key, 'path': name, 'stage': stage, 'seconds': value})
            outputs[name] = bundle.frame
            if name == 'optimized':
                frame = evaluated_frame(bundle, state)
                frame['study_version'] = '0.8-engineering'
                frame['run_id'] = 'v8-engineering-20261014'
                all_frames.append(frame)
                records.append(metrics_for(frame, config))
        delta = outputs['reference'].pred_log_qc.to_numpy() - outputs['optimized'].pred_log_qc.to_numpy()
        comparison.append({**state.metadata, 'state': state.key, 'rows': len(delta),
                           'max_abs_delta': float(np.abs(delta).max()),
                           'mean_abs_delta': float(np.abs(delta).mean()),
                           'label_changes': int(np.count_nonzero((outputs['reference'].pred_log_qc.to_numpy()<0) != (outputs['optimized'].pred_log_qc.to_numpy()<0)))})
        print(state.key, comparison[-1]['max_abs_delta'], flush=True)
    predictions = pd.concat(all_frames, ignore_index=True)
    predictions.to_parquet(output/'predictions.parquet', index=False)
    candidate = output / MODEL_ID
    candidate.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(candidate/'predictions.parquet', index=False)
    write_json(candidate/'protocol.json', {'model_id': MODEL_ID, 'reference': 'residual_ft_crps_guarded_a050',
        'scope': 'engineering_equivalence_only', 'states': len(comparison), 'rows': len(predictions),
        'same_weights_estimators_precision_query_batch': True, 'native_event_probability': False})
    pd.DataFrame(records).to_csv(output/'metrics.csv', index=False)
    pd.DataFrame(comparison).to_csv(output/'equivalence.csv', index=False)
    pd.DataFrame(timings).to_csv(output/'execution_timings.csv', index=False)
    write_json(output/'summary.json', {'model_id': MODEL_ID, 'reference': 'residual_ft_crps_guarded_a050',
        'states': len(comparison), 'rows': len(predictions), 'max_abs_delta': max(r['max_abs_delta'] for r in comparison),
        'label_changes': sum(r['label_changes'] for r in comparison), 'weights_changed': False,
        'query_batch_changed': False, 'n_estimators': 1, 'precision_changed': False,
        'timing_scope': 'GPU synchronized, separate initialization, actual fresh condition and forward on all 60 states; concurrent CPU activity may affect timings',
        'optimization': ['one neighbour calculation per whole profile', 'reuse feature IDW for support targets and query baseline',
                         'decode only required median, same native inverse-CDF', 'checkpoint loaded once per fold on both paths'],
        'no_prediction_cache': True})
    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=RUN)
    args = parser.parse_args()
    evaluate(args.output)
