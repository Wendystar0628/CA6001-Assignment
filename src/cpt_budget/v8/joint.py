"""Batched whole-profile decisions, with exact reference and support-only variants.

The runtime improvement caches a geometry/group-aware LOO operator once during
condition(), then evaluates all query profile atoms in one NumPy batch. Timing
this operator with frozen amplitude bases is deliberately a component benchmark.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from threadpoolctl import threadpool_limits

from cpt_budget.config import PROJECT_ROOT, write_json
from cpt_budget.diagnostics.common import STATE_KEYS, iter_validation_states
from cpt_budget.models.base import validate_observed, validate_query_site
from cpt_budget.types import QUERY_COLUMNS
from cpt_budget.v7.profile_distribution import (
    WholeProfileMixture, mixture_summary, thickness_decision,
)

N_DEPTHS = 24
MARGIN = .05
RUN = PROJECT_ROOT / 'runs/v8-engineering-20261014'
BASE = 'v7_simplex_structural_s050_gp'
MODELS = {
    'v8_joint_reference_fast': 'reference',
    'v8_joint_support_alpha': 'support_alpha',
    'v8_joint_count_shrink': 'count_shrink',
    'v8_joint_count_bias': 'count_bias',
}


def _query_batch(query, base, observed, locations, xy):
    if set(query) - set(QUERY_COLUMNS):
        raise ValueError('Query must contain geometry only')
    validate_query_site(query, observed)
    if query.empty or query.duplicated(['location_id', 'depth_index']).any():
        raise ValueError('Unique complete query profiles required')
    if set(query.location_id) & set(locations):
        raise ValueError('Query profile must not be in revealed support')
    if ('near_group_id' in query and
            set(query.near_group_id) & set(observed.near_group_id)):
        raise ValueError('Query near group must not overlap revealed support')
    if query.groupby('location_id')[['x_m', 'y_m']].nunique().ne(1).any().any():
        raise ValueError('Query geometry must be constant across depth')
    base = np.asarray(base, float)
    if base.shape != (len(query),) or not np.isfinite(base).all():
        raise ValueError('Finite base values aligned to query rows are required')
    ordered = query.reset_index(drop=True).sort_values(['location_id', 'depth_index'], kind='stable')
    order = ordered.index.to_numpy()
    groups = ordered.groupby('location_id', sort=False)
    if groups.size().ne(N_DEPTHS).any():
        raise ValueError('Complete 24-depth query profiles required')
    if not np.array_equal(ordered.depth_index.to_numpy(), np.tile(np.arange(N_DEPTHS), len(groups))):
        raise ValueError('Complete 24-depth query profiles required')
    geometry = ordered.drop_duplicates('location_id')
    query_xy = geometry[['x_m', 'y_m']].to_numpy(float)
    if not np.isfinite(query_xy).all():
        raise ValueError('Finite query geometry required')
    distance = cdist(query_xy, xy)
    if (distance <= 1e-9).any():
        raise ValueError('Query cannot be colocated with revealed support')
    return ordered, order, geometry.location_id.tolist(), base[order].reshape(-1, N_DEPTHS), distance


def _lower_weighted_quantile(values, weights, q=.5):
    """Batched atom-axis lower empirical quantile; stable atom ordering."""
    values = np.asarray(values)
    weights = np.asarray(weights, float)
    if values.ndim == 2:
        values = values[:, :, None]
        squeeze = True
    else:
        squeeze = False
    indices = np.argsort(values, axis=1, kind='stable')
    ordered_values = np.take_along_axis(values, indices, axis=1)
    ordered_weights = np.take_along_axis(np.broadcast_to(weights[:, :, None], values.shape), indices, axis=1)
    passed = np.cumsum(ordered_weights, axis=1) >= q
    first = np.argmax(passed, axis=1)
    answer = np.take_along_axis(ordered_values, first[:, None, :], axis=1)[:, 0, :]
    return answer[:, 0] if squeeze else answer


def batch_thickness_decision(base, probability, count):
    """Vectorized version of the fixed-count event assignment and crossing rule."""
    base = np.asarray(base, float)
    probability = np.asarray(probability, float)
    count = np.asarray(count, int)
    if (base.ndim != 2 or base.shape != probability.shape or count.shape != (len(base),)
            or not np.isfinite(base).all() or not np.isfinite(probability).all()
            or ((probability < 0) | (probability > 1)).any()
            or ((count < 0) | (count > base.shape[1])).any()):
        raise ValueError('Finite aligned point/probability arrays and valid counts required')
    rank = np.lexsort((np.broadcast_to(np.arange(base.shape[1]), base.shape), base, -probability), axis=1)
    low = np.zeros_like(base, bool)
    np.put_along_axis(low, rank, np.arange(base.shape[1])[None, :] < count[:, None], axis=1)
    return np.where(low & (base >= 0), -MARGIN, np.where(~low & (base < 0), MARGIN, base))


def _finish(atoms, weights, base, order, locations, *, count_override=None):
    weights = weights / weights.sum(axis=1, keepdims=True)
    low = atoms < 0
    probability = np.clip(np.einsum('ns,nsd->nd', weights, low), 0., 1.)
    atom_counts = low.sum(axis=2)
    count = _lower_weighted_quantile(atom_counts, weights).astype(int)
    effective_count = count if count_override is None else np.asarray(count_override, int)
    point = batch_thickness_decision(base, probability, effective_count)
    point_flat, probability_flat = np.empty(len(order)), np.empty(len(order))
    point_flat[order], probability_flat[order] = point.ravel(), probability.ravel()
    return {'point': point_flat, 'probability': probability_flat, 'atoms': atoms,
            'weights': weights, 'atom_counts': atom_counts,
            'count_median': count, 'count_expected': (weights*atom_counts).sum(axis=1),
            'base_count': (base < 0).sum(axis=1), 'decision_count': effective_count,
            'locations': locations, 'order': order, 'base_matrix': base,
            'probability_matrix': probability}


class FastJointAdapter:
    """Condition once, predict arbitrary batches without per-depth pandas work."""
    def condition(self, observed):
        frame = validate_observed(observed)
        if 'near_group_id' not in frame or frame.near_group_id.isna().any():
            raise ValueError('Known near groups are required')
        if 'z_center_m' not in frame or not np.isfinite(frame.z_center_m).all():
            raise ValueError('Finite support depth coordinates are required')
        if frame.groupby('location_id')[['x_m', 'y_m', 'near_group_id']].nunique().ne(1).any().any():
            raise ValueError('Support profile geometry and near groups must be constant')
        self.observed = frame
        self.locations = sorted(frame.location_id.unique())
        profile = frame.pivot(index='location_id', columns='depth_index', values='log_qc')
        if list(profile.columns) != list(range(N_DEPTHS)) or profile.isna().any().any():
            raise ValueError('Complete 24-depth revealed profiles are required')
        geometry = frame.drop_duplicates('location_id').set_index('location_id').loc[self.locations]
        self.curves = profile.loc[self.locations].to_numpy(float)
        self.xy = geometry[['x_m', 'y_m']].to_numpy(float)
        self.groups = geometry.near_group_id.to_numpy()
        self.group_size = geometry.near_group_id.map(geometry.near_group_id.value_counts()).to_numpy(float)
        distance = cdist(self.xy, self.xy)
        nonzero = pdist(self.xy)
        nonzero = nonzero[nonzero > 1e-9]
        if not len(nonzero):
            raise ValueError('At least two distinct support coordinates required')
        self.length = float(np.median(nonzero))
        allowed = (distance > 1e-9) & (self.groups[:, None] != self.groups[None, :])
        candidate_distance = np.where(allowed, distance, np.inf)
        nearest = np.argsort(candidate_distance, axis=1, kind='stable')[:, :min(3, len(self.xy))]
        closest = np.take_along_axis(candidate_distance, nearest, axis=1)
        reference_count = np.isfinite(closest).sum(axis=1)
        if not reference_count.all():
            raise ValueError('Empirical errors require another independent revealed CPT at each depth')
        weight = (closest[:, :1] / closest)**2
        weight /= weight.sum(axis=1, keepdims=True)
        # Same normalized three-neighbour IDW; depth is a matrix axis, not a loop.
        self.error_curves = self.curves - np.einsum('nk,nkd->nd', weight, self.curves[nearest])
        self.loo_min_reference_count = int(reference_count.min())
        self._adaptation = None
        return self

    def batch_predict(self, query, base_values, kind='error', *, shrinkage=.5, policy='reference'):
        if not np.isfinite(shrinkage) or shrinkage < 0:
            raise ValueError('Finite nonnegative shrinkage required')
        _, order, locations, base, distance = _query_batch(query, base_values, self.observed, self.locations, self.xy)
        if policy != 'reference':
            if self._adaptation is None:
                self._adaptation = self.fit_support_adaptation()
            if policy == 'support_alpha':
                shrinkage = self._adaptation['alpha']
            elif policy not in ('count_shrink', 'count_bias'):
                raise ValueError('Unknown support adaptation policy')
        log_weight = -.5*(distance/self.length)**2 - np.log(self.group_size)[None, :]
        weights = np.exp(log_weight-log_weight.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        if kind == 'error':
            deviation = np.broadcast_to(self.error_curves, (len(base), *self.error_curves.shape))
        elif kind == 'analog':
            mean = weights @ self.curves
            deviation = self.curves[None, :, :] - mean[:, None, :]
        else:
            raise ValueError('Unknown complete-curve mixture kind')
        atoms = base[:, None, :] + shrinkage*deviation
        result = _finish(atoms, weights, base, order, locations)
        if policy in ('count_shrink', 'count_bias'):
            if policy == 'count_shrink':
                mix = self._adaptation['count_weight']
                desired = np.floor(mix*result['count_median'] + (1-mix)*result['base_count'] + .5).astype(int)
            else:
                desired = np.clip(result['count_median'] + self._adaptation['count_bias'], 0, N_DEPTHS)
            result = _finish(atoms, weights, base, order, locations, count_override=desired)
        return result

    def fit_support_adaptation(self):
        """Select modest count corrections using whole-group held-out support.

        The surrogate amplitude predictor is 3NN IDW, not the full structural
        ensemble. Its count transport may be inaccurate; this limitation is
        explicit in experiment metadata. Held groups never enter error atoms.
        """
        records = []
        for group in np.unique(self.groups):
            held = self.observed.near_group_id.eq(group)
            training = self.observed.loc[~held]
            if training.near_group_id.nunique() < 2:
                continue
            model = FastJointAdapter().condition(training)
            truth = self.observed.loc[held].sort_values(['location_id', 'depth_index'])
            query = truth[[c for c in QUERY_COLUMNS if c in truth]]
            geometry = truth.drop_duplicates('location_id')
            distance = cdist(geometry[['x_m', 'y_m']], model.xy)
            if (distance <= 1e-9).any():
                # Coordinate aliases must also be excluded from the context.
                alias = np.any(cdist(model.xy, geometry[['x_m', 'y_m']]) <= 1e-9, axis=1)
                training = training.loc[~training.location_id.isin(np.asarray(model.locations)[alias])]
                if training.near_group_id.nunique() < 2:
                    continue
                model = FastJointAdapter().condition(training)
                distance = cdist(geometry[['x_m', 'y_m']], model.xy)
            nearest = np.argsort(distance, axis=1, kind='stable')[:, :3]
            closest = np.take_along_axis(distance, nearest, axis=1)
            weight = (closest[:, :1]/closest)**2
            weight /= weight.sum(axis=1, keepdims=True)
            base = np.einsum('nk,nkd->nd', weight, model.curves[nearest])
            actual = truth.log_qc.to_numpy(float).reshape(-1, N_DEPTHS)
            for alpha in (.25, .5, .75):
                result = model.batch_predict(query, base.ravel(), shrinkage=alpha)
                for j, location in enumerate(result['locations']):
                    records.append({'group': str(group), 'location': location, 'alpha': alpha,
                                    'actual_count': int((actual[j] < 0).sum()),
                                    'predicted_count': int(result['count_median'][j]),
                                    'base_count': int((base[j] < 0).sum()),
                                    'group_weight': 1./len(geometry)})
        if not records:
            return dict(alpha=.5, count_weight=1., count_bias=0, held_profiles=0, losses=[])
        frame = pd.DataFrame(records)
        losses = []
        for alpha in (.25, .5, .75):
            sample = frame.loc[frame.alpha.eq(alpha)]
            loss = np.average(np.abs(sample.predicted_count-sample.actual_count), weights=sample.group_weight)
            losses.append(dict(alpha=alpha, count_mae=float(loss)))
        selected_alpha = min(losses, key=lambda r: (r['count_mae'], abs(r['alpha']-.5)))['alpha']
        sample = frame.loc[frame.alpha.eq(.5)]
        shrink_loss = []
        for mixing in (0., .5, 1.):
            prediction = np.floor(mixing*sample.predicted_count + (1-mixing)*sample.base_count + .5)
            loss = np.average(np.abs(prediction-sample.actual_count), weights=sample.group_weight)
            shrink_loss.append((float(loss), -mixing))
        selected_weight = -min(shrink_loss)[1]
        # Equal group mass, lower empirical median, clipped to at most 0.5 m.
        bias_values = sample.actual_count.to_numpy()-sample.predicted_count.to_numpy()
        bias_weights = sample.group_weight.to_numpy(float)
        bias_order = np.argsort(bias_values, kind='stable')
        cut = np.searchsorted(np.cumsum(bias_weights[bias_order]/bias_weights.sum()), .5, side='left')
        bias = int(np.clip(bias_values[bias_order[min(cut, len(bias_order)-1)]], -2, 2))
        return dict(alpha=float(selected_alpha), count_weight=float(selected_weight), count_bias=bias,
                    held_profiles=len(sample), losses=losses)


class ReferenceJointAdapter:
    """Legacy pandas/profile implementation under the same public API."""
    def condition(self, observed):
        self.model = WholeProfileMixture(observed)
        self.observed = self.model.observed
        return self

    def batch_predict(self, query, base_values, kind='error', *, shrinkage=.5, policy='reference'):
        if shrinkage != .5 or policy != 'reference':
            raise ValueError('The exact legacy reference has fixed shrinkage 0.5')
        ordered, order, locations, base, _ = _query_batch(
            query, base_values, self.observed, self.model.locations, self.model.xy)
        atoms, weights, points, probs, counts, expected = [], [], [], [], [], []
        for j, (_, profile) in enumerate(ordered.groupby('location_id', sort=False)):
            a, w = self.model.atoms(profile, base[j], kind)
            summary = mixture_summary(a, w)
            atoms.append(a); weights.append(w)
            counts.append(summary['count_median']); expected.append(summary['count_expected'])
            probs.append(summary['probability'])
            points.append(thickness_decision(base[j], summary['probability'], summary['count_median']))
        point_flat, p_flat = np.empty(len(order)), np.empty(len(order))
        point_flat[order], p_flat[order] = np.asarray(points).ravel(), np.asarray(probs).ravel()
        return dict(point=point_flat, probability=p_flat, atoms=np.asarray(atoms), weights=np.asarray(weights),
                    atom_counts=(np.asarray(atoms)<0).sum(axis=2), count_median=np.asarray(counts),
                    count_expected=np.asarray(expected), base_count=(base<0).sum(axis=1),
                    decision_count=np.asarray(counts), locations=locations, order=order,
                    base_matrix=base, probability_matrix=np.asarray(probs))


def make_adapter(implementation='fast'):
    if implementation == 'fast':
        return FastJointAdapter()
    if implementation == 'reference':
        return ReferenceJointAdapter()
    raise ValueError('Unknown joint adapter implementation')


class LiveJointPredictor:
    """Real structural inference plus whole-curve point/native-joint decisions.

    This implements the joint-risk-head variant. The historical empirical-head
    sibling has identical points but a separate Graph/TabPFN risk path, whose
    time must not be silently attributed to this pipeline.
    """
    def __init__(self, fold_id, *, implementation='fast', policy='reference'):
        from cpt_budget.v8.structural import new_factory, old_factory
        self.base = (new_factory if implementation == 'fast' else old_factory)(fold_id)
        self.joint = make_adapter(implementation)
        self.policy = policy
        self.model_id = f'v8_live_joint_{implementation}_{policy}'

    def condition(self, observed):
        self.base.condition(observed)
        self.joint.condition(observed)

    def predict(self, query_grid):
        from cpt_budget.models.distribution import point_prediction_frame
        from cpt_budget.types import PredictionBundle
        query = query_grid[[c for c in QUERY_COLUMNS if c in query_grid]].copy()
        base = self.base.predict(query).frame.pred_log_qc.to_numpy(float)
        result = self.joint.batch_predict(query, base, policy=self.policy)
        frame = point_prediction_frame(query, result['point'])
        frame['p_low_1'] = result['probability']
        return PredictionBundle(frame, self.model_id)


def old_factory(fold_id):
    return LiveJointPredictor(fold_id, implementation='reference')


def new_factory(fold_id):
    return LiveJointPredictor(fold_id, implementation='fast')


def protocol():
    return {'family': 'batched_error_joint', 'base': BASE, 'models': MODELS,
            'runtime': 'Single support validation/pivot; cache full group/coordinate-excluded LOO; vectorize all query atoms and decisions',
            'timing_scope': 'Joint operator only, with frozen amplitude inputs; not end-to-end',
            'quality_parameters_frozen_before_scores': True,
            'support_alpha': [.25, .5, .75], 'count_shrink_weight': [0., .5, 1.],
            'count_bias': 'Group-equal weighted support LOO median signed count residual, clipped to [-2,2] cells',
            'support_selector': 'Whole near-group held-out support; additional coordinate exclusion; 3NN IDW surrogate amplitude; count absolute error',
            'selector_ties': 'Alpha closest to 0.5; count weight closest to 1.0',
            'adaptation_limitation': 'Support LOO uses IDW surrogate, not the full structural amplitude; transferred corrections are exploratory',
            'point_count_vs_native_probability': 'Count adaptations change point labels, not native marginal probabilities',
            'outer_test_used': False, 'query_truth_in_predict': False,
            'evaluation_scope': 'Repeated frozen validation; support-only adaptation does not create independent validation'}


def run(output=None):
    from cpt_budget.v8.common import load_frozen, emit_candidate
    output = Path(output or RUN/'joint')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'method_protocol.json', protocol())
    wide = load_frozen()
    points = {name: np.full(len(wide), np.nan) for name in MODELS}
    probabilities = {name: np.full(len(wide), np.nan) for name in MODELS}
    benchmark, adaptations = [], []
    for state in iter_validation_states():
        mask = np.ones(len(wide), bool)
        for key in STATE_KEYS:
            mask &= wide[key].eq(state.metadata[key]).to_numpy()
        index = np.flatnonzero(mask)
        block = wide.iloc[index]
        query = block[[c for c in QUERY_COLUMNS if c in block]]
        base = block[BASE].to_numpy(float)
        timing = {}
        for label, cls in [('reference', ReferenceJointAdapter), ('fast', FastJointAdapter)]:
            start = perf_counter(); model = cls().condition(state.observed)
            timing[label+'_condition_ms'] = 1000*(perf_counter()-start)
            start = perf_counter(); result = model.batch_predict(query, base)
            timing[label+'_predict_ms'] = 1000*(perf_counter()-start)
            if label == 'reference':
                reference = result
            else:
                fast = model
                np.testing.assert_allclose(result['point'], reference['point'], rtol=0, atol=1e-12)
                np.testing.assert_allclose(result['probability'], reference['probability'], rtol=0, atol=1e-12)
                np.testing.assert_array_equal(result['count_median'], reference['count_median'])
                timing['point_max_abs_difference'] = float(np.max(np.abs(result['point']-reference['point'])))
                timing['probability_max_abs_difference'] = float(np.max(np.abs(result['probability']-reference['probability'])))
        for name, policy in MODELS.items():
            result = fast.batch_predict(query, base, policy=policy)
            points[name][index] = result['point']
            probabilities[name][index] = result['probability']
        benchmark.append({**state.metadata, **timing, 'query_profiles': len(result['locations'])})
        adaptations.append({**state.metadata, **{k: v for k, v in fast._adaptation.items() if k != 'losses'}})
        print(f'joint: {state.key}', flush=True)
    for name in MODELS:
        emit_candidate(name, points[name], wide=wide, output_root=output, probability=probabilities[name],
                       metadata={**protocol(), 'policy': MODELS[name]})
    pd.DataFrame(benchmark).to_csv(output/'component_benchmark.csv', index=False)
    pd.DataFrame(adaptations).to_csv(output/'support_adaptation.csv', index=False)
    write_json(output/'execution_summary.json', {'states': len(benchmark), 'rows_per_model': len(wide),
               'full_matrix_reference_equivalence': True, 'models': list(MODELS),
               'timing_scope': 'Component operator using cached structural amplitude; not end-to-end'})
    return pd.DataFrame(benchmark)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output)
