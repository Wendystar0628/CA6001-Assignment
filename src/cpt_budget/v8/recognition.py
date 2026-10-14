"""Support-only localization and amplitude repair for the recognition finalist.

The original point decision is the analog whole-curve count median around the
support-adaptive structural stack. Its independent empirical-graph risk head is
kept fixed in exports, allowing point improvements to be interpreted separately.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from threadpoolctl import threadpool_limits

from cpt_budget.config import write_json
from cpt_budget.diagnostics.common import STATE_KEYS, iter_validation_states
from cpt_budget.models.base import validate_observed
from cpt_budget.types import QUERY_COLUMNS
from cpt_budget.v7.profile_distribution import thickness_decision, weighted_quantile


MAX_LOCAL_WEIGHT = .35
PROJECTION_WEIGHT = .5
VARIANTS = ('localize', 'amplitude', 'recall60')
MODELS = {
    'v8_recognition_fast': 'exact',
    'v8_recognition_localize': 'localize',
    'v8_recognition_amplitude': 'amplitude',
    'v8_recognition_recall60': 'recall60',
}


def depth_stencil(n=24):
    """Unit-sum three-cell kernel; no extrapolation at the two depth edges."""
    d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    weight = np.where(d == 0, .5, np.where(d == 1, .25, 0.))
    return weight / weight.sum(axis=1, keepdims=True)


class LocalEventEvidence:
    """Gaussian geometry/depth event evidence with whole-group LOO reliability.

Reliability measures whether spatial weighting predicts revealed events better
than a group-balanced depth climatology. No held-out query observation enters
this computation. Negative skill yields zero local correction.
"""

    def __init__(self, observed):
        self.observed = validate_observed(observed)
        frame = self.observed
        if 'near_group_id' not in frame or frame.near_group_id.isna().any():
            raise ValueError('Known support near groups are required')
        self.locations = sorted(frame.location_id.unique())
        curve = frame.pivot(index='location_id', columns='depth_index', values='log_qc')
        if list(curve.columns) != list(range(24)) or curve.isna().any().any():
            raise ValueError('Complete 24-cell support profiles are required')
        self.curves = curve.loc[self.locations].to_numpy(float)
        geometry = frame.drop_duplicates('location_id').set_index('location_id').loc[self.locations]
        self.xy = geometry[['x_m', 'y_m']].to_numpy(float)
        self.groups = geometry.near_group_id.to_numpy()
        self.group_size = geometry.near_group_id.map(geometry.near_group_id.value_counts()).to_numpy(float)
        distance = pdist(self.xy)
        positive = distance[distance > 1e-9]
        if not len(positive):
            raise ValueError('Independent support positions are required')
        self.length = float(np.median(positive))
        self.stencil = depth_stencil()
        self.events = (self.curves < 0.).astype(float)
        self.smooth_events = self.events @ self.stencil.T
        pairwise = cdist(self.xy, self.xy)
        allowed = ((self.groups[:, None] != self.groups[None, :]) & (pairwise > 1e-9))
        if not allowed.any(axis=1).all():
            raise ValueError('Whole-group LOO requires another independent support')
        spatial = np.exp(-.5 * (pairwise/self.length)**2) / self.group_size[None, :]
        spatial *= allowed
        spatial /= spatial.sum(axis=1, keepdims=True)
        climatology = allowed / self.group_size[None, :]
        climatology /= climatology.sum(axis=1, keepdims=True)
        self.loo_local_probability = spatial @ self.smooth_events
        self.loo_reference_probability = climatology @ self.smooth_events
        unit_weight = 1./self.group_size
        unit_weight /= unit_weight.sum()
        local_error = unit_weight @ (self.loo_local_probability-self.events)**2
        reference_error = unit_weight @ (self.loo_reference_probability-self.events)**2
        # Neighbor smoothing stabilizes skill estimation with only 5--13 CPTs.
        local_error = self.stencil @ local_error
        reference_error = self.stencil @ reference_error
        self.skill = np.clip(1.-(local_error+1e-8)/(reference_error+1e-8), 0., 1.)
        self.local_weight = MAX_LOCAL_WEIGHT*self.skill
        self.loo_min_reference_count = int(allowed.sum(axis=1).min())

    def blend(self, probability, weights):
        p, w = np.asarray(probability, float), np.asarray(weights, float)
        if p.shape != (24,) or w.shape != (len(self.locations),):
            raise ValueError('A full event profile and aligned support weights are required')
        local = w @ self.smooth_events
        return np.clip((1.-self.local_weight)*p+self.local_weight*local, 0., 1.)

    def project_crossings(self, base, decision, weights):
        """Repair only newly crossed amplitudes using observed same-class values."""
        base, decision = np.asarray(base, float), np.asarray(decision, float)
        output = decision.copy()
        changed = (base < 0.) != (decision < 0.)
        for depth in np.flatnonzero(changed):
            desired_low = decision[depth] < 0.
            weight = np.asarray(weights, float)[:, None]*self.stencil[depth][None, :]
            selected = ((self.curves < 0.) == desired_low) & (weight > 0.)
            if selected.any():
                target = weighted_quantile(self.curves[selected], weight[selected])
                output[depth] = (1.-PROJECTION_WEIGHT)*decision[depth]+PROJECTION_WEIGHT*target
        return output


class ReferenceRecognition:
    """Original point operator, rebuilt from observed support at every condition."""
    def condition(self, observed):
        from cpt_budget.v8.joint import ReferenceJointAdapter
        self.adapter = ReferenceJointAdapter()
        self.adapter.condition(observed)
        return self

    def predict(self, query, base):
        return self.adapter.batch_predict(query, base, kind='analog', shrinkage=.5)


class FastRecognition:
    """Exact batched operator, or one of three fixed support-only refinements."""
    def __init__(self, variant='exact'):
        if variant not in ('exact', *VARIANTS):
            raise ValueError('Unknown recognition variant')
        self.variant = variant

    def condition(self, observed):
        from cpt_budget.v8.joint import FastJointAdapter
        self.adapter = FastJointAdapter()
        self.adapter.condition(observed)
        self.evidence = None if self.variant == 'exact' else LocalEventEvidence(observed)
        return self

    def predict(self, query, base):
        result = self.adapter.batch_predict(query, base, kind='analog', shrinkage=.5)
        if self.variant == 'exact':
            return result
        order = np.asarray(result['order'], int)
        bases = np.asarray(base, float)[order].reshape(-1, 24)
        probabilities = np.asarray(result['probability'], float)[order].reshape(-1, 24)
        points, local_probabilities, counts = [], [], []
        for index, current in enumerate(bases):
            weights = result['weights'][index]
            p = self.evidence.blend(probabilities[index], weights)
            count = int(result['count_median'][index])
            if self.variant == 'recall60':
                atom_count = (result['atoms'][index] < 0.).sum(axis=1)
                count = int(weighted_quantile(atom_count, weights, .6))
            point = thickness_decision(current, p, count)
            if self.variant == 'amplitude':
                point = self.evidence.project_crossings(current, point, weights)
            points.append(point)
            local_probabilities.append(p)
            counts.append(count)
        output = dict(result)
        output['point'] = np.empty(len(query), float)
        output['point'][order] = np.asarray(points).reshape(-1)
        output['localizer_probability'] = np.empty(len(query), float)
        output['localizer_probability'][order] = np.asarray(local_probabilities).reshape(-1)
        output['chosen_count'] = np.asarray(counts)
        output['local_weight_mean'] = float(self.evidence.local_weight.mean())
        output['local_weight_max'] = float(self.evidence.local_weight.max())
        return output


def protocol():
    return {
        'family': 'recognition_engineering', 'models': MODELS,
        'reference_point': 'v7_joint_stack_analog_hmedian',
        'reference_risk': 'v6_empirical_graph_union_local',
        'base': 'support_stack_structural', 'mixture': 'analog complete curves with shrinkage .5',
        'depth_kernel': [.25, .5, .25], 'max_local_weight': MAX_LOCAL_WEIGHT,
        'reliability': 'Nonnegative per-depth Brier skill from support whole-near-group LOO against group-balanced depth climatology, smoothed one cell each direction',
        'projection_weight': PROJECTION_WEIGHT,
        'crossing_projection': 'Changed labels only, support same-class weighted median within depth +/-1 cell; preserves chosen class/count',
        'count_quantiles': {'exact': .5, 'localize': .5, 'amplitude': .5, 'recall60': .6},
        'risk_head': 'Same existing native empirical-graph probability for all variants; separate task head, not the refined point CDF',
        'parameter_freeze': 'Three variants and numeric constants communicated to root before scoring',
        'leakage_boundary': 'No query labels, no site-specific exception, no calibration labels for point fit',
        'independent_test': False, 'exploratory_reused_validation': True,
        'benchmark_scope': 'Condition and point-decision operator only; amplitude-base inference and separate risk head excluded',
    }


def run(output=None):
    from cpt_budget.v8.common import RUN, emit_candidate, load_frozen, load_native
    output = Path(output or RUN/'recognition')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'protocol.json', protocol())
    wide = load_frozen()
    base = wide.support_stack_structural.to_numpy(float)
    risk = load_native('v6_empirical_graph_union_local')
    points = {name: np.full(len(wide), np.nan) for name in MODELS}
    diagnostics, timing = [], []
    for state in iter_validation_states():
        mask = np.ones(len(wide), bool)
        for key in STATE_KEYS:
            mask &= wide[key].eq(state.metadata[key]).to_numpy()
        index = np.flatnonzero(mask)
        query = wide.iloc[index][[name for name in QUERY_COLUMNS if name in wide]]
        point = base[index]
        for name, variant in MODELS.items():
            started = perf_counter()
            estimator = FastRecognition(variant).condition(state.observed)
            result = estimator.predict(query, point)
            elapsed = perf_counter()-started
            points[name][index] = result['point']
            timing.append({**{key: state.metadata[key] for key in STATE_KEYS},
                           'model_id': name, 'condition_predict_seconds': elapsed,
                           'query_rows': len(query), 'support_profiles': state.observed.location_id.nunique()})
            order = np.asarray(result['order'])
            predicted_count = (result['point'][order].reshape(-1, 24) < 0.).sum(axis=1)
            selected = result.get('chosen_count', result['count_median'])
            if not np.array_equal(predicted_count, selected):
                raise ValueError('The emitted point prediction changed its chosen thickness count')
            for profile_index, location in enumerate(result['locations']):
                diagnostics.append({**{key: state.metadata[key] for key in STATE_KEYS},
                    'model_id': name, 'location_id': location,
                    'original_count_median': int(result['count_median'][profile_index]),
                    'chosen_count': int(selected[profile_index]),
                    'base_count': int((point[order].reshape(-1, 24)[profile_index] < 0.).sum()),
                    'local_weight_mean': result.get('local_weight_mean', 0.),
                    'local_weight_max': result.get('local_weight_max', 0.)})
        print(f'recognition: {state.key}', flush=True)
    for name in MODELS:
        emit_candidate(name, points[name], wide=wide, output_root=output, probability=risk,
                       metadata={**protocol(), 'variant': MODELS[name]})
    pd.DataFrame(diagnostics).to_csv(output/'profile_diagnostics.csv', index=False)
    pd.DataFrame(timing).to_csv(output/'component_timings.csv', index=False)
    write_json(output/'execution_summary.json', {
        'models': list(MODELS), 'states_per_model': 60, 'rows_per_model': len(wide),
        'model_count': len(MODELS), 'all_chosen_counts_equal_emitted_point_counts': True,
        'real_inference_operator': True, 'base_recomputation_in_component_timing': False,
    })
    return pd.DataFrame(diagnostics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output)
