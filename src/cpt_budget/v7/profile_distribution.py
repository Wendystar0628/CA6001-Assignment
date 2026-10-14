"""Finite whole-profile mixtures and distinct amplitude/thickness decisions.

Each component is a complete 24-depth curve. The component index is shared
across depth, so the joint thickness distribution is calculated from genuine
curve atoms rather than fabricated by multiplying marginal event probabilities.
These support-transport models are exploratory, not calibrated posteriors.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from threadpoolctl import threadpool_limits

from cpt_budget.config import write_json
from cpt_budget.diagnostics.common import STATE_KEYS, iter_validation_states
from cpt_budget.models.base import validate_observed, validate_query_site
from cpt_budget.types import QUERY_COLUMNS
from cpt_budget.v6.empirical_risk import support_error_frame


SHRINKAGE = .5
MARGIN = .05
N_DEPTHS = 24
MODELS = {
    'v7_joint_analog_median': ('analog', 'point_median', 'best'),
    'v7_joint_error_median': ('error', 'point_median', 'best'),
    'v7_joint_analog_hmedian': ('analog', 'thickness_median', 'best'),
    'v7_joint_error_hmedian': ('error', 'thickness_median', 'best'),
    'v7_joint_error_hmedian_fpca': ('error', 'thickness_median', 'fpca50'),
}


def weighted_quantile(values, weights, q=.5):
    """Lower inverse empirical CDF, one column at a time, no interpolation."""
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    scalar = values.ndim == 1
    if scalar:
        values = values[:, None]
    if (values.ndim != 2 or weights.shape != (len(values),)
            or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or (weights < 0).any() or weights.sum() <= 0 or not 0 < q < 1):
        raise ValueError('Finite atom values, nonnegative mass and interior quantile required')
    weights = weights / weights.sum()
    result = []
    for column in values.T:
        order = np.argsort(column, kind='stable')
        index = min(int(np.searchsorted(np.cumsum(weights[order]), q, side='left')), len(order)-1)
        result.append(column[order[index]])
    result = np.asarray(result)
    return float(result[0]) if scalar else result


def mixture_summary(atoms, weights):
    """Exact marginal events, whole-curve count PMF, and their Bayes decisions."""
    atoms = np.asarray(atoms, float)
    weights = np.asarray(weights, float)
    if (atoms.ndim != 2 or not len(atoms) or weights.shape != (len(atoms),)
            or not np.isfinite(atoms).all() or not np.isfinite(weights).all()
            or (weights < 0).any() or weights.sum() <= 0):
        raise ValueError('Valid complete curve atoms and weights are required')
    weights = weights / weights.sum()
    count = (atoms < 0.).sum(axis=1)
    p = np.clip(weights @ (atoms < 0.), 0., 1.)
    median_point = weighted_quantile(atoms, weights)
    count_pmf = np.bincount(count, weights=weights, minlength=atoms.shape[1]+1)
    return {
        'probability': p,
        'point_median': median_point,
        'count_pmf': count_pmf,
        'atom_counts': count,
        'count_median': int(weighted_quantile(count, weights)),
        'count_expected': float(weights @ count),
        'point_median_count': int((median_point < 0.).sum()),
    }


def thickness_decision(base, probability, count):
    """Fixed-count minimum expected Hamming loss, with minimum-change ties.

Select the largest marginal event probabilities. Among equal probabilities,
ascending base logqc minimizes total absolute modification under the frozen
+/-0.05 crossing rule. This is not a global minimum-change point estimator.
Only cells with a changed event label are modified.
"""
    base, probability = np.asarray(base, float), np.asarray(probability, float)
    if (base.ndim != 1 or probability.shape != base.shape
            or not np.isfinite(base).all() or not np.isfinite(probability).all()
            or ((probability < 0) | (probability > 1)).any()
            or int(count) != count or not 0 <= count <= len(base)):
        raise ValueError('Finite point/event profiles and an integer in-range count required')
    rank = np.lexsort((np.arange(len(base)), base, -probability))
    low = np.zeros(len(base), bool)
    low[rank[:int(count)]] = True
    result = base.copy()
    result[low & (base >= 0.)] = -MARGIN
    result[~low & (base < 0.)] = MARGIN
    return result


class WholeProfileMixture:
    """Condition on revealed curves; queries supply geometry only."""

    def __init__(self, observed):
        self.observed = validate_observed(observed)
        frame = self.observed
        if 'near_group_id' not in frame or frame.near_group_id.isna().any():
            raise ValueError('Known near groups are required')
        if (frame.groupby('location_id')[['x_m', 'y_m', 'near_group_id']].nunique() != 1).any().any():
            raise ValueError('Support profile geometry and near groups must be constant')
        self.locations = frame.location_id.drop_duplicates().sort_values().tolist()
        support = frame.pivot(index='location_id', columns='depth_index', values='log_qc')
        if list(support.columns) != list(range(N_DEPTHS)) or support.isna().any().any():
            raise ValueError('Complete 24-depth revealed profiles are required')
        self.curves = support.loc[self.locations].to_numpy(float)
        geometry = frame.drop_duplicates('location_id').set_index('location_id').loc[self.locations]
        self.xy = geometry[['x_m', 'y_m']].to_numpy(float)
        self.group_size = geometry.near_group_id.map(geometry.near_group_id.value_counts()).to_numpy(float)
        distances = pdist(self.xy)
        nonzero = distances[distances > 1e-9]
        if not len(nonzero):
            raise ValueError('At least two distinct support coordinates required')
        self.length = float(np.median(nonzero))
        errors = support_error_frame(frame)
        self.error_curves = errors.pivot(index='location_id', columns='depth_index',
            values='error_log_qc').loc[self.locations, list(range(N_DEPTHS))].to_numpy(float)
        self.loo_min_reference_count = int(errors.loo_reference_count.min())

    def atoms(self, query_profile, base, kind):
        if set(query_profile) - set(QUERY_COLUMNS):
            raise ValueError('Query must contain geometry only')
        validate_query_site(query_profile, self.observed)
        if (query_profile.location_id.nunique() != 1 or len(query_profile) != N_DEPTHS
                or query_profile.depth_index.tolist() != list(range(N_DEPTHS))):
            raise ValueError('Query must be one complete depth-sorted profile')
        if query_profile[['x_m', 'y_m']].nunique().ne(1).any():
            raise ValueError('Query geometry must be constant across depth')
        if set(query_profile.location_id) & set(self.locations):
            raise ValueError('Query profile must not be in revealed support')
        if ('near_group_id' in query_profile and
                set(query_profile.near_group_id) & set(self.observed.near_group_id)):
            raise ValueError('Query near group must not overlap revealed support')
        xy = query_profile[['x_m', 'y_m']].iloc[0].to_numpy(float)
        base = np.asarray(base, float)
        if not np.isfinite(xy).all() or base.shape != (N_DEPTHS,) or not np.isfinite(base).all():
            raise ValueError('Finite geometry and complete base predictions required')
        distances = np.linalg.norm(self.xy-xy, axis=1)
        if (distances <= 1e-9).any():
            raise ValueError('Query cannot be colocated with revealed support')
        log_weight = -.5*(distances/self.length)**2 - np.log(self.group_size)
        weight = np.exp(log_weight-log_weight.max())
        weight /= weight.sum()
        if kind == 'analog':
            deviation = self.curves - weight @ self.curves
        elif kind == 'error':
            deviation = self.error_curves
        else:
            raise ValueError('Unknown complete-curve mixture kind')
        return base[None, :] + SHRINKAGE*deviation, weight


def protocol():
    return {
        'models': MODELS, 'parameter_message_before_run': True,
        'family': 'finite_whole_profile_distribution', 'shrinkage': SHRINKAGE,
        'analog_atoms': 'base + 0.5*(revealed whole curve - weighted mean revealed whole curve)',
        'error_atoms': 'base + 0.5*(revealed whole curve - 3NN-IDW whole-group-held-out reference)',
        'weights': 'Gaussian horizontal kernel; scale median nonzero support pair distance; divide by near-group size',
        'loo_exclusion': 'Entire location, near group and colocated coordinates; recomputed from support only',
        'joint_model': 'One atom index is shared by every depth; enumerate finite support exactly, no random simulation',
        'quantile_convention': 'Lower inverse empirical CDF; no interpolation between counts',
        'point_median': 'Marginal logqc median minimizes expected absolute point error under the stipulated mixture',
        'thickness_median': 'Median of whole-atom low-cell counts minimizes expected absolute thickness error under the same mixture',
        'event_location_decision': 'At selected count, top marginal probabilities minimize expected Hamming loss; base-value ties minimize changes',
        'crossing_margin_log_qc': MARGIN, 'probability': 'Exact marginal P(atom logqc < 0), before common external calibration',
        'expected_thickness_is_point_thickness': False,
        'prediction_interval_or_coverage_claim': False, 'calibrated_bayesian_posterior': False,
        'transport_assumptions': 'Analog deviations or LOO-IDW whole-profile errors transfer to the frozen amplitude base; not its own calibrated residuals',
        'limits': 'Only 5-13 support profiles; horizontal nonstationarity and source-model mismatch can invalidate transport',
        'outer_test_used': False, 'validation_scope': 'Exploration on repeatedly used frozen validation sites',
        'sources': [
            'https://fpp.robjhyndman.com/publications/forecasting-functional-time-series-2/',
            'https://arxiv.org/abs/2004.03971',
        ],
        'source_relationship': 'Functional/bootstrap curve dependence inspires the construction; this spatial finite mixture does not reproduce their estimators or inherit their validity guarantees',
    }


def run(output=None):
    from cpt_budget.v7.common import BEST, KEYS, RUN, emit_candidate, load_frozen

    output = Path(output or RUN/'profile_distribution')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'method_protocol.json', protocol())
    wide = load_frozen()
    bases = {'best': wide[BEST].to_numpy(float),
             'fpca50': .5*(wide[BEST].to_numpy(float)+wide.functional_fpca_krr.to_numpy(float))}
    predicted = {model: np.full(len(wide), np.nan) for model in MODELS}
    probabilities = {model: np.full(len(wide), np.nan) for model in MODELS}
    decisions, atom_records = [], []
    for state in iter_validation_states():
        model = WholeProfileMixture(state.observed)
        mask = np.ones(len(wide), bool)
        for key in STATE_KEYS:
            mask &= wide[key].eq(state.metadata[key]).to_numpy()
        indices = np.flatnonzero(mask)
        for location, block in wide.iloc[indices].groupby('location_id', sort=False):
            block = block.sort_values('depth_index')
            index = block.index.to_numpy()
            query = block[[column for column in QUERY_COLUMNS if column in block]]
            cache = {}
            for name, (kind, decision, base_name) in MODELS.items():
                base = bases[base_name][index]
                key = (kind, base_name)
                if key not in cache:
                    atoms, weights = model.atoms(query, base, kind)
                    summary = mixture_summary(atoms, weights)
                    cache[key] = atoms, weights, summary
                    for atom_index, (weight, count) in enumerate(zip(weights, summary['atom_counts'])):
                        atom_records.append({**{k: state.metadata[k] for k in STATE_KEYS},
                            'location_id': location, 'kind': kind, 'base': base_name,
                            'source_location_id': model.locations[atom_index],
                            'weight': weight, 'low_cell_count': int(count),
                            'low_event_mask': ''.join('1' if value < 0 else '0' for value in atoms[atom_index])})
                atoms, weights, summary = cache[key]
                point = summary['point_median'] if decision == 'point_median' else thickness_decision(
                    base, summary['probability'], summary['count_median'])
                predicted[name][index] = point
                probabilities[name][index] = summary['probability']
                decisions.append({**{k: state.metadata[k] for k in STATE_KEYS},
                    'model_id': name, 'location_id': location,
                    'joint_median_count': summary['count_median'],
                    'marginal_median_count': summary['point_median_count'],
                    'expected_count': summary['count_expected'],
                    'base_count': int((base < 0.).sum()), 'final_count': int((point < 0.).sum()),
                    'changed_labels': int(((base < 0.) != (point < 0.)).sum()),
                    'absolute_point_change': float(np.abs(point-base).sum()),
                    'effective_atom_count': float(1./np.sum(weights**2)),
                    'loo_min_reference_count': model.loo_min_reference_count,
                    'geometry_length_m': model.length})
        print(f'whole-profile mixtures: {state.key}', flush=True)
    for name, (kind, decision, base_name) in MODELS.items():
        emit_candidate(name, predicted[name], wide=wide, output_root=output,
            probability=probabilities[name], metadata={**protocol(),
                'kind': kind, 'decision': decision, 'base': base_name})
    pd.DataFrame(decisions).to_csv(output/'profile_decisions.csv', index=False)
    pd.DataFrame(atom_records).to_csv(output/'joint_atoms.csv', index=False)
    write_json(output/'execution_summary.json', {
        'models': list(MODELS), 'states_per_model': 60, 'rows_per_model': len(wide),
        'profile_decisions': len(decisions), 'atom_records': len(atom_records),
        'joint_vs_marginal_median_count_disagreements': sum(
            item['joint_median_count'] != item['marginal_median_count'] for item in decisions),
        'all_thickness_decisions_match_joint_median': all(
            row['final_count'] == row['joint_median_count'] for row in decisions
            if MODELS[row['model_id']][1] == 'thickness_median'),
    })
    return pd.DataFrame(decisions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    arguments = parser.parse_args()
    with threadpool_limits(limits=1):
        run(arguments.output)
