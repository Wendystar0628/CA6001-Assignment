"""Observed-only kernel empirical residual distributions for fixed point bases.

This is an approximate error transport model, NOT a conformal predictive system:
LOO-IDW errors do not become the calibration errors of a different predictor.
Spatial dependence, kernel weighting and the present reuse of validation sites
also prevent attaching IID conformal coverage guarantees to these probabilities.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.spatial.distance import cdist, pdist

from cpt_budget.config import write_json
from cpt_budget.diagnostics.common import diagnostic_config, iter_validation_states, STATE_KEYS
from cpt_budget.features import idw_value
from cpt_budget.models.base import validate_observed, validate_query_site
from cpt_budget.types import QUERY_COLUMNS
from cpt_budget.v6.common import BEST, KEYS, RUN, emit_candidate, load_frozen


DEPTH_BANDWIDTH_M = .5
ERROR_BANDWIDTH_LOG = .15
MODES = ('depth', 'local')
BASES = ('best', 'fpca_guard', 'graph_union')
SOURCES = ['https://proceedings.mlr.press/v60/vovk17a.html',
           'https://arxiv.org/abs/1911.00941']


def support_error_frame(observed):
    """Current support LOO-IDW errors, excluding full near groups and coordinates."""
    support = validate_observed(observed).reset_index(drop=True)
    if 'z_center_m' not in support or not np.isfinite(support.z_center_m).all():
        raise ValueError('Finite support depth coordinates are required')
    by_depth = {depth: rows.sort_values('location_id', kind='stable')
                for depth, rows in support.groupby('depth_index', sort=False)}
    references, counts = [], []
    for row in support.itertuples(index=False):
        candidates = by_depth[row.depth_index]
        distance = np.linalg.norm(candidates[['x_m', 'y_m']].to_numpy(float) - [row.x_m, row.y_m], axis=1)
        allowed = candidates.location_id.ne(row.location_id).to_numpy() & (distance > 1e-9)
        if 'near_group_id' in support:
            if pd.isna(row.near_group_id) or support.near_group_id.isna().any():
                raise ValueError('Known near groups are required when that column is present')
            allowed &= candidates.near_group_id.ne(row.near_group_id).to_numpy()
        distances = distance[allowed]
        values = candidates.log_qc.to_numpy(float)[allowed]
        nearest = np.argsort(distances, kind='stable')[:3]
        if not len(nearest):
            raise ValueError('Empirical errors require another independent revealed CPT at each depth')
        references.append(idw_value(values[nearest], distances[nearest], power=2.))
        counts.append(len(nearest))
    result = support[[c for c in QUERY_COLUMNS if c in support]].copy()
    result['loo_reference_log_qc'] = references
    result['error_log_qc'] = support.log_qc.to_numpy(float) - references
    result['loo_reference_count'] = counts
    return result


def empirical_low_probability(error_frame, query, point, *, mode):
    """Gaussian-smoothed empirical P(Y<0) at each query without accessing truth.

    Depth weights are normalized separately inside each observed profile; those
    profile distributions then have equal weights, or a horizontal Gaussian
    weight in local mode. The independent information unit remains the CPT,
    never the number of correlated depth rows.
    """
    if mode not in MODES:
        raise ValueError(f'Unknown frozen weighting mode {mode}')
    if set(query) - set(QUERY_COLUMNS):
        raise ValueError('Query must contain geometry only')
    validate_query_site(query, error_frame)
    point = np.asarray(point, float).reshape(-1)
    if len(point) != len(query) or not np.isfinite(point).all():
        raise ValueError('One finite point prediction per geometry row is required')
    profiles = error_frame[['location_id', 'x_m', 'y_m']].drop_duplicates().sort_values('location_id')
    if profiles.location_id.duplicated().any() or not len(profiles):
        raise ValueError('Each support profile requires consistent geometry')
    pair_distance = pdist(profiles[['x_m', 'y_m']].to_numpy(float))
    nonzero = pair_distance[pair_distance > 1e-9]
    length = float(np.median(nonzero)) if len(nonzero) else 1.
    distances = cdist(query[['x_m', 'y_m']].to_numpy(float), profiles[['x_m', 'y_m']].to_numpy(float))
    log_profile_weight = -.5*(distances/length)**2 if mode == 'local' else np.zeros_like(distances)
    log_profile_weight -= log_profile_weight.max(axis=1, keepdims=True)
    profile_weight = np.exp(log_profile_weight)
    profile_weight /= profile_weight.sum(axis=1, keepdims=True)
    values = np.zeros(len(query), dtype=float)
    z_query = query.z_center_m.to_numpy(float)
    for column, location in enumerate(profiles.location_id):
        rows = error_frame.loc[error_frame.location_id.eq(location)]
        errors = rows.error_log_qc.to_numpy(float)
        if not np.isfinite(errors).all():
            raise ValueError('Empirical residuals must be finite')
        depth_log_weight = -.5*((z_query[:, None] - rows.z_center_m.to_numpy(float)[None, :]) / DEPTH_BANDWIDTH_M)**2
        depth_log_weight -= depth_log_weight.max(axis=1, keepdims=True)
        depth_weight = np.exp(depth_log_weight)
        depth_weight /= depth_weight.sum(axis=1, keepdims=True)
        local_cdf = ndtr((-point[:, None] - errors[None, :]) / ERROR_BANDWIDTH_LOG)
        values += profile_weight[:, column]*(depth_weight*local_cdf).sum(axis=1)
    values = np.clip(values, 0., 1.)
    profile_ess = 1./np.sum(profile_weight**2, axis=1)
    diagnostics = {
        'support_profiles': len(profiles),
        'support_near_groups': int(error_frame.near_group_id.nunique()) if 'near_group_id' in error_frame else len(profiles),
        'loo_min_reference_count': int(error_frame.loo_reference_count.min()),
        'local_length_m': length, 'profile_ess_min': float(profile_ess.min()),
        'profile_ess_mean': float(profile_ess.mean()),
        'small_support_fewer_than_10_profiles': len(profiles) < 10,
        'support_error_mean': float(error_frame.error_log_qc.mean()),
        'support_error_std': float(error_frame.error_log_qc.std()),
    }
    return values, diagnostics


def point_bases(wide):
    """Reconstruct frozen formulas from truth-free member points only."""
    residual = wide[BEST].to_numpy(float)
    fpca = .5*(residual + wide.functional_fpca_krr.to_numpy(float))
    fpca = np.where(residual < 0, np.minimum(fpca, -1e-9), np.maximum(fpca, 0.))
    graph = wide.graph_huber.to_numpy(float)
    mixture = .5*(residual + graph)
    graph_union = np.where((residual < 0) | (graph < 0), np.minimum(mixture, -1e-9), np.maximum(mixture, 0.))
    return dict(best=residual, fpca_guard=fpca, graph_union=graph_union)


def protocol():
    return {
        'family': 'observed_empirical_risk', 'candidate_count': 6,
        'parameter_freeze': 'Sent and agreed in parallel design messages before new scores',
        'point_bases': {'best': BEST, 'fpca_guard': 'v6_fpca_residual_sign_guard',
                        'graph_union': 'v6_graph_residual_low_union'},
        'modes': MODES, 'depth_bandwidth_m': DEPTH_BANDWIDTH_M,
        'error_CDF_bandwidth_log_qc': ERROR_BANDWIDTH_LOG,
        'horizontal_length': 'Median nonzero pairwise distance of currently revealed locations',
        'residual_predictor': '3NN IDW, power2, excluding whole CPT, near group and coincident coordinates',
        'probability': 'Sum of profile/depth kernel weights times Phi((-point - error)/0.15)',
        'point_predictions_unchanged': True, 'uses_old_OOF_probabilities': False,
        'query_truth_for_parameter_fit': False, 'outer_test_used': False,
        'calibration_scope': 'Native empirical support-error CDF; common crossfit risk applied externally',
        'error_transport_assumption': 'LOO-IDW errors approximate uncertainty of other point bases, not their own calibration residuals',
        'coverage_guarantee': False,
        'limitations': 'Only5-13CPTs; correlated depth rows not independent samples; spatial weighting and surrogate error transport lack conformal coverage',
        'sources': SOURCES,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(RUN/'empirical_risk'))
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    settings = protocol()
    write_json(output/'protocol.json', settings)
    wide = load_frozen()
    bases = point_bases(wide)
    probability_parts = {(base, mode): [] for base in BASES for mode in MODES}
    diagnostics = []
    for state in iter_validation_states(diagnostic_config()):
        errors = support_error_frame(state.observed)
        selection = np.ones(len(wide), dtype=bool)
        for key in STATE_KEYS:
            selection &= wide[key].eq(state.metadata[key]).to_numpy()
        positions = np.flatnonzero(selection)
        state_wide = wide.iloc[positions]
        query = state.query.set_index(['location_id', 'depth_index']).loc[
            pd.MultiIndex.from_frame(state_wide[['location_id', 'depth_index']])].reset_index()
        for base, point in bases.items():
            for mode in MODES:
                probability, diag = empirical_low_probability(errors, query, point[positions], mode=mode)
                part = state_wide[KEYS].copy()
                part['native_probability'] = probability
                probability_parts[base, mode].append(part)
                diagnostics.append({**{k: state.metadata[k] for k in STATE_KEYS},
                                    'base': base, 'mode': mode, **diag})
        print(f'empirical risk: {state.key}', flush=True)
    for (base, mode), parts in probability_parts.items():
        probability = pd.concat(parts, ignore_index=True)
        aligned = wide[KEYS].merge(probability, on=KEYS, how='left', validate='one_to_one', sort=False)
        if len(probability) != len(wide) or aligned.native_probability.isna().any():
            raise ValueError('Empirical probabilities must cover the complete frozen matrix')
        emit_candidate(f'v6_empirical_{base}_{mode}', bases[base], wide=wide, output_root=output,
                       probability=aligned.native_probability.to_numpy(float),
                       metadata={**settings, 'base': base, 'mode': mode})
    pd.DataFrame(diagnostics).to_csv(output/'support_diagnostics.csv', index=False)


if __name__ == '__main__':
    main()
