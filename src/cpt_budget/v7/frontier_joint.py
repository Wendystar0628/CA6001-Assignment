"""Targeted final-round whole-profile decisions around two observed frontiers.

This adaptive round was proposed after 85 configurations were evaluated. Four
point constructions receive either their own finite-mixture event probability
or one frozen, independent empirical-risk head. The latter is not the native
joint distribution of the new point construction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from cpt_budget.config import write_json
from cpt_budget.diagnostics.common import STATE_KEYS, iter_validation_states
from cpt_budget.types import QUERY_COLUMNS
from cpt_budget.v7.common import RUN, emit_candidate, load_frozen, load_native, read_candidate
from cpt_budget.v7.profile_distribution import (
    WholeProfileMixture, mixture_summary, thickness_decision, protocol as mixture_protocol,
)


BASES = {
    'structural': 'v7_simplex_structural_s050_gp',
    'graph_union': 'v7_simplex_graph_union_s075_gp',
}
POINTS = {f'v7_frontier_{base}_{kind}': (base, kind)
          for base in BASES for kind in ('analog', 'error')}
MODELS = {f'{point}_{head}': (point, head)
          for point in POINTS for head in ('joint', 'empirical')}
EMPIRICAL_MODEL = 'v6_empirical_graph_union_local'


def expand_risk_heads(points, joint_probabilities, empirical_probability):
    """Duplicate a point decision exactly while declaring its distinct heads."""
    empirical = np.asarray(empirical_probability, float)
    if (empirical.ndim != 1 or not np.isfinite(empirical).all()
            or ((empirical < 0.) | (empirical > 1.)).any()):
        raise ValueError('Finite aligned event probability in [0,1] required')
    if set(points) != set(joint_probabilities):
        raise ValueError('Point and joint probability names must agree')
    result = {}
    for name, raw in points.items():
        point, joint = np.asarray(raw, float), np.asarray(joint_probabilities[name], float)
        if (point.shape != empirical.shape or joint.shape != empirical.shape
                or not np.isfinite(point).all() or not np.isfinite(joint).all()
                or ((joint < 0.) | (joint > 1.)).any()):
            raise ValueError('Point and event vectors must be finite and aligned')
        result[name+'_joint'] = (point.copy(), joint.copy())
        result[name+'_empirical'] = (point.copy(), empirical.copy())
    return result


def protocol():
    return {
        'round': 'final_targeted', 'models': MODELS, 'point_bases': BASES,
        'selection_provenance': 'Nominated after 85 configurations were evaluated; adaptive frontier repair, not independent confirmation',
        'motivation': 'Improve thickness and point low-qc recognition while retaining strong continuous amplitude bases',
        'distribution_protocol': mixture_protocol(),
        'parameters_changed_from_first_round': [],
        'new_bandwidth_alpha_or_margin_search': False,
        'point_inputs': 'Only frozen keys and pred_log_qc from the two named point caches',
        'joint_head': 'Exact marginal event probability of the same finite whole-curve mixture',
        'empirical_head': EMPIRICAL_MODEL,
        'empirical_source': 'Frozen native p_low_1 only; no OOF risk probability or intervals',
        'empirical_head_is_joint_probability_of_new_point_model': False,
        'empirical_head_interpretation': 'Independent existing support-error event head paired with a new point decision',
        'head_switch_changes_point_predictions': False,
        'query_truth_in_prediction': False, 'outer_test_used': False,
    }


def run(output=None):
    output = Path(output or RUN/'frontier_joint')
    output.mkdir(parents=True, exist_ok=True)
    settings = protocol()
    write_json(output/'method_protocol.json', settings)
    wide = load_frozen()
    bases = {name: read_candidate('simplex_combinations', model, wide=wide)
             for name, model in BASES.items()}
    empirical = load_native(EMPIRICAL_MODEL)
    points = {name: np.full(len(wide), np.nan) for name in POINTS}
    probabilities = {name: np.full(len(wide), np.nan) for name in POINTS}
    decisions, atoms_all = [], []
    for state in iter_validation_states():
        mixture = WholeProfileMixture(state.observed)
        mask = np.ones(len(wide), bool)
        for key in STATE_KEYS:
            mask &= wide[key].eq(state.metadata[key]).to_numpy()
        for location, block in wide.loc[mask].groupby('location_id', sort=False):
            block = block.sort_values('depth_index')
            index = block.index.to_numpy()
            query = block[[column for column in QUERY_COLUMNS if column in block]]
            for name, (base_name, kind) in POINTS.items():
                base = bases[base_name][index]
                atoms, weights = mixture.atoms(query, base, kind)
                summary = mixture_summary(atoms, weights)
                point = thickness_decision(base, summary['probability'], summary['count_median'])
                points[name][index], probabilities[name][index] = point, summary['probability']
                decisions.append({**{key: state.metadata[key] for key in STATE_KEYS},
                    'point_id': name, 'location_id': location,
                    'joint_median_count': summary['count_median'],
                    'marginal_median_count': summary['point_median_count'],
                    'expected_count': summary['count_expected'],
                    'base_count': int((base < 0.).sum()), 'final_count': int((point < 0.).sum()),
                    'changed_labels': int(((base < 0.) != (point < 0.)).sum()),
                    'absolute_point_change': float(np.abs(point-base).sum()),
                    'loo_min_reference_count': mixture.loo_min_reference_count})
                for atom_index, (weight, count) in enumerate(zip(weights, summary['atom_counts'])):
                    atoms_all.append({**{key: state.metadata[key] for key in STATE_KEYS},
                        'point_id': name, 'location_id': location,
                        'source_location_id': mixture.locations[atom_index],
                        'weight': weight, 'low_cell_count': int(count),
                        'low_event_mask': ''.join('1' if value < 0 else '0' for value in atoms[atom_index])})
        print(f'frontier joint: {state.key}', flush=True)
    candidates = expand_risk_heads(points, probabilities, empirical)
    for name, (point, probability) in candidates.items():
        point_name, head = MODELS[name]
        emit_candidate(name, point, wide=wide, output_root=output,
            probability=probability, metadata={**settings, 'point_id': point_name,
                'head': head, 'probability_is_native_joint_marginal': head == 'joint'})
    pd.DataFrame(decisions).to_csv(output/'profile_decisions.csv', index=False)
    pd.DataFrame(atoms_all).to_csv(output/'joint_atoms.csv', index=False)
    write_json(output/'execution_summary.json', {
        'models': list(MODELS), 'unique_point_models': len(POINTS),
        'states_per_model': 60, 'rows_per_model': len(wide),
        'profile_decisions': len(decisions), 'atom_records': len(atoms_all),
        'all_actual_point_counts_equal_joint_median': all(
            row['final_count'] == row['joint_median_count'] for row in decisions),
        'joint_vs_marginal_median_disagreements': sum(
            row['joint_median_count'] != row['marginal_median_count'] for row in decisions),
        'all_head_pairs_have_identical_points': all(np.array_equal(
            candidates[name+'_joint'][0], candidates[name+'_empirical'][0]) for name in POINTS),
        'all_empirical_probabilities_equal_frozen_native_source': all(np.array_equal(
            candidates[name+'_empirical'][1], empirical) for name in POINTS),
    })
    return pd.DataFrame(decisions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output)
