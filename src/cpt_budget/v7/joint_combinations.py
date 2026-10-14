"""Round-two amplitude bases inside fixed whole-profile predictive mixtures.

The eight combinations were nominated after seeing round-one results. They
reuse the same finite-curve distributions and loss decisions; no kernel scale,
shrinkage, crossing margin, or calibration parameter is searched here.
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
from cpt_budget.v7.common import BEST, RUN, emit_candidate, load_frozen, read_candidate
from cpt_budget.v7.profile_distribution import (
    WholeProfileMixture, mixture_summary, thickness_decision, protocol as mixture_protocol,
)


BASE_NAMES = ('simplex', 'stack', 'simplex_best50')
MODELS = {
    **{f'v7_joint_{base}_{kind}_hmedian': (base, kind, 'thickness_median')
       for base in BASE_NAMES for kind in ('analog', 'error')},
    **{f'v7_joint_simplex_{kind}_median': ('simplex', kind, 'point_median')
       for kind in ('analog', 'error')},
}


def compose_bases(simplex, best, stack):
    values = [np.asarray(item, float) for item in (simplex, best, stack)]
    if (any(item.ndim != 1 for item in values) or
            any(item.shape != values[0].shape for item in values) or
            any(not np.isfinite(item).all() for item in values)):
        raise ValueError('Three finite row-aligned point vectors are required')
    simplex, best, stack = values
    return {'simplex': simplex, 'stack': stack, 'simplex_best50': .5*(simplex+best)}


def predict_profile(mixture, query, bases):
    """Only frozen points, revealed-support mixtures and geometry enter."""
    cache, predictions = {}, {}
    for name, (base_name, kind, decision) in MODELS.items():
        key = base_name, kind
        if key not in cache:
            atoms, weights = mixture.atoms(query, bases[base_name], kind)
            cache[key] = atoms, weights, mixture_summary(atoms, weights)
        atoms, weights, summary = cache[key]
        base = bases[base_name]
        point = summary['point_median'] if decision == 'point_median' else thickness_decision(
            base, summary['probability'], summary['count_median'])
        predictions[name] = point, summary['probability'], summary
    return predictions, cache


def protocol():
    return {
        'round': 2, 'models': MODELS,
        'selection_provenance': 'Nominated after round-one local-geometry and whole-profile-distribution results; adaptive exploration, not independent confirmation',
        'bases': {'simplex': 'v7_simplex_linear', 'stack': 'support_stack_structural',
                  'simplex_best50': f'0.5*v7_simplex_linear + 0.5*{BEST}'},
        'distribution_protocol': mixture_protocol(),
        'parameters_changed_from_round1': [],
        'amplitude_source_allowlist': 'Frozen cell keys and pred_log_qc only; never truth, old risk calibration or intervals',
        'probability': 'Exact marginal of the same complete-curve finite mixture around each new base',
        'query_truth_in_prediction': False, 'independent_test_used': False,
        'new_hyperparameter_search': False,
    }


def run(output=None):
    output = Path(output or RUN/'joint_combinations')
    output.mkdir(parents=True, exist_ok=True)
    settings = protocol()
    write_json(output/'method_protocol.json', settings)
    wide = load_frozen()
    simplex = read_candidate('local_geometry', 'v7_simplex_linear', wide=wide)
    bases = compose_bases(simplex, wide[BEST].to_numpy(float), wide.support_stack_structural.to_numpy(float))
    points = {name: np.full(len(wide), np.nan) for name in MODELS}
    probabilities = {name: np.full(len(wide), np.nan) for name in MODELS}
    diagnostics, atom_records = [], []
    for state in iter_validation_states():
        mixture = WholeProfileMixture(state.observed)
        mask = np.ones(len(wide), bool)
        for key in STATE_KEYS:
            mask &= wide[key].eq(state.metadata[key]).to_numpy()
        for location, block in wide.loc[mask].groupby('location_id', sort=False):
            block = block.sort_values('depth_index')
            index = block.index.to_numpy()
            query = block[[column for column in QUERY_COLUMNS if column in block]]
            profile_bases = {key: value[index] for key, value in bases.items()}
            predictions, mixtures = predict_profile(mixture, query, profile_bases)
            for (base_name, kind), (atoms, weights, summary) in mixtures.items():
                for atom_index, (weight, count) in enumerate(zip(weights, summary['atom_counts'])):
                    atom_records.append({**{key: state.metadata[key] for key in STATE_KEYS},
                        'location_id': location, 'base': base_name, 'kind': kind,
                        'source_location_id': mixture.locations[atom_index],
                        'weight': weight, 'low_cell_count': int(count),
                        'low_event_mask': ''.join('1' if value < 0 else '0' for value in atoms[atom_index])})
            for name, (point, probability, summary) in predictions.items():
                base_name, kind, decision = MODELS[name]
                base = profile_bases[base_name]
                points[name][index], probabilities[name][index] = point, probability
                diagnostics.append({**{key: state.metadata[key] for key in STATE_KEYS},
                    'model_id': name, 'location_id': location,
                    'base': base_name, 'kind': kind, 'decision': decision,
                    'joint_median_count': summary['count_median'],
                    'marginal_median_count': summary['point_median_count'],
                    'expected_count': summary['count_expected'],
                    'base_count': int((base < 0.).sum()), 'final_count': int((point < 0.).sum()),
                    'changed_labels': int(((base < 0.) != (point < 0.)).sum()),
                    'absolute_point_change': float(np.abs(point-base).sum()),
                    'loo_min_reference_count': mixture.loo_min_reference_count})
        print(f'joint combinations: {state.key}', flush=True)
    for name, (base_name, kind, decision) in MODELS.items():
        emit_candidate(name, points[name], wide=wide, output_root=output,
            probability=probabilities[name], metadata={**settings,
                'base': base_name, 'kind': kind, 'decision': decision})
    pd.DataFrame(diagnostics).to_csv(output/'profile_decisions.csv', index=False)
    pd.DataFrame(atom_records).to_csv(output/'joint_atoms.csv', index=False)
    write_json(output/'execution_summary.json', {
        'models': list(MODELS), 'states_per_model': 60, 'rows_per_model': len(wide),
        'profile_decisions': len(diagnostics), 'atom_records': len(atom_records),
        'all_thickness_decisions_match_joint_median': all(
            row['joint_median_count'] == row['final_count'] for row in diagnostics
            if row['decision'] == 'thickness_median'),
        'joint_vs_marginal_median_disagreements': sum(
            row['joint_median_count'] != row['marginal_median_count'] for row in diagnostics),
    })
    return pd.DataFrame(diagnostics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output)
