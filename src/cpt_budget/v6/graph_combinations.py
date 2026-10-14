"""Frozen graph/TabPFN-residual combinations and observed-only graph bias.

All weights and threshold rules were fixed in the parallel design message
before generating these candidates. No evaluation label determines a mixing
weight, risk threshold, graph boundary or sign projection. These candidates
provide point predictions; their native uncertainties are not combined.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cpt_budget.config import PROJECT_ROOT, write_json
from cpt_budget.diagnostics.common import diagnostic_config, iter_validation_states, source_folds
from cpt_budget.diagnostics.residual_adapter import leave_location_out_idw
from cpt_budget.types import load_dataset
from cpt_budget.v5.graph import GraphProfilePredictor, PARAMETERS


BEST_RESIDUAL = 'residual_ft_crps_guarded_a050'
WEIGHTS = (.25, .5, .75)
STATE_KEYS = ['fold_id', 'site_id', 'seed', 'budget_n']
ROW_KEYS = [*STATE_KEYS, 'location_id', 'depth_index']
EPSILON = 1e-9
CANDIDATES = [*(f'v6_graph_residual_a{int(w*100):03d}' for w in WEIGHTS),
              'v6_graph_residual_sign_guard', 'v6_graph_residual_low_union',
              'v6_graph_loo_bias_a050']


def mix_graph_residual(graph, residual, graph_weight):
    """Convex mixture of two already legal, aligned log-resistance fields."""
    graph, residual = np.asarray(graph, float), np.asarray(residual, float)
    if graph.shape != residual.shape or not np.isfinite(graph).all() or not np.isfinite(residual).all():
        raise ValueError('Finite, aligned graph and residual predictions are required')
    if graph_weight not in WEIGHTS:
        raise ValueError(f'Graph weight must be one of the frozen values {WEIGHTS}')
    return graph_weight*graph + (1-graph_weight)*residual


def project_low_mask(values, low_mask, epsilon=EPSILON):
    """Closest threshold-side projection for an explicitly provided low mask.

    The threshold is log(1 MPa)=0. Epsilon keeps the strict low comparison
    numerically unambiguous. This changes point classifications but does not
    manufacture a probability or claim that the supplied mask is correct.
    """
    values, low_mask = np.asarray(values, float), np.asarray(low_mask, bool)
    if values.shape != low_mask.shape or not np.isfinite(values).all():
        raise ValueError('Finite values and an aligned low mask are required')
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('A finite positive margin is required')
    return np.where(low_mask, np.minimum(values, -epsilon), np.maximum(values, 0.))


def observed_bias_context(observed, config=None):
    """LOO-IDW errors on revealed complete profiles, excluding coincident CPTs."""
    support = observed.copy(deep=True)
    reference = leave_location_out_idw(support, config)
    support['log_qc'] = support.log_qc.to_numpy(float) - reference
    # Graph only consumes log_qc, but avoid retaining stale observed qc units.
    if 'qc_mpa' in support:
        support['qc_mpa'] = np.exp(support.log_qc.to_numpy(float))
    return support


def main():
    from cpt_budget.v6.common import load_frozen, emit_candidate

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(PROJECT_ROOT/'runs/v6-combinations-20261014/graph'))
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'protocol.json', {
        'candidate_ids': CANDIDATES,
        'freeze_status': 'Parameters fixed in parallel-agent design messages before new candidate scores',
        'reference_model': BEST_RESIDUAL, 'graph_model': 'graph_huber',
        'graph_weights': WEIGHTS, 'threshold_log_qc': 0., 'sign_margin': EPSILON,
        'sign_guard': 'Project 0.5 mixture to the original best residual low/nonlow mask',
        'low_union': 'Project 0.5 mixture to the union of graph and residual low masks',
        'loo_bias': 'Best residual + 0.5 times graph-Huber extension of observed minus LOO-IDW',
        'loo_exclusion': 'Whole location and all coincident support locations',
        'graph_parameters_unchanged': PARAMETERS,
        'prediction_scope': 'Point-only; no native uncertainty fusion',
        'label_boundary': 'Only currently observed support can enter graph LOO bias; no query truth',
        'sources': ['https://mlg.eng.cam.ac.uk/zoubin/ssl.html',
                    'https://arxiv.org/abs/1901.09838'],
    })
    wide = load_frozen()
    residual = wide[BEST_RESIDUAL].to_numpy(float)
    graph = wide.graph_huber.to_numpy(float)
    for weight in WEIGHTS:
        model = f'v6_graph_residual_a{int(weight*100):03d}'
        emit_candidate(model, mix_graph_residual(graph, residual, weight),
                       wide=wide, output_root=output,
                       metadata={'family': 'graph', 'graph_weight': weight,
                                 'reference_model': BEST_RESIDUAL, 'mechanism': 'convex_log_qc_mix'})
        print(model, flush=True)
    mixed = mix_graph_residual(graph, residual, .5)
    for model, mask in [
        ('v6_graph_residual_sign_guard', residual < 0),
        ('v6_graph_residual_low_union', (residual < 0) | (graph < 0)),
    ]:
        emit_candidate(model, project_low_mask(mixed, mask), wide=wide, output_root=output,
                       metadata={'family': 'graph', 'graph_weight': .5, 'sign_margin': EPSILON,
                                 'reference_model': BEST_RESIDUAL,
                                 'mechanism': 'residual_sign' if model.endswith('sign_guard') else 'low_union'})
        print(model, flush=True)

    config = diagnostic_config()
    dataset = load_dataset(config)
    sites = sorted({site for fold in source_folds() for site in fold['validation_sites']})
    ids = dataset.locations().loc[lambda x: x.site_id.isin(sites), 'location_id']
    predictor = GraphProfilePredictor(dataset.query(ids), variant='huber')
    bias_parts, diagnostics = [], []
    for state in iter_validation_states(config):
        predictor.condition(observed_bias_context(state.observed, config))
        frame = predictor.predict(state.query).frame[['location_id', 'depth_index', 'pred_log_qc']]
        for name in STATE_KEYS:
            frame[name] = state.metadata[name]
        bias_parts.append(frame.rename(columns={'pred_log_qc': 'graph_loo_bias'}))
        diagnostics.append({**{k: state.metadata[k] for k in STATE_KEYS},
                            **predictor.solver_diagnostics,
                            'support_bias_mean': float(predictor.values.mean()),
                            'support_bias_abs_mean': float(np.abs(predictor.values).mean())})
        print(f'graph_loo_bias: {state.key}', flush=True)
    bias = pd.concat(bias_parts, ignore_index=True)
    aligned = wide[ROW_KEYS].merge(bias, on=ROW_KEYS, how='left', validate='one_to_one', sort=False)
    if aligned.graph_loo_bias.isna().any() or len(bias) != len(wide):
        raise ValueError('The graph LOO correction did not cover the frozen query matrix')
    emit_candidate('v6_graph_loo_bias_a050', residual + .5*aligned.graph_loo_bias.to_numpy(float),
                   wide=wide, output_root=output,
                   metadata={'family': 'graph', 'bias_weight': .5, 'reference_model': BEST_RESIDUAL,
                             'mechanism': 'graph_extension_of_revealed_whole_location_LOO_IDW_error',
                             'graph_parameters': PARAMETERS,
                             'solver_converged_states': sum(d['graph_converged'] for d in diagnostics),
                             'solver_total_states': len(diagnostics)})
    bias.to_parquet(output/'graph_loo_bias_values.parquet', index=False)
    pd.DataFrame(diagnostics).to_csv(output/'solver_diagnostics.csv', index=False)


if __name__ == '__main__':
    main()
