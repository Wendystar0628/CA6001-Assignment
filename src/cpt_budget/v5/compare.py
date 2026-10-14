"""Compare new structural families with frozen strong and weak controls."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from cpt_budget.config import PROJECT_ROOT, write_json
from cpt_budget.diagnostics.common import diagnostic_config, metrics_for, STATE_KEYS
from cpt_budget.diagnostics.analyze import curve_metrics, gate_values, METRICS
from cpt_budget.v2.pipeline import _gate, _plain
from cpt_budget.v2.risk import crossfit_risk, evaluate_risk
from cpt_budget.v4.evaluation import _expected_frame, validate_prediction_matrix

RUN = PROJECT_ROOT / 'runs/v5-exploration-20261014'
REFERENCE = PROJECT_ROOT / 'runs/v4-residual-20261014/comparison'
MODELS = {
    'target_constant_median': 'weak_baselines/target_constant_median',
    'target_depth_median': 'weak_baselines/target_depth_median',
    'functional_fpca_krr': 'functional/functional_fpca_krr',
    'functional_fpca_linear': 'functional/functional_fpca_linear',
    'graph_harmonic': 'graph/graph_harmonic',
    'graph_huber': 'graph/graph_huber',
    'registered_profile_log': 'layer_alignment/registered_profile_log',
    'registered_profile_shape': 'layer_alignment/registered_profile_shape',
    'cnp_profile': 'neural_process/validation',
}
WEAK = ('target_constant_median', 'target_depth_median')


def select_new_candidates(curves, settings):
    """Frozen v2 gate for new scientific methods, excluding the weak controls."""
    candidates = [m for m in MODELS if m not in WEAK]
    fold = curves.groupby(['fold_id', 'site_id', 'model_id'])[METRICS].mean().groupby(['fold_id', 'model_id']).mean()
    selected, gates, paired_parts = {}, [], []
    for fid in (0, 1, 2):
        base = min(['idw', 'gp'], key=lambda m: fold.loc[(fid, m), 'log_mae'])
        eligible = []
        for model in candidates:
            checks = _gate(gate_values(fold.loc[(fid, model)]), gate_values(fold.loc[(fid, base)]), settings)
            gates.append({'fold_id': fid, 'model_id': model, 'baseline_model': base,
                          **gate_values(fold.loc[(fid, model)]), **checks, 'passed': all(checks.values())})
            if all(checks.values()):
                eligible.append(model)
        chosen = min(eligible, key=lambda m: fold.loc[(fid, m), 'log_mae']) if eligible else base
        selected[str(fid)] = {'baseline_model': base, 'selected_model': chosen,
                              'eligible_candidates': eligible, 'fallback': not eligible}
        subset = curves[curves.fold_id.eq(fid)]
        paired_parts.append(subset[subset.model_id.eq(chosen)].merge(
            subset[subset.model_id.eq(base)], on=['fold_id', 'site_id', 'seed'],
            suffixes=('_method', '_baseline'), validate='one_to_one'))
    paired = pd.concat(paired_parts, ignore_index=True)
    names = ['log_mae', 'h_point_mae_m', 'recall', 'false_positive_rate', 'risk_recall', 'risk_fpr', 'risk_brier']
    columns = [f'{m}_{s}' for m in names for s in ['method', 'baseline']]
    sites = paired.groupby(['site_id', 'fold_id'])[columns].mean().groupby('site_id').mean()
    method = {m: float(sites[f'{m}_method'].mean()) for m in names}
    baseline = {m: float(sites[f'{m}_baseline'].mean()) for m in names}
    checks = _gate(gate_values(method), gate_values(baseline), settings)
    improved = int((sites.log_mae_method < sites.log_mae_baseline).sum())
    checks.update(majority_unique_sites_improve=improved > len(sites)/2, all_folds_complete=True)
    result = {'scope': 'exploratory_validation_only', 'folds': selected, 'method': method,
              'baseline': baseline, 'global_checks': checks, 'passed': all(checks.values()),
              'improved_sites': improved, 'unique_sites': len(sites),
              'relative_log_mae_reduction': 1-method['log_mae']/baseline['log_mae'],
              'weak_controls_excluded_from_promotion': list(WEAK), 'outer_test_executed': False}
    return result, pd.DataFrame(gates), paired, sites


def pairwise_comparisons(curves):
    rows, site_rows = [], []
    references = ['target_constant_median', 'idw', 'gp', 'residual_ft_crps_guarded_a050']
    for model in MODELS:
        for baseline in references:
            if baseline == model:
                continue
            pair = curves[curves.model_id.eq(model)].merge(curves[curves.model_id.eq(baseline)],
                on=['fold_id', 'site_id', 'seed'], suffixes=('_method', '_baseline'), validate='one_to_one',
                how='outer', indicator=True)
            if not pair['_merge'].eq('both').all():
                raise ValueError('Unpaired model comparison')
            names = ['log_mae', 'h_point_mae_m', 'recall', 'false_positive_rate', 'risk_recall', 'risk_fpr', 'risk_brier']
            cols = [f'{m}_{s}' for m in names for s in ['method', 'baseline']]
            sites = pair.groupby(['site_id', 'fold_id'])[cols].mean().groupby('site_id').mean()
            for name in names:
                sites[f'{name}_delta'] = sites[f'{name}_method']-sites[f'{name}_baseline']
            means = sites.mean().to_dict()
            row = {'model_id': model, 'reference_model': baseline, **means,
                   'relative_log_mae_reduction': 1-means['log_mae_method']/means['log_mae_baseline'],
                   'improved_unique_sites': int((sites.log_mae_delta < 0).sum()), 'unique_sites': len(sites)}
            rows.append(row)
            site_rows.append(sites.reset_index().assign(model_id=model, reference_model=baseline))
    return pd.DataFrame(rows), pd.concat(site_rows, ignore_index=True)


def main():
    config = diagnostic_config()
    output = RUN/'comparison'
    output.mkdir(parents=True, exist_ok=True)
    _, expected = _expected_frame(config)
    references = pd.read_parquet(REFERENCE/'predictions_oof.parquet')
    ref_models = sorted(references.model_id.unique())
    validate_prediction_matrix(references, expected, ref_models)
    metrics = pd.read_csv(REFERENCE/'metrics.csv')
    if len(metrics) != 60*len(ref_models):
        raise ValueError('Frozen reference metrics are incomplete')
    prediction_parts, metric_parts, calibrators = [references], [metrics], {}
    for model, relative in MODELS.items():
        path = RUN/relative/'predictions.parquet'
        if not path.exists():
            raise FileNotFoundError(f'Registered candidate missing: {path}')
        raw = pd.read_parquet(path)
        if set(raw.model_id) != {model}:
            raise ValueError(f'Model name mismatch: {path}')
        validate_prediction_matrix(raw, expected, [model])
        if any(c in raw and raw[c].notna().any() for c in ['p_low_1', 'q05_log', 'q95_log']):
            raise ValueError(f'{model} was registered as point-only')
        for fid, block in raw.groupby('fold_id'):
            calibrated, final = crossfit_risk(block.dropna(axis=1, how='all'))
            calibrated['risk_label'] = calibrated.risk_probability >= .2
            calibrated['risk_threshold'] = .2
            calibrated['calibration_scope'] = 'oof'
            calibrators.setdefault(str(fid), {})[model] = final
            prediction_parts.append(calibrated)
            records = []
            for _, state in calibrated.groupby(STATE_KEYS):
                record = metrics_for(state, config)
                truth = state[['site_id','location_id','depth_index','truth_log_qc','truth_qc_mpa']].rename(
                    columns={'truth_log_qc':'log_qc','truth_qc_mpa':'qc_mpa'})
                risk = evaluate_risk(state.drop(columns=['truth_log_qc','truth_qc_mpa']), truth, config)
                records.append({**record, **risk, 'study_version':'0.5-exploration'})
            metric_parts.append(pd.DataFrame(records))
    all_predictions = pd.concat(prediction_parts, ignore_index=True)
    all_models = ref_models+list(MODELS)
    validate_prediction_matrix(all_predictions, expected, all_models)
    all_predictions.to_parquet(output/'predictions_oof.parquet', index=False)
    write_json(output/'new_calibrators.json', calibrators)
    metrics = pd.concat(metric_parts, ignore_index=True)
    metrics.to_csv(output/'metrics.csv', index=False)
    curves = curve_metrics(metrics)
    curves.to_csv(output/'curves.csv', index=False)
    folds = curves.groupby(['fold_id','site_id','model_id'])[METRICS].mean().groupby(['fold_id','model_id']).mean()
    folds.to_csv(output/'fold_macro.csv')
    sites = curves.groupby(['fold_id','site_id','model_id'])[METRICS].mean().groupby(['site_id','model_id']).mean()
    sites.to_csv(output/'unique_site_metrics.csv')
    macro = sites.groupby('model_id')[METRICS].mean().sort_values('log_mae')
    macro.to_csv(output/'all_model_ranking.csv')
    macro.loc[list(MODELS)].sort_values('log_mae').to_csv(output/'new_model_ranking.csv')
    pairwise, site_pairs = pairwise_comparisons(curves)
    pairwise.to_csv(output/'pairwise_summary.csv', index=False)
    site_pairs.to_csv(output/'pairwise_by_site.csv', index=False)
    selection, gates, paired, chosen_sites = select_new_candidates(curves, config['v2']['gate'])
    gates.to_csv(output/'candidate_gates.csv', index=False)
    paired.to_csv(output/'selected_system_paired.csv', index=False)
    chosen_sites.to_csv(output/'selected_system_sites.csv')
    selection.update(status='complete', new_models=list(MODELS), reference_models=ref_models,
                     model_count=len(all_models), metric_rows=len(metrics), states_per_model=60,
                     recommendations_are_exploratory=True, reference_risk='frozen_v0.4_OOF_same_procedure')
    write_json(output/'selection.json', _plain(selection))
    print(macro.loc[list(MODELS)].sort_values('log_mae').to_string())
    print(json.dumps(_plain(selection), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
