"""Complete-matrix comparisons, held-group-out calibration and frozen gates."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from cpt_budget.types import PredictionBundle
from cpt_budget.v2.risk import crossfit_risk, evaluate_risk
from cpt_budget.v2.pipeline import _gate, _plain
from cpt_budget.config import write_json
from .common import OUTPUT_ROOT, SOURCE_RUN, STATE_KEYS, BUDGETS, diagnostic_config, metrics_for

NEW_MODELS = ['xgb24', 'tabpfn_residual_a000', 'tabpfn_residual_a025',
              'tabpfn_residual_a050', 'tabpfn_residual_a100',
              'tabpfn_source_context', 'tabpfn_honest_context']
METRICS = ['log_mae', 'h_point_mae_m', 'recall', 'false_positive_rate',
           'risk_recall', 'risk_fpr', 'risk_brier', 'coverage_90', 'width_90_log']


def curve_metrics(metrics):
    rows = []
    for key, group in metrics.groupby(['fold_id', 'site_id', 'seed', 'model_id'], sort=True):
        group = group.sort_values('budget_n')
        if group.budget_n.tolist() != list(BUDGETS):
            raise ValueError(f'Missing or duplicated budgets in {key}')
        row = dict(zip(['fold_id', 'site_id', 'seed', 'model_id'], key))
        x = group.budget_n.to_numpy(float)
        for name in METRICS:
            y = pd.to_numeric(group[name], errors='coerce').to_numpy(float)
            row[name] = float(np.trapezoid(y, x) / (x[-1]-x[0])) if np.isfinite(y).all() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def gate_values(row):
    return {'aubc_log_mae': row['log_mae'], 'aubc_h_point_mae': row['h_point_mae_m'],
            'risk_recall': row['risk_recall'], 'risk_fpr': row['risk_fpr']}


def main():
    config = diagnostic_config()
    out = OUTPUT_ROOT / 'comparison'
    out.mkdir(exist_ok=True)
    reference = pd.read_parquet(SOURCE_RUN / 'predictions_oof.parquet')
    reference = reference[(reference.role == 'evaluation') & (reference.split_role == 'validation')]
    # Existing calibrated columns are not used as candidate input or refit scores.
    reference = reference.drop(columns=[c for c in reference if c.startswith('risk_') or c == 'calibration_scope'])
    parts = [reference]
    paths = [OUTPUT_ROOT / 'xgb24/validation/predictions.parquet',
             *sorted((OUTPUT_ROOT / 'residual_adapter/validation').glob('*/predictions.parquet')),
             OUTPUT_ROOT / 'context_adaptation/source_bank/validation/predictions.parquet',
             OUTPUT_ROOT / 'context_adaptation/honest/validation/predictions.parquet']
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f'Complete registered comparison required: {path}')
        parts.append(pd.read_parquet(path))
    combined = pd.concat(parts, ignore_index=True)
    expected = set(reference.loc[reference.model_id.eq('idw'), STATE_KEYS].itertuples(index=False, name=None))
    if len(expected) != 60 or not set(NEW_MODELS).issubset(set(combined.model_id)):
        raise ValueError('Registered candidate matrix is incomplete')
    identity = ['fold_id', 'site_id', 'seed', 'budget_n', 'location_id', 'depth_index']
    for model, block in combined.groupby('model_id'):
        actual = set(block[STATE_KEYS].itertuples(index=False, name=None))
        if actual != expected or block.duplicated(identity).any():
            raise ValueError(f'Invalid complete matrix for {model}')
    idw = combined.loc[combined.model_id.eq('idw'), identity+['pred_log_qc']]
    zero = combined.loc[combined.model_id.eq('tabpfn_residual_a000'), identity+['pred_log_qc']]
    check = zero.merge(idw, on=identity, suffixes=('_zero', '_idw'), validate='one_to_one')
    zero_difference = float(abs(check.pred_log_qc_zero-check.pred_log_qc_idw).max())
    if zero_difference > 1e-10:
        raise ValueError('Alpha zero failed exact IDW recovery')
    frames, calibrators, records = [], {}, []
    for (fid, model), block in combined.groupby(['fold_id', 'model_id']):
        block = block.dropna(axis=1, how='all').copy()
        calibrated, final = crossfit_risk(block)
        calibrated['risk_label'] = calibrated.risk_probability >= .2
        calibrated['calibration_scope'] = 'leave_investigation_group_out'
        calibrators.setdefault(str(fid), {})[model] = final
        frames.append(calibrated)
        for _, rows in calibrated.groupby(STATE_KEYS):
            # Legacy snapshots use observed_set_id only in states; recover it by key.
            if 'observed_set_id' not in rows or rows.observed_set_id.isna().all():
                rows = rows.copy()
                rows['observed_set_id'] = 'frozen_v0.2_maximin'
            record = metrics_for(rows, config)
            truth = rows[['site_id', 'location_id', 'depth_index', 'truth_log_qc', 'truth_qc_mpa']].rename(
                columns={'truth_log_qc': 'log_qc', 'truth_qc_mpa': 'qc_mpa'})
            risk = evaluate_risk(rows.drop(columns=['truth_log_qc', 'truth_qc_mpa']), truth, config)
            records.append({**record, **risk})
    pd.concat(frames, ignore_index=True).to_parquet(out/'predictions_oof.parquet', index=False)
    write_json(out/'calibrators.json', calibrators)
    metrics = pd.DataFrame(records)
    metrics.to_csv(out/'metrics.csv', index=False)
    curves = curve_metrics(metrics)
    curves.to_csv(out/'curves.csv', index=False)
    fold = curves.groupby(['fold_id', 'model_id'])[METRICS].mean()
    fold.to_csv(out/'fold_macro.csv')
    sites = curves.groupby(['fold_id', 'site_id', 'model_id'])[METRICS].mean().groupby(['site_id', 'model_id']).mean()
    sites.to_csv(out/'unique_site_metrics.csv')
    macro = sites.groupby('model_id')[METRICS].mean().sort_values('log_mae')
    macro.to_csv(out/'unique_site_macro.csv')
    selections, paired, candidate_records = {}, [], []
    for fid in [0, 1, 2]:
        base_id = min(['idw', 'gp'], key=lambda name: fold.loc[(fid, name), 'log_mae'])
        baseline = gate_values(fold.loc[(fid, base_id)])
        eligible = []
        for model in NEW_MODELS:
            values = gate_values(fold.loc[(fid, model)])
            checks = _gate(values, baseline, config['v2']['gate'])
            row = {'fold_id': fid, 'model_id': model, 'baseline_id': base_id,
                   **values, **checks, 'passed': all(checks.values())}
            candidate_records.append(row)
            if row['passed']:
                eligible.append(model)
        selected = min(eligible, key=lambda name: fold.loc[(fid, name), 'log_mae']) if eligible else base_id
        selections[str(fid)] = {'baseline_model': base_id, 'selected_model': selected,
                                'fallback': not eligible, 'eligible_candidates': eligible}
        subset = curves[curves.fold_id.eq(fid)]
        a, b = subset[subset.model_id.eq(selected)], subset[subset.model_id.eq(base_id)]
        paired.append(a.merge(b, on=['fold_id', 'site_id', 'seed'], suffixes=('_method', '_baseline'), validate='one_to_one'))
    pd.DataFrame(candidate_records).to_csv(out/'candidate_gates.csv', index=False)
    paired = pd.concat(paired, ignore_index=True)
    paired.to_csv(out/'selected_system_paired.csv', index=False)
    columns = [f'{name}_{suffix}' for name in METRICS for suffix in ['method', 'baseline']]
    selected_sites = paired.groupby(['site_id', 'fold_id'])[columns].mean().groupby('site_id').mean()
    selected_sites.to_csv(out/'selected_system_sites.csv')
    method = {name: float(selected_sites[f'{name}_method'].mean()) for name in METRICS}
    baseline = {name: float(selected_sites[f'{name}_baseline'].mean()) for name in METRICS}
    global_checks = _gate(gate_values(method), gate_values(baseline), config['v2']['gate'])
    improved = int((selected_sites.log_mae_method < selected_sites.log_mae_baseline).sum())
    global_checks['majority_unique_sites_improve'] = improved > len(selected_sites)/2
    global_checks['all_folds_complete'] = set(selections) == {'0', '1', '2'}
    result = {'scope': 'exploratory_validation_only', 'folds': selections,
              'global_checks': global_checks, 'passed': all(global_checks.values()),
              'improved_sites': improved, 'unique_sites': len(selected_sites),
              'method': method, 'baseline': baseline, 'alpha_zero_max_difference': zero_difference,
              'relative_log_mae_reduction': 1-method['log_mae']/baseline['log_mae'],
              'candidates': NEW_MODELS, 'no_test_used': True}
    write_json(out/'selection.json', _plain(result))
    print(macro[['log_mae', 'h_point_mae_m', 'recall', 'false_positive_rate', 'risk_recall', 'risk_fpr']].to_string())
    print(json.dumps(_plain(result), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
