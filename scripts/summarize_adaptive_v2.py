"""Consolidate completed v2 experiments; produces CSV/JSON, never a report."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

from cpt_budget.config import write_json
from cpt_budget.adaptive_run import _frozen_inputs, _split_fingerprint, read_config
from run_adaptive_v2 import load_splits, summarize


def main():
    output = ROOT / 'results/adaptive_v2'
    output.mkdir(exist_ok=True)
    frozen = json.loads((ROOT/'results/adaptive_v1/frozen_config.json').read_text(encoding='utf-8'))
    original_inputs_unchanged = _frozen_inputs(read_config()) == frozen['frozen_inputs_sha256']
    original_split_unchanged = _split_fingerprint(ROOT/'results/adaptive_v1/splits.json') == frozen['split_sha256']
    if not original_inputs_unchanged or not original_split_unchanged:
        raise ValueError('Original frozen inputs or splits changed during v2 exploration')
    frames, all_predictions, selection_frames = [], [], []
    expected = []
    for seed in (17, 29, 43):
        expected.extend(('gp', seed, policy, 'runs/adaptive_v2_acquisition')
                        for policy in ('maximin', 'latent_ivr', 'target_ivr'))
        expected.extend(('target_et_residual', seed, policy, 'runs/adaptive_v2')
                        for policy in ('maximin', 'target_ivr'))
    missing = []
    for model, seed, policy, work in expected:
        directory = ROOT / work / f'newlock__s{seed}__{model}__{policy}'
        required = ['quality.csv', 'predictions.parquet', 'selections.csv', 'metadata.json']
        if not all((directory / name).exists() for name in required):
            missing.append(str(directory.relative_to(ROOT)))
            continue
        meta = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
        if meta['audit_labels_used_online'] or meta['stop_rule'] is not None:
            raise ValueError('This acquisition sweep must not contain an online audit/stop rule')
        frame = pd.read_csv(directory / 'quality.csv')
        frame['source_directory'] = str(directory.relative_to(ROOT))
        frames.append(frame)
        all_predictions.append(pd.read_parquet(directory / 'predictions.parquet'))
        selection_frames.append(pd.read_csv(directory / 'selections.csv'))
    if missing:
        raise ValueError('Incomplete fixed experiment; do not publish partial ranking: ' + ', '.join(missing))
    quality = pd.concat(frames, ignore_index=True).sort_values(['model_id', 'seed', 'policy', 'round'])
    predictions = pd.concat(all_predictions, ignore_index=True)
    selections = pd.concat(selection_frames, ignore_index=True)
    splits = {(s['site_id'], s['seed']): s for s in load_splits()}
    initial_max_difference, full_max_difference = 0., 0.
    for _, trace in quality.groupby('run_id'):
        split = splits[(trace.site_id.iloc[0], int(trace.seed.iloc[0]))]
        if trace.iloc[-1].n_observed != split['n_full'] or not trace.n_observed.is_monotonic_increasing:
            raise ValueError('A full sweep must increase support through the full endpoint')
        for _, state in trace.iterrows():
            known, selected = set(json.loads(state.known_ids)), set(json.loads(state.selected_ids))
            if len(known) != state.n_observed or not known <= set(split['full_ids']):
                raise ValueError('Invalid revealed location count or identity')
            if (known | selected) & set(split['audit_ids']) or known & selected:
                raise ValueError('Audit/known location entered acquisition candidates')
            for members in split['near_groups'].values():
                members = set(members)
                if (known & members and not members <= known) or (selected & members and not members <= selected):
                    raise ValueError('An acquisition step split a near group')
        run_predictions = predictions.loc[predictions.run_id.eq(trace.run_id.iloc[0])]
        if set(run_predictions.location_id) != set(split['audit_ids']):
            raise ValueError('Audit prediction set disagrees with frozen split')
    for (_, _), group in quality.groupby(['model_id', 'seed']):
        reference = group.loc[group.policy.eq('maximin')]
        for policy, states in group.groupby('policy'):
            for index, field in [(0, 'initial'), (-1, 'full')]:
                a = predictions.loc[predictions.run_id.eq(reference.run_id.iloc[0]) & predictions['round'].eq(reference.iloc[index]['round'])]
                b = predictions.loc[predictions.run_id.eq(states.run_id.iloc[0]) & predictions['round'].eq(states.iloc[index]['round'])]
                joined = a[['location_id', 'depth_index', 'pred_log_qc']].merge(
                    b[['location_id', 'depth_index', 'pred_log_qc']], on=['location_id', 'depth_index'], validate='one_to_one')
                difference = float(abs(joined.pred_log_qc_x - joined.pred_log_qc_y).max())
                if field == 'initial':
                    initial_max_difference = max(initial_max_difference, difference)
                else:
                    full_max_difference = max(full_max_difference, difference)
    if initial_max_difference > 1e-7 or full_max_difference > 1e-7:
        raise ValueError('Identical initial/full contexts produced policy-dependent predictions')
    per_run = summarize(quality)
    curves = []
    for run_id, trace in quality.groupby('run_id'):
        trace = trace.sort_values('n_observed')
        # Integrating the measured error curve uses actual CPT counts. No
        # synthetic prediction state or quality-passing stop is interpolated.
        x = trace.n_observed.to_numpy(float)
        width = x[-1] - x[0]
        curves.append(dict(run_id=run_id,
            budget_normalized_log_error_auc=float(trapezoid(trace.log_mae, x)/width),
            budget_normalized_thickness_error_auc=float(trapezoid(trace.thickness_mae_m, x)/width),
            best_log_budget=int(trace.loc[trace.log_mae.idxmin(), 'n_observed']),
            best_log_thickness_mae_m=float(trace.loc[trace.log_mae.idxmin(), 'thickness_mae_m']),
            passing_reconstruction_states=int(trace.reconstruction_pass.sum()),
            passing_complete_task_states=int(trace.complete_task_pass.sum())))
    per_run = per_run.merge(pd.DataFrame(curves), on='run_id', validate='one_to_one')
    reference = per_run.loc[per_run.policy.eq('maximin')].set_index(['model_id', 'seed'])
    for metric, label in [('budget_normalized_log_error_auc', 'log_auc'),
                          ('budget_normalized_thickness_error_auc', 'thickness_auc')]:
        per_run[f'{label}_improvement_vs_maximin_fraction'] = [
            1 - row[metric]/reference.loc[(row.model_id, row.seed), metric]
            for _, row in per_run.iterrows()]
    metrics = ['minimum_log_mae', 'full_log_mae', 'full_thickness_mae_m', 'full_recall', 'full_fpr',
               'budget_normalized_log_error_auc', 'budget_normalized_thickness_error_auc',
               'log_auc_improvement_vs_maximin_fraction', 'thickness_auc_improvement_vs_maximin_fraction',
               'oracle_max_saved_fraction_reconstruction', 'oracle_max_saved_fraction_complete_task']
    aggregate = per_run.groupby(['model_id', 'policy'], sort=True)[metrics].mean().reset_index()
    aggregate['seeds'] = 3
    quality.to_csv(output/'quality_states.csv', index=False)
    per_run.to_csv(output/'acquisition_runs.csv', index=False)
    aggregate.to_csv(output/'acquisition_comparison.csv', index=False)
    selections.to_csv(output/'selections.csv', index=False)
    write_json(output/'correctness.json', dict(trajectories=len(per_run),
        states=len(quality), all_frozen_splits_preserved=True, all_near_groups_atomic=True,
        all_full_endpoints_reached=True, audit_ids_never_revealed=True,
        initial_prediction_max_abs_difference=initial_max_difference,
        full_prediction_max_abs_difference=full_max_difference,
        original_frozen_inputs_unchanged=original_inputs_unchanged,
        original_frozen_split_unchanged=original_split_unchanged,
        truth_used_only_for_offline_scores=True, no_online_early_stop_used=True,
        no_runtime_comparison=True, no_report_generated=True, no_display_edited=True))
    write_json(output/'experiment_summary.json', dict(scope='post-unblind finite exploration',
        primary_site='newlock', seeds=[17,29,43], trajectories=len(per_run), states=len(quality),
        quality_targets_unchanged=True, reconstruction_pass_states=int(quality.reconstruction_pass.sum()),
        complete_task_pass_states=int(quality.complete_task_pass.sum()),
        online_verified_saved_cpt=0,
        savings_scope='All new acquisition paths are full diagnostic sweeps. Oracle opportunities are not online savings.',
        model_probe_states=102+105+153,
        model_probe_scope='Model probes overlap controls/contexts; 360 rows are not 360 independent experiments.',
        gp_initialization_fits=6, gp_kernel_change_probe_states=2,
        stopping_experiment='results/adaptive_v2_diagnostics/stopping_v2_frozen_selection.json',
        promotion='No candidate is promoted solely on log-MAE, stability or offline oracle opportunity.'))
    print(aggregate.to_string(index=False))
    print(f'States={len(quality)}, reconstruction passes={int(quality.reconstruction_pass.sum())}, '
          f'complete task passes={int(quality.complete_task_pass.sum())}')


if __name__ == '__main__':
    main()
