"""Run fixed, post-unblind v2 acquisition sweeps. CSV/JSON evidence only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from cpt_budget.adaptive_data import audit_truth
from cpt_budget.adaptive_run import _score
from cpt_budget.adaptive_v2_replay import POLICIES, run_v2_trajectory
from cpt_budget.config import load_config, write_json
from cpt_budget.types import load_dataset


def load_splits():
    splits = json.loads((ROOT / 'results/adaptive_v1/splits.json').read_text(encoding='utf-8'))
    return splits.get('splits', splits) if isinstance(splits, dict) else splits


def score_sweep(result, truth, goals):
    """Read audit labels only after the online sweep has returned."""
    records = []
    for round_index, predictions in result['predictions'].groupby('round', sort=True):
        row = result['trajectories'].loc[result['trajectories']['round'].eq(round_index)].iloc[0].to_dict()
        scores = _score(predictions, truth)
        records.append({**row, **scores})
    quality = pd.DataFrame(records)
    full = quality.iloc[-1]
    if full.n_observed != full.n_full:
        raise ValueError('A diagnostic sweep must reach its isolated full endpoint')
    quality['absolute_pass'] = (quality.log_mae <= goals['log_mae']) & (quality.thickness_mae_m <= goals['thickness_mae_m'])
    quality['relative_pass'] = (quality.log_mae <= full.log_mae + .05) & (quality.thickness_mae_m <= full.thickness_mae_m + .25)
    quality['reconstruction_pass'] = quality.absolute_pass & quality.relative_pass
    quality['low_qc_pass'] = (quality.recall >= .5) & (quality.fpr <= .1)
    quality['complete_task_pass'] = quality.reconstruction_pass & quality.low_qc_pass
    return quality


def summarize(quality):
    rows = []
    keys = ['site_id', 'seed', 'model_id', 'policy', 'run_id']
    for key, states in quality.groupby(keys, sort=True):
        states = states.sort_values('round')
        full = states.iloc[-1]
        row = {**dict(zip(keys, key)), 'n_full': int(full.n_full),
               'minimum_log_mae': float(states.log_mae.min()),
               'full_log_mae': float(full.log_mae), 'full_thickness_mae_m': float(full.thickness_mae_m),
               'full_recall': float(full.recall), 'full_fpr': float(full.fpr),
               'full_reconstruction_pass': bool(full.reconstruction_pass),
               'full_complete_task_pass': bool(full.complete_task_pass),
               'n_online_stop': int(full.n_observed), 'online_saved_fraction': 0.,
               'opportunity_is_offline_oracle': True}
        for metric in ['reconstruction', 'complete_task']:
            passing = states.loc[states[f'{metric}_pass']]
            n = int(passing.n_observed.min()) if len(passing) else None
            row[f'oracle_min_n_{metric}'] = n
            row[f'oracle_max_saved_fraction_{metric}'] = (1 - n / full.n_full) if n is not None else 0.
        rows.append(row)
    return pd.DataFrame(rows)


def protocol(args, goals):
    return dict(version='adaptive_v2_acquisition_sweep', scope='post_unblind_exploration',
        purpose='test whether legal acquisition and revealed-only adaptation create quality-passing states',
        sites=args.sites, seeds=args.seeds, models=args.models, policies=args.policies,
        split_source='results/adaptive_v1/splits.json', absolute_goals=goals,
        relative_goals=dict(log_mae_tolerance=.05, thickness_mae_tolerance_m=.25),
        low_qc_goals=dict(recall=.5, fpr=.1), batch_size=2, atomic_near_groups=True,
        coverage_weight=.15, stop_rule=None, online_early_savings_claimed=False,
        model_selection='revealed_group_cv_for_target_models; immutable_fold_source_for_v1_models',
        helper_for_UDS_and_target_ivr='gp; revealed_depth_median_when_primary_is_gp',
        monitor='current_remaining_plan_only', support_max_groups=5,
        support_diagnostics=not args.no_support, audit_labels_used_for_decisions=False,
        future_labels_used_for_acquisition=False, audit_geometry_used_for_acquisition=False,
        oracle_metrics='offline opportunity upper bound only; not an online stopping policy',
        no_runtime_comparison=True, no_report=True, no_display_changes=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sites', nargs='+', default=['newlock'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[17, 29, 43])
    parser.add_argument('--models', nargs='+', default=['gp'])
    parser.add_argument('--policies', nargs='+', choices=POLICIES, default=['maximin', 'latent_ivr', 'target_ivr'])
    parser.add_argument('--output', default='results/adaptive_v2')
    parser.add_argument('--work', default='runs/adaptive_v2')
    parser.add_argument('--no-support', action='store_true', help='Acquisition-only sweep; no online stop is permitted')
    parser.add_argument('--publish-only', action='store_true')
    args = parser.parse_args()
    output, work = ROOT / args.output, ROOT / args.work
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((ROOT / 'results/adaptive_v1/frozen_config.json').read_text(encoding='utf-8'))
    # Read numeric targets without changing the original frozen experiment.
    goals = dict(log_mae=float(frozen['quality']['absolute_log_mae_max']),
                 thickness_mae_m=float(frozen['quality']['absolute_thickness_mae_max_m']))
    spec = protocol(args, goals)
    name = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    spec_path = output / f'protocol_{name}.json'
    if not spec_path.exists():
        write_json(spec_path, spec)
    dataset = load_dataset(load_config())
    if not args.publish_only:
        for split in load_splits():
            if split['site_id'] not in args.sites or int(split['seed']) not in args.seeds:
                continue
            for model_id in args.models:
                for policy in args.policies:
                    run_id = f"{split['site_id']}__s{split['seed']}__{model_id}__{policy}"
                    directory = work / run_id
                    if (directory / 'quality.csv').exists():
                        print(f'{run_id} already complete', flush=True)
                        continue
                    with threadpool_limits(limits=1):
                        result = run_v2_trajectory(dataset, split, model_id, policy,
                            diagnostics=not args.no_support, log=lambda message: print(message, flush=True))
                    # Audit scoring is a separate operation after all decisions.
                    quality = score_sweep(result, audit_truth(dataset, split), goals)
                    directory.mkdir(parents=True, exist_ok=True)
                    for key in ['trajectories', 'selections', 'support']:
                        result[key].to_csv(directory / f'{key}.csv', index=False)
                    result['predictions'].to_parquet(directory / 'predictions.parquet', index=False)
                    quality.to_csv(directory / 'quality.csv', index=False)
                    write_json(directory / 'metadata.json', {**result['metadata'], 'protocol': spec_path.name})
                    print(f'{run_id} complete; min audit log-MAE={quality.log_mae.min():.6f}; '
                          f'passing reconstruction states={int(quality.reconstruction_pass.sum())}', flush=True)
    # Partition workers publish only their own manifest/CSV; parent aggregates
    # once all partitions are done by calling --publish-only with the full set.
    directories = [p for p in work.iterdir() if p.is_dir() and (p / 'quality.csv').exists()]
    frames = []
    for directory in directories:
        meta = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
        if (meta['site_id'] in args.sites and meta['seed'] in args.seeds
                and meta['model_id'] in args.models and meta['policy'] in args.policies):
            frames.append(pd.read_csv(directory / 'quality.csv'))
    if frames:
        quality = pd.concat(frames, ignore_index=True).sort_values(['site_id', 'seed', 'model_id', 'policy', 'round'])
        quality.to_csv(output / f'quality_{name}.csv', index=False)
        summary = summarize(quality)
        summary.to_csv(output / f'summary_{name}.csv', index=False)
        write_json(output / f'findings_{name}.json', dict(protocol=spec_path.name, trajectories=len(summary),
            completed_sweeps=len(directories), audit_quality_passing_states=int(quality.reconstruction_pass.sum()),
            complete_task_passing_states=int(quality.complete_task_pass.sum()),
            online_early_stop_used=False, online_savings_claimed=False,
            interpretation='Oracle opportunity metrics do not constitute deployable savings.'))


if __name__ == '__main__':
    main()
