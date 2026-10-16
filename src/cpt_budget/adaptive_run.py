"""Run prediction comparison, development-only selection, and adaptive evaluation."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
from importlib.metadata import version
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from .config import PROJECT_ROOT, load_config, write_json
from .evaluate import evaluate
from .types import load_dataset

METRICS = ['log_mae', 'thickness_mae_m', 'precision', 'recall', 'f1', 'fpr', 'log_error_p90']


def read_config(path='configs/adaptive_v1.yaml'):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    config['_path'] = str(path)
    return config


def _paths(config):
    output = PROJECT_ROOT / config['experiment']['output']
    work = PROJECT_ROOT / config['experiment']['work']
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    return output, work


def _score(frame, truth):
    result = evaluate(frame, truth, load_config())
    result['thickness_mae_m'] = result.pop('h_point_mae_m')
    result['fpr'] = result.pop('false_positive_rate')
    errors = frame[['location_id', 'depth_index', 'pred_log_qc']].merge(
        truth[['location_id', 'depth_index', 'log_qc']], on=['location_id', 'depth_index'], validate='one_to_one')
    result['log_error_p90'] = float(np.quantile(abs(errors.pred_log_qc - errors.log_qc), .9))
    return {key: value for key, value in result.items() if not key.endswith('_reason')}


def _environment():
    import torch
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'cpu': platform.processor(), 'logical_cpu_count': os.cpu_count(),
            'packages': {name: version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'tabpfn']},
            'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'peak_cuda_allocated_mb': torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None}


def _frozen_inputs(config):
    paths = [Path(config['_path']), PROJECT_ROOT / 'data_processed_v2/grid.parquet',
             PROJECT_ROOT / 'data_processed_v2/profiles.parquet']
    for fold in range(3):
        for name in ['tabpfn.ckpt', 'gp_trend.json', 'xgb.json', 'residual_trees.ubj']:
            paths.append(PROJECT_ROOT / 'artifacts/models' / f'fold_{fold}' / name)
    result = {}
    for path in paths:
        payload = (json.dumps(yaml.safe_load(path.read_text(encoding='utf-8')), sort_keys=True).encode()
                   if path == Path(config['_path']) else path.read_bytes())
        result[str(path.relative_to(PROJECT_ROOT))] = hashlib.sha256(payload).hexdigest()
    return result


def _split_fingerprint(path):
    # Git's LF/CRLF conversion does not change a JSON split's meaning.
    return hashlib.sha256(json.dumps(json.loads(path.read_text(encoding='utf-8')), sort_keys=True).encode()).hexdigest()


def prepare(config):
    from .adaptive_data import prepare_adaptive_data
    output, _ = _paths(config)
    return prepare_adaptive_data(output=output, seeds=config['experiment']['seeds'])


def _splits(config, role=None):
    output, _ = _paths(config)
    if not (output / 'splits.json').exists():
        prepare(config)
    splits = json.loads((output / 'splits.json').read_text(encoding='utf-8'))
    if isinstance(splits, dict):
        splits = splits['splits']
    return [s for s in splits if role is None or s['role'] == role]


def _e1_states(split, dataset, config):
    from .adaptive_policy import select_batch
    known = list(split['initial_ids'])
    geometry = dataset.locations(split['site_id'])
    checkpoints = list(config['models']['observation_checkpoints'])
    visited = set()
    while True:
        requested = [n for n in checkpoints if n <= len(known)]
        if requested or len(known) == split['n_full']:
            key = tuple(sorted(known))
            if key not in visited:
                yield list(known), requested
                visited.add(key)
            checkpoints = [n for n in checkpoints if n > len(known)]
        remaining = sorted(set(split['full_ids']) - set(known))
        if not remaining:
            return
        selected, _ = select_batch(geometry, known, remaining, split['near_groups'],
                                  policy='maximin', batch_size=config['experiment']['batch_size'])
        known.extend(selected)


def compare(config, role=None, sites=None, seeds=None, include_ablations=True):
    """Every model receives exactly the same complete-group revealed states."""
    from .adaptive_data import audit_truth
    from .adaptive_models import AdaptiveModelSuite
    output, work = _paths(config)
    dataset = load_dataset(load_config())
    wanted = list(config['models']['main'])
    if include_ablations:
        wanted += list(config['models']['ablations'])
    rows, runtime = [], []
    previous_fold = previous_role = None
    suite = None
    for split in _splits(config, role):
        if sites and split['site_id'] not in sites or seeds and split['seed'] not in seeds:
            continue
        if (previous_fold, previous_role) != (split['fold_id'], split['role']):
            suite = None
            gc.collect()
            suite = AdaptiveModelSuite(split['fold_id'], wanted, role=split['role'],
                                       device=config['experiment']['device']).initialize()
            previous_fold, previous_role = split['fold_id'], split['role']
        query = dataset.query(split['audit_ids'])
        # E1 evaluates saved predictions after the forward pass; labels never enter the models.
        for known, requested in _e1_states(split, dataset, config):
            start = perf_counter()
            suite.condition(dataset.observed(known))
            predictions = suite.predict(query)
            elapsed = perf_counter() - start
            truth = audit_truth(dataset, split)
            for model_id, bundle in predictions.items():
                rows.append({'split_id': split['split_id'], 'site_id': split['site_id'],
                             'seed': split['seed'], 'fold_id': split['fold_id'], 'role': split['role'],
                             'model_id': model_id, 'budget_n': len(known), 'n_full': split['n_full'],
                             'requested_checkpoints': json.dumps(requested),
                             'full_endpoint': len(known) == split['n_full'], **_score(bundle.frame, truth)})
            runtime.append({'stage': 'E1', 'site_id': split['site_id'], 'seed': split['seed'],
                            'role': split['role'], 'fold_id': split['fold_id'], 'n_observed': len(known),
                            'model_id': '+'.join(wanted), 'elapsed_s': elapsed})
            print(f"E1 {split['role']} {split['site_id']} seed={split['seed']} n={len(known)}/{split['n_full']} {elapsed:.2f}s", flush=True)
        pd.DataFrame(rows).to_csv(work / f'e1_{role or "all"}.csv', index=False)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError('No E1 splits matched')
    name = 'pilot_model_comparison.csv' if sites or seeds else 'model_comparison.csv'
    path = output / name
    ablation_path = output / ('pilot_ablation.csv' if sites or seeds else 'ablation.csv')
    if role is not None and not (sites or seeds):
        previous = [pd.read_csv(p) for p in (path, ablation_path) if p.exists()]
        if previous:
            old = pd.concat(previous, ignore_index=True)
            frame = pd.concat([old[old.role != role], frame], ignore_index=True)
    frame[frame.model_id.isin(config['models']['main'])].to_csv(path, index=False)
    frame[~frame.model_id.isin(config['models']['main'])].to_csv(ablation_path, index=False)
    pd.DataFrame(runtime).to_csv(work / f'e1_runtime_{role or "all"}.csv', index=False)
    return frame


def quality_flags(row, full, quality):
    finite = all(pd.notna(row[k]) for k in ['log_mae', 'thickness_mae_m'])
    absolute = finite and row['log_mae'] <= quality['absolute_log_mae_max'] and row['thickness_mae_m'] <= quality['absolute_thickness_mae_max_m']
    relative = finite and row['log_mae'] <= full['log_mae'] + quality['relative_log_tolerance'] and row['thickness_mae_m'] <= full['thickness_mae_m'] + quality['relative_thickness_tolerance_m']
    low = None if pd.isna(row.get('recall')) else bool(row['recall'] >= quality['low_qc_recall_target'] and pd.notna(row.get('fpr')) and row['fpr'] <= quality['low_qc_fpr_target'])
    return {'absolute_quality_pass': bool(absolute), 'relative_quality_pass': bool(relative),
            'audit_pass': bool(absolute and relative), 'low_qc_audit_pass': low,
            'log_mae_delta_full': row['log_mae'] - full['log_mae'],
            'thickness_mae_delta_full_m': row['thickness_mae_m'] - full['thickness_mae_m']}


def score_trajectory(result, dataset, split):
    from .adaptive_data import audit_truth
    truth = audit_truth(dataset, split)
    predictions = result['predictions']
    rows = []
    group_keys = [key for key in ['run_id', 'policy_id', 'round'] if key in predictions]
    for key, group in predictions.groupby(group_keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        meta = dict(zip(group_keys, key))
        meta.update({k: split[k] for k in ['site_id', 'seed', 'fold_id', 'role']})
        rows.append({**meta, **_score(group, truth)})
    return pd.DataFrame(rows)


def _store_result(result, split, dataset, work, label):
    folder = work / label
    folder.mkdir(parents=True, exist_ok=True)
    result['runtime']['stage'] = 'E3_' + label.split('/')[0]
    result['runtime']['process_peak_cuda_allocated_mb'] = _environment()['peak_cuda_allocated_mb']
    for name in ['trajectories', 'selections', 'runtime']:
        result[name].to_csv(folder / f'{name}.csv', index=False)
    result['predictions'].to_parquet(folder / 'predictions.parquet', index=False)
    scored = score_trajectory(result, dataset, split)
    scored.to_csv(folder / 'quality.csv', index=False)
    return scored


def _read_runs(work, phase):
    pairs = []
    for folder in sorted((work / phase).glob('*')):
        path = folder / 'trajectories.csv'
        if path.exists():
            pairs.append((pd.read_csv(path), pd.read_csv(folder / 'quality.csv')))
    return pairs


def _attach_quality(pairs, quality):
    audits = []
    full_by_split = {}
    for trajectory, scored in pairs:
        if trajectory.policy_id.iloc[0] == 'B_full':
            full_by_split[(trajectory.site_id.iloc[0], int(trajectory.seed.iloc[0]))] = scored.iloc[-1]
    for trajectory, scored in pairs:
        meta = trajectory[[c for c in ['run_id', 'policy_id', 'round', 'model_id', 'n_observed', 'n_full', 'variant'] if c in trajectory]]
        joined = scored.merge(meta, on=['run_id', 'policy_id', 'round'], validate='one_to_one')
        full = full_by_split[(trajectory.site_id.iloc[0], int(trajectory.seed.iloc[0]))]
        flags = pd.DataFrame([quality_flags(row, full, quality) for _, row in joined.iterrows()])
        audits.append(pd.concat([joined.reset_index(drop=True), flags], axis=1))
    return pd.concat(audits, ignore_index=True)


def _decision(trajectory, audit, rule=None):
    from .adaptive_policy import StopRule, StopState, update_stop
    trace = trajectory.sort_values('round').copy()
    state = StopState()
    chosen = trace.iloc[-1]
    recorded = trace[trace.stopped.eq(True)] if trace.role.iloc[0] == 'evaluation' and 'stopped' in trace else pd.DataFrame()
    if len(recorded):
        chosen = recorded.iloc[0]
    else:
        for _, row in trace.iterrows():
            if row.policy_id == 'B_full':
                chosen = row
                break
            if rule is not None:
                signals = {k: row[k] for k in ['delta_q90', 'delta_h90', 'support_log_mae', 'uncertainty_q90', 'coverage_ok']}
                state = update_stop(StopRule(**rule), state, signals, int(row.n_observed), int(row.n_full - row.n_observed))
                if state.stopped:
                    chosen = row
                    break
    quality = audit[audit.run_id.eq(chosen.run_id) & audit['round'].eq(chosen['round'])].iloc[0]
    early = int(chosen.n_observed) < int(chosen.n_full)
    reason = 'online_stability' if early else 'full_plan_complete'
    return {k: chosen[k] for k in ['run_id', 'site_id', 'seed', 'fold_id', 'role', 'model_id', 'policy_id', 'variant']} | {
        'round': int(chosen['round']), 'n_full': int(chosen.n_full), 'n_stop': int(chosen.n_observed),
        'n_initial': int(trace.n_observed.iloc[0]), 'n_batches': int(chosen['round']) + 1,
        'audit_pass': bool(quality.audit_pass), 'low_qc_audit_pass': None if pd.isna(quality.low_qc_audit_pass) else bool(quality.low_qc_audit_pass),
        'is_early_stop': early, 'false_early_stop': bool(early and not quality.audit_pass),
        'stop_reason': reason, 'saved_cpt': int(chosen.n_full - chosen.n_observed),
        'saved_fraction': float((chosen.n_full - chosen.n_observed) / chosen.n_full),
        **{k: quality[k] for k in METRICS}}


def calibrate(config, run_replays=True):
    from .adaptive_replay import run_trajectory
    from .adaptive_models import AdaptiveModelSuite, make_predictor
    output, work = _paths(config)
    e1 = pd.read_csv(output / 'model_comparison.csv')
    if e1.role.eq('evaluation').any():
        raise ValueError('Evaluation has begun; use a new experiment directory for a new development selection')
    development = e1[e1.role.eq('development')]
    if development.empty:
        raise ValueError('Run development E1 before calibration')
    ranking = development.groupby(['model_id', 'site_id', 'seed'])[METRICS].mean().groupby(['model_id', 'site_id']).mean().groupby('model_id').mean().sort_values(['log_mae', 'thickness_mae_m'])
    ranking.to_csv(output / 'development_model_ranking.csv')
    model_id = str(ranking.index[0])
    full_e1 = development[development.model_id.eq(model_id) & development.full_endpoint.astype(bool)]
    quality = dict(config['quality'])
    quality['absolute_log_mae_max'] = float(full_e1.log_mae.quantile(.75) + quality['absolute_log_margin'])
    quality['absolute_thickness_mae_max_m'] = float(full_e1.thickness_mae_m.quantile(.75) + quality['absolute_thickness_margin_m'])
    dataset = load_dataset(load_config())
    for split in _splits(config, 'development') if run_replays else []:
        helper = 'tabpfn_joint' if model_id == 'xgb' else 'xgb'
        needed = tuple(dict.fromkeys((model_id, 'gp', helper)))
        geometry = dataset.query([*split['full_ids'], *split['audit_ids']])
        suite = AdaptiveModelSuite(split['fold_id'], needed, role=split['role'],
                                   geometry=geometry, device=config['experiment']['device']).initialize()
        support = make_predictor(model_id, split['fold_id'], role=split['role'],
                                 geometry=geometry, device=config['experiment']['device']).initialize()
        policies = [('B_full', 'full', (.5, .3, .2)), ('B_seq', 'maximin', (.5, .3, .2))]
        policies += [('P_adapt', variant, tuple(weights)) for variant, weights in config['acquisition']['candidates'].items()]
        for policy, variant, weights in policies:
            label = f"development/{split['site_id']}_s{split['seed']}_{policy}_{variant}"
            print(f"E3 development {split['site_id']} seed={split['seed']} {policy}/{variant}", flush=True)
            result = run_trajectory(dataset, split, model_id, policy, weights=weights, stop_rule=None,
                                    suite=None if policy == 'B_full' else suite,
                                    support_predictor=None if policy == 'B_full' else support)
            for key in ['trajectories', 'predictions', 'selections', 'runtime']:
                result[key]['variant'] = variant
                if 'run_id' in result[key]:
                    result[key]['run_id'] = result[key].run_id.astype(str) + '_' + variant
            _store_result(result, split, dataset, work, label)
    pairs = _read_runs(work, 'development')
    expected = len(_splits(config, 'development')) * (2 + len(config['acquisition']['candidates']))
    if len(pairs) != expected:
        raise ValueError(f'Expected {expected} complete development trajectories, found {len(pairs)}')
    audit = _attach_quality(pairs, quality)
    eligible = []
    for trajectory, _ in pairs:
        if trajectory.policy_id.iloc[0] == 'B_full':
            continue
        joined = trajectory.merge(audit[['run_id', 'policy_id', 'round', 'audit_pass']], on=['run_id', 'policy_id', 'round'])
        eligible.append(joined[joined.n_observed.ge(config['stopping']['min_observed']) & joined.audit_pass])
    eligible = pd.concat(eligible, ignore_index=True)
    all_signals = pd.concat([t for t, _ in pairs if t.policy_id.iloc[0] != 'B_full'], ignore_index=True)
    source = eligible if len(eligible) else all_signals
    trials, rule_by_trial = [], {}
    for quantile in config['stopping']['candidate_quantiles']:
        rule = {}
        for key in ['delta_q90', 'delta_h90', 'support_log_mae', 'uncertainty_q90']:
            values = source[key].replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                values = all_signals[key].replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                raise ValueError(f'No finite development values for stopping signal {key}')
            rule[f'{key}_max'] = float(values.quantile(quantile))
        rule.update(min_observed=config['stopping']['min_observed'], consecutive_rounds=config['stopping']['consecutive_rounds'])
        for variant in config['acquisition']['candidates']:
            decisions = pd.DataFrame([_decision(t, audit, rule) for t, _ in pairs if t.policy_id.iloc[0] == 'P_adapt' and t.variant.iloc[0] == variant])
            decisions['verified_saved_fraction'] = np.where(decisions.audit_pass, decisions.saved_fraction, 0.)
            by_site = decisions.groupby('site_id')[['verified_saved_fraction', 'false_early_stop']].mean()
            score = float(by_site.verified_saved_fraction.mean() - by_site.false_early_stop.mean())
            trial_id = f'{variant}_q{quantile:g}'
            trials.append({'trial_id': trial_id, 'variant': variant, 'quantile': quantile, 'score': score,
                           'verified_saved_fraction': by_site.verified_saved_fraction.mean(),
                           'false_early_stop_rate': by_site.false_early_stop.mean(), **rule})
            rule_by_trial[trial_id] = rule
    trials = pd.DataFrame(trials).sort_values(['score', 'false_early_stop_rate', 'variant', 'quantile'], ascending=[False, True, False, True])
    trials.to_csv(output / 'stop_calibration.csv', index=False)
    winner = trials.iloc[0]
    environment = _environment()
    if not run_replays and (output / 'frozen_config.json').exists():
        environment.update(json.loads((output / 'frozen_config.json').read_text(encoding='utf-8')).get('environment', {}))
    frozen = {'created_at': datetime.now().astimezone().isoformat(), 'experiment': config,
              'primary_model': model_id, 'acquisition_variant': winner.variant,
              'acquisition_weights': config['acquisition']['candidates'][winner.variant],
              'stop_rule': rule_by_trial[winner.trial_id], 'quality': quality,
              'selection_source': 'development sites only; no evaluation outcomes used',
              'environment': environment,
              'frozen_inputs_sha256': _frozen_inputs(config),
              'split_sha256': _split_fingerprint(output / 'splits.json')}
    write_json(output / 'frozen_config.json', frozen)
    print(json.dumps({'primary_model': model_id, 'variant': winner.variant, 'stop_rule': frozen['stop_rule'], 'quality': quality}), flush=True)
    return frozen


def evaluate_replays(config, sites=None, seeds=None, publish_outputs=True):
    from .adaptive_policy import StopRule
    from .adaptive_replay import run_trajectory
    from .adaptive_models import AdaptiveModelSuite, make_predictor
    output, work = _paths(config)
    frozen = json.loads((output / 'frozen_config.json').read_text(encoding='utf-8'))
    if _split_fingerprint(output / 'splits.json') != frozen['split_sha256']:
        raise ValueError('Adaptive splits differ from the frozen selection')
    if _frozen_inputs(config) != frozen['frozen_inputs_sha256']:
        raise ValueError('Configuration, prepared data or source weights changed after selection')
    dataset = load_dataset(load_config())
    for split in _splits(config, 'evaluation'):
        if sites and split['site_id'] not in sites or seeds and split['seed'] not in seeds:
            continue
        model_id = frozen['primary_model']
        helper = 'tabpfn_joint' if model_id == 'xgb' else 'xgb'
        needed = tuple(dict.fromkeys((model_id, 'gp', helper)))
        geometry = dataset.query([*split['full_ids'], *split['audit_ids']])
        suite = AdaptiveModelSuite(split['fold_id'], needed, role=split['role'],
                                   geometry=geometry, device=config['experiment']['device']).initialize()
        support = make_predictor(model_id, split['fold_id'], role=split['role'],
                                 geometry=geometry, device=config['experiment']['device']).initialize()
        for policy, variant in [('B_full', 'full'), ('B_seq', 'maximin'), ('P_adapt', frozen['acquisition_variant'])]:
            label = f"evaluation/{split['site_id']}_s{split['seed']}_{policy}_{variant}"
            completed = work / label
            if all((completed / name).exists() for name in ['trajectories.csv', 'quality.csv', 'predictions.parquet', 'runtime.csv', 'selections.csv']):
                print(f'Reuse completed frozen trajectory {label}', flush=True)
                continue
            print(f"E3 evaluation {split['site_id']} seed={split['seed']} {policy}/{variant}", flush=True)
            result = run_trajectory(dataset, split, frozen['primary_model'], policy,
                                    weights=frozen['acquisition_weights'],
                                    stop_rule=StopRule(**frozen['stop_rule']) if policy != 'B_full' else None,
                                    suite=None if policy == 'B_full' else suite,
                                    support_predictor=None if policy == 'B_full' else support)
            for key in ['trajectories', 'predictions', 'selections', 'runtime']:
                result[key]['variant'] = variant
                if 'run_id' in result[key]:
                    result[key]['run_id'] = result[key].run_id.astype(str) + '_' + variant
            _store_result(result, split, dataset, work, label)
    return publish(config) if publish_outputs else None


def publish(config):
    output, work = _paths(config)
    frozen = json.loads((output / 'frozen_config.json').read_text(encoding='utf-8'))
    comparison = pd.read_csv(output / 'model_comparison.csv')
    comparison[comparison.full_endpoint.eq(True)].to_csv(output / 'full_plan_model_sensitivity.csv', index=False)
    pairs = _read_runs(work, 'development') + _read_runs(work, 'evaluation')
    pairs = [(t, a) for t, a in pairs if t.policy_id.iloc[0] != 'P_adapt' or t.variant.iloc[0] == frozen['acquisition_variant']]
    expected = len(_splits(config)) * 3
    if len(pairs) != expected:
        raise ValueError(f'Formal publication requires all {expected} trajectories, found {len(pairs)}')
    audit = _attach_quality(pairs, frozen['quality'])
    signal_checks = []
    for trajectory, _ in pairs:
        if trajectory.policy_id.iloc[0] == 'B_full':
            continue
        linked = trajectory.merge(audit[['run_id', 'policy_id', 'round', 'log_mae', 'thickness_mae_m']],
                                  on=['run_id', 'policy_id', 'round'], validate='one_to_one')
        values = linked[['uncertainty_q90', 'log_mae', 'thickness_mae_m']].dropna()
        record = {key: trajectory.iloc[0][key] for key in ['run_id', 'site_id', 'seed', 'role', 'policy_id']}
        record['n_rounds'] = len(values)
        for metric in ['log_mae', 'thickness_mae_m']:
            record[f'gp_uncertainty_vs_{metric}_spearman'] = (float(values.uncertainty_q90.corr(values[metric], method='spearman'))
                if len(values) >= 3 and values.uncertainty_q90.nunique() > 1 and values[metric].nunique() > 1 else None)
        signal_checks.append(record)
    pd.DataFrame(signal_checks).to_csv(output / 'signal_diagnostics.csv', index=False)
    decisions = pd.DataFrame([_decision(t, audit, frozen['stop_rule']) for t, _ in pairs])
    traces, selected, timings = [], [], []
    for t, _ in pairs:
        decision = decisions[decisions.run_id.eq(t.run_id.iloc[0])].iloc[0]
        trace = t[t['round'].le(decision['round'])].copy()
        trace['stopped'] = trace['round'].eq(decision['round'])
        trace.loc[trace.stopped, 'stop_reason'] = decision.stop_reason
        trace.loc[trace.stopped, 'selected_ids'] = '[]'
        traces.append(trace)
    trace = pd.concat(traces, ignore_index=True)
    keep = trace[['run_id', 'policy_id', 'round']]
    audit = audit.merge(keep, on=['run_id', 'policy_id', 'round'], validate='one_to_one')
    for phase in ['development', 'evaluation']:
        for folder in (work / phase).glob('*'):
            if not (folder / 'trajectories.csv').exists():
                continue
            for name, collector in [('selections', selected), ('runtime', timings)]:
                frame = pd.read_csv(folder / f'{name}.csv')
                if frame.empty or 'run_id' not in frame:
                    continue
                collector.append(frame)
    selections = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    if len(selections):
        # A selected batch at state t belongs to the next observation state.
        allowed = trace[['run_id', 'round']].copy()
        selections = selections.merge(allowed, on=['run_id', 'round'], how='inner')
    trace.to_csv(output / 'trajectories.csv', index=False)
    audit.to_csv(output / 'quality_audit.csv', index=False)
    decisions.to_csv(output / 'decisions.csv', index=False)
    selections.to_csv(output / 'selections.csv', index=False)
    e1_runtime = [pd.read_csv(p) for p in work.glob('e1_runtime_*.csv')]
    runtime = pd.concat(timings + e1_runtime, ignore_index=True)
    runtime.to_csv(output / 'runtime.csv', index=False)
    manifest = pd.read_csv(output / 'data_manifest.csv')
    locations = manifest.copy()
    locations['role'] = locations['partition'].map({'S0': 'initial', 'C': 'candidate', 'A': 'audit'}) if 'partition' in locations else locations.get('side', '').map({'S0': 'initial', 'C': 'candidate', 'A': 'audit'})
    locations.to_csv(output / 'locations.csv', index=False)
    economics(config)
    environment = _environment()
    if 'process_peak_cuda_allocated_mb' in runtime:
        recorded_peak = runtime.process_peak_cuda_allocated_mb.max()
        if pd.notna(recorded_peak):
            environment['peak_cuda_allocated_mb'] = float(recorded_peak)
    write_json(output / 'run_summary.json', {'primary_model': frozen['primary_model'],
               'acquisition_variant': frozen['acquisition_variant'], 'n_trajectories': len(decisions),
               'evaluation_sites': int(decisions[decisions.role.eq('evaluation')].site_id.nunique()),
               'seeds': config['experiment']['seeds'], 'environment': environment})
    return decisions


def economics(config):
    from .adaptive_economics import evaluate_economic_case, load_economics_config
    output, _ = _paths(config)
    results = evaluate_economic_case(pd.read_csv(output / 'decisions.csv'),
                                     load_economics_config(config['economics']['config']),
                                     trajectories=pd.read_csv(output / 'trajectories.csv'))
    for key, filename in [('economics', 'economic_results.csv'), ('sensitivity', 'sensitivity.csv'),
                          ('parameters', 'economic_inputs.csv'), ('social_user', 'social_user_benefit.csv')]:
        results[key].to_csv(output / filename, index=False)
    return results


def run(stage='all', config_path='configs/adaptive_v1.yaml', sites=None, seeds=None):
    config = read_config(config_path)
    with threadpool_limits(limits=int(config['experiment']['cpu_threads'])):
        if stage in ('prepare', 'all'):
            prepare(config)
        if stage == 'pilot':
            return compare(config, role='development', sites=sites or ['lower_hutt_town_hall'], seeds=seeds or [17])
        if stage in ('compare-development', 'all'):
            compare(config, role='development')
        if stage in ('calibrate', 'all'):
            calibrate(config)
        if stage == 'freeze':
            return calibrate(config, run_replays=False)
        if stage in ('compare-evaluation', 'all'):
            compare(config, role='evaluation')
        if stage in ('evaluate', 'all'):
            evaluate_replays(config)
        if stage == 'evaluate-partition':
            return evaluate_replays(config, sites, seeds, publish_outputs=False)
        if stage == 'publish':
            publish(config)
        if stage == 'economics':
            economics(config)
