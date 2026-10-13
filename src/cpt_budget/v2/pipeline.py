"""Versioned v0.2 experiments with group-held-out calibration and explicit gates."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from cpt_budget.config import write_json
from cpt_budget.types import PredictionBundle, load_dataset
from cpt_budget import pipeline as v1
from cpt_budget.evaluate import evaluate

MODEL_IDS = ('idw', 'gp', 'idw_residual', 'tabpfn_spatial', 'tabpfn_crps', 'tabpfn_task')
TABLES = ('predictions', 'metrics', 'recommendations', 'acquisition_scores', 'states')
KEYS = ['fold_id', 'site_id', 'seed', 'model_id', 'policy_id', 'budget_n']


def _read(path, fallback=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else fallback


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _save(path, value):
    write_json(path, _plain(value))


def prepare(config):
    dataset = v1._ensure_prepared(config)
    return {'locations': dataset.grid.location_id.nunique(), 'sites': len(dataset.sites),
            'prepared_dir': config['project']['prepared_dir'], 'study': 'exploratory_v2'}


def open_run(config, run_id=None, *, resume=False):
    run = v1.open_run(config, run_id, resume=resume)
    meta = _read(run/'run.json')
    meta.update(study_version='0.2', evidence_status='exploratory_second_round',
                original_results_preserved='v0.1-results')
    _save(run/'run.json', meta)
    return run


def _folder(run, fid):
    return Path(run)/'checkpoints'/f'fold_{fid}'


def _relocated(path, directory):
    path = Path(path)
    local = Path(directory)/path.name
    return local if local.exists() else path


def fit(config, run, fold_id=None, models=None):
    from cpt_budget.models.gp import fit_gp_trend
    config = {**config, '_source_commit': v1._git(config['_root'], 'rev-parse', 'HEAD')}
    dataset = v1._ensure_prepared(config)
    v1._bind_run_inputs(run, config)
    models = list(models or config['v2']['prediction_models'])
    output = []
    for fold in v1._folds(config):
        fid = fold['fold_id']
        if fold_id is not None and fid != fold_id:
            continue
        folder = _folder(run, fid)
        folder.mkdir(parents=True, exist_ok=True)
        manifest = _read(folder/'models.json', {'fold_id': fid, 'artifacts': {},
                                               'train_sites': fold['train_sites'],
                                               'validation_sites': fold['validation_sites']})
        trend_path = folder/'gp_trend.json'
        if not trend_path.exists():
            trend = fit_gp_trend(dataset, fold['train_sites'])
            _save(trend_path, {str(k): float(v) for k, v in trend.items()})
        manifest['artifacts'].update(idw={'kind': 'fixed_idw'}, gp={'trend_path': str(trend_path)})
        train = v1._episodes(config, fid, 'train')
        validation = v1._episodes(config, fid, 'validation')
        for model_id in models:
            if model_id in ('idw', 'gp'):
                continue
            if model_id == 'tabpfn_spatial':
                manifest['artifacts'][model_id] = {'kind': 'original_v2_weights_spatial_features'}
                continue
            target = folder/model_id
            known = manifest['artifacts'].get(model_id, {})
            filename = known.get('artifact_path') or known.get('checkpoint_path')
            if filename and _relocated(filename, target).exists():
                continue
            print(f'Fit fold {fid}: {model_id}', flush=True)
            started = time.perf_counter()
            try:
                if model_id == 'idw_residual':
                    from .residual import fit_residual
                    artifact = fit_residual(dataset, train, validation, config, target)
                elif model_id in ('tabpfn_crps', 'tabpfn_task'):
                    from .tabpfn import fit_spatial_tabpfn
                    artifact = fit_spatial_tabpfn(dataset, train, validation, config, target,
                                                 variant='task' if model_id.endswith('task') else 'crps')
                else:
                    raise ValueError(model_id)
            except Exception as exc:
                meta = _read(run/'run.json')
                meta.setdefault('failures', []).append({'stage': 'fit', 'fold_id': fid, 'model_id': model_id,
                    'error': f'{type(exc).__name__}: {exc}', 'source_commit': config['_source_commit']})
                meta['status'] = 'failed_training_preserved'
                _save(run/'run.json', meta)
                raise
            manifest['artifacts'][model_id] = artifact
            _save(folder/'models.json', manifest)
            v1._stage(run, f'fit_{fid}_{model_id}', {'seconds': time.perf_counter()-started,
                                                    'artifact': _plain(artifact)})
        _save(folder/'models.json', manifest)
        output.append({'fold_id': fid, 'models': list(manifest['artifacts'])})
    return output


def predictor(config, run, fid, model_id):
    folder = _folder(run, fid)
    manifest = _read(folder/'models.json', {'artifacts': {}})['artifacts']
    if model_id == 'idw':
        from cpt_budget.models.idw import IDWPredictor
        return IDWPredictor(config)
    if model_id == 'gp':
        from cpt_budget.models.gp import GPPredictor
        trend = pd.Series({int(k): float(v) for k, v in _read(folder/'gp_trend.json').items()})
        return GPPredictor(config, trend=trend)
    if model_id == 'idw_residual':
        from .residual import ResidualPredictor
        path = _relocated(manifest[model_id]['artifact_path'], folder/model_id)
        return ResidualPredictor(config, artifact=path)
    from .tabpfn import SpatialTabPFNPredictor
    path = None if model_id == 'tabpfn_spatial' else _relocated(manifest[model_id]['checkpoint_path'], folder/model_id)
    return SpatialTabPFNPredictor(config, artifact=path, model_id=model_id)


def _task_key(stage, fid, site, seed, model, policy):
    return f'{stage}:{fid}:{site}:{seed}:{model}:{policy}'


def _write_task(run, stage, key, tables):
    """Small per-task files avoid repeatedly rewriting every existing prediction."""
    import hashlib
    target = run/'tasks'/stage/hashlib.sha256(key.encode()).hexdigest()[:24]
    target.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        if frame is None:
            continue
        frame = frame.copy()
        frame['run_id'] = run.name
        frame['task_key'] = key
        frame['study_version'] = '0.2'
        path = target/f'{name}.parquet'
        temp = path.with_suffix('.tmp.parquet')
        frame.to_parquet(temp, index=False)
        temp.replace(path)
    _save(target/'complete.json', {'task_key': key})


def _consolidate(run, stage, prefix=''):
    root = run/'tasks'/stage
    completed = sorted(p.parent for p in root.glob('*/complete.json'))
    for name in TABLES:
        parts = [pd.read_parquet(folder/f'{name}.parquet') for folder in completed if (folder/f'{name}.parquet').exists()]
        parts = [frame for frame in parts if not frame.empty]
        if parts:
            path = run/f'{prefix}{name}.parquet'
            temp = path.with_suffix('.tmp.parquet')
            pd.concat(parts, ignore_index=True).to_parquet(temp, index=False)
            temp.replace(path)


def _completed(run, stage):
    return {_read(p)['task_key'] for p in (run/'tasks'/stage).glob('*/complete.json')}


def validate_models(config, run, fold_id=None, models=None):
    from .replay import run_replay_v2
    dataset = v1._ensure_prepared(config)
    v1._bind_run_inputs(run, config)
    seeds = config['v2']['validation_seeds']
    models = list(models or config['v2']['prediction_models'])
    done = _completed(run, 'validation_models')
    new_tasks = 0
    for fold in v1._folds(config):
        fid = fold['fold_id']
        if fold_id is not None and fid != fold_id:
            continue
        specs = [s for s in v1._replay_splits(config) if s['site_id'] in fold['validation_sites'] and s['seed'] in seeds]
        for model in models:
            pred = None
            for spec in specs:
                key = _task_key('validation_models', fid, spec['site_id'], spec['seed'], model, 'maximin')
                if key in done:
                    continue
                pred = pred or predictor(config, run, fid, model)
                local = {**spec, 'role': 'validation'}
                print(key, flush=True)
                tables = run_replay_v2(dataset, local, pred, config, policy='maximin', fold_id=fid)
                _write_task(run, 'validation_models', key, tables)
                done.add(key)
                new_tasks += 1
            del pred
    _consolidate(run, 'validation_models')
    v1._stage(run, 'validate_models', {'new_tasks': new_tasks, 'total_tasks': len(done), 'seeds': seeds})
    return {'new_tasks': new_tasks, 'total_tasks': len(done)}


def _metrics_from_predictions(frame, dataset, config, *, calibration_scope='oof'):
    from .risk import evaluate_risk
    records = []
    for key, rows in frame.groupby(KEYS, sort=True, dropna=False):
        common = dict(zip(KEYS, key))
        truth = dataset.truth(rows.location_id.drop_duplicates().tolist())
        pred = rows.drop(columns=['truth_qc_mpa', 'truth_log_qc'], errors='ignore').dropna(axis=1, how='all')
        metrics = evaluate(PredictionBundle(pred, str(common['model_id'])), truth, config)
        risk = evaluate_risk(pred, truth, config)
        records.append({**metrics, **risk, **common, 'role': 'validation',
                        'calibration_scope': calibration_scope,
                        'risk_calibration_mode': 'leave_investigation_group_out', 'run_id': rows.run_id.iloc[0]})
    return pd.DataFrame(records)


def calibrate(config, run):
    from .risk import crossfit_risk
    dataset = load_dataset(config)
    predictions = pd.read_parquet(run/'predictions.parquet')
    evaluation = predictions[(predictions.role=='evaluation') & (predictions.split_role=='validation')].copy()
    site_to_group = dataset.sites.set_index('site_id').investigation_group
    evaluation['investigation_group'] = evaluation.site_id.map(site_to_group)
    frames, calibrators = [], {}
    threshold = config['v2']['risk']['decision_threshold']
    for (fid, model), rows in evaluation.groupby(['fold_id', 'model_id'], sort=True):
        calibrated, final = crossfit_risk(rows)
        # The risk decision threshold is fixed before this experiment.
        calibrated['risk_label'] = calibrated.risk_probability >= threshold
        calibrated['risk_threshold'] = threshold
        calibrated['calibration_scope'] = 'oof'
        final['risk_threshold'] = threshold
        frames.append(calibrated)
        calibrators.setdefault(str(fid), {})[str(model)] = final
    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(run/'predictions_oof.parquet', index=False)
    metrics = _metrics_from_predictions(combined, dataset, config)
    metrics.to_parquet(run/'metrics_oof.parquet', index=False)
    _save(run/'calibrators.json', calibrators)
    v1._stage(run, 'calibrate', {'oof_metric_rows': len(metrics), 'method': 'leave_investigation_group_out',
                                'decision_threshold': threshold})
    return {'models': sum(len(v) for v in calibrators.values()), 'oof_metric_rows': len(metrics)}


def _curve_stats(metrics):
    rows = []
    for key, group in metrics.groupby(['fold_id', 'site_id', 'seed', 'model_id', 'policy_id']):
        group = group.sort_values('budget_n')
        x = group.budget_n.to_numpy(float)
        if len(x) != len(set(x)) or x.tolist() != [5, 7, 9, 11, 13]:
            raise ValueError('V0.2 selection requires each complete registered 5/7/9/11/13 curve')
        record = dict(zip(['fold_id', 'site_id', 'seed', 'model_id', 'policy_id'], key))
        for metric, target in [('log_mae', 'aubc_log_mae'), ('h_point_mae_m', 'aubc_h_point_mae'),
                               ('risk_recall', 'risk_recall'), ('risk_fpr', 'risk_fpr')]:
            y = group[metric].to_numpy(float)
            record[target] = (float(np.trapezoid(y, x)/(x[-1]-x[0])) if len(x)>1 and np.isfinite(y).all()
                              else None)
        rows.append(record)
    return pd.DataFrame(rows)


def _require_validation_matrix(metrics, config, *, model_selection=None):
    """Reject incomplete seed/site/model grids before choosing any winner."""
    if metrics.empty or 'role' not in metrics or not metrics.role.eq('validation').all():
        raise ValueError('Selection requires only explicit nonempty validation metrics')
    if 'calibration_scope' not in metrics or not metrics.calibration_scope.eq('oof').all():
        raise ValueError('Selection risk metrics must use held-group-out calibration')
    if metrics.duplicated(KEYS).any():
        raise ValueError('Duplicate validation keys would distort model selection')
    fids = set(int(x) for x in metrics.fold_id.unique())
    folds = {f['fold_id']: f for f in v1._folds(config)}
    if not fids.issubset(folds):
        raise ValueError('Unknown validation fold')
    seeds = config['v2']['validation_seeds']
    expected = set()
    for fid in fids:
        sites = {s['site_id'] for s in v1._replay_splits(config)
                 if s['site_id'] in folds[fid]['validation_sites'] and s['seed'] in seeds}
        if model_selection is None:
            models = config['v2']['prediction_models']
            policies = ['maximin']
        else:
            chosen = model_selection['folds'][str(fid)]
            models = list(dict.fromkeys(['gp', chosen['selected_model'], chosen['baseline_model']]))
            policies = config['v2']['policies']
        for site in sites:
            for seed in seeds:
                for model in models:
                    for policy in policies:
                        for budget in [5, 7, 9, 11, 13]:
                            expected.add((fid,site,seed,model,policy,budget))
    actual = set(metrics[KEYS].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError(f'Incomplete validation matrix: {len(expected-actual)} missing keys, {len(actual-expected)} unexpected keys')


def _gate(candidate, baseline, settings):
    tests = {
        'profile_improves': candidate['aubc_log_mae'] < baseline['aubc_log_mae']*(1-settings.get('min_relative_improvement', 0)),
        'thickness_guard': candidate['aubc_h_point_mae'] <= baseline['aubc_h_point_mae']+settings['h_tolerance_m'],
        'recall_guard': np.isfinite(candidate['risk_recall']) and candidate['risk_recall'] >= baseline['risk_recall']-settings['recall_tolerance'],
        'fpr_guard': np.isfinite(candidate['risk_fpr']) and candidate['risk_fpr'] <= baseline['risk_fpr']+settings['fpr_tolerance'],
    }
    return {key: bool(value) for key, value in tests.items()}


def select_models(config, run):
    from .quality import freeze_joint_targets
    metrics = pd.read_parquet(run/'metrics_oof.parquet')
    _require_validation_matrix(metrics, config)
    curves = _curve_stats(metrics)
    columns = ['aubc_log_mae', 'aubc_h_point_mae', 'risk_recall', 'risk_fpr']
    means = curves.groupby(['fold_id', 'model_id', 'site_id'])[columns].mean().groupby(['fold_id', 'model_id']).mean()
    settings = config['v2']['gate']
    selection = {'scope': 'oof_validation', 'folds': {}, 'gate_settings': settings,
                 'aggregation': 'CPT metrics, budget trapezoid, seed mean, site mean within fold'}
    for fid in sorted(metrics.fold_id.unique()):
        available = means.loc[fid]
        base_id = available.loc[[m for m in ['idw', 'gp'] if m in available.index], 'aubc_log_mae'].idxmin()
        base = available.loc[base_id].to_dict()
        gates = []
        for model, values in available.iterrows():
            if model in ('idw', 'gp'):
                continue
            candidate = values.to_dict()
            checks = _gate(candidate, base, settings)
            gates.append({'model_id': model, **candidate, 'passed': all(checks.values()),
                          'checks': checks, 'reason': ', '.join(k for k, v in checks.items() if not v) or 'passed'})
        passed = [row for row in gates if row['passed']]
        selected = min(passed, key=lambda row: row['aubc_log_mae'])['model_id'] if passed else base_id
        selection['folds'][str(fid)] = {'baseline_model': base_id, 'baseline_metrics': base,
                                       'selected_model': selected, 'fallback_to_baseline': not passed,
                                       'candidate_gates': gates}
    _save(run/'model_selection.json', selection)
    curves.to_csv(run/'model_validation_curves.csv', index=False)
    means.to_csv(run/'model_validation_macro.csv')
    targets = freeze_joint_targets(metrics, config)
    _save(run/'quality_targets.json', targets)
    v1._stage(run, 'select_models', {'selection': _plain(selection)})
    return _plain(selection)


def _site_calibrator(calibrators, fid, model_id, group, role):
    final = calibrators[str(fid)][model_id]
    if role == 'validation':
        return final['oof_calibrators'][str(group)]
    return final


def validate_policies(config, run, fold_id=None, *, role='validation'):
    from .replay import run_replay_v2
    dataset = load_dataset(config)
    selection = _read(run/'model_selection.json')
    calibrators = _read(run/'calibrators.json')
    group_of = dataset.sites.set_index('site_id').investigation_group
    stage = 'validation_policies' if role=='validation' else 'exploratory_test'
    policy_selection = _read(run/'policy_selection.json', {})
    if role == 'test' and not policy_selection.get('expand_test', False):
        return {'new_tasks': 0, 'total_tasks': 0, 'reason': 'global_validation_gate_not_passed'}
    done = _completed(run, stage)
    seeds = config['v2']['validation_seeds'] if role=='validation' else config['replay']['seeds']
    new_tasks = 0
    for fold in v1._folds(config):
        fid = fold['fold_id']
        if fold_id is not None and fid != fold_id:
            continue
        specs = [s for s in v1._replay_splits(config) if s['site_id'] in fold[f'{role}_sites'] and s['seed'] in seeds]
        chosen = selection['folds'][str(fid)]['selected_model']
        chosen_base = selection['folds'][str(fid)]['baseline_model']
        models = list(dict.fromkeys(['gp', chosen, chosen_base]))
        if role == 'test':
            decision = policy_selection['folds'][str(fid)]
        for model_id in models:
            pred = None
            for spec in specs:
                group = group_of[spec['site_id']]
                cal = _site_calibrator(calibrators, fid, model_id, group, role)
                proxy_cal = _site_calibrator(calibrators, fid, 'gp', group, role)
                if role=='validation':
                    policies = config['v2']['policies']
                else:
                    policies = list(dict.fromkeys(['maximin', decision['selected_policy']]))
                for policy in policies:
                    key = _task_key(stage, fid, spec['site_id'], spec['seed'], model_id, policy)
                    if key in done:
                        continue
                    pred = pred or predictor(config, run, fid, model_id)
                    proxy = pred if model_id=='gp' else (predictor(config, run, fid, 'gp') if policy!='maximin' else None)
                    print(key, flush=True)
                    local = {**spec, 'role': role}
                    tables = run_replay_v2(dataset, local, pred, config, policy=policy, fold_id=fid,
                                          proxy=proxy, calibrator=cal, proxy_calibrator=proxy_cal)
                    for name, frame in tables.items():
                        frame['calibration_scope'] = 'oof' if role=='validation' else 'source_validation_fitted'
                    _write_task(run, stage, key, tables)
                    new_tasks += 1
                    done.add(key)
                    if proxy is not pred:
                        del proxy
            del pred
    _consolidate(run, stage, prefix='policy_' if role=='validation' else 'test_')
    v1._stage(run, stage, {'new_tasks': new_tasks, 'total_tasks': len(done)})
    return {'new_tasks': new_tasks, 'total_tasks': len(done)}


def select_policies(config, run):
    metrics = pd.read_parquet(run/'policy_metrics.parquet')
    model_selection = _read(run/'model_selection.json')
    _require_validation_matrix(metrics, config, model_selection=model_selection)
    curves = _curve_stats(metrics)
    cols = ['aubc_log_mae', 'aubc_h_point_mae', 'risk_recall', 'risk_fpr']
    means = curves.groupby(['fold_id', 'model_id', 'policy_id', 'site_id'])[cols].mean().groupby(['fold_id', 'model_id', 'policy_id']).mean()
    selection = {'scope': 'oof_validation', 'folds': {}, 'evidence_status': 'exploratory_second_round'}
    for fid in sorted(metrics.fold_id.unique()):
        model = model_selection['folds'][str(fid)]['selected_model']
        base_model = model_selection['folds'][str(fid)]['baseline_model']
        baseline = means.loc[(fid, base_model, 'maximin')].to_dict()
        same_model = means.loc[(fid, model, 'maximin')].to_dict()
        rows = []
        site_means = curves.groupby(['fold_id','model_id','policy_id','site_id'])[cols].mean()
        for policy in config['v2']['policies']:
            if policy=='maximin':
                continue
            candidate = means.loc[(fid, model, policy)].to_dict()
            checks = _gate(candidate, baseline, config['v2']['gate'])
            checks['beats_same_predictor_maximin'] = candidate['aubc_log_mae'] < same_model['aubc_log_mae']
            paired = site_means.loc[(fid, model, policy), 'aubc_log_mae'] - site_means.loc[(fid, model, 'maximin'), 'aubc_log_mae']
            checks['majority_sites_improve'] = bool((paired < 0).sum() > len(paired)/2)
            rows.append({'policy_id': policy, **candidate, 'passed': all(checks.values()), 'checks': checks,
                         'improved_sites': int((paired < 0).sum()), 'n_validation_sites': len(paired)})
        passed = [row for row in rows if row['passed']]
        chosen = min(passed, key=lambda row: row['aubc_log_mae'])['policy_id'] if passed else 'maximin'
        selection['folds'][str(fid)] = {'selected_model': model, 'baseline_model': base_model,
                                       'selected_policy': chosen, 'expand_test': bool(passed),
                                       'candidate_gates': rows, 'baseline_metrics': baseline}
    # A validation-successful fold cannot silently remove the unsuccessful
    # folds from the final test. The adaptive system includes their maximin
    # fallback, and expands only after a global, complete validation decision.
    paired_rows = []
    for fid, chosen in selection['folds'].items():
        f = int(fid)
        method = curves[(curves.fold_id==f) & (curves.model_id==chosen['selected_model']) &
                        (curves.policy_id==chosen['selected_policy'])]
        baseline = curves[(curves.fold_id==f) & (curves.model_id==chosen['baseline_model']) &
                          (curves.policy_id=='maximin')]
        paired = method[['site_id','seed',*cols]].merge(baseline[['site_id','seed',*cols]],
                    on=['site_id','seed'], suffixes=('_method','_baseline'), validate='one_to_one')
        paired['fold_id'] = f
        paired_rows.append(paired)
    paired = pd.concat(paired_rows, ignore_index=True)
    numbers = [f'{name}_{suffix}' for name in cols for suffix in ['method','baseline']]
    # Repeated validation sites across outer folds do not gain extra weight.
    site_means = paired.groupby(['site_id','fold_id'])[numbers].mean().groupby('site_id').mean()
    method_mean = {name: float(site_means[f'{name}_method'].mean()) for name in cols}
    base_mean = {name: float(site_means[f'{name}_baseline'].mean()) for name in cols}
    checks = _gate(method_mean, base_mean, config['v2']['gate'])
    difference = site_means.aubc_log_mae_method - site_means.aubc_log_mae_baseline
    checks['majority_unique_validation_sites_improve'] = bool((difference<0).sum()>len(difference)/2)
    checks['all_outer_folds_validated'] = set(map(int, selection['folds'])) == {f['fold_id'] for f in v1._folds(config)}
    selection['global_gate'] = {'passed': all(checks.values()), 'checks': checks,
                                'method_metrics': method_mean, 'baseline_metrics': base_mean,
                                'n_unique_validation_sites': len(difference),
                                'improved_sites': int((difference<0).sum()),
                                'failed_folds_retained_as_maximin_fallback': True}
    selection['expand_test'] = all(checks.values())
    _save(run/'policy_selection.json', selection)
    curves.to_csv(run/'policy_validation_curves.csv', index=False)
    means.to_csv(run/'policy_validation_macro.csv')
    v1._stage(run, 'select_policies', {'selection': _plain(selection)})
    return _plain(selection)


def summarize(config, run):
    from .quality import summarize_v2_quality
    summary = run/'summary'
    summary.mkdir(exist_ok=True)
    targets = _read(run/'quality_targets.json')
    files = [('prediction_validation', run/'metrics_oof.parquet'),
             ('policy_validation', run/'policy_metrics.parquet'), ('exploratory_test', run/'test_metrics.parquet')]
    counts = {}
    for prefix, path in files:
        if not path.exists():
            continue
        tables = summarize_v2_quality(pd.read_parquet(path), targets, config)
        counts[prefix] = {}
        for name, frame in tables.items():
            frame.to_csv(summary/f'{prefix}_{name}.csv', index=False, encoding='utf-8-sig')
            counts[prefix][name] = len(frame)
    _save(summary/'coverage.json', counts)
    v1._stage(run, 'summarize_v2', {'tables': counts})
    return counts


def export(config, run):
    from .figures import export_v2
    paths = export_v2(run, load_dataset(config), config)
    v1._stage(run, 'export_v2', {'files': [str(p.relative_to(run)) for p in paths]})
    return [str(p) for p in paths]


def run_all(config, run, fold_id=None, models=None):
    fit(config, run, fold_id, models)
    validate_models(config, run, fold_id, models)
    calibrate(config, run)
    select_models(config, run)
    validate_policies(config, run, fold_id)
    decision = select_policies(config, run)
    if decision['expand_test']:
        validate_policies(config, run, fold_id, role='test')
    summarize(config, run)
    return export(config, run)
