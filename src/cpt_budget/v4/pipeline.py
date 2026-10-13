"""Direct-main, versioned orchestration for source-trained residual TabPFN."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import pandas as pd

from cpt_budget.config import write_json
from cpt_budget.types import load_dataset
from cpt_budget.v2.pipeline import _plain


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def _commit(config):
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=config['_root'], text=True).strip()


def open_run(config, name='v4-residual-20261014'):
    if not name or Path(name).name != name or name in ('.', '..') or '/' in name or '\\' in name:
        raise ValueError('Run name must be a simple directory name')
    run = Path(config['_root']) / config['project']['output_dir'] / name
    frozen = {key: value for key, value in config.items() if not key.startswith('_')}
    if (run / 'run.json').exists():
        meta = _read(run/'run.json')
        if meta['config'] != frozen:
            raise ValueError('Run configuration changed; use a new run name')
    else:
        run.mkdir(parents=True, exist_ok=True)
        write_json(run/'run.json', {'run_id': name, 'study_version': '0.4',
            'scope': 'exploratory_source_trained_residual', 'created_at': _now(),
            'config': frozen, 'source_commit': _commit(config), 'status': 'created',
            'stages': {}, 'failures': [], 'outer_test_executed': False})
    return run


def record(run, stage, value, *, status=None):
    meta = _read(run/'run.json')
    meta['stages'][stage] = {'completed_at': _now(), **_plain(value)}
    if status:
        meta['status'] = status
    write_json(run/'run.json', meta)


def folds(config):
    source = Path(config['_root']) / config['v4']['evaluation']['source_run']
    document = _read(source/'inputs/folds.json')
    return document['folds'] if isinstance(document, dict) else document


def prepare(config, run):
    from .episodes import build_trajectory_episodes, save_episode_plan
    dataset = load_dataset(config)
    output = []
    for fold in folds(config):
        fid = fold['fold_id']
        for role in ['train', 'validation']:
            path = run/'episodes'/f'fold_{fid}_{role}.parquet'
            if path.exists():
                episodes = pd.read_parquet(path)
                if not episodes.role.eq(role).all() or not episodes.fold_id.eq(fid).all():
                    raise ValueError('Saved episode roles or folds do not match')
            else:
                episodes = build_trajectory_episodes(dataset, fold, config, role=role)
                save_episode_plan(episodes, path)
            output.append({'fold_id': fid, 'role': role, 'episodes': len(episodes),
                           'sites': episodes.site_id.nunique(), 'path': str(path)})
    record(run, 'prepare', {'plans': output}, status='prepared')
    return output


def fit(config, run, fold_id=None, variants=None):
    from .training import fit_residual_tabpfn
    dataset = load_dataset(config)
    config = copy.deepcopy(config)
    config['_source_commit'] = _commit(config)
    config['v4']['training']['source_commit'] = config['_source_commit']
    output = []
    for fold in folds(config):
        fid = fold['fold_id']
        if fold_id is not None and fid != fold_id:
            continue
        train = pd.read_parquet(run/'episodes'/f'fold_{fid}_train.parquet')
        validation = pd.read_parquet(run/'episodes'/f'fold_{fid}_validation.parquet')
        for variant in variants or config['v4']['variants']:
            if variant not in ('crps', 'task'):
                raise ValueError(variant)
            target = run/'checkpoints'/f'fold_{fid}'/f'residual_{variant}'
            summary_file = target/'summary.json'
            if summary_file.exists():
                known = _read(summary_file)
                checkpoint = Path(known['checkpoint_path'])
                if not checkpoint.is_absolute():
                    checkpoint = target/checkpoint
                if checkpoint.exists() and known.get('fold_id') == fid and known.get('variant') == variant:
                    output.append(known)
                    continue
                raise ValueError(f'Incomplete or mismatched saved training artifact: {summary_file}')
            meta = _read(run/'run.json')
            meta['status'] = 'training'
            meta['active_training'] = {'fold_id': fid, 'variant': variant, 'started_at': _now()}
            write_json(run/'run.json', meta)
            print(f'Training fold {fid}: residual {variant}', flush=True)
            try:
                summary = fit_residual_tabpfn(dataset, train, validation, config, target, variant=variant)
            except Exception as exc:
                meta = _read(run/'run.json')
                meta['status'] = 'failed_training_preserved'
                meta['failures'].append({'stage': 'training', 'fold_id': fid, 'variant': variant,
                                         'time': _now(), 'error': f'{type(exc).__name__}: {exc}'})
                write_json(run/'run.json', meta)
                raise
            summary.update(fold_id=fid, variant=variant, source_commit=config['_source_commit'])
            write_json(summary_file, _plain(summary))
            record(run, f'fit_{fid}_{variant}', {'summary_path': str(summary_file),
                   'checkpoint_path': summary['checkpoint_path'], 'actual_steps': summary.get('actual_steps'),
                   'best_step': summary.get('best_step')})
            output.append(summary)
    meta = _read(run/'run.json')
    completed = list((run/'checkpoints').glob('fold_*/residual_*/summary.json'))
    meta['status'] = 'training_complete' if len(completed) == 6 else 'partial_training'
    meta.pop('active_training', None)
    write_json(run/'run.json', meta)
    return output


def evaluate(config, run, fold_id=None, variants=None):
    from .evaluation import evaluate_trained_residuals
    if not (run/'matched_xgb/validation/predictions.parquet').exists():
        from .matched_baseline import run_matched_baseline
        run_matched_baseline(config, run)
    record(run, 'matched_xgb', {'summary_path': str(run/'matched_xgb/summary.json'),
                               'model_id': 'xgb_residual_matched'})
    meta = _read(run/'run.json')
    meta['status'] = 'evaluating'
    write_json(run/'run.json', meta)
    try:
        summary = evaluate_trained_residuals(config, run, variants=tuple(variants or config['v4']['variants']),
                                             fold_ids=None if fold_id is None else [fold_id])
    except Exception as exc:
        meta = _read(run/'run.json')
        meta['status'] = 'failed_evaluation_preserved'
        meta['failures'].append({'stage': 'evaluation', 'time': _now(), 'error': f'{type(exc).__name__}: {exc}'})
        write_json(run/'run.json', meta)
        raise
    status = ('completed_validation_passed' if summary.get('passed') else 'completed_validation_gate_failed')
    if summary.get('status') != 'complete':
        status = 'partial_evaluation'
    record(run, 'evaluation', summary, status=status)
    return summary
