"""Shared, frozen-state validation; predictors never receive evaluation truth."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd

from cpt_budget.config import PROJECT_ROOT, write_json
from cpt_budget.evaluate import evaluate, join_evaluation
from cpt_budget.types import PredictionBundle, load_dataset

SOURCE_RUN = PROJECT_ROOT / 'runs/v2/v2-validation-20261013-r2'
OUTPUT_ROOT = PROJECT_ROOT / 'runs/diagnostics-20261014'
STATE_KEYS = ['fold_id', 'site_id', 'seed', 'budget_n']
BUDGETS = (5, 7, 9, 11, 13)


def diagnostic_config():
    config = json.loads((SOURCE_RUN / 'run.json').read_text(encoding='utf-8'))['config']
    config['_root'] = str(PROJECT_ROOT)
    config['run']['mode'] = 'exploratory_diagnostics_v3'
    return config


def source_folds():
    value = json.loads((SOURCE_RUN / 'inputs/folds.json').read_text(encoding='utf-8'))
    return value['folds'] if isinstance(value, dict) else value


@dataclass
class ValidationState:
    metadata: dict
    observed: pd.DataFrame
    query: pd.DataFrame
    truth: pd.DataFrame

    @property
    def key(self):
        m = self.metadata
        return f"f{m['fold_id']}_{m['site_id']}_s{m['seed']}_n{m['budget_n']}"


def iter_validation_states(config=None, *, fold_ids=None, seeds=(17, 29), budgets=BUDGETS):
    """Reuse the exact v0.2 maximin states, including observed ID order."""
    config = config or diagnostic_config()
    dataset = load_dataset(config)
    states = pd.read_parquet(SOURCE_RUN / 'states.parquet')
    states = states[(states.model_id == 'idw') & (states.policy_id == 'maximin')]
    if not states.role.eq('validation').all():
        raise ValueError('Only validation states are allowed')
    if states.duplicated(STATE_KEYS).any():
        raise ValueError('Duplicate frozen states')
    fids = list(fold_ids) if fold_ids is not None else [0, 1, 2]
    states = states[states.fold_id.isin(fids) & states.seed.isin(seeds) & states.budget_n.isin(budgets)]
    expected = {(f, site, seed, n) for f in fids
                for site in states.loc[states.fold_id.eq(f), 'site_id'].unique()
                for seed in seeds for n in budgets}
    actual = set(states[STATE_KEYS].itertuples(index=False, name=None))
    if actual != expected or set(states.fold_id) != set(fids):
        raise ValueError('Incomplete frozen validation matrix')
    if fold_ids is None and tuple(seeds) == (17, 29) and tuple(budgets) == BUDGETS and len(states) != 60:
        raise ValueError('Registered matrix must contain exactly 60 states')
    for row in states.sort_values(STATE_KEYS).itertuples(index=False):
        observed_ids, evaluation_ids = list(row.observed_ids), list(row.evaluation_ids)
        if len(observed_ids) != row.budget_n or set(observed_ids) & set(evaluation_ids):
            raise ValueError('Invalid support/evaluation boundary')
        observed, query = dataset.observed(observed_ids), dataset.query(evaluation_ids)
        if set(observed.near_group_id) & set(query.near_group_id):
            raise ValueError('A near group crosses the support/evaluation boundary')
        meta = {k: getattr(row, k) for k in STATE_KEYS}
        meta.update(investigation_group=row.investigation_group, policy_id='maximin',
                    observed_set_id=row.observed_set_id, role='evaluation', split_role='validation',
                    study_version='0.3-diagnostics', run_id='diagnostics-20261014')
        yield ValidationState(meta, observed, query, dataset.truth(evaluation_ids))


def evaluated_frame(bundle, state):
    frame = join_evaluation(bundle, state.truth)
    for name, value in state.metadata.items():
        frame[name] = value
    frame['model_id'] = bundle.model_id
    return frame


def metrics_for(frame, config):
    """Mixed point/distribution tables may contain entirely absent columns."""
    frame = frame.dropna(axis=1, how='all')
    truth = frame[['site_id', 'location_id', 'depth_index', 'truth_log_qc', 'truth_qc_mpa']].rename(
        columns={'truth_log_qc': 'log_qc', 'truth_qc_mpa': 'qc_mpa'})
    pred = frame.drop(columns=['truth_log_qc', 'truth_qc_mpa'])
    result = evaluate(PredictionBundle(pred, str(frame.model_id.iloc[0])), truth, config)
    for key in [*STATE_KEYS, 'model_id', 'policy_id', 'investigation_group', 'observed_set_id']:
        result[key] = frame[key].iloc[0]
    result.update(role='validation', split_role='validation', study_version='0.3-diagnostics')
    return result


def run_predictor(factory, output_dir, *, config=None, fold_ids=None, resume=True):
    """factory(fold_id) -> predictor; return consolidated metrics/predictions."""
    config = config or diagnostic_config()
    output = Path(output_dir)
    (output / 'states').mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, text=True).strip()
    write_json(output / 'evaluation_protocol.json', {
        'source_run': str(SOURCE_RUN), 'source_commit': commit,
        'scope': 'exploratory_validation_only', 'seeds': [17, 29], 'budgets': BUDGETS,
        'truth_boundary': 'Only the evaluator receives truth; predict receives geometry-only query.'})
    predictor, last_fold = None, None
    parts, records = [], []
    for state in iter_validation_states(config, fold_ids=fold_ids):
        path = output / 'states' / f'{state.key}.parquet'
        if resume and path.exists():
            frame = pd.read_parquet(path)
            if set(frame.observed_set_id) != {state.metadata['observed_set_id']}:
                raise ValueError('Cached prediction support differs from frozen support')
        else:
            fid = state.metadata['fold_id']
            if fid != last_fold:
                predictor = None
                predictor = factory(fid)
                last_fold = fid
            started = time.perf_counter()
            predictor.condition(state.observed.copy())
            bundle = predictor.predict(state.query.copy())
            frame = evaluated_frame(bundle, state)
            frame['prediction_seconds'] = time.perf_counter() - started
            temp = path.with_suffix('.tmp.parquet')
            frame.to_parquet(temp, index=False)
            temp.replace(path)
            print(f'{bundle.model_id}: {state.key}', flush=True)
        parts.append(frame)
        records.append(metrics_for(frame, config))
    predictions, metrics = pd.concat(parts, ignore_index=True), pd.DataFrame(records)
    predictions.to_parquet(output / 'predictions.parquet', index=False)
    metrics.to_csv(output / 'metrics.csv', index=False)
    return {'predictions': predictions, 'metrics': metrics}
