"""Exploratory CPU probe; fold 0 validation only, no production mutations."""
from pathlib import Path
import json
import time
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from threadpoolctl import threadpool_limits
from cpt_budget.types import load_dataset, PredictionBundle
from cpt_budget.features import build_features
from cpt_budget.models.xgb import episode_training_data, XGBPredictor
from cpt_budget.models.idw import IDWPredictor
from cpt_budget.models.gp import GPPredictor
from cpt_budget.models.distribution import point_prediction_frame
from cpt_budget.evaluate import evaluate

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/main-20261013'
START = time.perf_counter()
config = json.loads((RUN / 'run.json').read_text(encoding='utf-8'))['config']
config['_root'] = str(ROOT)
dataset = load_dataset(config)
fold = json.loads((RUN / 'inputs/folds.json').read_text(encoding='utf-8'))['folds'][0]
episodes = pd.read_parquet(RUN / 'inputs/episodes.parquet')
episodes = episodes[(episodes.fold_id == 0) & episodes.role.isin(['train', 'validation'])]
assert set(episodes.loc[episodes.role.eq('train'), 'site_id']).isdisjoint(fold['validation_sites'])
states = pd.read_parquet(RUN / 'states.parquet')
states = states[(states.fold_id == 0) & states.role.eq('validation') & (states.seed == 17)
                & states.policy_id.eq('maximin') & states.model_id.eq('gp') & states.budget_n.isin([5, 9, 13])]
assert set(states.site_id).issubset(fold['validation_sites'])
assert states.groupby('site_id').budget_n.apply(list).apply(lambda x: sorted(x) == [5, 9, 13]).all()
states.to_parquet(OUT / 'validation_states.parquet', index=False)

print('Building existing source-episode features', flush=True)
t = time.perf_counter()
xtr, ytr, wtr, mtr = episode_training_data(dataset, episodes[episodes.role.eq('train')])
xva, yva, wva, mva = episode_training_data(dataset, episodes[episodes.role.eq('validation')])
feature_seconds = time.perf_counter() - t
baseline_training = json.loads((RUN / 'checkpoints/fold_0/xgb_training.json').read_text(encoding='utf-8'))
params = baseline_training['training_parameters']
params.update(n_jobs=4, device='cpu')
model = XGBRegressor(**params)
t = time.perf_counter()
with threadpool_limits(limits=4):
    model.fit(xtr, ytr-xtr.idw_log_qc.to_numpy(), sample_weight=wtr,
              eval_set=[(xva, yva-xva.idw_log_qc.to_numpy())], sample_weight_eval_set=[wva], verbose=False)
train_seconds = time.perf_counter() - t
model.save_model(OUT / 'residual_xgb.json')
episode_rows = []
for name, prediction in [('idw', xva.idw_log_qc.to_numpy()),
                         ('residual_xgb', xva.idw_log_qc.to_numpy()+model.predict(xva))]:
    meta = mva.copy()
    meta['absolute_error'] = np.abs(prediction-yva)
    by_site = meta.groupby(['site_id', 'episode_id', 'location_id']).absolute_error.mean().groupby(['site_id', 'episode_id']).mean().groupby('site_id').mean()
    for site, mae in by_site.items():
        episode_rows.append({'model_id': name, 'site_id': site, 'log_mae': mae})
pd.DataFrame(episode_rows).to_csv(OUT / 'source_episode_validation.csv', index=False)

saved = pd.read_parquet(RUN / 'metrics.parquet')
saved = saved[(saved.fold_id == 0) & saved.role.eq('validation') & (saved.seed == 17)
              & saved.policy_id.eq('maximin') & saved.model_id.eq('gp') & saved.budget_n.isin([5, 9, 13])].copy()
saved['seconds'] = np.nan
saved['evidence'] = 'frozen_run_metric'
rows = saved.to_dict('records')
predictions = []
warnings_rows = []

def add_result(name, bundle, truth, state, seconds):
    row = evaluate(bundle, truth, config)
    row.update(fold_id=0, site_id=state.site_id, seed=17, model_id=name,
               policy_id='maximin', budget_n=int(state.budget_n), role='validation',
               seconds=seconds, evidence='exploratory_validation_only')
    rows.append(row)
    pred = bundle.frame.copy()
    pred['model_id'] = name
    pred['budget_n'] = state.budget_n
    predictions.append(pred)
    print(f'{state.site_id} n={state.budget_n} {name}: log_mae={row["log_mae"]:.5f}, H={row["h_point_mae_m"]:.5f}, {seconds:.2f}s', flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'metrics.csv', index=False)

for state in states.itertuples(index=False):
    observed_ids = json.loads(state.observed_ids) if isinstance(state.observed_ids, str) else list(state.observed_ids)
    eval_ids = json.loads(state.evaluation_ids) if isinstance(state.evaluation_ids, str) else list(state.evaluation_ids)
    assert len(observed_ids) == state.budget_n and set(observed_ids).isdisjoint(eval_ids)
    observed = dataset.observed(observed_ids)
    query = dataset.query(eval_ids)
    truth = dataset.truth(eval_ids)
    assert set(query.site_id) == {state.site_id}
    for name, predictor in [
        ('idw', IDWPredictor(config)),
        ('xgb', XGBPredictor(config, artifact=RUN / 'checkpoints/fold_0/xgb_model.json')),
        ('gp_local_depth_mean', GPPredictor(config, trend=observed.groupby('depth_index').log_qc.mean())),
    ]:
        t = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=4):
            warnings.simplefilter('always')
            predictor.condition(observed)
            bundle = predictor.predict(query)
        for warning in caught:
            warnings_rows.append({'site_id': state.site_id, 'budget_n': int(state.budget_n), 'model_id': name, 'warning': str(warning.message)})
        add_result(name, bundle, truth, state, time.perf_counter()-t)
    t = time.perf_counter()
    features = build_features(observed, query)
    pred = features.idw_log_qc.to_numpy()+model.predict(features)
    bundle = PredictionBundle(point_prediction_frame(query, pred), 'residual_xgb')
    add_result('residual_xgb', bundle, truth, state, time.perf_counter()-t)

metrics = pd.DataFrame(rows)
columns = ['log_mae', 'log_rmse', 'qc_mae_mpa', 'h_point_mae_m', 'recall', 'false_positive_rate', 'coverage_90', 'h_expected_mae_m', 'seconds']
macro = metrics.groupby(['model_id', 'budget_n'])[columns].mean().reset_index()
macro.to_csv(OUT / 'macro_by_budget.csv', index=False)
pd.concat(predictions, ignore_index=True).to_parquet(OUT / 'predictions.parquet', index=False)
metadata = {
    'purpose': 'exploratory fixed validation-state comparison, not confirmatory generalization evidence',
    'fold_id': 0, 'role': 'validation', 'seed': 17, 'budgets': [5,9,13], 'policy': 'maximin',
    'replay_sites': sorted(states.site_id.unique().tolist()),
    'source_train_sites': sorted(mtr.site_id.unique().tolist()),
    'early_stopping_validation_sites': sorted(mva.site_id.unique().tolist()),
    'excluded_validation_site_reason': 'kaiapoi_near_river has no eligible stored replay',
    'source_feature_seconds': feature_seconds, 'residual_training_seconds': train_seconds,
    'residual_best_iteration': int(model.best_iteration),
    'residual_best_validation_mae': float(model.best_score),
    'residual_parameters': model.get_params(),
    'total_seconds': time.perf_counter()-START,
    'warnings': warnings_rows,
    'limitations': ['Only two eligible validation sites and one seed', 'No new test-set prediction or hyperparameter search',
                   'No acquisition/cost claim from a fixed-state predictor probe',
                   'Local GP trend estimation uncertainty is not propagated into intervals',
                   'Any next model choice requires separate confirmatory evaluation; reported test results already seen'],
}
(OUT / 'probe_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
print(macro.to_string(index=False), flush=True)
print(json.dumps({k:v for k,v in metadata.items() if k.endswith('seconds')}, indent=2), flush=True)
