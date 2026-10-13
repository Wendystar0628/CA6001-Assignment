"""CPU-only read-only diagnosis of frozen feature processing and training exposure."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cpt_budget.config import load_config
from cpt_budget.types import load_dataset
from cpt_budget.features import build_features, FEATURE_COLUMNS
from tabpfn.preprocessing.steps import RemoveConstantFeaturesStep
from tabpfn.preprocessing.datamodel import FeatureSchema

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
config = load_config(ROOT/'configs/main.yaml')
dataset = load_dataset(config)
episodes = pd.read_parquet(ROOT/'runs/main-20261013/inputs/episodes.parquet')
records = []
for fold in range(3):
    checkpoint = ROOT/f'runs/main-20261013/checkpoints/fold_{fold}'
    summary = json.loads((checkpoint/'tabpfn_training_summary.json').read_text())
    history = json.loads((checkpoint/'tabpfn_training_history.json').read_text())
    e = episodes[(episodes.fold_id == fold) & (episodes.role == 'train')]
    available = {(r.episode_id, str(q)) for r in e.itertuples() for q in r.query_location_ids}
    visited = {(r['episode_id'], q) for r in history['steps'] for q in r['query_location_ids']}
    seen_at_best = {(r['episode_id'],q) for r in history['steps'] if r['epoch']<=summary['best_epoch'] for q in r['query_location_ids']}
    per_site_exposure = []
    for site, site_episodes in e.groupby('site_id'):
        possible_site = {(r.episode_id,str(q)) for r in site_episodes.itertuples() for q in r.query_location_ids}
        per_site_exposure.append({'site':site,'pairs_possible':len(possible_site),'pairs_seen_at_best':len(possible_site & seen_at_best),'pairs_seen_at_end':len(possible_site & visited)})
    records.append({
        'fold': fold, 'steps': summary['gradient_steps'], 'elapsed_s': summary['elapsed_seconds'],
        'best_epoch': summary['best_epoch'], 'n_episodes_available': len(e),
        'episode_query_pairs_available': len(available), 'episode_query_pairs_seen': len(visited),
        'pair_fraction_seen': len(visited)/len(available),
        'pair_fraction_seen_at_best':len(seen_at_best)/len(available),
        'per_site_exposure':per_site_exposure,
        'min_gradient_norm': min(s['gradient_norm_before_clip'] for s in history['steps']),
        'median_gradient_norm': float(np.median([s['gradient_norm_before_clip'] for s in history['steps']])),
        'fraction_gradient_clipped': float(np.mean([s['gradient_norm_before_clip'] > 1 for s in history['steps']])),
        'validation_baseline_rmse': summary['baseline_validation']['macro_log_rmse'],
        'validation_best_rmse': summary['best_validation_macro_log_rmse'],
        'last_epochs': [{k:v for k,v in h.items() if k in ['epoch','mean_train_nll','macro_log_rmse']} for h in history['epochs'][-4:]],
    })

row = episodes[(episodes.fold_id == 0)&(episodes.role == 'train')&(episodes.site_id == 'newlock')].iloc[0]
observed = dataset.observed(row.support_location_ids)
query = dataset.query(row.query_location_ids)
X = build_features(observed, dataset.query(row.support_location_ids), leave_location_out=True)
Xq = build_features(observed, query)
schema = FeatureSchema.from_only_categorical_indices([], len(FEATURE_COLUMNS))
preprocessor = RemoveConstantFeaturesStep()
filtered = preprocessor.fit_transform(X.to_numpy(), schema).X
filtered_query = preprocessor.transform(Xq.to_numpy()).X
scaled_observed, scaled_query = observed.copy(), query.copy()
for frame in [scaled_observed, scaled_query]:
    frame['x_m'] *= 100
    frame['y_m'] *= 100
    frame['geometry_scale_m'] *= 100
scaled_context = scaled_observed[[c for c in query.columns if c in scaled_observed]].copy()
Xs = build_features(scaled_observed, scaled_context, leave_location_out=True)
Xqs = build_features(scaled_observed, scaled_query)
newpreprocessor = RemoveConstantFeaturesStep()
filtered_scaled = newpreprocessor.fit_transform(Xs.to_numpy(), schema).X
filtered_query_scaled = newpreprocessor.transform(Xqs.to_numpy()).X
feature_result = {
    'site': row.site_id, 'support_n': len(row.support_location_ids), 'query_n': len(row.query_location_ids),
    'removed_columns': np.array(FEATURE_COLUMNS)[~np.asarray(preprocessor.sel_)].tolist(),
    'context_shape_after_constant_removal': list(filtered.shape),
    'scaled_geometry_factor': 100,
    'max_context_abs_change_after_constant_removal': float(np.nanmax(np.abs(filtered-filtered_scaled))),
    'max_query_abs_change_after_constant_removal': float(np.nanmax(np.abs(filtered_query-filtered_query_scaled))),
    'support_count_support': X.support_count.unique().tolist(), 'support_count_query': Xq.support_count.unique().tolist(),
    'note':'Exact upstream RemoveConstantFeaturesStep called on real features; no neural weights loaded or inference performed.'
}
result = {'training_exposure':records,'feature_erasure':feature_result}
(OUT/'diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
