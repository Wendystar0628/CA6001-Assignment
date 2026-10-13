"""Read-only exploratory diagnosis of the frozen 20261013 experiment.

Outputs remain in this research folder. Seeds and cells are never treated as
independent cross-site replications. No thresholds or original data are changed.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
metrics = pd.read_csv(ROOT / 'results/main-20261013/test_metrics.csv')
costs = pd.read_csv(ROOT / 'results/main-20261013/costs.csv')
attain = pd.read_csv(ROOT / 'results/main-20261013/attainment.csv')
targets = json.loads((ROOT / 'results/main-20261013/quality_targets.json').read_text())['folds']
site_keys = ['fold_id', 'site_id']
curve_keys = site_keys + ['seed', 'model_id', 'policy_id']
core = ['log_mae', 'h_point_mae_m', 'h_expected_mae_m', 'recall', 'precision', 'false_positive_rate', 'average_precision', 'coverage_90']
final = metrics[metrics.budget_n.eq(13)].copy()
by_site = final.groupby(site_keys + ['model_id', 'policy_id'])[core].mean().reset_index()
by_site.to_csv(OUT / 'site_final_metrics.csv', index=False)
labels = metrics[metrics.model_id.eq('idw') & metrics.budget_n.eq(5)][site_keys + ['seed', 'n_eval_locations', 'n_eval_cells', 'n_true_positive']].copy()
labels['positive_prevalence'] = labels.n_true_positive / labels.n_eval_cells
labels.to_csv(OUT / 'holdout_labels.csv', index=False)
label_site = labels.groupby('site_id').agg(n_eval_min=('n_eval_locations', 'min'), n_eval_max=('n_eval_locations', 'max'), positive_cells_min=('n_true_positive','min'), positive_cells_max=('n_true_positive','max'), prevalence_mean=('positive_prevalence','mean'), prevalence_min=('positive_prevalence','min'), prevalence_max=('positive_prevalence','max'), zero_positive_seeds=('n_true_positive',lambda x:int(x.eq(0).sum())))
label_site.to_csv(OUT / 'holdout_label_site.csv')

for label, metric, op in [('E','log_mae',np.less_equal),('H','h_point_mae_m',np.less_equal),('R','recall',np.greater_equal)]:
    final['target_'+label] = final.fold_id.map(lambda x:targets[str(x)][label+'_star'])
    final['pass_'+label] = op(final[metric], final['target_'+label])
final.loc[final.n_true_positive.eq(0), 'pass_R'] = True
final['failure_set'] = final.apply(lambda x: ''.join(k for k in ['E','H','R'] if not x['pass_'+k]) or 'PASS', axis=1)
final[curve_keys + core + ['target_E','target_H','target_R','pass_E','pass_H','pass_R','failure_set']].to_csv(OUT / 'final_quality_components.csv', index=False)
failed = final.groupby(['model_id','policy_id','failure_set']).size().unstack(fill_value=0)
failed.to_csv(OUT / 'failure_counts.csv')
focus_cost = costs[costs.model_id.eq('tabpfn_finetuned') & costs.policy_id.eq('threshold_uncertainty')]
pair_site = focus_cost.groupby(['site_id','pair_status']).size().unstack(fill_value=0)
pair_site.to_csv(OUT / 'attainment_pair_by_site.csv')

# Preserve whole site as statistical unit; display seeds only as within-site variability.
rows=[]
for model,policy in [('gp','maximin'),('idw','maximin'),('tabpfn_finetuned','maximin'),('tabpfn_finetuned','uncertainty'),('tabpfn_finetuned','threshold_uncertainty')]:
    a=attain[attain.model_id.eq(model)&attain.policy_id.eq(policy)]
    rows.append({'model':model,'policy':policy,'first_hit_count':int(a.n_first_hit.notna().sum()),'stable_count':int(a.n_stable.notna().sum()),'first_hit_lost_by_end':int((a.n_first_hit.notna()&a.n_stable.isna()).sum()),'stable_at_5':int(a.n_stable.le(5).sum()),'stable_at_7':int(a.n_stable.le(7).sum()),'stable_at_9':int(a.n_stable.le(9).sum()),'stable_at_11':int(a.n_stable.le(11).sum()),'stable_at_13':int(a.n_stable.le(13).sum()),'restricted_mean_capped13':float(a.n_stable.fillna(13).mean())})
att_summary=pd.DataFrame(rows)
att_summary.to_csv(OUT / 'attainment_summary.csv',index=False)

grid=pd.read_parquet(ROOT / 'data_processed/grid.parquet')
sites=pd.read_parquet(ROOT / 'data_processed/sites.parquet')
spatial=[]
for sid,g in grid.groupby('site_id'):
    p=g.pivot(index='location_id',columns='depth_index',values='log_qc')
    xy=g.drop_duplicates('location_id').set_index('location_id').loc[p.index,['x_m','y_m']].to_numpy()
    dist=np.sqrt(((xy[:,None]-xy[None,:])**2).sum(-1)); np.fill_diagonal(dist,np.inf)
    nearest=dist.argmin(axis=1)
    delta=np.mean(np.abs(p.to_numpy()[:,None,:]-p.to_numpy()[None,:,:]),axis=2)
    allpair=delta[np.triu_indices(len(p),1)]
    spatial.append({'site_id':sid,'n_locations':len(p),'prevalence':float(g.low_1.mean()),'mean_true_thickness_m':float(g.groupby('location_id').low_1.sum().mean()*.25),'nn_profile_log_mae':float(delta[np.arange(len(p)),nearest].mean()) if len(p)>1 else None,'allpair_profile_log_mae':float(allpair.mean()) if len(allpair) else None,'nn_distance_m_median':float(np.median(np.min(dist,axis=1))) if len(p)>1 else None,'recorded_depth_reference':g.depth_reference.iloc[0]})
spatial=pd.DataFrame(spatial).merge(sites[['site_id','replay_eligible','grouping_verified']],on='site_id')
spatial.to_csv(OUT / 'spatial_coherence.csv',index=False)

val=pd.read_parquet(ROOT / 'runs/main-20261013/metrics.parquet')
val=val[val.role.eq('validation') & val.budget_n.eq(13)]
val.groupby(['fold_id','site_id'])[core].mean().to_csv(OUT / 'validation_reference_site_metrics.csv')
joint_validation=[]
for fold,g in val.groupby('fold_id'):
    target=targets[str(fold)]
    s=g.groupby('site_id')[['log_mae','h_point_mae_m','recall']].mean()
    seed_pass=(g.log_mae<=target['E_star'])&(g.h_point_mae_m<=target['H_star'])&(g.recall>=target['R_star'])
    site_pass=(s.log_mae<=target['E_star'])&(s.h_point_mae_m<=target['H_star'])&(s.recall>=target['R_star'])
    joint_validation.append({'fold_id':fold,'reference_final_seed_instances':len(g),'reference_final_seed_instances_joint_pass':int(seed_pass.sum()),'reference_final_sites':len(s),'reference_final_site_means_joint_pass':int(site_pass.sum())})
joint_validation=pd.DataFrame(joint_validation)
joint_validation.to_csv(OUT / 'validation_joint_attainability.csv',index=False)

comparison=by_site[by_site.policy_id.eq('maximin')].pivot(index='site_id',columns='model_id',values='log_mae')
delta=(comparison.tabpfn_finetuned-comparison.idw).to_numpy()
bootstrap=np.random.default_rng(20261013).choice(delta,size=(100000,len(delta)),replace=True).mean(axis=1)
ci={'comparison':'FT minus IDW, maximin, 13 CPT; lower favors FT', 'exploratory':True,'resampling_unit':'site; five seeds first averaged within each site','n_sites':len(delta),'n_sites_favoring_ft':int((delta<0).sum()),'mean_difference':float(delta.mean()),'site_bootstrap_percentile_95':np.quantile(bootstrap,[.025,.975]).tolist()}
(OUT/'exploratory_site_bootstrap.json').write_text(json.dumps(ci,indent=2),encoding='utf-8')

parts=[]
def show(title, frame):
    parts.extend([title, frame.to_string(index=True), ''])
show('Holdout labels (seeds are within-site repetitions)',label_site)
show('Eight replay site spatial coherence (all hidden values used only in exploratory diagnosis)',spatial[spatial.replay_eligible])
show('Final quality failure sets (E=log-MAE, H=point thickness, R=recall)',failed)
show('Focus method paired attainment by site',pair_site)
show('Attainment with no deletion of nonattainers (capped13 is not actual cost or savings)',att_summary)
show('Final site maximin model metrics',by_site[by_site.policy_id.eq('maximin')])
show('Macro by method/policy/budget (site mean then equal site weight)',metrics.groupby(['site_id','model_id','policy_id','budget_n'])[core].mean().groupby(['model_id','policy_id','budget_n']).mean())
show('Validation reference final means',val.groupby(['fold_id','site_id'])[core].mean())
show('Validation reference joint attainability',joint_validation)
show('Both attained focus rows',focus_cost[focus_cost.pair_status.eq('both_reached')][['site_id','seed','n_stable','baseline_n_stable','saved_cpt']])
(OUT/'diagnostics.txt').write_text('\n'.join(parts),encoding='utf-8')
print('\n'.join(parts[:15]))
print('Written exploratory diagnostics to',OUT)
