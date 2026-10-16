"""Build a standalone project-state comparison from existing result archives."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    'constant': 'Constant', 'idw': 'IDW', 'gp': 'GP', 'xgb': 'XGBoost 残差',
    'tabpfn_joint': 'TabPFN joint', 'tabpfn_graph': 'TabPFN＋graph',
    'balanced': 'Balanced', 'thickness': 'Thickness', 'recognition': 'Recognition',
    'simplex': 'Simplex', 'count_shrink': 'Count-shrink',
    'thickness_guarded': 'Thickness guarded', 'cnp': 'CNP',
    'target_et_residual': 'ExtraTrees 深度残差',
}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, 'item'):
        return clean(value.item())
    if isinstance(value, float):
        return round(value, 8) if math.isfinite(value) else None
    return value


def build():
    old = pd.read_csv(ROOT/'results/final_test/unblind/test_ranking.csv')
    old = old.rename(columns={'h_point_mae_m': 'thickness_mae_m', 'false_positive_rate': 'fpr'})
    metrics = ['log_mae', 'thickness_mae_m', 'recall', 'fpr']
    current = pd.read_csv(ROOT/'results/adaptive_v1/model_comparison.csv')
    current = current.loc[current.role.eq('evaluation')]
    current = (current.groupby(['site_id', 'seed', 'model_id'])[metrics].mean()
               .groupby(['site_id', 'model_id']).mean().groupby('model_id').mean().reset_index())
    models = ['gp', 'xgb', 'cnp', 'tabpfn_joint', 'tabpfn_graph']
    old_models = old.loc[old.model_id.isin(models), ['model_id', *metrics]]
    new_models = current.loc[current.model_id.isin(models), ['model_id', *metrics]]
    decisions = pd.read_csv(ROOT/'results/adaptive_v1/decisions.csv')
    decisions = decisions.loc[decisions.role.eq('evaluation')]
    economics = pd.read_csv(ROOT/'results/adaptive_v1/economic_results.csv')
    economics = economics.loc[economics.role.eq('evaluation') & economics.alpha.eq(1)]
    locations = pd.read_csv(ROOT/'results/adaptive_v1/locations.csv')
    traces = pd.read_csv(ROOT/'results/adaptive_v1/trajectories.csv')
    site_names = {
        'newlock': ('New Lock', '荷兰新船闸'),
        'wellington_airport': ('Wellington Airport', '机场场地'),
        'hollywood': ('Hollywood', 'Hollywood 场地'),
        'christchurch_stadium': ('Christchurch Stadium', '体育场场地'),
    }
    sites = []
    for site_id, (name, label) in site_names.items():
        d = decisions.loc[decisions.site_id.eq(site_id)]
        a = d.loc[d.policy_id.eq('P_adapt')]
        s = d.loc[d.policy_id.eq('B_seq')]
        e = economics.loc[economics.site_id.eq(site_id) & economics.policy_id.eq('P_adapt')]
        last = (traces.loc[traces.site_id.eq(site_id) & traces.seed.eq(17) & traces.policy_id.eq('P_adapt')]
                .sort_values('round').iloc[-1])
        known = set(json.loads(last.known_ids))
        loc = locations.loc[locations.site_id.eq(site_id) & locations.seed.eq(17)]
        points = [dict(id=r.location_id, x=r.x_m, y=r.y_m, audit=r.partition == 'A',
                       measured=r.location_id in known, initial=r.partition == 'S0')
                  for r in loc.itertuples()]
        sites.append(dict(id=site_id, name=name, label=label, n_full=int(a.n_full.iloc[0]),
            n_seq=float(s.n_stop.mean()), n_adapt=float(a.n_stop.mean()),
            saved_fraction=float(a.saved_fraction.mean()), passed=int(a.audit_pass.sum()),
            net=float(e.net_benefit.mean()), direct=float(e.direct_saving.mean()),
            schedule=float(e.project_time_saving.mean()), opening=float(e.early_opening_benefit.mean()),
            overhead=float(e.ai_incremental_cost.mean()), points=points,
            map_n=int(last.n_observed), map_audit=int(loc.partition.eq('A').sum()),
            low_qc_defined=bool(a.recall.notna().any())))
    states = pd.read_csv(ROOT/'results/adaptive_v2/quality_states.csv')
    curves = states[['seed', 'model_id', 'policy', 'n_observed', *metrics]].to_dict('records')
    full = pd.read_csv(ROOT/'results/adaptive_v2_diagnostics/model_choice_newlock_summary.csv')
    full = full.loc[full.state.eq('full') & full.model_id.isin(['gp', 'tabpfn_joint', 'tabpfn_graph'])]
    comparison = pd.read_csv(ROOT/'results/adaptive_v2/acquisition_comparison.csv')
    et = comparison.loc[comparison.model_id.eq('target_et_residual') & comparison.policy.eq('maximin')].iloc[0]
    newlock = full[['model_id', *metrics]].to_dict('records')
    newlock.append(dict(model_id='target_et_residual', log_mae=et.full_log_mae,
        thickness_mae_m=et.full_thickness_mae_m, recall=et.full_recall, fpr=et.full_fpr))
    frozen = json.loads((ROOT/'results/adaptive_v1/frozen_config.json').read_text(encoding='utf-8'))
    summary = json.loads((ROOT/'results/adaptive_v2/experiment_summary.json').read_text(encoding='utf-8'))
    catalog = json.loads((ROOT/'configs/catalog.json').read_text(encoding='utf-8'))
    stop_dev = pd.read_csv(ROOT/'results/adaptive_v2_diagnostics/stopping_v2_development_selection.csv').set_index('candidate_id')
    stop_eval = pd.read_csv(ROOT/'results/adaptive_v2_diagnostics/stopping_v2_evaluation_summary.csv').set_index('method')
    stop_choice = json.loads((ROOT/'results/adaptive_v2_diagnostics/stopping_v2_frozen_selection.json').read_text(encoding='utf-8'))
    stop = dict(dev_before=stop_dev.loc['original', 'site_equal_verified_saved_fraction'],
        dev_after=stop_dev.loc[stop_choice['selected_candidate_id'], 'site_equal_verified_saved_fraction'],
        eval_before=stop_eval.loc['original_adaptive', 'site_equal_verified_saved_fraction'],
        eval_after=stop_eval.loc['development_selected_adaptive', 'site_equal_verified_saved_fraction'])
    data = clean(dict(names=NAMES, old=old_models.to_dict('records'), new=new_models.to_dict('records'),
        old_all=old[['model_id', *metrics]].to_dict('records'), sites=sites, curves=curves,
        newlock=newlock, acquisition=comparison.to_dict('records'), summary=summary,
        quality=frozen['quality'], stopping=stop, catalog=[dict(id=m['id'], group=m['group']) for m in catalog]))
    template = (ROOT/'src/cpt_budget/project_comparison_template.html').read_text(encoding='utf-8')
    html = template.replace('/*__PROJECT_COMPARISON_DATA__*/', json.dumps(data, ensure_ascii=False, separators=(',', ':')))
    target = ROOT/'CA6001_项目阶段对比.html'
    target.write_text(html, encoding='utf-8')
    print(str(target))


if __name__ == '__main__':
    build()
