"""Common evaluation of explicitly registered complete v6 candidate families."""
import argparse
import json
import numpy as np
import pandas as pd
from cpt_budget.config import write_json
from cpt_budget.diagnostics.common import diagnostic_config, metrics_for, STATE_KEYS
from cpt_budget.diagnostics.analyze import curve_metrics, gate_values, METRICS
from cpt_budget.v2.pipeline import _gate, _plain
from cpt_budget.v2.risk import crossfit_risk, evaluate_risk
from cpt_budget.v4.evaluation import _expected_frame, validate_prediction_matrix
from cpt_budget.v5 import compare as v5
from .common import RUN, REFERENCE, BEST

CORE = ['log_mae', 'h_point_mae_m', 'recall', 'false_positive_rate', 'risk_recall', 'risk_fpr', 'risk_brier']

def candidate_global_checks(curves, models, settings):
    folds = curves.groupby(['fold_id','site_id','model_id'])[CORE].mean().groupby(['fold_id','model_id']).mean()
    strong = {f: min(['idw','gp'], key=lambda m: folds.loc[(f,m),'log_mae']) for f in (0,1,2)}
    base = pd.concat([curves[curves.fold_id.eq(f)&curves.model_id.eq(m)] for f,m in strong.items()])
    rows, sites_all = [], []
    for model in models:
        a = curves[curves.model_id.eq(model)]
        for ref_name, ref in [('fold_strong', base), ('idw',curves[curves.model_id.eq('idw')]),
                              ('gp',curves[curves.model_id.eq('gp')]), (BEST,curves[curves.model_id.eq(BEST)])]:
            paired = a.merge(ref, on=['fold_id','site_id','seed'], suffixes=('_method','_baseline'), validate='one_to_one')
            if len(paired) != len(a):
                raise ValueError('Incomplete pairing')
            cols = [f'{m}_{s}' for m in CORE for s in ('method','baseline')]
            sites = paired.groupby(['site_id','fold_id'])[cols].mean().groupby('site_id').mean()
            method = {m:float(sites[f'{m}_method'].mean()) for m in CORE}
            baseline = {m:float(sites[f'{m}_baseline'].mean()) for m in CORE}
            checks = _gate(gate_values(method),gate_values(baseline),settings)
            improve = int((sites.log_mae_method < sites.log_mae_baseline).sum())
            checks['majority_unique_sites_improve'] = improve > len(sites)/2
            er = 1-method['log_mae']/baseline['log_mae']
            hr = 1-method['h_point_mae_m']/baseline['h_point_mae_m']
            row = {'model_id':model,'reference':ref_name, **{k+'_method':v for k,v in method.items()},
                   **{k+'_baseline':v for k,v in baseline.items()}, **checks,
                   'original_global_gate':all(checks.values()),'relative_log_mae_reduction':er,
                   'relative_thickness_reduction':hr,'improved_sites':improve,
                   'clear_win':bool(all(checks.values()) and er>=.05 and hr>=.10 and method['recall']>=baseline['recall'])}
            rows.append(row)
            sites_all.append(sites.reset_index().assign(model_id=model, reference=ref_name))
    return pd.DataFrame(rows),pd.concat(sites_all,ignore_index=True)

def run(families, output_name):
    config = diagnostic_config()
    _,expected = _expected_frame(config)
    paths = []
    for family in families:
        found = sorted((RUN/family).glob('*/predictions.parquet'))
        if not found:
            raise FileNotFoundError(f'No completed candidates: {family}')
        paths.extend(found)
    model_paths = {}
    for p in paths:
        model = p.parent.name
        if model in model_paths:
            raise ValueError(f'Duplicate model {model}')
        model_paths[model]=p
    metrics_parts = [pd.read_csv(REFERENCE/'metrics.csv')]
    all_prediction_parts = [pd.read_parquet(REFERENCE/'predictions_oof.parquet')]
    calibrators = {}
    for model,path in model_paths.items():
        raw = pd.read_parquet(path).dropna(axis=1,how='all')
        validate_prediction_matrix(raw,expected,[model])
        raw=raw.drop(columns=[c for c in raw if c.startswith('risk_') or c=='calibration_scope'])
        records=[]
        for fid,block in raw.groupby('fold_id'):
            calibrated,final=crossfit_risk(block)
            calibrated['risk_threshold']=.2
            calibrated['risk_label']=calibrated.risk_probability>=.2
            calibrated['calibration_scope']='leave_investigation_group_out'
            calibrators.setdefault(model,{})[str(fid)]=final
            all_prediction_parts.append(calibrated)
            for _,state in calibrated.groupby(STATE_KEYS):
                record=metrics_for(state,config)
                truth=state[['site_id','location_id','depth_index','truth_log_qc','truth_qc_mpa']].rename(columns={'truth_log_qc':'log_qc','truth_qc_mpa':'qc_mpa'})
                risk=evaluate_risk(state.drop(columns=['truth_log_qc','truth_qc_mpa']),truth,config)
                records.append({**record,**risk,'study_version':'0.6-combinations'})
        metrics_parts.append(pd.DataFrame(records))
        print(f'evaluated {model}',flush=True)
    output=RUN/output_name
    output.mkdir(parents=True,exist_ok=True)
    metrics=pd.concat(metrics_parts,ignore_index=True)
    metrics.to_csv(output/'metrics.csv',index=False)
    pd.concat(all_prediction_parts,ignore_index=True).to_parquet(output/'predictions_oof.parquet',index=False)
    curves=curve_metrics(metrics)
    curves.to_csv(output/'curves.csv',index=False)
    sites=curves.groupby(['fold_id','site_id','model_id'])[METRICS].mean().groupby(['site_id','model_id']).mean()
    sites.to_csv(output/'unique_site_metrics.csv')
    macro=sites.groupby('model_id')[METRICS].mean().sort_values('log_mae')
    macro.to_csv(output/'all_model_ranking.csv')
    macro.loc[list(model_paths)].sort_values('log_mae').to_csv(output/'new_model_ranking.csv')
    # Reuse the unchanged v5 selector with only this round's declared candidates.
    v5.MODELS=model_paths
    selection,gates,paired,selected_sites=v5.select_new_candidates(curves,config['v2']['gate'])
    gates.to_csv(output/'candidate_fold_gates.csv',index=False)
    paired.to_csv(output/'selected_system_pairs.csv',index=False)
    selected_sites.to_csv(output/'selected_system_sites.csv')
    global_checks,site_pairs=candidate_global_checks(curves,list(model_paths),config['v2']['gate'])
    global_checks.to_csv(output/'candidate_global_gates.csv',index=False)
    site_pairs.to_csv(output/'candidate_site_pairs.csv',index=False)
    selection.update(candidates=list(model_paths),families=families,new_model_count=len(model_paths),
                     metric_rows=len(metrics),scope='adaptive_exploration_reused_validation',
                     clear_win_definition='E reduction >=5%, H reduction >=10%, point recall >=baseline, risk guards and majority sites',
                     clear_winners=global_checks.loc[global_checks.reference.eq('fold_strong')&global_checks.clear_win,'model_id'].tolist())
    write_json(output/'selection.json',_plain(selection))
    write_json(output/'calibrators.json',calibrators)
    print(macro.loc[list(model_paths)][CORE].sort_values('log_mae').to_string())
    print(json.dumps(_plain(selection),ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--families',nargs='+',default=['graph','functional','alignment'])
    parser.add_argument('--output',default='round1')
    args=parser.parse_args()
    run(args.families,args.output)
