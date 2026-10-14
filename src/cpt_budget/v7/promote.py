"""Result-informed exploratory shortlist; not a confirmatory significance gate."""
import argparse
import json
import pandas as pd
from cpt_budget.config import write_json
from .common import RUN

RULE={
    'timing':'Post-hoc exploratory decision after new-family and combination results; explicitly requested by user.',
    'comparison':'IDW is the common strong anchor; GP, v4 and v6 remain fully reported references.',
    'profile_and_thickness':'Both below IDW; no minimum percentage improvement.',
    'point_balanced_error_tolerance':.01,'point_fpr_tolerance':.02,
    'risk_recall_tolerance':.02,'risk_fpr_tolerance':.02,
    'spatial_consistency':'At least 3 of 4 unique sites improve continuous error.',
    'non_dominance':'Non-dominated among all current and historical fixed models on E,H,point BER,risk BER.',
    'reason':'Observed useful frontier trades small identification changes for profile/thickness gains; retain explicit small error tolerances instead of a large percentage hurdle.',
    'interpretation':'Promoted for continued development and frozen independent evaluation; not proven generalization.',
    'specialist':'Remaining Pareto candidates are retained for their explicit tradeoff, not declared overall winners.',
}

def run(output_name='final'):
    output=RUN/output_name
    table=pd.read_csv(output/'discovery_ranking.csv')
    baseline=table.set_index('model_id').loc['idw']
    sites=pd.read_csv(output/'unique_site_metrics.csv')
    base_sites=sites[sites.model_id.eq('idw')][['site_id','log_mae']].rename(columns={'log_mae':'idw_log_mae'})
    pairs=sites.merge(base_sites,on='site_id',validate='many_to_one')
    counts=pairs.assign(improved=pairs.log_mae<pairs.idw_log_mae).groupby('model_id').improved.sum()
    table['improved_e_sites_vs_idw']=table.model_id.map(counts).astype(int)
    table['e_improves']=table.log_mae<baseline.log_mae
    table['h_improves']=table.h_point_mae_m<baseline.h_point_mae_m
    table['point_ber_guard']=table.point_balanced_error<=baseline.point_balanced_error+.01
    table['point_fpr_guard']=table.false_positive_rate<=baseline.false_positive_rate+.02
    table['risk_recall_guard']=table.risk_recall>=baseline.risk_recall-.02
    table['risk_fpr_guard']=table.risk_fpr<=baseline.risk_fpr+.02
    table['site_consistency']=table.improved_e_sites_vs_idw>=3
    checks=['e_improves','h_improves','point_ber_guard','point_fpr_guard','risk_recall_guard','risk_fpr_guard','site_consistency']
    table['promote_mainline']=table.is_new & table.pareto_all & table[checks].all(axis=1)
    table['retain_specialist']=table.is_new & table.pareto_all & ~table.promote_mainline
    promoted=table[table.promote_mainline].sort_values('log_mae')
    representatives={}
    if len(promoted):
        for task,metric in [('balanced','log_mae'),('thickness','h_point_mae_m'),('identification','point_balanced_error')]:
            representatives[task]=str(promoted.sort_values([metric,'log_mae']).iloc[0].model_id)
    table.to_csv(output/'promotion_decisions.csv',index=False)
    pairs.to_csv(output/'site_comparisons_vs_idw.csv',index=False)
    result={'rule':RULE,'new_configurations':int(table.is_new.sum()),'total_fixed_models':len(table),
            'promoted_mainline':promoted.model_id.tolist(),'representatives':representatives,
            'retained_specialists':table.loc[table.retain_specialist,'model_id'].tolist(),
            'legacy_gates_changed':False,'outer_test_executed':False,'selection_bias_acknowledged':True,
            'anchor_metrics':{k:float(baseline[k]) for k in ['log_mae','h_point_mae_m','recall',
                'false_positive_rate','risk_recall','risk_fpr','point_balanced_error','risk_balanced_error']}}
    write_json(output/'promotion.json',result)
    discovery_path=output/'discovery.json'
    if discovery_path.exists():
        discovery=json.loads(discovery_path.read_text(encoding='utf-8'))
        discovery.update(promotion_status='Complete: result-informed exploratory shortlist, not independent confirmation.',
                         promoted_mainline=result['promoted_mainline'],representatives=representatives)
        write_json(discovery_path,discovery)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='final')
    run(parser.parse_args().output)
