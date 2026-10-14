"""Same scientific metrics; old promotion gates are reference-only in v7."""
import argparse
import json
import numpy as np
import pandas as pd
from cpt_budget.config import write_json
from cpt_budget.v6 import evaluate as core
from .common import RUN,REFERENCE

def pareto_flags(values):
    cost=np.asarray(values,float)
    if not np.isfinite(cost).all():
        raise ValueError('Pareto costs must be finite')
    return np.array([not np.any(np.all(cost<=row+1e-12,axis=1)&np.any(cost<row-1e-12,axis=1)) for row in cost])

def describe(output_name):
    output=RUN/output_name
    selection=json.loads((output/'legacy_selection.json').read_text(encoding='utf-8'))
    ranking=pd.read_csv(output/'all_model_ranking.csv')
    ranking['point_balanced_error']=.5*(1-ranking.recall+ranking.false_positive_rate)
    ranking['risk_balanced_error']=.5*(1-ranking.risk_recall+ranking.risk_fpr)
    ranking['is_new']=ranking.model_id.isin(selection['candidates'])
    costs=['log_mae','h_point_mae_m','point_balanced_error','risk_balanced_error']
    ranking['pareto_all']=pareto_flags(ranking[costs])
    ranking.to_csv(output/'discovery_ranking.csv',index=False)
    ranking.loc[ranking.pareto_all].to_csv(output/'pareto_frontier.csv',index=False)
    ranking.loc[ranking.is_new].sort_values('h_point_mae_m').to_csv(output/'new_thickness_ranking.csv',index=False)
    summary={'scope':'adaptive_exploration_reused_validation','new_models':selection['new_model_count'],
             'reference_models':len(ranking)-selection['new_model_count'],'metrics_rows':selection['metric_rows'],
             'evaluation':'Same complete states, unique-site aggregation and leave-investigation-group-out calibration.',
             'legacy_gate_role':'reference only; neither v6 3/10 goal nor old gates veto v7 exploratory promotion',
             'promotion_status':'Pending result-based decision after all planned families and combinations.',
             'pareto_axes':costs,'new_pareto_models':ranking.loc[ranking.pareto_all&ranking.is_new,'model_id'].tolist(),
             'independent_test':False,'post_hoc_selection':True}
    write_json(output/'discovery.json',summary)
    print(ranking.loc[ranking.is_new,['model_id',*costs,'recall','false_positive_rate','pareto_all']].sort_values('log_mae').to_string(index=False))
    return summary

def run(families,output_name):
    # Reuse verified common calibration/metrics, without changing any v6 files.
    core.RUN,core.REFERENCE=RUN,REFERENCE
    core.run(families,output_name)
    output=RUN/output_name
    for old,new in [('selection.json','legacy_selection.json'),
                    ('candidate_global_gates.csv','legacy_candidate_global_gates.csv'),
                    ('candidate_fold_gates.csv','legacy_candidate_fold_gates.csv')]:
        (output/old).replace(output/new)
    return describe(output_name)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--families',nargs='+',required=True)
    parser.add_argument('--output',default='round1')
    args=parser.parse_args()
    run(args.families,args.output)
