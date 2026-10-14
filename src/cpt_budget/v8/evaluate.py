"""Quality comparisons with matched risk heads; speed is measured separately."""
import argparse
import json
import pandas as pd
from cpt_budget.config import write_json
from cpt_budget.v6 import evaluate as core
from .common import RUN,REFERENCE,KEYS,load_frozen,load_native,emit_candidate

PAIRS={
 'v8_structural_exact':'v7_simplex_structural_s050_gp',
 'v8_structural_loo_simplex':'v7_simplex_structural_s050_gp',
 'v8_structural_reliability':'v7_simplex_structural_s050_gp',
 'v8_haar_fast':'v7_pair_haar_thickness_empirical_graph',
 'v8_haar_loo_bias':'v7_pair_haar_thickness_empirical_graph',
 'v8_haar_loo_blend':'v7_pair_haar_thickness_empirical_graph',
 'v8_haar_loo_localblend':'v7_pair_haar_thickness_empirical_graph',
 'v8_tabpfn_guarded_fast':'residual_ft_crps_guarded_a050',
}

def prepare_matched_heads():
    wide=load_frozen()
    gp=load_native('gp')
    empirical=load_native('v6_empirical_graph_union_local')
    for path in sorted((RUN/'structural').glob('*/predictions.parquet')):
        model=path.parent.name
        raw=pd.read_parquet(path,columns=KEYS+['pred_log_qc'])
        values=wide[KEYS].merge(raw,on=KEYS,validate='one_to_one').pred_log_qc.to_numpy(float)
        emit_candidate(model,values,wide=wide,output_root=RUN/'structural',probability=gp,
            metadata={'engineering_scope':'structural point base','matched_risk_head':'gp',
                      'head_is_point_cdf':False,'point_values_changed_by_head_attachment':False})
    for path in sorted((RUN/'joint').glob('*/predictions.parquet')):
        model=path.parent.name
        raw=pd.read_parquet(path,columns=KEYS+['pred_log_qc'])
        values=wide[KEYS].merge(raw,on=KEYS,validate='one_to_one').pred_log_qc.to_numpy(float)
        name=model.replace('v8_joint_reference_fast','v8_balanced_fast').replace('v8_joint_','v8_balanced_')
        emit_candidate(name,values,wide=wide,output_root=RUN/'balanced',probability=empirical,
            metadata={'point_source':model,'matched_risk_head':'v6_empirical_graph_union_local',
                      'head_is_point_cdf':False,'risk_head_runtime':'v8.runtime.EmpiricalRiskPipeline recomputes the same head live'})

def run():
    prepare_matched_heads()
    core.RUN,core.REFERENCE=RUN,REFERENCE
    families=['structural','joint','balanced','haar','recognition','tabpfn']
    core.run(families,'comparison')
    output=RUN/'comparison'
    for old,new in [('selection.json','legacy_selection.json'),('candidate_global_gates.csv','legacy_candidate_global_gates.csv'),
                    ('candidate_fold_gates.csv','legacy_candidate_fold_gates.csv')]:
        (output/old).replace(output/new)
    metrics=pd.read_csv(output/'metrics.csv',low_memory=False)
    metrics.loc[metrics.model_id.str.startswith('v8_'),'study_version']='0.8-engineering'
    metrics.to_csv(output/'metrics.csv',index=False)
    ranking=pd.read_csv(output/'all_model_ranking.csv').set_index('model_id')
    candidates=[name for name in ranking.index if name.startswith('v8_')]
    pairs=[]
    for name in candidates:
        if name.startswith('v8_balanced_'):
            ref='v7_frontier_structural_error_empirical'
        elif name.startswith('v8_joint_'):
            ref='v7_frontier_structural_error_joint'
        elif name.startswith('v8_recognition_'):
            ref='v7_pair_joint_recognition_empirical_graph'
        else:
            ref=PAIRS.get(name)
        if ref is None:
            raise ValueError(f'Missing documented before/after reference for {name}')
        row={'model_id':name,'reference_model':ref}
        for metric in core.CORE:
            row[metric]=float(ranking.loc[name,metric]);row[metric+'_before']=float(ranking.loc[ref,metric])
            row[metric+'_delta']=row[metric]-row[metric+'_before']
        pairs.append(row)
    pd.DataFrame(pairs).to_csv(output/'optimization_pairs.csv',index=False)
    write_json(output/'engineering_summary.json',{
        'candidate_count':len(candidates),'reference_count':len(ranking)-len(candidates),
        'states_per_model':60,'metric_rows':len(metrics),'unique_validation_sites':4,
        'comparison_scope':'Fixed models, equal unique sites and identical budget curves; same risk head for matched before/after.',
        'runtime_scope':'Separately measured by serial v8.benchmark; prediction caches are not runtime measurements.',
        'independent_test':False,'legacy_gate_role':'historical reference only','new_models':candidates})

if __name__=='__main__':
    run()
