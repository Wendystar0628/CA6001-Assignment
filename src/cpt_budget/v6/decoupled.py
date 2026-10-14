"""Separate continuous and risk heads, with optional coherent event decoding.

The GP native event probability is an existing legitimate prediction, not its
OOF calibrated risk or any evaluation label. This control measures whether
separate task heads solve the multiobjective conflict without claiming a new GP.
"""
import numpy as np
import pandas as pd
from cpt_budget.config import write_json
from cpt_budget.diagnostics.common import iter_validation_states, STATE_KEYS
from .common import RUN, REFERENCE, KEYS, BEST, load_frozen, emit_candidate
from .indicator import IndicatorLayerPredictor, structured_event_profile

BASES = {'residual': BEST, 'fpca_guard': 'v6_fpca_residual_sign_guard',
         'graph_union': 'v6_graph_residual_low_union', 'adaptive_structure': 'support_stack_structural'}
SETTINGS = {
    'round': 3, 'source': 'fixed combinations after observing rounds 1 and 2; exploratory',
    'risk_head': 'frozen GP native p_low_1, never OOF risk_probability',
    'point_heads': BASES, 'event_decoding_heads': ['fpca_guard','graph_union','adaptive_structure'],
    'event_tau': .5, 'potts': .25, 'cardinality': .25, 'margin': .05,
    'count_prior': '3NN IDW2 weighted median revealed low-value cell count',
    'interpretation': 'Distinct task heads; GP probability is not the CDF of the combined point predictor.',
}

def align_column(wide,path,column,model=None):
    cols=KEYS+[column]+(['model_id'] if model is not None else [])
    frame=pd.read_parquet(path,columns=cols)
    if model is not None:
        frame=frame.loc[frame.model_id.eq(model)].drop(columns='model_id')
    frame=wide[KEYS].merge(frame,on=KEYS,how='left',validate='one_to_one')
    value=frame[column].to_numpy(float)
    if not np.isfinite(value).all():
        raise ValueError('Incomplete or invalid cached head')
    return value

def run():
    output=RUN/'decoupled'
    write_json(output/'protocol.json',SETTINGS)
    wide=load_frozen()
    gp=align_column(wide,REFERENCE/'predictions_oof.parquet','p_low_1','gp')
    points={
        'residual':wide[BEST].to_numpy(float),
        'fpca_guard':align_column(wide,RUN/'functional/v6_fpca_residual_sign_guard/predictions.parquet','pred_log_qc'),
        'graph_union':align_column(wide,RUN/'graph/v6_graph_residual_low_union/predictions.parquet','pred_log_qc'),
        'adaptive_structure':align_column(wide,RUN/'adaptive/support_stack_structural/predictions.parquet','pred_log_qc'),
    }
    for name,value in points.items():
        emit_candidate(f'v6_twohead_{name}_gp',value,wide=wide,probability=gp,output_root=output,metadata=SETTINGS)
    event_names=SETTINGS['event_decoding_heads']
    predicted={n:np.full(len(wide),np.nan) for n in event_names}
    probability={n:np.full(len(wide),np.nan) for n in event_names}
    rows=[]
    for state in iter_validation_states():
        mask=np.ones(len(wide),dtype=bool)
        for key in STATE_KEYS:
            mask &= wide[key].eq(state.metadata[key]).to_numpy()
        event=IndicatorLayerPredictor(mode='cardinality')
        event.condition(state.observed)
        for _,profile in wide.loc[mask].groupby('location_id',sort=False):
            profile=profile.sort_values('depth_index')
            index=profile.index.to_numpy()
            _,_,target_count=event._spatial(profile[['x_m','y_m']].iloc[0].to_numpy(float))
            for name in event_names:
                values,p,labels=structured_event_profile(points[name][index],gp[index],target_count,
                                     tau=.5,potts=.25,cardinality=.25,margin=.05)
                predicted[name][index],probability[name][index]=values,p
                rows.append({**state.metadata,'base':name,'location_id':profile.location_id.iloc[0],
                             'support_count_prior':target_count,'predicted_count':int(labels.sum()),
                             'changed_cells':int(np.sum((points[name][index]<0)!=labels))})
    for name in event_names:
        emit_candidate(f'v6_gp_event_{name}',predicted[name],wide=wide,probability=probability[name],
                       output_root=output,metadata=SETTINGS)
    pd.DataFrame(rows).to_csv(output/'decoding_diagnostics.csv',index=False)
    print('Completed seven decoupled/GP-event candidates; no query truth used.')

if __name__=='__main__':
    run()
