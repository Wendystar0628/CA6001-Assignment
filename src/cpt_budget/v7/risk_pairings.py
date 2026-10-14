"""Explicit independent risk heads for promising point/count decisions."""
from .common import RUN,load_frozen,load_native,read_candidate,emit_candidate
from cpt_budget.config import write_json

BASES={
    'joint_simplex':('joint_combinations','v7_joint_simplex_best50_error_hmedian'),
    'joint_recognition':('joint_combinations','v7_joint_stack_analog_hmedian'),
    'haar_thickness':('multiscale_combinations','v7_hcount_simplex_median_a100_gp'),
    'transfer':('transfer_combinations','v7_transfer_simplex_residual_gprisk'),
}
HEADS={'gp':'gp','empirical_graph':'v6_empirical_graph_union_local'}

def run():
    output=RUN/'risk_pairings'
    metadata={'round':3,'point_bases':BASES,'risk_heads':HEADS,
        'selection':'Point bases chosen after round-2 point results; heads from previously evaluated legitimate native predictions.',
        'meaning':'Separate point/count and risk tasks; probability is not claimed to be the combined point predictor CDF.',
        'labels_used_for_prediction':False,'old_calibrated_risk_used':False,
        'warning':'Adaptive reused validation, not independent confirmation.'}
    write_json(output/'protocol.json',metadata)
    wide=load_frozen()
    for name,(family,model) in BASES.items():
        values=read_candidate(family,model,wide=wide)
        for head,member in HEADS.items():
            emit_candidate(f'v7_pair_{name}_{head}',values,wide=wide,output_root=output,
                           probability=load_native(member),metadata={**metadata,'point_member':model,'native_risk_member':member})
    print('Eight independent-head pairings exported without changing any point predictions.')

if __name__=='__main__':
    run()
