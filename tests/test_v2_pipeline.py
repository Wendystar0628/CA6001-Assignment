import pandas as pd
import pytest
from cpt_budget.config import write_json
from cpt_budget.v2 import pipeline


def test_partial_budget_cannot_receive_auc_for_promotion():
    data=pd.DataFrame([dict(fold_id=0,site_id='a',seed=17,model_id='gp',policy_id='maximin',
         budget_n=n,log_mae=.4,h_point_mae_m=.2,risk_recall=.5,risk_fpr=.1) for n in [5,13]])
    with pytest.raises(ValueError,match='complete registered'):
        pipeline._curve_stats(data)


def test_missing_validation_seed_cannot_be_silently_averaged(monkeypatch):
    config={'v2':{'validation_seeds':[17,29],'prediction_models':['gp']}}
    monkeypatch.setattr(pipeline.v1,'_folds',lambda c:[{'fold_id':0,'validation_sites':['a']}])
    monkeypatch.setattr(pipeline.v1,'_replay_splits',lambda c:[{'site_id':'a','seed':s} for s in [17,29]])
    rows=[dict(fold_id=0,site_id='a',seed=17,model_id='gp',policy_id='maximin',budget_n=n,
               role='validation',calibration_scope='oof') for n in [5,7,9,11,13]]
    with pytest.raises(ValueError,match='missing keys'):
        pipeline._require_validation_matrix(pd.DataFrame(rows),config)


def test_successful_partial_fold_cannot_authorize_selective_test(monkeypatch,tmp_path):
    config={'v2':{'validation_seeds':[17],'prediction_models':['gp'],
        'policies':['maximin','gp_sur'],'gate':{'min_relative_improvement':0.,
        'h_tolerance_m':.05,'recall_tolerance':.02,'fpr_tolerance':.02}}}
    folds=[{'fold_id':i,'validation_sites':['a','b']} for i in range(3)]
    monkeypatch.setattr(pipeline.v1,'_folds',lambda c:folds)
    monkeypatch.setattr(pipeline.v1,'_replay_splits',lambda c:[{'site_id':s,'seed':17} for s in ['a','b']])
    write_json(tmp_path/'model_selection.json',{'folds':{'0':{'selected_model':'gp','baseline_model':'gp'}}})
    write_json(tmp_path/'run.json',{'stages':{}})
    rows=[dict(fold_id=0,site_id=s,seed=17,model_id='gp',policy_id=p,budget_n=n,
          log_mae=1. if p=='maximin' else .8,h_point_mae_m=.2,risk_recall=.6,risk_fpr=.05,
          role='validation',calibration_scope='oof')
          for s in ['a','b'] for p in ['maximin','gp_sur'] for n in [5,7,9,11,13]]
    pd.DataFrame(rows).to_parquet(tmp_path/'policy_metrics.parquet')
    result=pipeline.select_policies(config,tmp_path)
    assert result['folds']['0']['expand_test']
    assert not result['expand_test']
    assert result['global_gate']['checks']['all_outer_folds_validated'] is False
