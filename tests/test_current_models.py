import json
import numpy as np
import pandas as pd
import pytest
from cpt_budget.config import PROJECT_ROOT
from cpt_budget.experiment import catalog,ASSETS
from cpt_budget.runtime import make_model
from cpt_budget.components.haar import project_counts,_FastCore,_SimplexGeometry,HAAR_MATRIX
from cpt_budget.components.geometry import simplex_weights
from cpt_budget.report import RISK_SOURCES
from cpt_budget import cli

REFERENCE=ASSETS/'evaluation/reference_predictions.parquet'

def test_catalog_has_only_requested_groups():
    rows=catalog()
    assert len(rows)==13 and len({r['id'] for r in rows})==13
    assert {g:sum(r['group']==g for r in rows) for g in ['baseline','tabpfn','selected','discussion']}=={
        'baseline':4,'tabpfn':2,'selected':3,'discussion':4}
    assert all(r['risk_source'] in RISK_SOURCES and r['device'] in ('cpu','cuda') for r in rows)
    with pytest.raises(ValueError):make_model('tabpfn_spatial')
    with pytest.raises(ValueError):make_model('tabpfn_fpca')

def test_ported_entries_keep_their_archived_identity():
    rows={r['id']:r for r in catalog()}
    assert rows['thickness']['source_model_id']=='v7_hcount_simplex_nearest_a100_gp'
    assert rows['thickness_guarded']['source_model_id']=='v8_haar_fast'
    assert rows['tabpfn_joint']['source_model_id']=='v7_pair_joint_simplex_empirical_graph'
    shared={r['id'] for r in rows.values() if r['risk_source']=='shared_empirical_head'}
    assert shared=={'balanced','thickness','recognition','count_shrink','thickness_guarded','tabpfn_joint'}
    assert {r['id'] for r in rows.values() if r['point_only_supported']}==shared

@pytest.mark.parametrize('model_id',['constant','idw','gp','xgb','tabpfn_graph','simplex','cnp'])
def test_point_only_is_rejected_without_separate_risk_head(model_id):
    with pytest.raises(ValueError,match='no separate risk head'):make_model(model_id,point_only=True)

def test_artifact_paths_are_project_relative():
    data=json.loads((ASSETS/'manifest.json').read_text(encoding='utf-8'))
    assert sum(r['path'].startswith('artifacts/models/fold_') for r in data['files'])==27
    assert all(r['path'].startswith('artifacts/') and '..' not in r['path'] for r in data['files'])
    assert all('bytes' in r for r in data['files'])

def test_count_projection_preserves_count_and_untouched_values():
    base=np.array([[-2.,-.1,.2,3.]])
    out=project_counts(base,np.array([3]))
    assert (out<0).sum()==3
    np.testing.assert_array_equal(out[0,[0,1,3]],base[0,[0,1,3]])

def test_simplex_partition_reproduces_affine_surface():
    xy=np.array([[0.,0.],[1.,0.],[0.,1.]])
    weights,_=simplex_weights(xy,np.array([.2,.3]))
    assert np.all(weights>=0) and np.isclose(weights.sum(),1)
    assert np.isclose(weights@np.array([2.,3.,4.]),2.8)

def test_cached_simplex_geometry_matches_per_query_weights():
    rng=np.random.default_rng(0)
    xy=rng.uniform(0,50,(9,2));profiles=rng.normal(size=(9,24))
    query=np.vstack([rng.uniform(-20,70,(40,2)),xy[:2]])
    cached=_SimplexGeometry(xy,profiles).predict(query)
    loop=np.asarray([simplex_weights(xy,q)[0] for q in query])@profiles
    np.testing.assert_allclose(cached,loop,atol=1e-12,rtol=0)

def test_haar_nearest_detail_uses_the_nearest_support_profile():
    rng=np.random.default_rng(1)
    xy=np.array([[0.,0.],[10.,0.],[0.,10.],[10.,10.]]);profiles=rng.normal(size=(4,24))
    query=np.array([[1.,2.],[9.,8.]])
    _,haar=_FastCore(xy,profiles,detail='nearest').predict(query)
    detail=(haar@HAAR_MATRIX)[:,3:]
    np.testing.assert_allclose(detail,(profiles@HAAR_MATRIX)[[0,3],3:],atol=1e-12)
    with pytest.raises(ValueError):_FastCore(xy,profiles,detail='mean')

def test_idw_prediction_does_not_use_query_measurements():
    rows=[]
    for loc,x in [('a',0.),('b',1.),('c',2.)]:
        for d in range(24):
            rows.append(dict(site_id='s',location_id=loc,depth_index=d,x_m=x,y_m=0.,geometry_scale_m=1.,z_center_m=2.125+.25*d,log_qc=x+.1*d,qc_mpa=np.exp(x+.1*d)))
    observed=pd.DataFrame(rows)
    query=observed[observed.location_id.eq('a')].drop(columns=['log_qc','qc_mpa']).assign(location_id='q',x_m=.3)
    model=make_model('idw');model.condition(observed)
    clean=model.predict(query).frame.pred_log_qc.to_numpy()
    dirty=model.predict(query.assign(log_qc=-999.,qc_mpa=1e-99)).frame.pred_log_qc.to_numpy()
    np.testing.assert_array_equal(clean,dirty)

def _native_frames():
    old=pd.DataFrame({'pred_log_qc':[0.,1.],'p_low_1':[.2,.3],'q05_log':[np.nan,np.nan]})
    return old,old.copy()

def test_verify_fails_when_a_native_column_disappears_or_appears():
    old,new=_native_frames()
    assert cli._native_columns('m',old,new)=={'p_low_1':0.0}
    with pytest.raises(AssertionError,match='disappeared'):cli._native_columns('m',old,new.drop(columns='p_low_1'))
    with pytest.raises(AssertionError,match='appeared'):cli._native_columns('m',old,new.assign(q05_log=[0.,1.]))
    with pytest.raises(AssertionError,match='partially'):cli._native_columns('m',old.assign(p_low_1=[.1,np.nan]),new)

@pytest.mark.skipif(not REFERENCE.is_file(),reason='local reference predictions are not in Git')
def test_evaluate_reproduces_published_ranking(tmp_path):
    from cpt_budget.report import run
    models=['idw','gp','thickness','tabpfn_joint']
    _,_,_,ranking=run('reference',models,str(tmp_path),log=None)  # absolute path: written outside results/
    assert (tmp_path/'all_model_ranking.csv').is_file()
    published=pd.read_csv(PROJECT_ROOT/'results/current/all_model_ranking.csv').set_index('model_id')
    new=ranking.set_index('model_id')
    columns=['log_mae','h_point_mae_m','recall','false_positive_rate','risk_recall','risk_fpr','risk_brier']
    np.testing.assert_allclose(new.loc[models,columns].to_numpy(float),published.loc[models,columns].to_numpy(float),atol=1e-12,rtol=0)
    assert (new.loc[models,'risk_source']==published.loc[models,'risk_source']).all()
    assert np.isclose(new.loc['thickness','h_point_mae_m'],0.4724934895833333,atol=1e-12)
