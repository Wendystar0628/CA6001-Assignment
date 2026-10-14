import json
import numpy as np
import pandas as pd
import pytest
from cpt_budget.experiment import catalog,ASSETS
from cpt_budget.runtime import make_model
from cpt_budget.components.haar import project_counts
from cpt_budget.components.geometry import simplex_weights

def test_catalog_has_only_requested_groups():
    rows=catalog()
    assert len(rows)==12 and len({r['id'] for r in rows})==12
    assert {g:sum(r['group']==g for r in rows) for g in ['baseline','tabpfn','selected','discussion']}=={
        'baseline':4,'tabpfn':2,'selected':3,'discussion':3}
    with pytest.raises(ValueError):make_model('tabpfn_spatial')

def test_artifact_paths_are_project_relative():
    data=json.loads((ASSETS/'manifest.json').read_text(encoding='utf-8'))
    assert sum(r['path'].startswith('artifacts/models/fold_') for r in data['files'])==27
    assert all(r['path'].startswith('artifacts/') and '..' not in r['path'] for r in data['files'])

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
