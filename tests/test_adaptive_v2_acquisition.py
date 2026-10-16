from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from cpt_budget.adaptive_v2_acquisition import latent_posterior, select_batch


def fixture():
    geometry = pd.DataFrame({'location_id': ['o', 'a', 'b', 'c', 'd'],
                             'x_m': [0., 3., 3.1, 3.2, 10.], 'y_m': 0.})
    query = geometry.assign(depth_index=0, z_center_m=2.125)
    observed = query.iloc[[0]].assign(log_qc=1.)
    transform = lambda frame: frame[['x_m', 'y_m', 'z_center_m']].to_numpy(float)
    gp = GaussianProcessRegressor(kernel=RBF(1.) + WhiteKernel(.2), optimizer=None, alpha=1.e-8)
    gp.fit(transform(observed), observed.log_qc)
    return geometry, query, SimpleNamespace(regressor=gp, observed=observed, _coordinates=transform)


def test_latent_covariance_removes_only_unreducible_white_noise():
    _, query, predictor = fixture()
    latent, noise = latent_posterior(predictor, query)
    _, observed_covariance = predictor.regressor.predict(predictor._coordinates(query), return_cov=True)
    assert np.allclose(observed_covariance - latent, np.eye(len(query)) * .2)
    assert noise == pytest.approx(.20000001)
    assert np.linalg.eigvalsh(latent).min() > -1.e-9
    wrapped = SimpleNamespace(predictor=predictor)
    assert np.allclose(latent_posterior(wrapped, query)[0], latent)


def test_ivr_targets_shared_covariance_and_updates_entire_batch_without_labels():
    geometry, query, predictor = fixture()
    selected, scores = select_batch(geometry, ['o'], ['a', 'b', 'c', 'd'],
                                   gp_predictor=predictor, query_grid=query,
                                   batch_size=2, coverage_weight=0.)
    # First choose the correlated cluster, then the isolated position; stale
    # marginal std or stale IVR would spend both CPTs in the same cluster.
    assert selected[0] in ['a', 'b', 'c']
    assert selected[1] == 'd'
    assert scores.loc[scores.selected, 'selection_order'].tolist().count(1) == 1
    assert not {'log_qc', 'qc_mpa'} & set(scores)
    poisoned = query.assign(log_qc=np.nan, qc_mpa=-999.)
    assert select_batch(geometry.iloc[::-1], ['o'], ['d', 'c', 'b', 'a'], gp_predictor=predictor,
                        query_grid=poisoned, batch_size=2, coverage_weight=0.)[0] == selected


def test_groups_are_indivisible_and_audit_grid_is_rejected():
    geometry, query, predictor = fixture()
    groups = {'o': ['o'], 'g': ['a', 'b', 'c'], 'd': ['d']}
    selected, _ = select_batch(geometry, ['o'], ['a', 'b', 'c', 'd'], groups,
                              gp_predictor=predictor, query_grid=query, batch_size=1, coverage_weight=0.)
    assert selected in [['a', 'b', 'c'], ['d']]
    with pytest.raises(ValueError, match='exactly the fixed plan'):
        select_batch(geometry, ['o'], ['a', 'b', 'c'], gp_predictor=predictor, query_grid=query)
    with pytest.raises(ValueError, match='splits near group'):
        select_batch(geometry, ['o'], ['a', 'b'], groups, gp_predictor=predictor, query_grid=query)


def test_continuous_disagreement_respects_keys_and_never_needs_hidden_labels():
    geometry, query, predictor = fixture()
    primary = query[['location_id', 'depth_index']].assign(pred_log_qc=.2)
    helper = primary.assign(pred_log_qc=[.2, .2, .2, .2, 5.])
    selected, _ = select_batch(geometry, ['o'], ['a', 'b', 'c', 'd'], gp_predictor=predictor,
                              query_grid=query, primary_predictions=primary.iloc[::-1],
                              helper_predictions=helper, batch_size=1, policy='target_ivr', coverage_weight=0.)
    assert selected == ['d']
    with pytest.raises(ValueError, match='requires primary'):
        select_batch(geometry, ['o'], ['a', 'b', 'c', 'd'], gp_predictor=predictor,
                     query_grid=query, policy='target_ivr')
