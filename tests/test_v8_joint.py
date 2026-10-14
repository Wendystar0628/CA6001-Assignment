import numpy as np
import pandas as pd
import pytest

from cpt_budget.v8.joint import FastJointAdapter, ReferenceJointAdapter, batch_thickness_decision
from cpt_budget.v7.profile_distribution import thickness_decision


def support():
    return pd.DataFrame([
        dict(site_id='s', location_id=f'l{i}', near_group_id='a' if i < 2 else f'g{i}',
             depth_index=k, x_m=float(i*10), y_m=float((i%2)*7), z_center_m=2.125+.25*k,
             log_qc=np.sin(k/3.) + .2*i)
        for i in range(6) for k in range(24)])


def query():
    template = support().loc[lambda f: f.location_id.eq('l0')].drop(columns='log_qc')
    return pd.concat([template.assign(location_id=f'q{i}', near_group_id=f'q{i}', x_m=15.+i*15, y_m=10.)
                      for i in range(3)], ignore_index=True)


@pytest.mark.parametrize('kind', ['error', 'analog'])
def test_batched_reference_equivalence_and_row_order(kind):
    observed = support()
    q = query().sample(frac=1, random_state=17).reset_index(drop=True)
    base = np.sin(q.depth_index.to_numpy()/5.)-.3
    fast = FastJointAdapter().condition(observed)
    reference = ReferenceJointAdapter().condition(observed)
    actual, expected = fast.batch_predict(q, base, kind), reference.batch_predict(q, base, kind)
    np.testing.assert_allclose(actual['point'], expected['point'], rtol=0, atol=1e-14)
    np.testing.assert_allclose(actual['probability'], expected['probability'], rtol=0, atol=1e-14)
    np.testing.assert_allclose(actual['atoms'], expected['atoms'], rtol=0, atol=1e-14)
    np.testing.assert_array_equal(actual['count_median'], expected['count_median'])


def test_group_and_coordinate_loo_exclusion_and_query_boundary():
    observed = support()
    model = FastJointAdapter().condition(observed)
    changed = observed.copy()
    changed.loc[changed.location_id.eq('l1'), 'log_qc'] += 1000.
    revised = FastJointAdapter().condition(changed)
    np.testing.assert_allclose(model.error_curves[0], revised.error_curves[0])
    with pytest.raises(ValueError, match='geometry only'):
        model.batch_predict(query().assign(truth_log_qc=2), np.zeros(72))
    with pytest.raises(ValueError, match='near group'):
        model.batch_predict(query().assign(near_group_id='a'), np.zeros(72))
    with pytest.raises(ValueError, match='colocated'):
        model.batch_predict(query().assign(x_m=0., y_m=0.), np.zeros(72))


def test_stable_probability_base_depth_ties_match_reference():
    base = np.array([[-.7, .4, .1, -.2], [1., 1., -1., -1.]])
    p = np.array([[.1, .9, .6, .6], [.5, .5, .5, .5]])
    counts = np.array([2, 1])
    actual = batch_thickness_decision(base, p, counts)
    expected = np.stack([thickness_decision(b, probability, c) for b, probability, c in zip(base, p, counts)])
    np.testing.assert_array_equal(actual, expected)


def test_support_policies_are_finite_and_preserve_declared_count():
    observed = support()
    before = observed.copy(deep=True)
    model = FastJointAdapter().condition(observed)
    base = np.sin(query().depth_index.to_numpy()/5.)-.3
    for policy in ('support_alpha', 'count_shrink', 'count_bias'):
        result = model.batch_predict(query(), base, policy=policy)
        np.testing.assert_array_equal((result['point'].reshape(-1,24)<0).sum(axis=1), result['decision_count'])
        assert np.isfinite(result['point']).all()
    assert model._adaptation['alpha'] in (.25, .5, .75)
    assert model._adaptation['count_weight'] in (0., .5, 1.)
    assert -2 <= model._adaptation['count_bias'] <= 2
    pd.testing.assert_frame_equal(observed, before)
