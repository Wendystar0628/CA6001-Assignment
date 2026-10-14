import numpy as np
import pandas as pd
import pytest

from cpt_budget.v7.joint_combinations import MODELS, compose_bases, predict_profile
from cpt_budget.v7.profile_distribution import WholeProfileMixture


def geometry_support():
    return pd.DataFrame([
        dict(site_id='s', location_id=f'l{i}', near_group_id=f'g{i}',
             depth_index=k, x_m=float(i*10), y_m=0., z_center_m=2.125+.25*k,
             log_qc=np.sin(k/3.+i/4.) + .3*i)
        for i in range(5) for k in range(24)])


def test_new_base_composition_rejects_nonfinite_or_misaligned_vectors():
    simplex, best, stack = np.arange(24.), np.arange(24.)+1, -np.arange(24.)
    bases = compose_bases(simplex, best, stack)
    np.testing.assert_allclose(bases['simplex_best50'], simplex+.5)
    with pytest.raises(ValueError, match='row-aligned'):
        compose_bases(simplex[:-1], best, stack)
    with pytest.raises(ValueError, match='finite'):
        compose_bases(simplex*np.nan, best, stack)


def test_same_joint_distribution_supports_distinct_decisions_and_rejects_query_labels():
    observed = geometry_support()
    model = WholeProfileMixture(observed)
    query = observed.loc[observed.location_id.eq('l0')].drop(columns='log_qc').assign(
        location_id='q', near_group_id='q', x_m=12., y_m=20.)
    bases = compose_bases(np.linspace(-.2, .3, 24), np.linspace(-.1, .5, 24), np.linspace(-.6, .1, 24))
    predicted, mixtures = predict_profile(model, query, bases)
    assert len(predicted) == 8 and len(mixtures) == 6
    for kind in ('analog', 'error'):
        a = predicted[f'v7_joint_simplex_{kind}_hmedian']
        b = predicted[f'v7_joint_simplex_{kind}_median']
        np.testing.assert_array_equal(a[1], b[1])
        assert a[2] is b[2]
    for name, (point, probability, summary) in predicted.items():
        assert np.isfinite(point).all() and ((0 <= probability) & (probability <= 1)).all()
        if MODELS[name][2] == 'thickness_median':
            assert (point < 0.).sum() == summary['count_median']
        assert np.isclose(probability.sum(), summary['count_expected'])
    with pytest.raises(ValueError, match='geometry only'):
        predict_profile(model, query.assign(truth_log_qc=1.), bases)
