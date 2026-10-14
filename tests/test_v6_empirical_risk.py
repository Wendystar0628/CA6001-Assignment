import numpy as np
import pandas as pd
import pytest

from cpt_budget.v6.empirical_risk import support_error_frame, empirical_low_probability


def support_grid():
    return pd.DataFrame([
        dict(site_id='s', location_id=f'l{i}', near_group_id='a' if i < 2 else f'g{i}',
             depth_index=k, x_m=i*10., y_m=0., z_center_m=2.125+.25*k,
             log_qc=float(i)+.3*k)
        for i in range(4) for k in range(4)
    ])


def test_loo_excludes_entire_near_group_and_preserves_inputs():
    observed = support_grid()
    before = observed.copy(deep=True)
    errors = support_error_frame(observed)
    changed = observed.copy()
    changed.loc[changed.location_id.eq('l1'), 'log_qc'] += 1000.
    changed_errors = support_error_frame(changed)
    np.testing.assert_allclose(errors.loc[errors.location_id.eq('l0'), 'loo_reference_log_qc'],
                               changed_errors.loc[changed_errors.location_id.eq('l0'), 'loo_reference_log_qc'])
    pd.testing.assert_frame_equal(observed, before)
    assert errors.loc[errors.location_id.eq('l0'), 'loo_reference_count'].eq(2).all()


def test_empirical_probability_is_bounded_monotone_and_query_order_invariant():
    observed = support_grid()
    errors = support_error_frame(observed)
    query = observed.drop(columns='log_qc').copy()
    for mode in ('depth', 'local'):
        low, diag = empirical_low_probability(errors, query, np.full(len(query), -5.), mode=mode)
        high, _ = empirical_low_probability(errors, query, np.full(len(query), 5.), mode=mode)
        assert np.all((0 <= high) & (high <= low) & (low <= 1))
        assert 1 <= diag['profile_ess_min'] <= 4
        probability, _ = empirical_low_probability(errors, query, np.linspace(-1., 1., len(query)), mode=mode)
        reverse, _ = empirical_low_probability(errors, query.iloc[::-1], np.linspace(-1., 1., len(query))[::-1], mode=mode)
        np.testing.assert_allclose(probability, reverse[::-1])
        with pytest.raises(ValueError, match='geometry'):
            empirical_low_probability(errors, query.assign(truth_qc_mpa=1.), np.zeros(len(query)), mode=mode)


def test_zero_residual_distribution_is_symmetric_and_no_independent_support_fails():
    observed = support_grid().assign(log_qc=2.)
    errors = support_error_frame(observed)
    p, _ = empirical_low_probability(errors, observed.drop(columns='log_qc'), np.zeros(len(observed)), mode='depth')
    np.testing.assert_allclose(p, .5)
    with pytest.raises(ValueError, match='independent'):
        support_error_frame(observed.assign(near_group_id='same'))
