import itertools

import numpy as np
import pandas as pd
import pytest

from cpt_budget.v7.profile_distribution import (
    WholeProfileMixture, mixture_summary, thickness_decision, weighted_quantile,
)
from cpt_budget.v6.empirical_risk import support_error_frame


def support():
    return pd.DataFrame([
        dict(site_id='s', location_id=f'l{i}', near_group_id='a' if i < 2 else f'g{i}',
             depth_index=k, x_m=float(i*10), y_m=0., z_center_m=2.125+.25*k,
             log_qc=np.sin(k/3.) + .3*i)
        for i in range(5) for k in range(24)])


def query():
    return support().loc[lambda frame: frame.location_id.eq('l0')].drop(columns='log_qc').assign(
        location_id='q', near_group_id='q', x_m=15., y_m=10.)


def test_joint_count_median_differs_from_marginal_median_count():
    # Both whole curves contain exactly one low cell. Their lower coordinatewise
    # median has two: reconstructing independence would corrupt the count law.
    summary = mixture_summary(np.array([[-1., 1.], [1., -1.]]), [.5, .5])
    assert summary['count_median'] == 1
    assert summary['point_median_count'] == 2
    np.testing.assert_array_equal(summary['count_pmf'], [0., 1., 0.])
    np.testing.assert_allclose(summary['probability'], [.5, .5])
    assert summary['count_expected'] == summary['probability'].sum()
    assert weighted_quantile([0., 2., 4.], [.2, .6, .2]) == 2.


def test_fixed_count_decision_is_bayes_hamming_optimal_and_preserves_agreeing_values():
    base = np.array([-.7, .4, .1, -.2])
    p = np.array([.1, .9, .6, .6])
    decision = thickness_decision(base, p, 2)
    chosen = decision < 0.
    assert chosen.sum() == 2
    loss = np.where(chosen, 1-p, p).sum()
    for selected in itertools.combinations(range(4), 2):
        labels = np.isin(np.arange(4), selected)
        assert loss <= np.where(labels, 1-p, p).sum() + 1e-12
    # Tie at positions 2/3 chooses the already-low cell 3.
    assert chosen.tolist() == [False, True, False, True]
    np.testing.assert_array_equal(decision, [.05, -.05, .1, -.2])


def test_joint_atoms_are_finite_geometry_only_and_analog_mean_is_base():
    observed = support()
    original = observed.copy(deep=True)
    model = WholeProfileMixture(observed)
    base = np.linspace(-.4, .8, 24)
    atoms, weight = model.atoms(query(), base, 'analog')
    np.testing.assert_allclose(weight @ atoms, base, atol=1e-14)
    assert atoms.shape == (5, 24)
    assert np.isfinite(atoms).all() and np.isclose(weight.sum(), 1.)
    pd.testing.assert_frame_equal(observed, original)
    with pytest.raises(ValueError, match='geometry only'):
        model.atoms(query().assign(truth_log_qc=2.), base, 'error')
    with pytest.raises(ValueError, match='near group'):
        model.atoms(query().assign(near_group_id='a'), base, 'error')


def test_whole_group_loo_exclusion_and_profile_atom_identity():
    observed = support()
    model = WholeProfileMixture(observed)
    modified = observed.copy()
    modified.loc[modified.location_id.eq('l1'), 'log_qc'] += 1000.
    revised = WholeProfileMixture(modified)
    # l1 shares l0's near group and cannot contribute to its LOO reference.
    np.testing.assert_allclose(model.error_curves[0], revised.error_curves[0])
    errors = support_error_frame(observed)
    expected = errors.loc[errors.location_id.eq('l0')].sort_values('depth_index').error_log_qc
    atoms, _ = model.atoms(query(), np.zeros(24), 'error')
    np.testing.assert_allclose(atoms[0], .5*expected)
