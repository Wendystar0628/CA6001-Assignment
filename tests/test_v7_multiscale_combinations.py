import numpy as np
import pytest
from cpt_budget.v7.multiscale import haar_forward
from cpt_budget.v7.multiscale_combinations import transfer_low_count, combine_haar_bands


@pytest.mark.parametrize("alpha", [0., 0.5, 1.])
def test_transfer_count_and_no_unnecessary_amplitude_change(alpha):
    base = np.array([[-2., -1., 0., 0., 1.], [-3., -2., -1., -0.2, 0.4]])
    donor = np.array([[-3., -2., -1., -0.2, 0.4], [-2., 0., 0., 0., 0.]])
    result, count = transfer_low_count(base, donor, alpha)
    np.testing.assert_array_equal((result < 0).sum(axis=1), count)
    changed = (result < 0) != (base < 0)
    np.testing.assert_array_equal(result[~changed], base[~changed])
    assert np.all(np.abs(result[changed]) == 0.05)
    if alpha == 0.:
        np.testing.assert_array_equal(result, base)
    if alpha == 1.:
        np.testing.assert_array_equal(count, (donor < 0).sum(axis=1))
    if alpha == 0.5:
        np.testing.assert_array_equal(count, [3, 3])  # 2.5 rounded up.
        assert result[0, 2] < 0 and result[0, 3] == 0  # depth breaks equal-value tie.


def test_haar_combination_preserves_declared_bands():
    rng = np.random.default_rng(42)
    base, donor = rng.normal(size=(8, 24)), rng.normal(size=(8, 24))
    actual = haar_forward(combine_haar_bands(base, donor))
    np.testing.assert_allclose(actual[:, :3], haar_forward(base)[:, :3], atol=1e-13)
    np.testing.assert_allclose(actual[:, 3:], haar_forward(donor)[:, 3:], atol=1e-13)
    np.testing.assert_allclose(combine_haar_bands(base, base), base, atol=1e-13)


def test_count_extremes_reproduce_zero_and_full_predicted_thickness():
    base = np.tile(np.linspace(-2, 2, 24), (2, 1))
    donor = np.vstack([np.ones(24), -np.ones(24)])
    values, counts = transfer_low_count(base, donor, 1.)
    np.testing.assert_array_equal(counts, [0, 24])
    assert np.all(values[0] >= 0) and np.all(values[1] < 0)
