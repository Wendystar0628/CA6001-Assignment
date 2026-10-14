"""Check depth identity, amplitude transfer, and low-value fusion semantics."""
import numpy as np
import pytest

from cpt_budget.v6.alignment_combinations import MODELS, combine_profiles


def test_identical_profiles_are_preserved_by_every_combination():
    x = np.array([[2., 2., -.5, -1., 0., 1., 1.], [0.]*7])
    for mode in MODELS.values():
        np.testing.assert_allclose(combine_profiles(x, x, mode), x, atol=1e-14)


def test_amplitude_transfer_uses_shape_location_and_residual_scale():
    shape = np.array([[2., 2., -1., -1., 2., 2., 2.]])
    residual = np.roll(shape*1.5 + .2, 2, axis=1)
    prediction = combine_profiles(residual, shape, "amplitude")
    np.testing.assert_allclose(prediction.mean(axis=1), residual.mean(axis=1))
    np.testing.assert_allclose(prediction.std(axis=1), residual.std(axis=1))
    np.testing.assert_array_equal(np.argsort(prediction), np.argsort(shape))


def test_softmin_is_symmetric_bounded_and_favors_lower_prediction():
    a = np.array([[2., -.5, .1, 3., -1.]])
    b = np.array([[-1., 1., .1, 0., -2.]])
    result = combine_profiles(a, b, "softmin")
    np.testing.assert_allclose(result, combine_profiles(b, a, "softmin"))
    assert np.all(result >= np.minimum(a, b)-1e-14)
    assert np.all(result <= (a+b)/2+1e-14)
    assert (result < 0).sum() >= (((a+b)/2) < 0).sum()


def test_nonfinite_or_mismatched_members_are_rejected():
    a = np.ones((2, 24))
    with pytest.raises(ValueError):
        combine_profiles(a, a[:, :23], "trend")
    with pytest.raises(ValueError):
        combine_profiles(a, a*np.nan, "blend050")
