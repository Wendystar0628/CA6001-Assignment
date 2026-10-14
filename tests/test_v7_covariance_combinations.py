import numpy as np
import pytest

from cpt_budget.v7.covariance_combinations import MEMBERS, combine_points


def test_six_fixed_blends_match_registered_formula_and_stay_in_convex_hull():
    bases = {"simplex": np.array([-1.0, 2.0, 3.0]), "residual": np.array([1.0, 0.0, 4.0])}
    members = {name: np.array([-2.0, 4.0, 2.0]) + i for i, name in enumerate(MEMBERS)}
    result = combine_points(bases, members)
    assert len(result) == 6
    for base, a in bases.items():
        for member, b in members.items():
            prediction = result[f"v7_{base}_{member}_b075_gp"]
            np.testing.assert_array_equal(prediction, .75 * a + .25 * b)
            assert np.all(prediction >= np.minimum(a, b))
            assert np.all(prediction <= np.maximum(a, b))


def test_missing_or_misaligned_members_fail():
    bases = {"simplex": np.ones(3), "residual": np.ones(3)}
    members = {name: np.ones(3) for name in MEMBERS}
    with pytest.raises(ValueError, match="Exactly two"):
        combine_points({"simplex": np.ones(3)}, members)
    members["cokfull"] = np.ones(2)
    with pytest.raises(ValueError, match="same frozen rows"):
        combine_points(bases, members)
