import numpy as np
import pytest

from cpt_budget.v7.simplex_combinations import combine_points, MEMBERS


def test_fixed_ten_constructions_and_control():
    simplex = np.array([-1., 0., 1.])
    members = {name: simplex + i + 1 for i, name in enumerate(MEMBERS)}
    result = combine_points(simplex, members)
    assert len(result) == 10
    np.testing.assert_array_equal(result["v7_simplex_point_gp"], simplex)
    np.testing.assert_allclose(result["v7_simplex_residual_s050_gp"], simplex + 0.5)
    np.testing.assert_allclose(result["v7_simplex_residual_s075_gp"], simplex + 0.25)
    np.testing.assert_array_equal(result["v7_simplex_median_fpca_graph_gp"], simplex + 2)
    assert not np.shares_memory(result["v7_simplex_point_gp"], simplex)


def test_convex_bounds_and_invalid_member_rejection():
    simplex = np.array([-2., 4.])
    members = {name: np.array([3., -1.]) for name in MEMBERS}
    for value in combine_points(simplex, members).values():
        assert np.all(value >= [-2., -1.]) and np.all(value <= [3., 4.])
    with pytest.raises(ValueError, match="four frozen"):
        combine_points(simplex, {})
    members["residual"] = np.array([1., np.nan])
    with pytest.raises(ValueError, match="finite"):
        combine_points(simplex, members)
