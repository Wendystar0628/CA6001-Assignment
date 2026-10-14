import numpy as np
import pytest

from cpt_budget.v7.frontier_joint import expand_risk_heads


def test_external_risk_head_preserves_points_and_is_not_joint_probability():
    point = np.array([-.5, .7, .1])
    joint = np.array([.7, .1, .3])
    empirical = np.array([.6, .2, .8])
    result = expand_risk_heads({'a': point}, {'a': joint}, empirical)
    np.testing.assert_array_equal(result['a_joint'][0], point)
    np.testing.assert_array_equal(result['a_empirical'][0], point)
    np.testing.assert_array_equal(result['a_joint'][1], joint)
    np.testing.assert_array_equal(result['a_empirical'][1], empirical)
    result['a_empirical'][1][0] = 0.
    np.testing.assert_array_equal(joint, [.7, .1, .3])
    np.testing.assert_array_equal(empirical, [.6, .2, .8])


def test_invalid_or_misaligned_head_cannot_be_attached():
    with pytest.raises(ValueError, match='aligned'):
        expand_risk_heads({'a': [1., 2.]}, {'a': [.2]}, [.1, .2])
    with pytest.raises(ValueError, match='probability'):
        expand_risk_heads({'a': [1.]}, {'a': [.2]}, [1.5])
    with pytest.raises(ValueError, match='agree'):
        expand_risk_heads({'a': [1.]}, {'b': [.2]}, [.1])
