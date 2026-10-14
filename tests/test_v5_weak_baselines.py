"""Key semantics of deliberately low-information (not degraded) references."""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from cpt_budget.v5.weak_baselines import (
    TargetConstantMedianPredictor,
    TargetDepthMedianPredictor,
)


def context():
    return pd.DataFrame({
        "site_id": ["a"] * 6,
        "location_id": ["one", "one", "two", "two", "three", "three"],
        "depth_index": [0, 1, 0, 1, 0, 1],
        "x_m": [0, 0, 5, 5, 20, 20],
        "y_m": [0] * 6,
        "log_qc": [-2., 4., 0., 6., 1., 20.],
    })


def query():
    return pd.DataFrame({
        "site_id": ["a"] * 4,
        "location_id": ["q1", "q1", "q2", "q2"],
        "depth_index": [1, 0, 1, 0],
        "x_m": [0., 0., 100., 100.], "y_m": [0.] * 4,
    })


def test_constant_matches_standard_median_dummy_and_has_no_distribution():
    observed, grid = context(), query()
    model = TargetConstantMedianPredictor()
    model.condition(observed)
    prediction = model.predict(grid).frame
    reference = DummyRegressor(strategy="median").fit(np.zeros((6, 1)), observed.log_qc)
    np.testing.assert_array_equal(prediction.pred_log_qc, reference.predict(np.zeros((4, 1))))
    np.testing.assert_allclose(prediction.pred_qc_mpa, np.exp(prediction.pred_log_qc))
    assert not any(c.startswith(("p_low_", "q05_", "q50_", "q95_")) for c in prediction)


def test_depth_median_keeps_depth_alignment_and_ignores_horizontal_geometry_and_query_truth():
    model = TargetDepthMedianPredictor()
    model.condition(context())
    original = model.predict(query()).frame.pred_log_qc.to_numpy()
    np.testing.assert_array_equal(original, [6., 0., 6., 0.])
    altered = query().assign(x_m=10000., y_m=-999., log_qc=-100., qc_mpa=1e-10)
    np.testing.assert_array_equal(original, model.predict(altered).frame.pred_log_qc)
    moved = context().assign(x_m=-100., y_m=900.)
    model.condition(moved)
    np.testing.assert_array_equal(original, model.predict(query()).frame.pred_log_qc)


def test_missing_depth_and_unconditioned_queries_fail_explicitly():
    model = TargetDepthMedianPredictor()
    with pytest.raises(RuntimeError, match="condition"):
        model.predict(query())
    model.condition(context())
    with pytest.raises(ValueError, match="No revealed resistance"):
        model.predict(query().assign(depth_index=2))
    with pytest.raises(ValueError, match="Query sites"):
        model.predict(query().assign(site_id="b"))
