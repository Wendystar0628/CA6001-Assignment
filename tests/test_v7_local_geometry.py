import numpy as np
import pandas as pd
import pytest

from cpt_budget.v7.local_geometry import affine_profiles, simplex_weights, LocalGeometryPredictor


def test_simplex_affine_reproduction_and_outside_projection():
    xy = np.array([[0., 0.], [2., 0.], [0., 2.]])
    values = 1 + 2 * xy[:, 0] - xy[:, 1]
    w, status = simplex_weights(xy, [0.5, 0.5])
    assert status == "inside_simplex"
    assert np.isclose(w @ values, 1.5)
    w, status = simplex_weights(xy, [1., -3.])
    assert status == "projected_hull_edge"
    np.testing.assert_allclose(w, [0.5, 0.5, 0.])
    assert np.isclose(w.sum(), 1.) and np.all(w >= 0)


def test_collinear_and_singleton_fallback():
    w, _ = simplex_weights(np.array([[0., 0.], [2., 0.], [4., 0.]]), [1., 2.])
    np.testing.assert_allclose(w, [0.5, 0.5, 0.])
    w, _ = simplex_weights(np.array([[2., 0.]]), [1., 2.])
    np.testing.assert_allclose(w, [1.])


def test_affine_losses_constant_invariance_and_lad_outlier_robustness():
    xy = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.], [0., 0.]])
    profile = np.ones((5, 3)) * np.array([0.1, -1., 2.])
    for loss in ["l1", "l2"]:
        np.testing.assert_allclose(affine_profiles(xy, profile, [0., 0.], loss=loss), profile[0], atol=1e-10)
    profile[-1] += 5
    robust = affine_profiles(xy, profile, [0., 0.], loss="l1")
    smooth = affine_profiles(xy, profile, [0., 0.], loss="l2")
    assert np.linalg.norm(robust - profile[0]) < np.linalg.norm(smooth - profile[0])


def test_predict_uses_only_support_labels_and_rejects_wrong_site():
    rows = [{"site_id": "s", "location_id": str(i), "depth_index": d,
             "x_m": x, "y_m": y, "log_qc": i - d / 2}
            for i, (x, y) in enumerate([(0., 0.), (2., 0.), (0., 2.)]) for d in range(3)]
    observed = pd.DataFrame(rows)
    query = observed.iloc[:3].copy()
    query["location_id"], query["x_m"], query["y_m"] = "q", 0.5, 0.5
    for variant in ["mls_affine", "lad_affine", "simplex_linear"]:
        predictor = LocalGeometryPredictor(variant)
        predictor.condition(observed)
        a = predictor.predict(query).frame.pred_log_qc.to_numpy()
        query["log_qc"] = 99999.
        b = predictor.predict(query).frame.pred_log_qc.to_numpy()
        np.testing.assert_array_equal(a, b)
        with pytest.raises(ValueError, match="Query sites"):
            predictor.predict(query.assign(site_id="other"))
