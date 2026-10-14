import numpy as np
import pandas as pd
import pytest

from cpt_budget.v6.functional_combinations import (
    RESIDUAL, FUNCTIONAL, SHAPE, MODEL_IDS, combine_predictions,
)


def frozen():
    frame = pd.DataFrame([
        dict(fold_id=f, site_id=f"site{f}", seed=17, budget_n=5,
             location_id=f"l{f}", depth_index=d)
        for f in (0, 1) for d in range(4)
    ])
    frame[RESIDUAL] = [-2., 2., -1., 1., 1., -1., 2., -2.]
    frame[FUNCTIONAL] = [4., -4., 3., -3., -3., 3., -4., 4.]
    frame[SHAPE] = [.1] * 8
    bases = {0: np.array([[1., 1., 1., 1.]]) / 2,
             1: np.eye(4)[:2]}
    depths = {f: np.arange(4) for f in (0, 1)}
    return frame, bases, depths


def test_transfer_retains_residual_complement_and_uses_functional_subspace():
    frame, bases, depths = frozen()
    predictions = combine_predictions(frame, bases, depths)[MODEL_IDS[5]]
    for fold_id, rows in frame.groupby("fold_id"):
        component = bases[fold_id]
        p = component.T @ component
        q = np.eye(4) - p
        result = predictions[rows.index]
        np.testing.assert_allclose(result @ p, rows[FUNCTIONAL].to_numpy() @ p, atol=1e-12)
        np.testing.assert_allclose(result @ q, rows[RESIDUAL].to_numpy() @ q, atol=1e-12)


def test_query_label_change_and_row_shuffle_cannot_change_predictions():
    frame, bases, depths = frozen()
    original = combine_predictions(frame, bases, depths)
    changed = frame.sample(frac=1., random_state=7).copy()
    changed["truth_log_qc"] = 1e9
    changed["risk_probability"] = 1.
    result = combine_predictions(changed, bases, depths)
    for model_id in MODEL_IDS:
        np.testing.assert_allclose(result[model_id], original[model_id][changed.index])


def test_sign_guard_preserves_residual_low_resistance_labels():
    frame, bases, depths = frozen()
    predictions = combine_predictions(frame, bases, depths)
    assert np.array_equal(predictions[MODEL_IDS[3]] < 0., frame[RESIDUAL].to_numpy() < 0.)
    assert np.any((predictions[MODEL_IDS[1]] < 0.) != (frame[RESIDUAL].to_numpy() < 0.))


def test_transfer_rejects_incomplete_profiles_and_nonorthonormal_basis():
    frame, bases, depths = frozen()
    with pytest.raises(ValueError, match="every depth"):
        combine_predictions(frame.iloc[1:], bases, depths)
    bases[0] *= 2.
    with pytest.raises(ValueError, match="orthonormal"):
        combine_predictions(frame, bases, depths)
