import numpy as np
import pandas as pd

from cpt_budget.v8.haar import HAAR_MATRIX, VARIANTS, make_predictor, project_counts
from cpt_budget.v7.multiscale_combinations import transfer_low_count


def observed():
    xy = [(0,0), (10,0), (0,10), (10,10), (5,4), (12,4)]
    return pd.DataFrame([dict(site_id="s", location_id=f"c{i}", near_group_id=f"g{i//2}", depth_index=d,
        x_m=x, y_m=y, log_qc=.5*np.sin(d/2 + i/3)+.08*i) for i,(x,y) in enumerate(xy) for d in range(24)])


def query():
    return pd.DataFrame([dict(site_id="s", location_id=f"q{i}", depth_index=d, x_m=x, y_m=y)
        for i,(x,y) in enumerate([(3,2), (30,-1), (10,10), (0,5)]) for d in range(24)])


def test_orthonormal_and_minimum_sign_change_projection():
    np.testing.assert_allclose(HAAR_MATRIX @ HAAR_MATRIX.T, np.eye(24), atol=1e-14)
    rng = np.random.default_rng(38)
    base, donor = rng.normal(size=(12,24)), rng.normal(size=(12,24))
    expected, counts = transfer_low_count(base, donor, 1.)
    np.testing.assert_array_equal(project_counts(base, counts), expected)
    np.testing.assert_array_equal((expected < 0).sum(axis=1), counts)


def test_reference_equivalence_for_hull_coincident_and_shuffled_query():
    slow, fast = make_predictor(optimized=False), make_predictor()
    for model in (slow, fast):
        model.condition(observed())
    clean = fast.predict(query()).frame
    np.testing.assert_allclose(slow.predict(query()).frame.pred_log_qc, clean.pred_log_qc, atol=1e-12)
    dirty = query().sample(frac=1, random_state=7).assign(log_qc=-999, truth_log_qc=-1000)
    shuffled = fast.predict(dirty).frame
    keys = ["location_id", "depth_index"]
    np.testing.assert_allclose(clean.set_index(keys).sort_index().pred_log_qc,
        shuffled.set_index(keys).sort_index().pred_log_qc, atol=1e-12)


def test_calibration_excludes_entire_near_group_and_ignores_query_truth():
    for variant in VARIANTS[1:]:
        model = make_predictor(variant)
        model.condition(observed())
        assert len(model.calibration_records) == 6
        for record in model.calibration_records:
            assert record["held_group"] not in record["support_groups"]
            assert record["location_id"] not in record["support_location_ids"]
            assert len(record["support_location_ids"]) == 4
        clean = model.predict(query()).frame.pred_log_qc.to_numpy()
        dirty = model.predict(query().assign(log_qc=-1e6, truth_log_qc=1e6)).frame.pred_log_qc.to_numpy()
        np.testing.assert_array_equal(clean, dirty)
        assert np.isfinite(clean).all()
        assert model.last_diagnostics.actual_point_count.between(0,24).all()
