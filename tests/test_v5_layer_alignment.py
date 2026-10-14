"""Physical warp constraints and hidden-query information boundary."""
import numpy as np
import pandas as pd

from cpt_budget.v5.layer_alignment import LayerAlignmentPredictor, bounded_warp


def grid(locations):
    rows = []
    z = np.arange(24)
    for name, x, shift in locations:
        values = 1.5 - 2.5 * np.exp(-0.5 * ((z - 11 - shift) / 1.5) ** 2)
        for j, v in enumerate(values):
            rows.append(dict(site_id="s", location_id=name, depth_index=j, x_m=x,
                             y_m=0., z_center_m=2.125 + 0.25*j, log_qc=v, qc_mpa=np.exp(v)))
    return pd.DataFrame(rows)


def test_warp_is_bounded_strictly_monotonic_and_endpoint_preserving():
    support = grid([("a", 0, 0), ("b", 20, 2)])
    a, b = [g.log_qc.to_numpy() for _, g in support.groupby("location_id")]
    warp = bounded_warp(a, b)
    assert warp[0] == 0 and warp[-1] == 23
    assert np.max(abs(warp - np.arange(24))) <= 2
    assert np.all(np.diff(warp) >= .5) and np.all(np.diff(warp) <= 1.5)
    assert np.mean((a - np.interp(warp, np.arange(24), b))**2) < np.mean((a-b)**2)


def test_hidden_values_are_ignored_and_colocated_profile_is_exact():
    observed = grid([("a", 0, -1), ("b", 20, 1), ("c", 40, 0)])
    query = grid([("q", 10, 9)]).drop(columns=["log_qc", "qc_mpa"])
    for mode in ("log", "shape"):
        model = LayerAlignmentPredictor(mode=mode)
        model.condition(observed)
        clean = model.predict(query).frame
        poisoned = model.predict(query.assign(log_qc=-100, qc_mpa=1000)).frame
        pd.testing.assert_frame_equal(clean, poisoned)
        exact = model.predict(query.assign(x_m=0)).frame.pred_log_qc
        np.testing.assert_allclose(exact, observed[observed.location_id.eq("a")].log_qc)


def test_identical_profiles_remain_exact_after_registration_and_reconstruction():
    observed = grid([("a", 0, 0), ("b", 20, 0), ("c", 40, 0)])
    query = grid([("q", 11, 0)]).drop(columns=["log_qc", "qc_mpa"])
    model = LayerAlignmentPredictor(mode="shape")
    model.condition(observed)
    np.testing.assert_allclose(model.predict(query).frame.pred_log_qc,
                               observed[observed.location_id.eq("a")].log_qc)
