import numpy as np
import pandas as pd

from cpt_budget.v7.local_transfer import independent_components, fit_monotone_platt, apply_platt, local_adjustment


def support():
    # a/b share near-group; b/c are colocated: the full transitive component
    # must leave together, even though a/c differ in both raw metadata fields.
    geometry = [("a", "ab", 0.), ("b", "ab", 10.), ("c", "c", 10.), ("d", "d", 20.)]
    return pd.DataFrame([dict(site_id="s", location_id=loc, near_group_id=group,
        x_m=x, y_m=0., depth_index=d, log_qc=-1. if d < 4 else 1.)
        for loc, group, x in geometry for d in range(24)])


def test_near_and_colocation_closure_is_order_invariant():
    frame = support()
    expected = [["a", "b", "c"], ["d"]]
    assert independent_components(frame) == expected
    assert independent_components(frame.sample(frac=1., random_state=12)) == expected


def test_one_class_calibration_is_finite_monotone_and_improves_fit():
    p = np.tile(np.linspace(.05, .95, 24), (5, 1))
    fitted = fit_monotone_platt(p, np.zeros_like(p), np.arange(5))
    result = apply_platt(p[0], fitted)
    assert fitted["success"] and fitted["a"] >= 0.
    assert fitted["objective_fitted"] <= fitted["objective_identity"]
    assert np.isfinite(result).all() and np.all(np.diff(result) >= -1e-12)
    assert result.mean() < p[0].mean()


def test_no_independent_context_uses_identity_without_predicting():
    frame = support().query("location_id in ['a', 'b', 'c']")
    result = local_adjustment(frame, {})
    assert result["loo_groups"] == 0 and result["count_adjustment"] == 0.
    assert result["calibration"]["a"] == 1. and result["calibration"]["b"] == 0.


def test_local_crossfit_excludes_full_components_and_uses_revealed_counts():
    class Count:
        def predict(self, xp):
            return np.repeat(2., len(xp))

    class Event:
        classes_ = np.array([0, 1])

        def predict_proba(self, xe):
            return np.tile([.75, .25], (len(xe), 1))

    result = local_adjustment(support(), {"count": Count(), "event": Event()})
    assert result["loo_groups"] == 2
    assert result["raw_count_bias"] == 2.
    assert np.isclose(result["count_adjustment"], 2.*2./12.)
    for record in result["loo_records"]:
        assert not set(record["held_locations"]) & set(record["support_locations"])
    assert result["loo_records"][0]["support_locations"] == ["d"]
    assert result["loo_records"][1]["support_locations"] == ["a", "b", "c"]
