import numpy as np
import pandas as pd
import pytest

from cpt_budget.adaptive_v2_stopping import (
    remaining_grid_signals, current_support_groups, support_quality_signals,
    calibrate_development_margin,
)


def geometry():
    return pd.DataFrame({"location_id": ["a", "b", "c", "d", "e", "f", "g"],
                         "x_m": [0., 0., 2., 4., 6., 8., 10.],
                         "y_m": [0.] * 7})


def test_remaining_changes_and_uncertainty_are_not_diluted_by_known_grid():
    keys = pd.DataFrame({"location_id": [f"p{i}" for i in range(100)], "depth_index": 0})
    old = keys.assign(pred_log_qc=0.)
    new = keys.assign(pred_log_qc=0., std_log_qc=0.)
    new.loc[99, ["pred_log_qc", "std_log_qc"]] = [.8, .9]
    signals = remaining_grid_signals(old, new, ["p99"])
    assert signals["delta_q90"] == .8
    assert signals["delta_h90"] == 0.
    assert signals["uncertainty_q90"] == .9
    assert np.isnan(remaining_grid_signals(new, new, [])["uncertainty_q90"])


def test_current_support_groups_update_and_preserve_atomic_groups():
    groups = {"pair": ["a", "b"], **{x: [x] for x in "cdefg"}}
    first = current_support_groups(geometry(), list("abcde"), groups, max_groups=3)
    later = current_support_groups(geometry(), list("abcdefg"), groups, max_groups=3)
    assert first != later
    assert any("g" in members for members in later.values())
    assert len(current_support_groups(geometry(), list("abcdefg"), groups, max_groups=None)) == 6
    with pytest.raises(ValueError, match="whole near groups"):
        current_support_groups(geometry(), ["a", "c"], groups)


class Predictor:
    def __init__(self):
        self.calls = []

    def condition(self, observed):
        self.context = observed

    def predict(self, query):
        assert set(query.location_id).isdisjoint(self.context.location_id)
        assert "log_qc" not in query
        self.calls.append((set(self.context.location_id), set(query.location_id)))
        return query.assign(pred_log_qc=.3)


def test_support_errors_use_revealed_truth_and_distance_matched_estimate():
    geo = geometry()
    groups = {"pair": ["a", "b"], **{x: [x] for x in "cdefg"}}
    observed = pd.DataFrame([{"location_id": x, "depth_index": d, "log_qc": -.2}
                             for x in "abcde" for d in [0, 1]])

    def query(ids):
        return pd.DataFrame([{"location_id": x, "depth_index": d} for x in ids for d in [0, 1]])

    predictor = Predictor()
    signals, detail = support_quality_signals(observed, geo, groups, predictor, query,
                                              remaining_ids=["f", "g"], max_groups=None)
    assert signals["support_log_mae"] == .5
    assert signals["support_thickness_mae_m"] == .5
    assert signals["matched_support_log_mae"] == .5
    assert not signals["candidate_geometry_supported"]  # g at 10 m, known max 6 m
    assert signals["support_groups"] == 4
    assert len(detail) == 5
    assert all(not held & context for context, held in predictor.calls)
    assert all(held <= set("abcde") for _, held in predictor.calls)
    assert any(held == set("ab") for _, held in predictor.calls)


def test_development_margin_rejects_evaluation_and_accounts_for_worst_site():
    frame = pd.DataFrame({"role": ["development"] * 3, "site_id": ["one", "one", "two"],
                          "audit_log_mae": [.3, .4, .5], "audit_thickness_mae_m": [.4, .7, .5],
                          "matched_support_log_mae": [.2, .4, .3],
                          "matched_support_thickness_mae_m": [.3, .4, .6]})
    margin = calibrate_development_margin(frame)
    assert margin.log_margin == pytest.approx(.2)
    assert margin.thickness_margin_m == pytest.approx(.3)
    assert margin.development_sites == ("one", "two")
    upper = margin.upper_estimates({"matched_support_log_mae": .3,
                                    "matched_support_thickness_mae_m": .5})
    assert upper["estimated_log_mae_upper"] == pytest.approx(.5)
    assert upper["estimated_thickness_mae_upper_m"] == pytest.approx(.8)
    frame.loc[0, "role"] = "evaluation"
    with pytest.raises(ValueError, match="development"):
        calibrate_development_margin(frame)
