import numpy as np
import pandas as pd
import pytest

from cpt_budget.adaptive_policy import (
    StopRule, StopSignals, StopState, compute_change, coverage_fraction,
    rank01, select_batch, support_leaveout_groups, update_stop,
)


def geometry():
    return pd.DataFrame({"location_id": ["o", "a", "b", "c"],
                         "x_m": [0., 10., 9., 2.], "y_m": 0.,
                         "near_group_id": ["o", "a", "b", "c"]})


def test_rank_ties_are_average_and_constants_do_not_invent_order():
    assert np.array_equal(rank01([9., 9., 9.]), [.5, .5, .5])
    assert np.array_equal(rank01([9.]), [.5])
    assert np.array_equal(rank01([1., 1., 3.]), [.25, .25, 1.])
    with pytest.raises(ValueError, match="finite"):
        rank01([1., np.nan])


def test_batch_recomputes_only_distance_and_is_independent_of_input_order():
    frame = geometry()
    uncertainty = {"a": 1., "b": 1., "c": 1.}
    disagreement = {"a": .5, "b": .5, "c": .5}
    chosen, scores = select_batch(frame, ["o"], ["a", "b", "c"], uncertainty=uncertainty,
                                  disagreement=disagreement)
    assert chosen == ["a", "c"]
    assert scores.set_index("location_id").loc["c", "selection_distance_m"] == 2.
    chosen_reordered, _ = select_batch(frame.iloc[::-1], ["o"], ["c", "b", "a"],
                                      uncertainty=uncertainty, disagreement=disagreement)
    assert chosen_reordered == chosen
    maximin, _ = select_batch(frame, ["o"], ["a", "b", "c"], policy="maximin")
    assert maximin == ["a", "c"]
    # Geometry filtering never reads labels, even if a caller supplies extras.
    changed = frame.assign(qc_mpa=[1.e30, -50., np.nan, -999.])
    assert select_batch(changed, ["o"], ["a", "b", "c"], uncertainty=uncertainty,
                        disagreement=disagreement)[0] == chosen


def test_group_atomicity_can_exceed_nominal_batch_and_split_groups_are_rejected():
    frame = geometry().assign(near_group_id=["o", "g", "g", "g"])
    chosen, scores = select_batch(frame, ["o"], ["a", "b", "c"],
                                  {"o": ["o"], "g": ["a", "b", "c"]},
                                  uncertainty={"a": 1., "b": 2., "c": 3.},
                                  disagreement={"a": 0., "b": 0., "c": 0.})
    assert chosen == ["a", "b", "c"]
    assert scores.selected.all() and (scores.group_size == 3).all()
    assert (scores.uncertainty == 2).all() and (scores.selection_order == 1).all()
    with pytest.raises(ValueError, match="splits near group"):
        select_batch(frame, ["o", "a"], ["b", "c"], policy="maximin")
    with pytest.raises(ValueError, match="splits near group"):
        select_batch(frame, ["o"], ["a", "b"], policy="maximin")


def test_tie_breaks_use_stable_group_id_and_zero_weight_omits_unused_inputs():
    frame = pd.DataFrame({"location_id": ["o", "a", "b"], "x_m": [0., -1., 1.], "y_m": 0.})
    chosen, _ = select_batch(frame, ["o"], ["a", "b"], {"o": ["o"], "z": ["a"], "g": ["b"]},
                             weights=(0, 0, 1), batch_size=1)
    assert chosen == ["b"]


def test_change_aligns_fixed_grid_keys_and_uses_curve_consistent_thickness():
    previous = pd.DataFrame({"location_id": ["p", "p", "q", "q"], "depth_index": [0, 1, 0, 1],
                             "pred_log_qc": [-.1, .1, -.2, -.2]})
    current = previous.assign(pred_log_qc=[.1, .2, -.1, .1])
    change = compute_change(previous, current.iloc[::-1])
    assert change["delta_q90"] == pytest.approx(np.quantile([.2, .1, .1, .3], .9))
    assert change["delta_h90"] == .25
    with pytest.raises(ValueError, match="same nonempty fixed grid"):
        compute_change(previous, current.iloc[:-1])


def test_coverage_uses_only_populated_quadrants_and_all_mandatory_points():
    frame = pd.DataFrame({"location_id": list("abcd"), "x_m": [-1., -1., 1., 1.], "y_m": [-1., 1., -1., 1.]})
    assert coverage_fraction(frame, list("abc"), list("abcd"), ["d"]) == .6
    assert coverage_fraction(frame, list("abcd"), list("abcd"), ["d"]) == 1.
    # A one-dimensional plan needs only its two populated half-planes.
    assert coverage_fraction(geometry(), ["o", "a"], list("oabc")) == 1.


def test_support_leaveout_set_is_fixed_and_cannot_split_near_groups():
    groups = {f"g{i}": [f"p{i}"] for i in range(7)}
    initial = support_leaveout_groups([f"p{i}" for i in range(6)], groups)
    assert len(initial) == 5
    later = support_leaveout_groups([f"p{i}" for i in range(7)], groups, fixed_group_ids=initial)
    assert later == initial
    with pytest.raises(ValueError, match="splits near group"):
        support_leaveout_groups(["a"], {"g": ["a", "b"]})


def test_stop_requires_minimum_coverage_and_two_consecutive_observation_rounds():
    rule = StopRule(.2, .3, .4, .5)
    good = StopSignals(.1, .2, .3, .4, True)
    state = update_stop(rule, StopState(), good, 7, 9)
    assert state.pass_streak == 0 and not state.stopped
    state = update_stop(rule, state, good, 9, 7)
    assert state.pass_streak == 1 and not state.stopped
    # Re-evaluating the same observations must not count as another round.
    with pytest.raises(ValueError, match="newly observed"):
        update_stop(rule, state, good, 9, 7)
    state = update_stop(rule, state, StopSignals(.1, .2, .3, .4, False), 11, 5)
    assert state.pass_streak == 0
    state = update_stop(rule, state, good, 13, 3)
    state = update_stop(rule, state, good, 15, 1)
    assert state.stopped and state.stop_reason == "online_stability"
    assert update_stop(rule, state, good, 16, 0) is state


def test_full_plan_has_distinct_reason_and_missing_signals_cannot_trigger_stop():
    rule = StopRule(.2, .3, .4, .5)
    invalid = StopSignals(np.nan, .2, .3, .4, True)
    state = update_stop(rule, StopState(), invalid, 9, 2)
    assert not state.stopped and not state.checks["delta_q90"]
    state = update_stop(rule, state, invalid, 11, 0)
    assert state.stopped and state.stop_reason == "full_plan_complete"
