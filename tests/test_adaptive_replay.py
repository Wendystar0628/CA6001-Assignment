"""Replay endpoints, label boundaries, and whole-near-group diagnostics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cpt_budget.adaptive_replay as replay
from cpt_budget.adaptive_data import validate_split
from cpt_budget.adaptive_policy import StopRule
from cpt_budget.models.distribution import point_prediction_frame
from cpt_budget.types import PredictionBundle, PreparedDataset


def fixture(initial=5):
    records, cells = [], []
    for i in range(16):
        record = dict(site_id="lower_hutt_town_hall", profile_id=f"p{i}",
                      location_id=f"l{i:02}", near_group_id="a_pair" if i in (5, 6) else f"g{i:02}",
                      x_m=float(i), y_m=float(i % 3), geometry_scale_m=1.)
        records.append(record)
        for depth in range(24):
            cells.append({**record, "depth_index": depth, "z_center_m": 2.125 + .25 * depth,
                          "log_qc": .2 + i / 100., "qc_mpa": np.exp(.2 + i / 100.)})
    profiles = pd.DataFrame(records)
    dataset = PreparedDataset(profiles, pd.DataFrame(cells),
                              pd.DataFrame([dict(site_id="lower_hutt_town_hall")]), Path("."))
    full = [f"l{i:02}" for i in range(13)]
    split = dict(site_id="lower_hutt_town_hall", seed=17, fold_id=0, role="development",
                 initial_ids=full[:initial], candidate_ids=full[initial:], full_ids=full,
                 audit_ids=[f"l{i:02}" for i in range(13, 16)], n_full=13,
                 near_groups={str(g): sorted(x.location_id) for g, x in profiles.groupby("near_group_id")},
                 budgets=[initial, 13])
    validate_split(split)
    dataset.truth = lambda ids: pytest.fail("Online replay must never call dataset.truth")
    original_observed = dataset.observed
    dataset.observed_calls = []

    def observed(ids):
        ids = list(ids)
        assert not set(ids) & set(split["audit_ids"])
        dataset.observed_calls.append(ids)
        return original_observed(ids)

    dataset.observed = observed
    return dataset, split


class FakeSuite:
    def __init__(self):
        self.conditions, self.queries = [], []

    def condition(self, observed):
        self.conditions.append(observed.copy())
        self.observed = observed.copy()
        return self

    def predict(self, query, model_ids):
        assert not {"log_qc", "qc_mpa"} & set(query)
        self.queries.append(query.copy())
        measured = self.observed.set_index(["location_id", "depth_index"]).log_qc
        keys = pd.MultiIndex.from_frame(query[["location_id", "depth_index"]])
        known = keys.isin(measured.index)
        values = np.full(len(query), .2)
        values[known] = measured.reindex(keys[known]).to_numpy()
        output = {}
        for model_id in model_ids:
            frame = point_prediction_frame(query, values)
            if model_id == "gp":
                frame["std_log_qc"] = np.where(known, 0., .5 + query.x_m.to_numpy() / 100.)
            output[model_id] = PredictionBundle(frame, model_id)
        return output


class FakePredictor:
    instances = []

    def __init__(self, *args, **kwargs):
        self.conditions, self.queries = [], []
        self.instances.append(self)

    def initialize(self):
        return self

    def condition(self, observed):
        self.conditions.append(observed.copy())
        return self

    def predict(self, query):
        assert not {"log_qc", "qc_mpa"} & set(query)
        self.queries.append(query.copy())
        return point_prediction_frame(query, np.full(len(query), .2))


@pytest.mark.parametrize("policy_id", ["B_seq", "P_adapt", "B_full"])
def test_replay_reaches_real_full_endpoint_and_never_reads_audit(policy_id):
    dataset, split = fixture()
    suite = FakeSuite()
    result = replay.run_trajectory(dataset, split, "gp", policy_id, diagnostics=False, suite=suite, log=None)
    trajectory = result["trajectories"]
    assert trajectory.iloc[-1].n_observed == split["n_full"]
    assert trajectory.iloc[-1].stop_reason == "full_plan_complete"
    assert trajectory.iloc[-1].stopped
    assert set(json.loads(trajectory.iloc[-1].known_ids)) == set(split["full_ids"])
    assert set(result["predictions"].location_id) == set(split["audit_ids"])
    assert not {"log_qc", "qc_mpa"} & set(result["predictions"])
    assert len(suite.conditions) == len(suite.queries) == len(trajectory)
    assert result["metadata"]["grid_cells"] == split["n_full"] * 24
    assert not result["metadata"]["audit_labels_read"]
    if policy_id == "B_full":
        assert len(trajectory) == 1
        assert set(suite.conditions[0].location_id) == set(split["full_ids"])
    else:
        for ids in trajectory.selected_ids.map(json.loads):
            assert not set(ids) & {"l05", "l06"} or {"l05", "l06"} <= set(ids)
        assert result["selections"].location_id.is_unique
        increments = trajectory.n_observed.diff().dropna().to_list()
        assert all(n >= 2 for n in increments[:-1])
        assert increments[-1] >= 1


def test_hidden_audit_and_future_labels_cannot_change_current_selection_or_stop(monkeypatch):
    monkeypatch.setattr(replay, "make_predictor", FakePredictor)
    dataset, split = fixture(initial=9)
    rule = StopRule(100., 100., 100., 100., min_observed=9, consecutive_rounds=1)
    first = replay.run_trajectory(dataset, split, "gp", "P_adapt", stop_rule=rule,
                                  suite=FakeSuite(), log=None)
    trajectory = first["trajectories"]
    assert trajectory.iloc[-1].stop_reason == "online_stability"
    assert trajectory.iloc[-1].n_observed < split["n_full"]
    last_known = set(json.loads(trajectory.iloc[-1].known_ids))
    poisoned, same_split = fixture(initial=9)
    future = (set(split["full_ids"]) - last_known) | set(split["audit_ids"])
    poisoned.grid.loc[poisoned.grid.location_id.isin(future), ["log_qc", "qc_mpa"]] = -999.
    second = replay.run_trajectory(poisoned, same_split, "gp", "P_adapt", stop_rule=rule,
                                   suite=FakeSuite(), log=None)
    columns = ["round", "n_observed", "known_ids", "selected_ids", "delta_q90", "delta_h90",
               "support_log_mae", "uncertainty_q90", "coverage_ok", "stopped", "stop_reason"]
    pd.testing.assert_frame_equal(trajectory[columns], second["trajectories"][columns])
    pd.testing.assert_frame_equal(first["selections"], second["selections"])
    pd.testing.assert_frame_equal(first["predictions"], second["predictions"])


def test_support_diagnostics_remove_whole_frozen_groups_and_use_single_model(monkeypatch):
    FakePredictor.instances = []
    monkeypatch.setattr(replay, "make_predictor", FakePredictor)
    dataset, split = fixture(initial=9)
    result = replay.run_trajectory(dataset, split, "gp", "B_seq", suite=FakeSuite(), log=None)
    assert len(FakePredictor.instances) == 1
    predictor = FakePredictor.instances[0]
    expected = result["metadata"]["support_fixed_group_ids"]
    assert len(expected) <= 5
    assert "a_pair" in expected
    for context, query in zip(predictor.conditions, predictor.queries):
        held = set(query.location_id)
        assert held.isdisjoint(context.location_id)
        assert any(held == set(split["near_groups"][key]) for key in expected)
        assert held <= set(split["initial_ids"])
    checked = result["trajectories"].query("n_support_audit_groups > 0")
    assert checked.n_support_audit_groups.eq(len(expected)).all()
    assert result["trajectories"].iloc[-1].n_support_audit_groups == 0
