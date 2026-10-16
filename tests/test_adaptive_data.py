"""Information and count boundaries for the adaptive-data stage."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cpt_budget.adaptive_data import (
    AdaptiveDataView, audit_truth, build_splits, fixed_query_grid,
    raw_low_thickness, validate_split,
)
from cpt_budget.types import PreparedDataset, QUERY_COLUMNS


def toy_dataset():
    profiles, cells = [], []
    for i in range(18):
        # Two close positions are an indivisible group; most groups are singleton.
        group = "pair" if i in (3, 4) else f"group_{i}"
        record = {"site_id": "lower_hutt_town_hall", "profile_id": f"p{i}",
                  "location_id": f"l{i:02}", "near_group_id": group,
                  "x_m": float(i), "y_m": float(i % 3), "geometry_scale_m": 1.,
                  "is_representative": True, "usable": True}
        profiles.append(record)
        for depth in range(3):
            cells.append({**record, "depth_index": depth, "z_center_m": 2.125 + depth * .25,
                          "qc_mpa": float(i + 1), "log_qc": np.log(i + 1), "low_1": False})
    sites = pd.DataFrame([{"site_id": "lower_hutt_town_hall", "n_locations": 18}])
    return PreparedDataset(pd.DataFrame(profiles), pd.DataFrame(cells), sites, Path("."))


def test_geometry_partition_is_group_atomic_and_contains_full_endpoint():
    dataset = toy_dataset()
    splits = build_splits(dataset, seeds=(17, 29, 43), site_ids=["lower_hutt_town_hall"])
    for split in splits:
        validate_split(split)
        assert split["n_full"] == len(split["initial_ids"]) + len(split["candidate_ids"])
        assert split["budgets"][-1] == split["n_full"]
        assert set(split["full_ids"]).isdisjoint(split["audit_ids"])
        assert split["role"] == "development"
    poisoned = copy.deepcopy(dataset)
    poisoned.grid[["qc_mpa", "log_qc"]] = 987654.
    assert splits == build_splits(poisoned, seeds=(17, 29, 43), site_ids=["lower_hutt_town_hall"])


def test_online_view_never_reveals_audit_and_rejects_partial_groups():
    dataset = toy_dataset()
    split = build_splits(dataset, seeds=(17,), site_ids=["lower_hutt_town_hall"])[0]
    view = AdaptiveDataView(dataset, split)
    assert set(view.query(split["audit_ids"])) <= set(QUERY_COLUMNS)
    assert "qc_mpa" not in view.geometry()
    with pytest.raises(ValueError):
        view.reveal(split["audit_ids"][:1])
    pair_ids = split["near_groups"]["pair"]
    if set(pair_ids) <= set(split["candidate_ids"]):
        with pytest.raises(ValueError):
            view.reveal(pair_ids[:1])
    for group in split["near_groups"].values():
        if set(group) <= set(split["candidate_ids"]):
            view.reveal(group)
    assert set(view.revealed_ids) == set(split["full_ids"])
    assert set(audit_truth(dataset, split).location_id) == set(split["audit_ids"])
    assert set(fixed_query_grid(dataset, split).location_id) == set(split["full_ids"])


def test_raw_thickness_interpolates_threshold_crossings_without_bridging_gaps():
    # Two linear threshold crossings bound a thin low-qc interval of exactly .1 m.
    curve = np.array([[2., 1.5], [2.1, .5], [2.2, 1.5], [3., .5], [3.2, .5]])
    result = raw_low_thickness(curve, threshold=1., start=2., end=3.2, max_gap=.5)
    assert result["raw_low_thickness_m"] == pytest.approx(.3)
    assert result["raw_covered_depth_m"] == pytest.approx(.4)
    assert result["low_intervals"][0] == pytest.approx((2.05, 2.15))


def test_validation_detects_group_crossing_support_and_audit():
    dataset = toy_dataset()
    split = build_splits(dataset, seeds=(17,), site_ids=["lower_hutt_town_hall"])[0]
    group = next(ids for ids in split["near_groups"].values() if ids[0] in split["audit_ids"])
    split["initial_ids"].append(group[0])
    with pytest.raises(ValueError):
        validate_split(split)
