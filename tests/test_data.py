"""Critical leakage boundaries and source/grid integrity, using small fixtures."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cpt_budget.data.normalize import _investigation_groups, midpoint_values, prepare_data
from cpt_budget.data.splits import make_splits, select_group_subset
from cpt_budget.types import PreparedDataset


def test_interpolation_never_crosses_large_gap_or_extrapolates():
    source = np.array([[1.9, 1], [2.0, 2], [2.25, 4], [3.0, 6], [3.25, 0], [3.5, -2]])
    actual = midpoint_values(source, np.array([1.8, 2.125, 2.25, 2.5, 3.0, 3.4]), .5)
    np.testing.assert_allclose(actual[[1, 2, 4]], [3, 4, 6])
    assert np.isnan(actual[[0, 3, 5]]).all()


def test_preparation_uses_explicit_array_keys_and_usable_lexical_representative(tmp_path):
    inputs = tmp_path / "Data"
    inputs.mkdir()
    depth = np.arange(2, 8.01, .1)
    arrays = {"key-z": np.c_[depth, np.full(len(depth), 10.)],
              "key-a": np.array([[2, 1.], [8, 2.]]),
              "key-w": np.c_[depth, np.full(len(depth), .5)],
              "key-b": np.c_[depth, np.full(len(depth), 2.)],
              "key-c": np.c_[depth, np.full(len(depth), 3.)]}
    # Manifest order and array-key order deliberately contradict source filename order.
    manifest = [{"site": "@Fixture", "file": f"fixture/{filename}.xlsx", "array_key": key,
                 "xy": [x, 0], "header": ["Depth(m)", "Cone resistance, qc(MPa)"]}
                for filename, key, x in [("c", "key-z", 0), ("a", "key-a", 0), ("b", "key-w", 0),
                                         ("d", "key-b", 1.9), ("e", "key-c", 3.8)]]
    (inputs / "cpt_screened_spatial_profiles.json").write_text(json.dumps(manifest))
    np.savez(inputs / "cpt_screened_depth_qc.npz", **arrays)
    dataset = prepare_data({"_root": str(tmp_path)})
    at_zero = dataset.profiles[dataset.profiles.x_m == 0]
    representative = at_zero[at_zero.is_representative].iloc[0]
    assert representative.source_file == "fixture/b.xlsx"
    assert not bool(at_zero.loc[at_zero.source_file == "fixture/a.xlsx", "usable"].iloc[0])
    assert (dataset.grid.loc[dataset.grid.location_id == representative.location_id, "qc_mpa"] == .5).all()
    assert len(dataset.grid) == 3 * 24
    # 0 -> 1.9 -> 3.8 is one transitive group without merging the paid positions.
    assert dataset.locations().near_group_id.nunique() == 1
    assert dataset.locations().location_id.nunique() == 3
    assert not {"qc_mpa", "log_qc", "low_1", "thickness_1_m"}.intersection(dataset.query(dataset.locations().location_id).columns)


def test_cross_site_groups_use_global_not_local_coordinates():
    profiles = pd.DataFrame([
        {"site_id": "a", "longitude": 172., "latitude": -43., "x_m": 0, "y_m": 0},
        {"site_id": "b", "longitude": 173., "latitude": -43., "x_m": 0, "y_m": 0},
        {"site_id": "c", "longitude": 172.001, "latitude": -43., "x_m": 1e6, "y_m": 1e6},
        {"site_id": "unknown", "longitude": np.nan, "latitude": np.nan, "x_m": 0, "y_m": 0},
    ])
    groups, edges = _investigation_groups(profiles, {}, 250)
    assert groups["a"] == groups["c"]
    assert len({groups["a"], groups["b"], groups["unknown"]}) == 3
    assert len(edges) == 1 and edges[0]["reason"] == "geographic_nearest_pair"


def _partition_fixture(root: Path) -> PreparedDataset:
    profiles, rows, sites = [], [], []
    for index in range(6):
        sid = f"site_{index}"
        sites.append({"site_id": sid, "investigation_group": sid, "n_locations": 20,
                      "replay_eligible": True, "grouping_verified": index != 0})
        for i in range(20):
            lid = f"{sid}_location_{i:02}"
            p = {"site_id": sid, "profile_id": lid, "location_id": lid,
                 "near_group_id": f"{sid}_group_{i // 2}", "x_m": (i // 2) * 20 + i % 2,
                 "y_m": 0., "geometry_scale_m": 1., "is_representative": True, "usable": True}
            profiles.append(p)
            for j in range(24):
                rows.append({**p, "depth_index": j, "z_center_m": 2.125 + j * .25,
                             "qc_mpa": float(1 + i), "log_qc": np.log(1 + i)})
    return PreparedDataset(pd.DataFrame(profiles), pd.DataFrame(rows), pd.DataFrame(sites), root)


def test_saved_partitions_are_label_blind_and_near_groups_never_cross_boundary(tmp_path):
    dataset = _partition_fixture(tmp_path)
    config = {"_root": str(tmp_path), "replay": {"seeds": [17]}, "split": {"seed": 17}}
    result = make_splits(dataset, config)
    modified = PreparedDataset(dataset.profiles.copy(), dataset.grid.copy(), dataset.sites.copy(), tmp_path)
    modified.grid["qc_mpa"] *= 1e5
    modified.grid["log_qc"] += 100
    again = make_splits(modified, config)
    assert result["folds"] == again["folds"]
    assert result["replay_splits"] == again["replay_splits"]
    pd.testing.assert_frame_equal(result["episodes"], again["episodes"])
    membership = dataset.locations().set_index("location_id").near_group_id.to_dict()
    for episode in result["episodes"].itertuples():
        support_groups = {membership[loc] for loc in episode.support_location_ids}
        query_groups = {membership[loc] for loc in episode.query_location_ids}
        assert not support_groups & query_groups
        assert not set(episode.withheld_support_side_ids) & set(episode.query_location_ids)
    for split in result["replay_splits"]:
        assert not {membership[loc] for loc in split["evaluation_ids"]} & {membership[loc] for loc in split["pool_ids"]}
        assert set(split["initial_ids"]).issubset(split["pool_ids"])
        assert len(split["initial_ids"]) == 5 and len(split["pool_ids"]) >= 13
    all_tests = [group for fold in result["folds"] for group in fold["test_groups"]]
    assert len(all_tests) == len(set(all_tests)) == 6
    for fold in result["folds"]:
        assert not set(fold["train_groups"]) & set(fold["validation_groups"])
        assert not (set(fold["train_groups"]) | set(fold["validation_groups"])) & set(fold["test_groups"])
    assert select_group_subset({"one_large_group": list(range(20))}, 4, 7, np.random.default_rng(17)) == []
