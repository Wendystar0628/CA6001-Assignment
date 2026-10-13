"""Ensure source trajectories reproduce replay geometry without label access."""
import json

import numpy as np
import pandas as pd
import pytest

from cpt_budget.acquisition import select_batch
from cpt_budget.data.splits import make_replay_splits
from cpt_budget.v4.episodes import (
    BUDGETS, TRAIN_SCENARIO_SEEDS, VALIDATION_SCENARIO_SEEDS,
    build_trajectory_episodes, save_episode_plan,
)


class GeometryOnlyDataset:
    def __init__(self, sizes, *, near_pairs=False):
        self.sites = pd.DataFrame([dict(site_id=s, investigation_group=f"group_{s}",
                                        grouping_verified=True) for s in sizes])
        self._locations = pd.DataFrame([
            dict(site_id=s, location_id=f"{s}_{i:02d}",
                 near_group_id=f"{s}_group_{i // 2 if near_pairs else i:02d}",
                 x_m=float((i % 5) * 10), y_m=float((i // 5) * 10))
            for s, n in sizes.items() for i in range(n)
        ])

    def locations(self):
        return self._locations.copy()

    def observed(self, ids):
        raise AssertionError("Episode planning must not read observations")

    def truth(self, ids):
        raise AssertionError("Episode planning must not read query labels")


def fold(train=("s",), validation=("v",), test=("t",)):
    result = {"fold_id": 0}
    for role, sites in (("train", train), ("validation", validation), ("test", test)):
        result[f"{role}_sites"] = list(sites)
        result[f"{role}_groups"] = [f"group_{s}" for s in sites]
    return result


def test_complete_trajectories_match_real_replay_selection_and_spacing():
    dataset = GeometryOnlyDataset({"s": 24, "v": 24, "t": 24}, near_pairs=True)
    config = {"replay": {"seeds": list(TRAIN_SCENARIO_SEEDS)}}
    plan = build_trajectory_episodes(dataset, fold(), config, role="train")
    replay, _ = make_replay_splits(dataset, config)
    geometry = dataset.locations()
    assert set(plan.scenario_seed) == set(TRAIN_SCENARIO_SEEDS)
    assert set(plan.budget) == set(BUDGETS)
    assert len(plan) == 8 * 5
    for split in [row for row in replay if row["site_id"] == "s"]:
        rows = plan.loc[plan.scenario_seed == split["seed"]].sort_values("budget")
        observed = split["initial_ids"].copy()
        pool = geometry.loc[geometry.location_id.isin(split["pool_ids"])]
        for row in rows.itertuples():
            while len(observed) < row.budget:
                chosen, _ = select_batch(pool.loc[pool.location_id.isin(observed)],
                                         pool.loc[~pool.location_id.isin(observed)], None, config,
                                         policy="maximin", min_spacing_m=split["min_spacing_m"], batch_size=2)
                observed.extend(chosen)
            assert row.support_location_ids == observed
            assert row.query_location_ids == split["evaluation_ids"]
            assert not set(row.support_groups) & set(row.query_near_groups)
            assert row.min_spacing_m == split["min_spacing_m"]


def test_small_sites_keep_feasible_budgets_and_explicit_skips():
    dataset = GeometryOnlyDataset({"small": 8, "tiny": 5, "v": 12, "t": 15}, near_pairs=True)
    plan = build_trajectory_episodes(dataset, fold(("small", "tiny")), {}, role="train")
    assert set(plan.site_id) == {"small"}
    assert set(plan.budget) == {5}
    assert all(len(ids) == 2 for ids in plan.query_location_ids)
    skips = plan.attrs["provenance"]["skipped"]
    assert len(skips) == 16
    assert all(row["missing_budgets"] == [7, 9, 11, 13] for row in skips if row["site_id"] == "small")
    assert all(row["missing_budgets"] == list(BUDGETS) for row in skips if row["site_id"] == "tiny")


def test_roles_are_frozen_and_no_test_plans_are_allowed():
    dataset = GeometryOnlyDataset({"s": 18, "v": 18, "t": 18})
    plan = build_trajectory_episodes(dataset, fold(), {}, role="validation")
    assert set(plan.site_id) == {"v"}
    assert set(plan.scenario_seed) == set(VALIDATION_SCENARIO_SEEDS)
    with pytest.raises(ValueError, match="test is forbidden"):
        build_trajectory_episodes(dataset, fold(), {}, role="test")
    with pytest.raises(ValueError, match="overlap"):
        build_trajectory_episodes(dataset, fold(validation=("s",)), {}, role="train")
    with pytest.raises(ValueError, match="assignments disagree"):
        build_trajectory_episodes(dataset, {**fold(), "train_groups": ["wrong"]}, {}, role="train")


def test_plan_is_deterministic_label_invariant_and_roundtrips(tmp_path):
    dataset = GeometryOnlyDataset({"s": 24, "v": 24, "t": 24})
    first = build_trajectory_episodes(dataset, fold(), {}, role="train")
    dataset._locations["qc_mpa"] = np.nan
    dataset._locations["log_qc"] = -999.
    dataset._locations = dataset._locations.sample(frac=1, random_state=901)
    second = build_trajectory_episodes(dataset, fold(), {}, role="train")
    pd.testing.assert_frame_equal(first, second)
    assert first.attrs == second.attrs
    saved = save_episode_plan(first, tmp_path / "fold_0_train.parquet")
    restored = pd.read_parquet(saved["episodes_path"])
    assert len(restored) == len(first)
    provenance = json.loads((tmp_path / "fold_0_train.provenance.json").read_text())
    assert provenance["labels_read"] is False
    assert provenance["n_trajectories"] == 8
    assert provenance["site_coverage"][0]["n_distinct_query_masks"] > 1
