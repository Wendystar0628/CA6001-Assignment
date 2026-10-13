"""Residual-space event mathematics, truth isolation and honest update coverage."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cpt_budget.diagnostics.residual_adapter import leave_location_out_idw
from cpt_budget.types import PreparedDataset
from cpt_budget.v4 import training


def toy_dataset(site_ids=("S",)):
    profiles, rows, sites = [], [], []
    for site in site_ids:
        sites.append(dict(site_id=site, investigation_group=f"group_{site}", geometry_scale_m=10.))
        for i in range(8):
            location = f"{site}_L{i}"
            profile = dict(site_id=site, location_id=location, profile_id=location,
                           near_group_id=location, x_m=float(i * 10), y_m=float(i % 2 * 10),
                           geometry_scale_m=10., usable=True, is_representative=True)
            profiles.append(profile)
            for depth in range(24):
                log_qc = .07 * i * i - .06 * depth
                rows.append({**profile, "depth_index": depth, "z_center_m": 2.125 + .25 * depth,
                             "log_qc": log_qc, "qc_mpa": np.exp(log_qc)})
    return PreparedDataset(pd.DataFrame(profiles), pd.DataFrame(rows), pd.DataFrame(sites), None)


def episode(site="S", role="train", episode_id="e", query=None):
    return dict(site_id=site, role=role, fold_id=0, episode_id=episode_id, seed=17,
                support_location_ids=[f"{site}_L{i}" for i in range(5)],
                query_location_ids=query or [f"{site}_L{i}" for i in range(5, 8)])


def test_query_truth_changes_only_loss_labels_and_residual_targets():
    dataset = toy_dataset()
    item = episode()
    first = training.build_residual_episode(dataset, item, ["S_L5", "S_L6"])
    altered = copy.deepcopy(dataset)
    mask = altered.grid.location_id.isin(item["query_location_ids"])
    altered.grid.loc[mask, "log_qc"] += 9.
    altered.grid.loc[mask, "qc_mpa"] *= np.exp(9.)
    second = training.build_residual_episode(altered, item, ["S_L5", "S_L6"])
    for name in ("X_support", "y_support", "X_query", "query_baseline_log_qc", "support_baseline_log_qc"):
        np.testing.assert_allclose(first[name], second[name], equal_nan=True)
    np.testing.assert_allclose(second["y_query"] - first["y_query"], 9.)
    np.testing.assert_allclose(first["y_query"] + first["query_baseline_log_qc"], first["raw_query_log_qc"])
    observed = dataset.observed(item["support_location_ids"])
    np.testing.assert_allclose(first["y_support"], observed.log_qc.to_numpy() - leave_location_out_idw(observed))
    assert first["X_support"].shape == (120, 24)
    assert first["X_query"].shape == (48, 24)


def test_row_dependent_thresholds_follow_both_official_target_transforms():
    torch = pytest.importorskip("torch")
    pytest.importorskip("tabpfn")
    from sklearn.preprocessing import PowerTransformer

    support = np.array([-1.2, -.4, .2, .5, 1.3, 3.1])
    baseline = np.array([-1.4, -.1, .3, 2.])
    standardized = (support - support.mean()) / support.std()
    transform = PowerTransformer().fit(standardized[:, None])
    configs = [[SimpleNamespace(target_transform=None)], [SimpleNamespace(target_transform=transform)]]
    actual = training.residual_event_thresholds_in_estimator_space(support, configs, baseline)
    expected_z = (-baseline - support.mean()) / support.std()
    expected_power = transform.transform(expected_z.astype(np.float32).reshape(-1, 1)).reshape(-1)
    np.testing.assert_allclose(actual[0].numpy(), expected_z, rtol=1e-6)
    np.testing.assert_allclose(actual[1].numpy(), expected_power, rtol=1e-6)
    assert actual.shape == (2, 4)
    assert torch.unique(actual[0]).numel() == 4
    with pytest.raises(ValueError, match="finite"):
        training.residual_event_thresholds_in_estimator_space(support, configs, [np.nan])


def test_official_cdf_respects_query_specific_thresholds_and_physical_labels():
    torch = pytest.importorskip("torch")
    pytest.importorskip("tabpfn")
    from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution

    bardist = FullSupportBarDistribution(torch.tensor([-3., -1., 0., 1., 3.]))
    logits = torch.tensor([[[.1, .8, .2, -.1], [.4, -.2, .5, .3], [.2, -.1, .6, -.7]]], requires_grad=True)
    thresholds = torch.tensor([[-1.2, .5, 1.7]])
    # Residual targets [-.2, -.2, .5] have different signs from these physical
    # targets [1.0, -.7, -1.2]; using residual<0 would silently invert labels.
    raw_targets = np.array([1., -.7, -1.2])
    loss = training.residual_native_event_brier(logits, bardist, thresholds, raw_targets)
    independent = torch.stack([
        bardist.cdf(logits[0, index], thresholds[0, index:index + 1])[0]
        for index in range(3)
    ])
    expected = (independent - torch.tensor([0., 1., 1.])).square().mean()
    torch.testing.assert_close(loss, expected)
    wrong = training.residual_native_event_brier(logits, bardist, thresholds, [-.2, -.2, .5])
    assert abs(loss.item() - wrong.item()) > .01
    permutation = [2, 0, 1]
    permuted = training.residual_native_event_brier(
        logits[:, permutation], bardist, thresholds[:, permutation], raw_targets[permutation])
    torch.testing.assert_close(loss, permuted)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="exactly one"):
        training.residual_native_event_brier(logits, bardist, torch.zeros(1), raw_targets)


def test_seeded_plan_covers_complete_query_blocks_once_per_site_cycle():
    items = [
        {**episode(site, episode_id=f"{site}_{i}"),
         "support_location_ids": [f"{site}_S{k}" for k in range(13)],
         "query_location_ids": [f"{site}_Q{k}" for k in range(19)]}
        for site in ("S", "T") for i in range(3)
    ]
    config = {"v4": {"training": {"seed": 17}}, "episodes": {"max_rows": 512}}
    plan = training.make_residual_training_plan(items, config)
    assert plan == training.make_residual_training_plan(list(reversed(items)), config)
    for site, tasks in plan.items():
        assert len(tasks) == 9 and len({key for _, _, key in tasks}) == 9
        for item in (row for row in items if row["site_id"] == site):
            blocks = [block for e, block, _ in tasks if e["episode_id"] == item["episode_id"]]
            assert sorted(value for block in blocks for value in block) == sorted(item["query_location_ids"])
            assert all((13 + len(block)) * 24 <= 512 for block in blocks)
    duplicate = [items[0], items[0]]
    with pytest.raises(ValueError, match="unique"):
        training.make_residual_training_plan(duplicate, config)


def test_checkpoint_validation_scores_reconstructed_qc_not_residual_error():
    data = toy_dataset()

    class Predictor:
        def condition(self, observed):
            pass

        def predict(self, query):
            frame = data.truth(query.location_id.drop_duplicates())
            frame["pred_log_qc"] = frame.log_qc + frame.location_id.map({"S_L5": 0., "S_L6": 2., "S_L7": 4.})
            frame["pred_qc_mpa"] = np.exp(frame.pred_log_qc)
            frame["p_low_1"] = .2
            return SimpleNamespace(frame=frame)

    adapter = SimpleNamespace(predictor=Predictor, use_amp=False)
    validation = [episode(query=["S_L5", "S_L6"]), episode(query=["S_L7"])]
    result = training.validate_residual(adapter, data, validation, {})
    assert result["macro_log_mae"] == pytest.approx(2.5)
    assert result["macro_native_event_brier"] == pytest.approx(.04)
    assert result["macro_score"] == pytest.approx(2.52)
    assert result["residual_alpha_for_checkpoint_selection"] == 1.0
    assert result["risk_calibration_used"] is False


def test_training_rejects_site_or_group_leakage_before_model_initialization(tmp_path):
    data = toy_dataset(("S", "T"))
    with pytest.raises(ValueError, match="sites overlap"):
        training.fit_residual_tabpfn(data, [episode()], [episode(role="validation")], {}, tmp_path)
    data.sites["investigation_group"] = "shared"
    with pytest.raises(ValueError, match="groups overlap"):
        training.fit_residual_tabpfn(data, [episode()], [episode("T", "validation")], {}, tmp_path)
    with pytest.raises(ValueError, match="Only train-role"):
        training.fit_residual_tabpfn(data, [episode(role="test")], [episode("T", "validation")], {}, tmp_path)


def test_successful_step_budget_site_balance_and_best_final_exposure_are_distinct(tmp_path, monkeypatch):
    data = toy_dataset(("S", "T", "V"))
    train = [episode(site, episode_id=f"{site}_{i}") for site in ("S", "T") for i in range(160)]
    validation = [episode("V", "validation")]

    class FakeAdapter:
        def __init__(self, config, variant):
            self.steps, self.use_amp, self.device = 0, False, "cpu"

        def step(self, **payload):
            self.steps += 1
            return {"step": self.steps, "optimizer_updates": 1}

        def save(self, path, metadata):
            path.write_text(json.dumps(metadata), encoding="utf-8")

    monkeypatch.setattr(training, "ResidualEpisodeAdapter", FakeAdapter)
    monkeypatch.setattr(training, "build_residual_episode", lambda data, item, block, config: {
        **dict.fromkeys(("X_support", "y_support", "X_query", "y_query", "query_baseline_log_qc", "raw_query_log_qc"), np.zeros(2)),
        "support_location_ids": item["support_location_ids"],
    })
    monkeypatch.setattr(training, "validate_residual", lambda adapter, *args: {
        "macro_score": 1. if adapter.steps == 0 else 2. + adapter.steps / 1000.,
        "selection_metric": "physical_log_mae_plus_half_native_brier",
    })
    config = {"v4": {"training": {"max_steps": 220, "validation_every": 100,
                                   "min_steps": 200, "patience": 1, "seed": 17}}}
    result = training.fit_residual_tabpfn(data, train, validation, config, tmp_path, variant="task")
    assert result["best_step"] == 100 and result["actual_steps"] == 200
    assert result["successful_optimizer_updates"] == 200
    assert result["best_checkpoint_exposure"]["seen_episodes"] == 100
    assert result["exposure"]["seen_episodes"] == 200
    assert not result["exposure"]["all_query_blocks_seen"]
    assert {v["updates"] for v in result["exposure"]["per_site"].values()} == {100}
    assert result["stop_reason"] == "early_stopping_after_minimum_successful_updates"
    assert result["validation_preferred_model"] == "tabpfn_residual_pretrained"
    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved["checkpoint_path"] == result["checkpoint_path"]
