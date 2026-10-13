"""Physical event space, fixed query coverage, and spatial label boundaries."""
import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cpt_budget.types import PreparedDataset
from cpt_budget.v2.tabpfn import (
    build_spatial_episode, event_thresholds_in_estimator_space, make_training_plan,
    native_event_brier, query_blocks, validate_spatial,
    optimizer_step_with_recovery,
)


def toy_dataset():
    profiles, rows = [], []
    for i in range(8):
        location = f"L{i}"
        profile = dict(site_id="S", location_id=location, profile_id=location,
                       near_group_id=location, x_m=float(i * 10), y_m=float(i % 2 * 10),
                       geometry_scale_m=10., usable=True, is_representative=True)
        profiles.append(profile)
        for depth in range(24):
            log_qc = .3 * i + .02 * depth
            rows.append({**profile, "depth_index": depth, "z_center_m": 2.125 + .25 * depth,
                         "log_qc": log_qc, "qc_mpa": np.exp(log_qc)})
    sites = pd.DataFrame([dict(site_id="S", investigation_group="G", geometry_scale_m=10.)])
    return PreparedDataset(pd.DataFrame(profiles), pd.DataFrame(rows), sites, None)


def episode():
    return dict(site_id="S", episode_id="a", support_location_ids=[f"L{i}" for i in range(5)],
                query_location_ids=["L5", "L6", "L7"])


def test_query_labels_cannot_change_spatial_inputs():
    dataset = toy_dataset()
    original = build_spatial_episode(dataset, episode(), ["L5", "L6"])
    altered = copy.deepcopy(dataset)
    mask = altered.grid.location_id.isin(["L5", "L6", "L7"])
    altered.grid.loc[mask, "log_qc"] += 10
    altered.grid.loc[mask, "qc_mpa"] *= np.exp(10)
    changed = build_spatial_episode(altered, episode(), ["L5", "L6"])
    assert original["X_support"].shape == (120, 24)
    assert original["X_query"].shape == (48, 24)
    for name in ["X_support", "y_support", "X_query"]:
        np.testing.assert_allclose(original[name], changed[name], equal_nan=True)
    np.testing.assert_allclose(changed["y_query"] - original["y_query"], 10.)


def test_blocks_cover_query_once_and_keep_intact_support_within_512_rows():
    item = {**episode(), "support_location_ids": [f"S{i}" for i in range(13)],
            "query_location_ids": [f"Q{i}" for i in range(19)]}
    blocks = query_blocks(item, {"episodes": {"query_profiles_per_batch": 8, "max_rows": 512}})
    assert list(map(len, blocks)) == [8, 8, 3]
    assert [q for block in blocks for q in block] == item["query_location_ids"]
    assert max((len(item["support_location_ids"]) + len(block)) * 24 for block in blocks) <= 512
    second = {**item, "episode_id": "b", "query_location_ids": ["X", "Y"]}
    plan = make_training_plan([item, second], {})["S"]
    assert [task[0]["episode_id"] for task in plan] == ["a", "b", "a", "a"]
    assert len({key for _, _, key in plan}) == 4


def test_event_threshold_follows_support_zscore_and_fitted_target_transform():
    torch = pytest.importorskip("torch")
    pytest.importorskip("tabpfn")
    from sklearn.preprocessing import PowerTransformer
    from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
    support = np.array([1., 1.1, 1.5, 2.2, 3.8, 7.])
    standardized = (support - support.mean()) / support.std()
    transform = PowerTransformer().fit(standardized[:, None])
    configs = [[SimpleNamespace(target_transform=None)], [SimpleNamespace(target_transform=transform)]]
    thresholds = event_thresholds_in_estimator_space(support, configs)
    expected_z = (0. - support.mean()) / support.std()
    expected_power = transform.transform(np.array([[expected_z]], dtype=np.float32))[0, 0]
    np.testing.assert_allclose(thresholds.numpy(), [expected_z, expected_power], rtol=1e-6)
    assert not torch.any(thresholds == 0.)
    bardist = FullSupportBarDistribution(torch.tensor([-3., -1., 0., 1., 3.]))
    logits = torch.tensor([[[.1, .8, .2, -.1], [.4, -.2, .5, .3]]], requires_grad=True)
    labels = np.array([-.4, .5])
    loss = native_event_brier(logits, bardist, thresholds[:1], labels)
    exact = ((bardist.cdf(logits, thresholds[:1]).squeeze(-1) - torch.tensor([[1., 0.]])) ** 2).mean()
    torch.testing.assert_close(loss, exact)
    wrong = native_event_brier(logits, bardist, torch.zeros(1), labels)
    assert abs(loss.item() - wrong.item()) > .01
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_validation_score_uses_scenario_and_site_macro_with_native_probability():
    data = toy_dataset()

    class Predictor:
        def condition(self, observed):
            pass

        def predict(self, query):
            frame = data.truth(query.location_id.drop_duplicates())
            frame["pred_log_qc"] = frame.log_qc + frame.location_id.map({"L5": 0., "L6": 2., "L7": 4.})
            frame["pred_qc_mpa"] = np.exp(frame.pred_log_qc)
            frame["p_low_1"] = .2
            return SimpleNamespace(frame=frame)

    adapter = SimpleNamespace(predictor=Predictor, use_amp=False)
    episodes = [{**episode(), "query_location_ids": ["L5", "L6"]},
                {**episode(), "query_location_ids": ["L7"]}]
    result = validate_spatial(adapter, data, episodes, {})
    assert result["macro_log_mae"] == pytest.approx(2.5)
    assert result["macro_native_event_brier"] == pytest.approx(.04)
    assert result["macro_score"] == pytest.approx(2.52)
    assert result["macro_recall"] is None
    assert result["risk_calibration_used"] is False


@pytest.mark.parametrize("overflow_attempts, expected_fallback", [(2, False), (4, True)])
def test_amp_overflow_retries_same_episode_without_partial_update(overflow_attempts, expected_fallback):
    torch = pytest.importorskip("torch")
    pytest.importorskip("tabpfn")
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scaler = torch.amp.GradScaler("cpu", init_scale=1024.)
    original = model.weight.detach().clone()
    calls, draws = [], []
    def compute(amp_enabled):
        # None of the failed attempts may have advanced weights or Adam state.
        torch.testing.assert_close(model.weight, original)
        assert not optimizer.state
        calls.append(amp_enabled)
        draws.append(torch.rand(3))
        prediction = model(torch.tensor([[1., 2.]]))
        if len(calls) <= overflow_attempts:
            prediction.register_hook(lambda grad: torch.full_like(grad, float("inf")))
        return {"loss": (prediction - 3.).square().mean()}
    _, result = optimizer_step_with_recovery(
        compute, model=model, optimizer=optimizer, scaler=scaler, use_amp=True)
    assert result["amp_overflow_count"] == overflow_attempts
    assert result["amp_fp32_fallback"] == expected_fallback
    assert result["optimizer_updates"] == 1
    assert result["updated_parameter_delta_l2"] > 0
    assert result["amp_scale_after"] == 1024. * .5 ** overflow_attempts
    assert len(calls) == overflow_attempts + 1
    for draw in draws:
        torch.testing.assert_close(draw, draws[0])
    assert all(int(state["step"].item()) == 1 for state in optimizer.state.values())


def test_nonfinite_fp32_fallback_fails_without_any_update():
    torch = pytest.importorskip("torch")
    pytest.importorskip("tabpfn")
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scaler = torch.amp.GradScaler("cpu", init_scale=1024.)
    original = [p.detach().clone() for p in model.parameters()]
    def compute(amp_enabled):
        prediction = model(torch.ones(1, 1))
        prediction.register_hook(lambda grad: torch.full_like(grad, float("inf")))
        return {"loss": prediction.square().mean()}
    with pytest.raises(FloatingPointError, match="FP32 fallback"):
        optimizer_step_with_recovery(compute, model=model, optimizer=optimizer, scaler=scaler, use_amp=True)
    assert not optimizer.state
    for before, after in zip(original, model.parameters()):
        torch.testing.assert_close(before, after)
