"""Critical boundaries; no pretrained model download in ordinary unit tests."""
import copy

import numpy as np
import pandas as pd
import pytest

from cpt_budget.models.tabpfn_adapter import EpisodeAdapter, decode_full_output
from cpt_budget.models.tabpfn_train import _validate, build_episode, fit_tabpfn
from cpt_budget.types import PreparedDataset


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
    return dict(site_id="S", support_location_ids=[f"L{i}" for i in range(5)], query_location_ids=["L5", "L6", "L7"])


def test_query_labels_only_change_loss_targets():
    data = toy_dataset()
    first = build_episode(data, episode(), query_limit=2)
    changed = copy.deepcopy(data)
    changed.grid.loc[changed.grid.location_id.isin(["L5", "L6", "L7"]), "log_qc"] += 100
    second = build_episode(changed, episode(), query_limit=2)
    for key in ("X_support", "y_support", "X_query"):
        np.testing.assert_allclose(first[key], second[key], equal_nan=True)
    np.testing.assert_allclose(second["y_query"] - first["y_query"], 100)
    assert first["X_support"].shape == (120, 16)
    assert first["X_query"].shape == (48, 16)
    rotated = build_episode(data, episode(), query_limit=2, query_cycle=1)
    assert rotated["query_location_ids"] == ["L7", "L5"]


def test_near_group_boundary_and_train_roles_are_enforced():
    data = toy_dataset()
    data.profiles.loc[data.profiles.location_id == "L5", "near_group_id"] = "L0"
    with pytest.raises(ValueError, match="Near groups"):
        build_episode(data, episode())
    with pytest.raises(ValueError, match="train-role"):
        fit_tabpfn(toy_dataset(), [{**episode(), "role": "test"}], [{**episode(), "role": "validation"}], {}, ".")


def test_native_cdf_uses_raw_log_thresholds_and_half_normal_tails():
    torch = pytest.importorskip("torch")
    pytest.importorskip("tabpfn")
    from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
    criterion = FullSupportBarDistribution(torch.tensor([-1., -.5, 0., .5, 1.]))
    logits = torch.tensor([[0., .2, .3, -.1], [.2, -.3, .1, 0.]])
    quantiles = [criterion.icdf(logits, p).numpy() for p in (.05, .10, .25, .50, .75, .90, .95)]
    query = pd.DataFrame({"site_id": ["S", "S"], "location_id": ["A", "A"], "depth_index": [0, 1], "log_qc": [999., 999.]})
    frame = decode_full_output(dict(criterion=criterion, logits=logits, quantiles=quantiles), query)
    expected = criterion.cdf(logits, torch.tensor(np.log([.5, 1., 2.]), dtype=logits.dtype)).numpy()
    np.testing.assert_allclose(frame[["p_low_0_5", "p_low_1", "p_low_2"]], expected)
    np.testing.assert_allclose(frame.pred_log_qc, quantiles[3])
    assert "log_qc" not in frame
    # Both tails are nonzero beyond finite outer borders; a clipped histogram
    # CDF would incorrectly return exactly zero/one here.
    tails = criterion.cdf(logits, torch.tensor([-1.1, 1.1]))
    assert torch.all(tails[:, 0] > 0) and torch.all(tails[:, 1] < 1)


def test_preprocessor_initialisation_sees_support_only(monkeypatch):
    pytest.importorskip("tabpfn")
    import tabpfn.finetuning.data_util as official
    adapter = EpisodeAdapter.__new__(EpisodeAdapter)
    adapter.config = {}
    adapter.estimator = object()
    support_x = np.arange(30.).reshape(5, 6)
    support_y = np.arange(5.)
    query_x = np.ones((2, 6)) * 999
    query_y = np.ones(2) * 777

    def capture(calling_instance, X, y, **kwargs):
        np.testing.assert_array_equal(X, support_x)
        np.testing.assert_array_equal(y, support_y)
        assert kwargs["shuffle"] is False and kwargs["max_data_size"] is None
        split = kwargs["split_fn"](X, y, stratify=None)
        np.testing.assert_array_equal(split[1], query_x)
        np.testing.assert_array_equal(split[3], query_y)
        return ["one_complete_episode"]

    monkeypatch.setattr(official, "get_preprocessed_dataset_chunks", capture)
    monkeypatch.setattr(official, "meta_dataset_collator", lambda items: items[0])
    assert adapter.preprocess(support_x, support_y, query_x, query_y, seed=42) == "one_complete_episode"


def test_validation_averages_cpt_rmse_then_scenarios_and_sites():
    from types import SimpleNamespace
    data = toy_dataset()

    class Predictor:
        def condition(self, observed):
            pass

        def predict(self, query):
            frame = data.truth(query.location_id.drop_duplicates())
            # Two CPTs with constant errors 0 and 2 have mean CPT RMSE 1,
            # not pooled RMSE sqrt(2). A one-query scenario has equal weight.
            frame["pred_log_qc"] = frame.log_qc + frame.location_id.map({"L5": 0., "L6": 2., "L7": 4.})
            return SimpleNamespace(frame=frame)

    adapter = SimpleNamespace(predictor=Predictor, use_amp=False)
    first = {**episode(), "query_location_ids": ["L5", "L6"]}
    second = {**episode(), "query_location_ids": ["L7"]}
    scores = _validate(adapter, data, [first])
    assert scores["macro_log_rmse"] == pytest.approx(1.)
    scores = _validate(adapter, data, [first, second])
    assert scores["macro_log_rmse"] == pytest.approx(2.5)
    assert scores["site_log_rmse"]["S"] == pytest.approx(2.5)
