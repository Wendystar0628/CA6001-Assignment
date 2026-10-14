"""Focused information-boundary and set/geometry behavior checks for profile CNP."""
import numpy as np
import pandas as pd
import pytest
import torch

from cpt_budget.v5.neural_process import ProfileCNP, NeuralProcessPredictor, _validate_plans


def _rows(xys, *, prefix="p", labels=True):
    rows = [{"site_id": "target", "location_id": f"{prefix}{i}",
             "near_group_id": f"{prefix}{i}", "depth_index": d,
             "z_center_m": 2.125+.25*d, "x_m": x, "y_m": y,
             "log_qc": .2*i+np.sin(d/4)}
            for i, (x, y) in enumerate(xys) for d in range(24)]
    frame = pd.DataFrame(rows)
    return frame if labels else frame.drop(columns="log_qc")


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(719)
    network = ProfileCNP()
    metadata = {"model_id": "cnp_profile", "n_depth": 24,
                "hidden_width": 64, "xy_scale_m": 50.,
                "depth_indices": list(range(24)), "depth_positions": np.linspace(-1, 1, 24).tolist(),
                "target_mean": .5, "target_scale": 1.2}
    path = tmp_path/"model.pt"
    torch.save({"metadata": metadata, "state_dict": network.state_dict()}, path)
    return path


def test_set_order_and_query_row_order_do_not_change_predictions(checkpoint):
    model = NeuralProcessPredictor(artifact=checkpoint)
    observed = _rows([(0, 0), (20, 0), (0, 30), (40, 25), (10, 50)])
    query = _rows([(15, 10), (5, 45)], prefix="q", labels=False)
    model.condition(observed)
    expected = model.predict(query).frame.set_index(["location_id", "depth_index"]).sort_index()
    model.condition(observed.sample(frac=1, random_state=7))
    actual = model.predict(query.sample(frac=1, random_state=4)).frame.set_index(["location_id", "depth_index"]).sort_index()
    np.testing.assert_allclose(expected.pred_log_qc, actual.pred_log_qc, atol=1e-7)
    assert not any(c.startswith("p_low") or c.startswith("q05") for c in actual)


def test_query_labels_are_ignored_and_translation_keeps_predictions(checkpoint):
    model = NeuralProcessPredictor(artifact=checkpoint)
    observed = _rows([(0, 0), (20, 0), (0, 30)])
    query = _rows([(15, 10)], prefix="q", labels=False)
    model.condition(observed)
    expected = model.predict(query).frame.pred_log_qc.to_numpy()
    poisoned = query.assign(log_qc=1e10, qc_mpa=1e10)
    np.testing.assert_allclose(model.predict(poisoned).frame.pred_log_qc, expected)
    for frame in [observed, query]:
        frame.loc[:, "x_m"] += 1750000.
        frame.loc[:, "y_m"] += 5400000.
    model.condition(observed)
    np.testing.assert_allclose(model.predict(query).frame.pred_log_qc, expected, atol=1e-7)


def test_context_is_complete_and_metric_scale_is_retained(checkpoint):
    model = NeuralProcessPredictor(artifact=checkpoint)
    observed = _rows([(0, 0), (20, 0), (0, 30)])
    query = _rows([(15, 10)], prefix="q", labels=False)
    with pytest.raises(ValueError, match="all trained depths"):
        model.condition(observed.iloc[1:])
    model.condition(observed)
    first = model.predict(query).frame.pred_log_qc.to_numpy()
    observed[["x_m", "y_m"]] *= 10
    query[["x_m", "y_m"]] *= 10
    model.condition(observed)
    second = model.predict(query).frame.pred_log_qc.to_numpy()
    assert not np.allclose(first, second, atol=1e-7)


def test_source_roles_cannot_include_outer_test_or_overlap():
    train = pd.DataFrame([{"role": "train", "fold_id": 0, "site_id": "source",
                           "investigation_group": "g1", "episode_id": "a"}])
    validation = pd.DataFrame([{"role": "validation", "fold_id": 0, "site_id": "validation",
                                "investigation_group": "g2", "episode_id": "b"}])
    _validate_plans(train, validation)
    with pytest.raises(ValueError, match="outer test"):
        _validate_plans(train, validation.assign(role="test"))
    with pytest.raises(ValueError, match="sites overlap"):
        _validate_plans(train, validation.assign(site_id="source"))
    with pytest.raises(ValueError, match="groups overlap"):
        _validate_plans(train, validation.assign(investigation_group="g1"))
