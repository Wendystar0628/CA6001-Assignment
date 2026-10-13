"""Focused safeguards for feature leakage, log-space decoding and source weights."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cpt_budget.features import FEATURE_COLUMNS, build_features
from cpt_budget.models.distribution import gaussian_prediction_frame, thickness_summary
from cpt_budget.models.gp import GPPredictor, fit_gp_trend
from cpt_budget.models.idw import IDWPredictor
from cpt_budget.models.xgb import XGBPredictor, episode_training_data, fit_xgb


def grid(site, locations, offset=0.0):
    records = []
    for location, x, value in locations:
        for depth_index in range(4):
            log_qc = value + 0.1 * depth_index + offset
            records.append(dict(site_id=site, location_id=location, near_group_id=location,
                                depth_index=depth_index, z_center_m=2.125 + 0.25 * depth_index,
                                x_m=x, y_m=0.0, geometry_scale_m=10.0,
                                log_qc=log_qc, qc_mpa=np.exp(log_qc)))
    return pd.DataFrame(records)


def query(frame):
    return frame.drop(columns=["log_qc", "qc_mpa"])


def test_whole_location_and_coincident_exclusion_blocks_self_leakage():
    observed = grid("s", [("a", 0, 100), ("a_repeat", 0, 200), ("b", 10, 1), ("c", 20, 3)])
    targets = query(observed.loc[observed.location_id == "a"])
    features = build_features(observed, targets, leave_location_out=True)
    changed = observed.copy()
    changed.loc[changed.location_id.isin(["a", "a_repeat"]), "log_qc"] = -1000
    pd.testing.assert_frame_equal(features, build_features(changed, targets, leave_location_out=True))
    assert list(features) == FEATURE_COLUMNS
    assert features.iloc[0].support_count == 2
    assert features.iloc[0].neighbor_mean_log_qc == 2
    assert features.iloc[0].neighbor_std_log_qc == 1
    assert features.iloc[0].idw_log_qc == pytest.approx(1.4)
    assert np.isnan(features.iloc[0].neighbor_3_log_qc)
    assert features.iloc[0].neighbor_3_valid == 0


def test_feature_query_labels_never_used_and_sites_do_not_mix():
    observed = grid("s", [("a", 0, 1), ("b", 10, 3)])
    target = grid("s", [("q", 5, 99)])
    foreign = grid("other", [("foreign", 5, -1000)])
    baseline = build_features(observed, query(target))
    pd.testing.assert_frame_equal(baseline, build_features(pd.concat([observed, foreign]), target))
    assert baseline.iloc[0].support_count == 2
    solo = build_features(observed.iloc[:4], query(observed.iloc[:4]), leave_location_out=True)
    assert (solo.support_count == 0).all()
    assert solo.idw_log_qc.isna().all()


def test_native_gaussian_outputs_use_log_thresholds_and_median_qc():
    target = query(grid("s", [("q", 5, 0)]))
    frame = gaussian_prediction_frame(target, np.zeros(4), np.ones(4))
    assert (frame.pred_qc_mpa == 1).all()
    assert (frame.p_low_1 == 0.5).all()
    assert np.all(frame.p_low_0_5 < frame.p_low_1)
    assert np.all(frame.p_low_1 < frame.p_low_2)
    assert frame.q05_log.iloc[0] == pytest.approx(-1.64485362695)
    thickness = thickness_summary(frame)
    assert thickness.h_point_m.iloc[0] == 0
    assert thickness.h_expected_m.iloc[0] == 0.5


def test_gp_source_trend_is_site_equal_and_predictions_include_noise():
    small = grid("small", [("a", 0, 0)])
    large = grid("large", [("b", 0, 10), ("c", 10, 10), ("d", 20, 10)])
    hidden = grid("test", [("e", 0, 100)])
    trend = fit_gp_trend(SimpleNamespace(grid=pd.concat([small, large, hidden])), ["small", "large"])
    assert trend.iloc[0] == 5.0
    with pytest.raises(ValueError, match="source-training trend"):
        GPPredictor()
    model = GPPredictor({"gp": {"optimize": False}}, trend=trend)
    model.condition(large)
    frame = model.predict(query(large)).frame
    assert np.all(frame.q95_log > frame.q05_log)
    assert "log_qc" not in frame and "qc_mpa" not in frame
    idw = IDWPredictor()
    idw.condition(large)
    point = idw.predict(query(large)).frame
    assert np.allclose(point.pred_log_qc, large.log_qc)
    assert "q05_log" not in point and "p_low_1" not in point


class EpisodeDataset:
    def __init__(self):
        self.grid = pd.concat([grid(site, [(f"{site}{i}", i * 10, i * 0.2) for i in range(5)])
                               for site in ("a", "b", "v")], ignore_index=True)

    def observed(self, ids):
        return self.grid.loc[self.grid.location_id.isin(ids)].copy()

    def query(self, ids):
        return query(self.observed(ids))

    def truth(self, ids):
        return self.observed(ids)


def test_xgb_equal_site_weights_and_validation_stopped_artifact(tmp_path):
    dataset = EpisodeDataset()
    episodes = pd.DataFrame([
        dict(site_id="a", episode_id="a1", support_location_ids=["a0", "a1"], query_location_ids=["a2"], role="train", fold_id=0),
        dict(site_id="a", episode_id="a2", support_location_ids=["a0", "a2"], query_location_ids=["a1", "a3", "a4"], role="train", fold_id=0),
        dict(site_id="b", episode_id="b1", support_location_ids=["b0", "b1"], query_location_ids=["b2", "b3"], role="train", fold_id=0),
        dict(site_id="v", episode_id="v1", support_location_ids=["v0", "v1"], query_location_ids=["v2", "v3"], role="validation", fold_id=0),
    ])
    _, _, weights, metadata = episode_training_data(dataset, episodes.loc[episodes.role == "train"])
    totals = metadata.assign(weight=weights).groupby("site_id").weight.sum()
    assert totals.loc["a"] == pytest.approx(totals.loc["b"])
    artifact = fit_xgb(dataset, episodes, {"xgb": {"n_estimators": 6, "n_jobs": 1, "early_stopping_rounds": 2}}, tmp_path)
    model = XGBPredictor(artifact=artifact)
    model.condition(dataset.observed(["v0", "v1"]))
    predictions = model.predict(dataset.query(["v2"])).frame
    assert len(predictions) == 4 and np.isfinite(predictions.pred_log_qc).all()
    assert "p_low_1" not in predictions
