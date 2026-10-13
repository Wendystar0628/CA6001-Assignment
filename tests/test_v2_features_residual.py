"""Focused safeguards for spatial representation and guarded residual fallback."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cpt_budget.features import FEATURE_COLUMNS, build_features
from cpt_budget.models.idw import IDWPredictor
from cpt_budget.types import PreparedDataset
from cpt_budget.v2.features import FEATURE_COLUMNS_V2, build_features_v2
from cpt_budget.v2.residual import ResidualPredictor, _choose_alpha, _episode_data, fit_residual


def _grid(site, locations):
    rows = []
    for location, x, value in locations:
        for depth in range(24):
            z = 2.125 + 0.25 * depth
            log_qc = value + 0.2 * depth + 0.15 * np.sin(depth / 2 + x / 20)
            rows.append(dict(site_id=site, location_id=location, near_group_id=location,
                             x_m=x, y_m=0.0, geometry_scale_m=10.0, depth_index=depth,
                             z_center_m=z, log_qc=log_qc, qc_mpa=np.exp(log_qc)))
    return pd.DataFrame(rows)


def _query(rows):
    return rows.drop(columns=["log_qc", "qc_mpa"])


def test_all_added_features_obey_whole_location_exclusion_and_ignore_query_truth():
    observed = _grid("s", [("a", 0, 10), ("repeat", 0, 20), ("b", 10, -2), ("c", 40, 1)])
    query = _query(observed.loc[observed.location_id == "a"])
    before = build_features_v2(observed, query, leave_location_out=True)
    changed = observed.copy()
    changed.loc[changed.location_id.isin(["a", "repeat"]), "log_qc"] = -100
    changed_query = query.assign(log_qc=999., qc_mpa=999.)
    after = build_features_v2(changed, changed_query, leave_location_out=True)
    pd.testing.assert_frame_equal(before, after)
    assert list(before) == FEATURE_COLUMNS_V2
    pd.testing.assert_frame_equal(before[FEATURE_COLUMNS], build_features(observed, query, leave_location_out=True))
    assert before.idw_above_log_qc.iloc[0] != before.idw_above_log_qc.iloc[0]
    assert np.isfinite(before.idw_vertical_gradient).all()


def test_physical_kernels_change_on_horizontal_rescaling_and_vertical_slope_is_physical():
    observed = _grid("s", [("a", 0, -2), ("b", 20, 1), ("c", 70, 3)])
    query = _query(_grid("s", [("q", 7, 99)]))
    before = build_features_v2(observed, query)
    scaled_observed, scaled_query = observed.copy(), query.copy()
    scaled_observed[["x_m", "y_m", "geometry_scale_m"]] *= 100
    scaled_query[["x_m", "y_m", "geometry_scale_m"]] *= 100
    after = build_features_v2(scaled_observed, scaled_query)
    np.testing.assert_allclose(before.idw_log_qc, after.idw_log_qc)
    assert np.max(abs(before.kernel_50m_log_qc - after.kernel_50m_log_qc)) > 0.1
    assert np.max(abs(before.kernel_50m_low_fraction - after.kernel_50m_low_fraction)) > 0.1
    assert before.idw_vertical_gradient.iloc[5] == pytest.approx(
        (before.idw_below_log_qc.iloc[5] - before.idw_above_log_qc.iloc[5]) / 0.5)


def test_guardrails_reject_lower_error_with_lost_recall_and_can_select_idw():
    rows = [dict(alpha=0., log_mae=0.5, h_point_mae_m=0.4, recall=0.3, false_positive_rate=0.1),
            dict(alpha=0.25, log_mae=0.45, h_point_mae_m=0.41, recall=0.27, false_positive_rate=0.1),
            dict(alpha=0.5, log_mae=0.48, h_point_mae_m=0.43, recall=0.29, false_positive_rate=0.11),
            dict(alpha=1., log_mae=0.4, h_point_mae_m=0.5, recall=0.35, false_positive_rate=0.1)]
    selected, reason = _choose_alpha(rows)
    assert selected["alpha"] == 0.5 and reason is None
    assert rows[1]["guardrail_failures"] == ["recall_drop_gt_0.02"]
    rows[2]["log_mae"] = 0.6
    selected, reason = _choose_alpha(rows)
    assert selected["alpha"] == 0 and reason == "no_eligible_nonzero_candidate_improved_log_mae"


def test_residual_fit_roundtrip_and_zero_alpha_exactly_preserves_idw(tmp_path):
    frame = pd.concat([_grid(site, [(f"{site}{i}", i * 12., i * .15 - 2.) for i in range(7)])
                       for site in ("a", "b", "v")], ignore_index=True)
    profiles = frame.drop_duplicates("location_id").drop(columns=["log_qc", "qc_mpa", "depth_index", "z_center_m"])
    sites = pd.DataFrame([dict(site_id=site, investigation_group=site) for site in ("a", "b", "v")])
    dataset = PreparedDataset(profiles, frame, sites, Path(tmp_path))
    records = [dict(site_id=site, episode_id=f"{site}0", fold_id=0,
                    support_location_ids=[f"{site}{i}" for i in range(5)],
                    query_location_ids=[f"{site}{i}" for i in (5, 6)], role=role)
               for site, role in (("a", "train"), ("b", "train"), ("v", "validation"))]
    train, valid = pd.DataFrame(records[:2]), pd.DataFrame(records[2:])
    config = {"v2": {"residual": {"n_estimators": 4, "n_jobs": 1, "early_stopping_rounds": 2}}}
    _, _, weights, _, meta = _episode_data(dataset, train, config)
    totals = meta.assign(weight=weights).groupby("site_id").weight.sum()
    assert totals.a == pytest.approx(totals.b)
    result = fit_residual(dataset, train, valid, config, tmp_path)
    fitted = ResidualPredictor(config, artifact=Path(result["artifact_path"]))
    support = dataset.observed([f"v{i}" for i in range(5)])
    query = dataset.query(["v5", "v6"])
    fitted.condition(support)
    output = fitted.predict(query).frame
    assert len(output) == len(query) and "p_low_1" not in output and "q05_log" not in output
    result["alpha"] = 0.
    fallback_path = tmp_path / "fallback.json"
    fallback_path.write_text(json.dumps(result), encoding="utf-8")
    fallback = ResidualPredictor(config, artifact=fallback_path)
    fallback.condition(support)
    idw = IDWPredictor(config)
    idw.condition(support)
    pd.testing.assert_frame_equal(fallback.predict(query).frame, idw.predict(query).frame)
    with pytest.raises(ValueError, match="validation-role"):
        fit_residual(dataset, train, valid.assign(role="test"), config, tmp_path)
