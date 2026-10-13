"""Targeted v2 risks: calibration leakage, class absence and joint attainment."""
import json

import numpy as np
import pandas as pd
import pytest

from cpt_budget.v2.risk import (
    apply_risk, crossfit_risk, evaluate_risk, fit_risk_calibrator, transform_probability,
)
from cpt_budget.v2.quality import (
    TARGET_VERSION, evaluate_joint_attainment, freeze_joint_targets, summarize_v2_quality,
)


def _predictions(native=True):
    rows = []
    for site in ["a", "b"]:
        for index, (p, qc) in enumerate(zip([.01, .2, .6, .99], [2., .5, 3., .3])):
            rows.append({"site_id": site, "investigation_group": "group_" + site,
                         "location_id": site, "depth_index": index,
                         "role": "evaluation", "split_role": "validation",
                         "model_id": "gp" if native else "idw", "fold_id": 0,
                         "truth_qc_mpa": qc, "p_low_1": p if native else np.nan,
                         "pred_qc_mpa": 1 / (p + .1), "pred_log_qc": np.log(1 / (p + .1))})
    return pd.DataFrame(rows)


def _config():
    return {"data": {"tau_mpa": 1., "depth_step_m": .25},
            "quality": {"reference_model": "gp", "reference_policy": "maximin", "reference_budget": 13,
                        "validation_seeds": [17, 29]},
            "replay": {"seeds": [17, 29, 43, 71, 101], "budgets": [5, 7, 9, 11, 13], "initial_locations": 5}}


def _reference():
    return pd.DataFrame([
        {"fold_id": 0, "site_id": site, "seed": seed, "model_id": "gp", "policy_id": "maximin",
         "budget_n": 13, "log_mae": e, "h_point_mae_m": h, "risk_recall": .8, "risk_fpr": .2,
         "recall": .5, "n_true_positive": 3, "n_true_negative": 10, "role": "validation"}
        for site, e, h in [("a", 1., 10.), ("b", 10., 1.)] for seed in [17, 29]])


def _curve(model, errors, role="test"):
    return pd.DataFrame([
        {"fold_id": 0, "site_id": "a", "seed": 17, "model_id": model, "policy_id": "maximin",
         "budget_n": budget, "log_mae": error, "h_point_mae_m": error,
         "risk_recall": .8, "risk_fpr": .2, "recall": .5,
         "n_true_positive": 3, "n_true_negative": 10, "role": role}
        for budget, error in zip([5, 7, 9, 11, 13], errors)])


def test_crossfit_never_uses_held_out_truth_and_rejects_non_validation():
    frame = _predictions()
    original, calibration = crossfit_risk(frame)
    changed = frame.copy()
    changed.loc[changed.site_id == "a", "truth_qc_mpa"] = .01
    second, _ = crossfit_risk(changed)
    np.testing.assert_allclose(original.loc[original.site_id == "a", "risk_probability"],
                               second.loc[second.site_id == "a", "risk_probability"])
    assert calibration["oof_calibrators"]["group_a"]["training_groups"] == ["group_b"]
    assert calibration["oof_calibrators"]["group_a"]["excluded_groups"] == ["group_a"]
    assert original.risk_calibration_mode.eq("leave_investigation_group_out").all()
    json.dumps(calibration, allow_nan=False)
    with pytest.raises(ValueError, match="validation"):
        fit_risk_calibrator(frame.assign(split_role="test"))
    with pytest.raises(ValueError, match="evaluation"):
        fit_risk_calibrator(frame.assign(role="candidate"))
    with pytest.raises(ValueError, match="two validation"):
        crossfit_risk(frame[frame.site_id == "a"])


def test_monotone_site_equal_fit_and_truth_free_application():
    frame = _predictions()
    frame.loc[frame.site_id == "a", "truth_qc_mpa"] = [2., 3., 1.5, .2]
    calibration = fit_risk_calibrator(frame)
    replicated = pd.concat([frame, *[frame[frame.site_id == "a"]] * 8], ignore_index=True)
    calibration_replicated = fit_risk_calibrator(replicated)
    queries = pd.DataFrame({"p_low_1": np.linspace(0, 1, 101)})
    first = apply_risk(queries, calibration)
    second = apply_risk(queries.assign(truth_qc_mpa=-999), calibration_replicated)
    np.testing.assert_allclose(first.risk_probability, second.risk_probability, atol=1e-7)
    assert np.diff(first.risk_probability).min() >= 0
    assert calibration["slope"] >= 0
    np.testing.assert_array_equal(first.p_low_1, queries.p_low_1)
    matrix = np.array([[.01, .1], [.5, .9]])
    transformed = transform_probability(matrix, calibration)
    assert transformed.shape == matrix.shape
    np.testing.assert_allclose(transformed.ravel(), apply_risk(pd.DataFrame({"p_low_1": matrix.ravel()}), calibration).risk_probability)


def test_point_score_all_nan_native_and_single_class_fallback_are_explicit():
    frame = _predictions(native=False)
    calibration = fit_risk_calibrator(frame.assign(truth_qc_mpa=3.))
    assert calibration["score_kind"] == "negative_point_log_prediction"
    assert calibration["fallback_reason"] == "single_class_validation"
    result = apply_risk(frame.drop(columns="truth_qc_mpa"), calibration)
    assert result.risk_probability.eq(0).all() and not result.risk_label.any()
    with pytest.raises(ValueError, match="native-probability"):
        transform_probability([.1], calibration)
    partial = _predictions()
    partial.loc[0, "p_low_1"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_risk_calibrator(partial)


def test_risk_metrics_preserve_undefined_class_semantics_and_point_prediction():
    predictions = _predictions().query("site_id == 'a'").copy()
    point_before = predictions.pred_qc_mpa.copy()
    predictions["risk_probability"], predictions["risk_label"] = .1, False
    truth = predictions[["site_id", "location_id", "depth_index"]].assign(qc_mpa=2.)
    metrics = evaluate_risk(predictions, truth, _config())
    assert metrics["risk_recall"] is None and metrics["risk_precision"] is None
    assert metrics["risk_recall_reason"] == "no_positive_labels"
    assert metrics["risk_fpr"] == 0 and metrics["risk_brier"] == pytest.approx(.01)
    assert metrics["h_risk_expected_mae_m"] == pytest.approx(.1)
    assert metrics["risk_n_true_positive"] == 0 and metrics["risk_n_true_negative"] == 4
    pd.testing.assert_series_equal(predictions.pred_qc_mpa, point_before)
    assert "recall" not in metrics and "h_point_mae_m" not in metrics


def test_joint_target_checks_actual_joint_fraction_and_preserves_v1_medians():
    targets = freeze_joint_targets(_reference(), _config())
    target = targets["folds"]["0"]
    assert targets["target_version"] == TARGET_VERSION
    assert target["candidate_history"][0]["reference_joint_pass_rate"] == 0
    assert target["selected_q"] == .6
    assert target["E_star"] == 10 and target["H_star"] == 10
    assert target["reference_joint_pass_rate"] == 1.
    assert target["n_reference_instances"] == target["n_reference_instances_total"] == 4
    assert target["reference_counts_by_site"]["a"] == {"passed": 2, "total": 2}
    assert targets["v1_marginal_targets"]["folds"]["0"]["E_star"] == 5.5
    with pytest.raises(ValueError, match="validation"):
        freeze_joint_targets(_reference().assign(role="test"), _config())
    with pytest.raises(ValueError, match="out-of-group"):
        freeze_joint_targets(_reference().assign(risk_calibration_mode="in_sample"), _config())


def test_quality_complete_budgets_and_full_cost_denominators():
    targets = freeze_joint_targets(_reference(), _config())
    metrics = pd.concat([_curve("gp", [12., 9., 12., 9., 9.]),
                         _curve("idw", [9., 9., 9., 9., 12.]),
                         _curve("partial", [9., 9., 9., 9.]),
                         _curve("better", [12., 12., 9., 9., 9.])], ignore_index=True)
    results = summarize_v2_quality(metrics, targets, _config())
    attainment = results["attainment"].set_index("model_id")
    assert attainment.loc["gp", "n_first_hit"] == 7 and attainment.loc["gp", "n_stable"] == 11
    assert attainment.loc["idw", "status"] == "not_reached_by_budget"
    assert attainment.loc["partial", "status"] == "incomplete_budget_curve"
    assert pd.isna(attainment.loc["partial", "n_stable"])
    assert pd.isna(attainment.loc["partial", "aubc_log_mae"])
    costs = results["costs"].set_index("model_id")
    assert costs.loc["partial", "pair_status"] == "incomplete_budget_curve"
    assert pd.isna(costs.loc["partial", "saved_cpt"])
    assert costs.loc["better", "saved_cpt"] == 2
    assert costs.loc["idw", "n_all_instances"] == 1
    assert costs.loc["idw", "n_both_reached"] == 0
    assert results["aggregate"].metric.eq("risk_recall").any()
    assert results["comparisons"].metric.eq("risk_fpr").any()
    for table in results.values():
        assert table.target_version.eq(TARGET_VERSION).all()
    with pytest.raises(ValueError, match="target versions"):
        evaluate_joint_attainment(metrics, targets["v1_marginal_targets"], _config())


def test_absent_class_does_not_invent_threshold_or_pass_applicable_missing_metric():
    reference = _reference().assign(n_true_positive=0, risk_recall=np.nan, recall=np.nan)
    targets = freeze_joint_targets(reference, _config())
    assert targets["folds"]["0"]["R_star"] is None
    negative = _curve("gp", [1.] * 5).assign(n_true_positive=0, risk_recall=np.nan)
    assert evaluate_joint_attainment(negative, targets, _config()).iloc[0].status == "reached"
    positive = _curve("gp", [1.] * 5)
    assert evaluate_joint_attainment(positive, targets, _config()).iloc[0].status == "quality_target_incomplete"
    known_target = freeze_joint_targets(_reference(), _config())
    positive.loc[0, "risk_recall"] = np.nan
    result = evaluate_joint_attainment(positive, known_target, _config()).iloc[0]
    assert result.status == "incomplete_quality_metrics" and pd.isna(result.n_stable)


def test_summary_does_not_mix_validation_and_test_instances():
    targets = freeze_joint_targets(_reference(), _config())
    both = pd.concat([_curve("gp", [1.] * 5, role="validation"), _curve("gp", [2.] * 5)])
    result = summarize_v2_quality(both, targets, _config())
    assert set(result["attainment"].summary_role) == {"validation", "test"}
    assert len(result["attainment"]) == 2
