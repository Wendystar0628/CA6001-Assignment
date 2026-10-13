"""Evaluation on a fixed holdout, with explicit undefined-label semantics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def probability_column(tau: float) -> str:
    return "p_low_" + format(float(tau), "g").replace(".", "_")


def join_evaluation(predictions, evaluation_truth: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.frame if hasattr(predictions, "frame") else predictions
    keys = ["location_id", "depth_index"]
    if "site_id" in frame and "site_id" in evaluation_truth:
        keys.insert(0, "site_id")
    truth = evaluation_truth.copy()
    if "truth_qc_mpa" not in truth:
        truth["truth_qc_mpa"] = truth.qc_mpa
    if "truth_log_qc" not in truth:
        truth["truth_log_qc"] = truth.log_qc if "log_qc" in truth else np.log(truth.truth_qc_mpa)
    required = keys + ["truth_qc_mpa", "truth_log_qc"]
    pred = frame.drop(columns=["truth_qc_mpa", "truth_log_qc", "qc_mpa", "log_qc"], errors="ignore")
    joined = pred.merge(truth[required], on=keys, how="outer", validate="one_to_one", indicator=True)
    if not (joined._merge == "both").all() or joined.empty:
        raise ValueError("Evaluation predictions must match every fixed holdout cell exactly")
    joined = joined.drop(columns="_merge")
    values = joined[["pred_log_qc", "pred_qc_mpa", "truth_log_qc", "truth_qc_mpa"]].to_numpy(float)
    if not np.isfinite(values).all() or (joined.truth_qc_mpa <= 0).any() or (joined.pred_qc_mpa <= 0).any():
        raise ValueError("Evaluation requires finite, positive qc and finite log predictions")
    return joined


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    labels, scores = labels[order].astype(int), scores[order]
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    tp = np.cumsum(labels)[ends]
    precision = tp / (ends + 1)
    return float(np.sum(np.diff(np.r_[0, tp]) / labels.sum() * precision))


def evaluate(predictions, evaluation_truth, config) -> dict:
    frame = join_evaluation(predictions, evaluation_truth)
    tau = float(config.get("data", {}).get("tau_mpa", 1.0))
    step = float(config.get("data", {}).get("depth_step_m", 0.25))
    if tau <= 0 or step <= 0:
        raise ValueError("tau_mpa and depth_step_m must be positive")
    group_keys = ["location_id"]
    if "site_id" in frame:
        group_keys.insert(0, "site_id")
    frame["abs_log_error"] = abs(frame.pred_log_qc - frame.truth_log_qc)
    frame["sq_log_error"] = (frame.pred_log_qc - frame.truth_log_qc) ** 2
    frame["abs_qc_error"] = abs(frame.pred_qc_mpa - frame.truth_qc_mpa)
    labels = (frame.truth_qc_mpa.to_numpy() < tau)
    pred_labels = (frame.pred_qc_mpa.to_numpy() < tau)
    frame["true_low"] = labels
    frame["pred_low"] = pred_labels
    by_location = frame.groupby(group_keys, sort=False)
    errors = by_location[["abs_log_error", "sq_log_error", "abs_qc_error"]].mean()
    thickness = by_location[["true_low", "pred_low"]].sum() * step
    tp, fp = int((labels & pred_labels).sum()), int((~labels & pred_labels).sum())
    fn, tn = int((labels & ~pred_labels).sum()), int((~labels & ~pred_labels).sum())
    reasons = {}
    result = {
        "log_mae": float(errors.abs_log_error.mean()),
        "log_rmse": float(np.sqrt(errors.sq_log_error).mean()),
        "qc_mae_mpa": float(errors.abs_qc_error.mean()),
        "h_point_mae_m": float(abs(thickness.pred_low - thickness.true_low).mean()),
        "n_eval_locations": len(errors), "n_eval_cells": len(frame),
        "n_true_positive": tp + fn, "n_true_negative": tn + fp,
        "n_predicted_positive": tp + fp, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "has_positive_labels": bool(tp + fn), "tau_mpa": tau,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
    }
    if not tp + fp:
        reasons["precision"] = "no_predicted_positive_labels"
    if not tp + fn:
        reasons["recall"] = "no_positive_labels"
        reasons["average_precision"] = "no_positive_labels"
    if not 2 * tp + fp + fn:
        reasons["f1"] = "true_and_predicted_all_negative"
    if not fp + tn:
        reasons["false_positive_rate"] = "no_negative_labels"
    prob_col = probability_column(tau)
    if prob_col in frame:
        p = frame[prob_col].to_numpy(float)
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError("Low-qc probabilities must be finite and in [0, 1]")
        h_expected = by_location[prob_col].sum() * step
        result["h_expected_mae_m"] = float(abs(h_expected - thickness.true_low).mean())
        ap_scores = p
        result["ap_score_kind"] = "low_qc_probability"
    else:
        result["h_expected_mae_m"] = None
        reasons["h_expected_mae_m"] = "predictive_probability_unavailable"
        ap_scores = -frame.pred_log_qc.to_numpy(float)
        result["ap_score_kind"] = "negative_point_log_prediction"
    result["average_precision"] = _average_precision(labels, ap_scores) if labels.any() else None
    for level, low, high in [(50, "q25_log", "q75_log"), (80, "q10_log", "q90_log"), (90, "q05_log", "q95_log")]:
        coverage_key, width_key = f"coverage_{level}", f"width_{level}_log"
        if low in frame and high in frame:
            bounds = frame[[low, high]].to_numpy(float)
            if not np.isfinite(bounds).all() or (bounds[:, 0] > bounds[:, 1]).any():
                raise ValueError(f"Invalid {level}% predictive interval")
            frame["covered"] = (frame.truth_log_qc >= frame[low]) & (frame.truth_log_qc <= frame[high])
            frame["interval_width"] = frame[high] - frame[low]
            result[coverage_key] = float(frame.groupby(group_keys).covered.mean().mean())
            result[width_key] = float(frame.groupby(group_keys).interval_width.mean().mean())
        else:
            result[coverage_key] = result[width_key] = None
            reasons[coverage_key] = reasons[width_key] = "predictive_interval_unavailable"
    for name in ("precision", "recall", "f1", "average_precision", "false_positive_rate", "h_expected_mae_m", "coverage_50", "coverage_80", "coverage_90"):
        result[f"{name}_reason"] = reasons.get(name)
    # A JSON string survives CSV and Parquet consistently and keeps all reasons.
    import json
    result["missing_reasons"] = json.dumps(reasons, sort_keys=True)
    return result
