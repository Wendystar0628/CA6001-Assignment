"""Jointly attainable validation quality targets, kept separate from v1 targets."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..economics import CURVE_KEYS, METRIC_COLUMNS, _costs, freeze_quality_targets


TARGET_VERSION = "joint_v2"
RISK_METRICS = ["risk_recall", "risk_precision", "risk_fpr", "risk_brier", "h_risk_expected_mae_m"]
ALL_METRICS = list(dict.fromkeys(METRIC_COLUMNS + RISK_METRICS))


def _quality_settings(config: dict) -> dict:
    """Resolve the public v2 options without silently discarding invalid edits."""
    settings = config.get("v2", {}).get("quality", {})
    if not isinstance(settings, dict):
        raise ValueError("v2.quality must be a mapping")
    legacy_version = config.get("quality", {}).get("target_version")
    version = settings.get("target_version", legacy_version if legacy_version is not None else TARGET_VERSION)
    if not isinstance(version, str) or not version.strip() or version != version.strip() or version == "v1_marginal_medians":
        raise ValueError("v2.quality.target_version must be a nonempty, distinct joint-target version")
    if legacy_version is not None and legacy_version != version:
        raise ValueError("quality.target_version and v2.quality.target_version must agree")
    grid = settings.get("q_grid", [.5, .6, .7, .8, .9, 1.0])
    if not isinstance(grid, (list, tuple)) or not grid or any(isinstance(q, bool) for q in grid):
        raise ValueError("v2.quality.q_grid must be a nonempty list of increasing quantiles")
    try:
        grid = np.asarray(grid, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("v2.quality.q_grid must contain numeric quantiles") from error
    if grid.ndim != 1 or not np.isfinite(grid).all() or ((grid < 0) | (grid > 1)).any() or (np.diff(grid) <= 0).any():
        raise ValueError("v2.quality.q_grid must be strictly increasing, finite, and within [0, 1]")
    rate = settings.get("min_joint_pass_rate", .5)
    if isinstance(rate, bool):
        raise ValueError("v2.quality.min_joint_pass_rate must lie in (0, 1]")
    try:
        rate = float(rate)
    except (TypeError, ValueError) as error:
        raise ValueError("v2.quality.min_joint_pass_rate must be numeric") from error
    if not np.isfinite(rate) or not 0 < rate <= 1:
        raise ValueError("v2.quality.min_joint_pass_rate must lie in (0, 1]")
    return {"target_version": version, "q_grid": grid.tolist(), "min_joint_pass_rate": rate}


def _class_masks(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    columns = ["n_true_positive", "n_true_negative"]
    if not set(columns).issubset(frame):
        raise ValueError("Joint quality requires true positive/negative counts to distinguish undefined from missing")
    counts = frame[columns].to_numpy(float)
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts.sum(axis=1) <= 0).any():
        raise ValueError("Joint quality requires finite, nonnegative counts and a nonempty holdout")
    return counts[:, 0] > 0, counts[:, 1] > 0


def _site_weights(frame: pd.DataFrame) -> np.ndarray:
    weights = 1 / frame.groupby("site_id").site_id.transform("size").to_numpy(float)
    return weights / weights.sum()


def _weighted_quantile(values, weights, quantile: float) -> float | None:
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return None
    values, weights = values[valid], weights[valid]
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) / weights.sum()
    index = min(int(np.searchsorted(cumulative, quantile, side="left")), len(values) - 1)
    return float(values[index])


def _joint_hits(frame: pd.DataFrame, target: dict) -> tuple[np.ndarray, np.ndarray]:
    positives, negatives = _class_masks(frame)
    errors = frame.log_mae.to_numpy(float)
    thickness = frame.h_point_mae_m.to_numpy(float)
    recall, fpr = frame.risk_recall.to_numpy(float), frame.risk_fpr.to_numpy(float)
    valid = np.isfinite(errors) & np.isfinite(thickness)
    valid &= (~positives | np.isfinite(recall)) & (~negatives | np.isfinite(fpr))
    hit = valid & (errors <= target["E_star"]) & (thickness <= target["H_star"])
    if target.get("R_star") is None:
        valid &= ~positives
        hit &= ~positives
    else:
        hit &= ~positives | (recall >= target["R_star"])
    if target.get("F_star") is None:
        valid &= ~negatives
        hit &= ~negatives
    else:
        hit &= ~negatives | (fpr <= target["F_star"])
    return hit, valid


def freeze_joint_targets(validation_metrics: pd.DataFrame, config: dict) -> dict:
    """Choose the smallest shared q attaining the configured joint pass rate.

    Weighted empirical inverse-CDF quantiles are used on complete site/seed
    reference instances. Seeds within a site have equal weight and all sites
    have equal total weight. Undefined-class conditions are checked only when
    that class exists; other missing metrics never count as success.
    """
    settings = _quality_settings(config)
    metrics = validation_metrics.copy()
    if "role" not in metrics or not metrics.role.eq("validation").all():
        raise ValueError("Joint quality freezing accepts explicit role='validation' only")
    if "risk_calibration_mode" in metrics and not metrics.risk_calibration_mode.eq("leave_investigation_group_out").all():
        raise ValueError("Joint validation quality must use out-of-group calibrated risk metrics")
    required = {"fold_id", "site_id", "seed", "model_id", "policy_id", "budget_n",
                "log_mae", "h_point_mae_m", "risk_recall", "risk_fpr", "recall"}
    if not required.issubset(metrics):
        raise ValueError(f"Missing quality metrics: {sorted(required - set(metrics))}")
    quality = config.get("quality", {})
    reference = {"model_id": quality.get("reference_model", "gp"),
                 "policy_id": quality.get("reference_policy", "maximin"),
                 "budget_n": int(quality.get("reference_budget", 13))}
    seeds = set(quality.get("validation_seeds", config.get("replay", {}).get("seeds", [])))
    v1_config = {**config, "replay": {**config.get("replay", {}), "seeds": sorted(seeds)}}
    output = {
        "target_version": settings["target_version"], "source_role": "validation", "reference": reference,
        "quantile_method": "weighted_empirical_inverse_cdf",
        "weighting": "equal_sites_then_equal_seeds_within_site",
        "joint_reference_pass_rate_required": settings["min_joint_pass_rate"],
        "quantile_candidates": settings["q_grid"], "folds": {},
        "v1_marginal_targets": {**freeze_quality_targets(metrics, v1_config),
                                "target_version": "v1_marginal_medians"},
    }
    for fold in sorted(metrics.fold_id.unique()):
        source = metrics[(metrics.fold_id == fold) & (metrics.model_id == reference["model_id"]) &
                         (metrics.policy_id == reference["policy_id"]) &
                         (metrics.budget_n == reference["budget_n"])].copy()
        if source.duplicated(["site_id", "seed"]).any():
            raise ValueError("Duplicate validation site/seed reference instances")
        excluded, keep = {}, []
        for site, group in source.groupby("site_id", sort=True):
            positive, negative = _class_masks(group)
            required_finite = np.isfinite(group[["log_mae", "h_point_mae_m"]].to_numpy(float)).all()
            required_finite &= np.isfinite(group.loc[positive, "risk_recall"].to_numpy(float)).all()
            required_finite &= np.isfinite(group.loc[negative, "risk_fpr"].to_numpy(float)).all()
            if seeds and set(group.seed) != seeds:
                excluded[str(site)] = "incomplete_reference_seeds"
            elif not required_finite:
                excluded[str(site)] = "missing_applicable_reference_metric"
            else:
                keep.append(group)
        complete = pd.concat(keep, ignore_index=True) if keep else source.iloc[:0]
        target = {
            "fold_id": int(fold), "target_version": settings["target_version"], "reference": reference,
            "complete": False, "status": "quality_target_incomplete",
            "E_star": None, "H_star": None, "R_star": None, "F_star": None, "selected_q": None,
            "n_reference_instances_total": len(source), "n_reference_sites_total": int(source.site_id.nunique()),
            "n_reference_instances": len(complete), "n_reference_sites": int(complete.site_id.nunique()),
            "excluded_sites": excluded, "validation_sites": sorted(complete.site_id.astype(str).unique()),
            "expected_seeds": sorted(seeds), "candidate_history": [], "reference_pass_rate_by_site": {},
            "missing_reasons": [],
        }
        if complete.empty:
            target["missing_reasons"] = ["no_complete_validation_reference"]
            output["folds"][str(fold)] = target
            continue
        weights = _site_weights(complete)
        positive, negative = _class_masks(complete)
        target["n_reference_positive_instances"] = int(positive.sum())
        target["n_reference_negative_instances"] = int(negative.sum())
        for q in output["quantile_candidates"]:
            candidate = {
                "E_star": _weighted_quantile(complete.log_mae, weights, q),
                "H_star": _weighted_quantile(complete.h_point_mae_m, weights, q),
                "R_star": _weighted_quantile(complete.risk_recall.where(positive), weights, 1 - q),
                "F_star": _weighted_quantile(complete.risk_fpr.where(negative), weights, q),
            }
            hits, _ = _joint_hits(complete, candidate)
            pass_rate = float(np.dot(weights, hits))
            per_site = complete.assign(_hit=hits).groupby("site_id")._hit.agg(["mean", "sum", "size"])
            target["candidate_history"].append({"q": q, **candidate, "reference_joint_pass_rate": pass_rate,
                                                 "n_reference_passed": int(hits.sum())})
            if pass_rate >= settings["min_joint_pass_rate"] - 1e-12:
                target.update(candidate)
                target.update(selected_q=q, complete=True, status="complete",
                              reference_joint_pass_rate=pass_rate, n_reference_passed=int(hits.sum()),
                              reference_joint_pass_rate_unweighted=float(hits.mean()),
                              reference_pass_rate_by_site={str(site): float(row["mean"]) for site, row in per_site.iterrows()},
                              reference_counts_by_site={str(site): {"passed": int(row["sum"]), "total": int(row["size"])}
                                                        for site, row in per_site.iterrows()})
                break
        if not target["complete"]:
            target["missing_reasons"] = ["joint_reference_pass_rate_not_attained"]
        if not positive.any():
            target["missing_reasons"].append("no_positive_validation_reference")
        if not negative.any():
            target["missing_reasons"].append("no_negative_validation_reference")
        output["folds"][str(fold)] = target
    return output


def _target_for(targets, fold):
    folds = targets.get("folds", {})
    return folds.get(str(fold), folds.get(fold, {}))


def evaluate_joint_attainment(metrics: pd.DataFrame, targets: dict, config: dict) -> pd.DataFrame:
    """Full-budget stable attainment with explicit target version and censoring."""
    settings = _quality_settings(config)
    if targets.get("target_version") != settings["target_version"]:
        raise ValueError("Joint attainment requires v2 joint targets; do not mix target versions")
    if "role" in metrics and metrics.role.nunique(dropna=False) > 1:
        raise ValueError("Summarize validation and test attainment separately")
    expected = list(map(int, config.get("replay", {}).get("budgets", [5, 7, 9, 11, 13])))
    rows = []
    for key, group in metrics.groupby(CURVE_KEYS, sort=True, dropna=False):
        group = group.sort_values("budget_n")
        if group.budget_n.duplicated().any():
            raise ValueError("Duplicate budgets within a replay curve")
        budgets = group.budget_n.to_numpy(int)
        row = dict(zip(CURVE_KEYS, key))
        row.update(target_version=settings["target_version"], last_budget=int(budgets[-1]), n_budgets=len(group),
                   n_expected_budgets=len(expected), n_first_hit=None, n_stable=None,
                   budgets=json.dumps(budgets.tolist()), budget_curve_complete=budgets.tolist() == expected,
                   missing_budgets=json.dumps([n for n in expected if n not in budgets]),
                   n_first_hit_scope="observed_budget_points")
        for metric, destination in [("log_mae", "aubc_log_mae"), ("h_point_mae_m", "aubc_h_point_mae"),
                                    ("risk_recall", "aubc_risk_recall"), ("risk_fpr", "aubc_risk_fpr")]:
            values = group[metric].to_numpy(float)
            reason = ("incomplete_budget_curve" if not row["budget_curve_complete"] else
                      "fewer_than_two_budgets" if len(group) < 2 else
                      "missing_metric_at_budget" if not np.isfinite(values).all() else None)
            row[destination] = (float(np.sum((values[:-1] + values[1:]) * np.diff(budgets) / 2) /
                                      (budgets[-1] - budgets[0])) if reason is None else None)
            row[f"{destination}_reason"] = reason
        target = _target_for(targets, row["fold_id"])
        positives, negatives = _class_masks(group)
        available = bool(target.get("complete")) and all(target.get(c) is not None for c in ["E_star", "H_star"])
        available &= not positives.any() or target.get("R_star") is not None
        available &= not negatives.any() or target.get("F_star") is not None
        if not available:
            row.update(status="quality_target_incomplete", hit_sequence=None,
                       attainment_reason="validation_quality_target_unavailable_for_applicable_condition")
        else:
            hits, valid = _joint_hits(group, target)
            row["hit_sequence"] = json.dumps(hits.tolist())
            row["n_valid_quality_budgets"] = int(valid.sum())
            if hits.any():
                row["n_first_hit"] = int(budgets[np.flatnonzero(hits)[0]])
            if not row["budget_curve_complete"]:
                row.update(status="incomplete_budget_curve", attainment_reason="configured_budget_points_not_fully_observed")
            elif not valid.all():
                row.update(status="incomplete_quality_metrics", attainment_reason="missing_quality_metric_at_budget")
            else:
                stable = np.logical_and.accumulate(hits[::-1])[::-1]
                if stable.any():
                    row["n_stable"] = int(budgets[np.flatnonzero(stable)[0]])
                row.update(status="reached" if row["n_stable"] is not None else "not_reached_by_budget",
                           attainment_reason=None)
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        grouping = result.groupby(["model_id", "policy_id"])
        result["n_all_instances"] = grouping.site_id.transform("size")
        result["n_complete_budget_instances"] = grouping.budget_curve_complete.transform("sum")
        result["n_reached_instances"] = grouping.status.transform(lambda s: s.eq("reached").sum())
    return result


def _aggregate(metrics, attainment):
    tables = []
    for frame, columns, budget_keys in [(metrics, ALL_METRICS, ["budget_n"]),
            (attainment, ["aubc_log_mae", "aubc_h_point_mae", "aubc_risk_recall", "aubc_risk_fpr"], [])]:
        for metric in columns:
            if metric not in frame:
                continue
            keys = ["model_id", "policy_id", *budget_keys]
            sites = frame.groupby(keys + ["fold_id", "site_id"], as_index=False).agg(
                value=(metric, "mean"), n_valid_seeds=(metric, "count"), n_total_seeds=(metric, "size"))
            sites["metric"], sites["level"], sites["n_total_sites"] = metric, "site", 1
            sites["n_sites"] = sites.value.notna().astype(int)
            tables.append(sites)
            macro = sites.groupby(keys, as_index=False).agg(
                value=("value", "mean"), n_sites=("value", "count"), n_total_sites=("value", "size"),
                n_valid_seeds=("n_valid_seeds", "sum"), n_total_seeds=("n_total_seeds", "sum"))
            macro["metric"], macro["level"] = metric, "macro"
            tables.append(macro)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _comparisons(metrics, config):
    quality = config.get("quality", {})
    reference = (quality.get("reference_model", "gp"), quality.get("reference_policy", "maximin"))
    pair_keys = ["fold_id", "site_id", "seed", "budget_n"]
    columns = [c for c in ALL_METRICS if c in metrics]
    frames = []
    for (model, policy), method in metrics.groupby(["model_id", "policy_id"]):
        references = [(*reference, "reference_baseline")]
        if policy != "maximin" and (model, "maximin") != reference:
            references.append((model, "maximin", "strategy_effect"))
        for reference_model, reference_policy, scope in references:
            if (model, policy) == (reference_model, reference_policy):
                continue
            baseline = metrics[(metrics.model_id == reference_model) & (metrics.policy_id == reference_policy)]
            paired = method[pair_keys + columns].merge(baseline[pair_keys + columns], on=pair_keys,
                        how="outer", suffixes=("_method", "_reference"), validate="one_to_one")
            for metric in columns:
                seed = paired[pair_keys].copy()
                seed["method_value"], seed["reference_value"] = paired[f"{metric}_method"], paired[f"{metric}_reference"]
                valid = seed[["method_value", "reference_value"]].notna().all(axis=1)
                seed["difference"] = seed.method_value - seed.reference_value
                seed["metric"], seed["model_id"], seed["policy_id"] = metric, model, policy
                seed["reference_model"], seed["reference_policy"] = reference_model, reference_policy
                seed["comparison_scope"], seed["level"] = scope, "seed"
                seed["n_pairs"], seed["n_total_pairs"], seed["n_sites"] = valid.astype(int), 1, valid.astype(int)
                seed["missing_reason"] = np.where(valid, None, "paired_metric_or_instance_unavailable")
                frames.append(seed)
                keys = ["model_id", "policy_id", "reference_model", "reference_policy", "comparison_scope", "metric", "budget_n"]
                # Identical validity masks keep the means on the same paired instances.
                paired_seed = seed.copy()
                paired_seed.loc[~valid, ["method_value", "reference_value"]] = np.nan
                sites = paired_seed.groupby(keys + ["fold_id", "site_id"], as_index=False).agg(
                    method_value=("method_value", "mean"), reference_value=("reference_value", "mean"),
                    difference=("difference", "mean"), n_pairs=("n_pairs", "sum"), n_total_pairs=("n_total_pairs", "sum"))
                sites["level"], sites["n_sites"] = "site", sites.difference.notna().astype(int)
                frames.append(sites)
                macro = sites.groupby(keys, as_index=False).agg(
                    method_value=("method_value", "mean"), reference_value=("reference_value", "mean"),
                    difference=("difference", "mean"), n_pairs=("n_pairs", "sum"),
                    n_total_pairs=("n_total_pairs", "sum"), n_sites=("difference", "count"), n_total_sites=("difference", "size"))
                macro["level"] = "macro"
                frames.append(macro)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not result.empty:
        result["delta"] = result.difference
    return result


def summarize_v2_quality(metrics: pd.DataFrame, targets: dict, config: dict) -> dict[str, pd.DataFrame]:
    """Keep roles separate and expose all paired instances, including failures."""
    settings = _quality_settings(config)
    names = ["attainment", "comparisons", "calibration", "costs", "aggregate"]
    results = {name: [] for name in names}
    if metrics.empty:
        return {name: pd.DataFrame() for name in names}
    if "role" not in metrics:
        raise ValueError("v2 summaries require explicit validation/test role")
    for role, frame in metrics.groupby("role", dropna=False):
        if frame.duplicated(CURVE_KEYS + ["budget_n"]).any():
            raise ValueError("Duplicate experiment keys cannot be summarized")
        attainment = evaluate_joint_attainment(frame, targets, config)
        # v1's cost arithmetic is unchanged. Missing quality curves are unavailable,
        # never evidence that a competing method alone attained a known target.
        cost_input = attainment.copy()
        cost_input.loc[cost_input.status.eq("incomplete_quality_metrics"), "status"] = "quality_target_incomplete"
        costs = _costs(cost_input, config)
        calibration = []
        for level in (50, 80, 90):
            if not {f"coverage_{level}", f"width_{level}_log", "n_eval_locations"}.issubset(frame):
                continue
            view = frame[CURVE_KEYS + ["budget_n", "n_eval_locations", f"coverage_{level}", f"width_{level}_log"]].copy()
            view.rename(columns={f"coverage_{level}": "empirical_coverage", f"width_{level}_log": "mean_width_log"}, inplace=True)
            view["nominal_coverage"] = level / 100
            view["missing_reason"] = np.where(view.empirical_coverage.isna(), "predictive_interval_unavailable", None)
            calibration.append(view)
        tables = {"attainment": attainment, "comparisons": _comparisons(frame, config), "costs": costs,
                  "aggregate": _aggregate(frame, attainment),
                  "calibration": pd.concat(calibration, ignore_index=True) if calibration else pd.DataFrame()}
        for name, table in tables.items():
            table["target_version"], table["summary_role"] = settings["target_version"], role
            results[name].append(table)
    return {name: pd.concat(tables, ignore_index=True) for name, tables in results.items()}
