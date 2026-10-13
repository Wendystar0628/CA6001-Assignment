"""Validation-frozen quality thresholds, honest attainment, and paired costs."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd


CURVE_KEYS = ["fold_id", "site_id", "seed", "model_id", "policy_id"]
METRIC_COLUMNS = ["log_mae", "log_rmse", "qc_mae_mpa", "h_point_mae_m", "h_expected_mae_m",
                  "precision", "recall", "f1", "average_precision", "false_positive_rate",
                  "coverage_50", "coverage_80", "coverage_90", "width_50_log", "width_80_log", "width_90_log"]


def freeze_quality_targets(metrics, config):
    """Freeze targets exclusively from explicitly labelled validation records."""
    if "role" not in metrics:
        raise ValueError("Quality freezing requires explicit role='validation'; test data cannot set targets")
    quality = config.get("quality", {})
    model = quality.get("reference_model", "gp")
    policy = quality.get("reference_policy", "maximin")
    budget = int(quality.get("reference_budget", 13))
    tolerance = float(quality.get("recall_tolerance", 0.05))
    reference = dict(model_id=model, policy_id=policy, budget_n=budget, recall_tolerance=tolerance)
    output = {"folds": {}, "reference": reference, "source_role": "validation"}
    if not 0 <= tolerance <= 1:
        raise ValueError("Recall tolerance must lie in [0, 1]")
    for fold in sorted(metrics.fold_id.unique()):
        source = metrics[(metrics.fold_id == fold) & (metrics.role == "validation") &
                         (metrics.model_id == model) & (metrics.policy_id == policy) & (metrics.budget_n == budget)].copy()
        if source.duplicated(["site_id", "seed"]).any():
            raise ValueError("Duplicate reference rows would overweight validation seeds")
        expected_seeds = set(config.get("replay", {}).get("seeds", []))
        excluded_sites = {}
        kept = []
        for site_id, group in source.groupby("site_id", sort=True):
            if expected_seeds and set(group.seed) != expected_seeds:
                excluded_sites[str(site_id)] = "incomplete_reference_seeds"
            elif not np.isfinite(group[["log_mae", "h_point_mae_m"]].to_numpy(float)).all():
                excluded_sites[str(site_id)] = "missing_reference_errors"
            else:
                kept.append(group)
        source = pd.concat(kept, ignore_index=True) if kept else source.iloc[:0]
        site_means = source.groupby("site_id")[["log_mae", "h_point_mae_m", "recall"]].mean()
        e_star = float(site_means.log_mae.median()) if len(site_means) else None
        h_star = float(site_means.h_point_mae_m.median()) if len(site_means) else None
        valid_recall = site_means.recall.dropna()
        r_star = float(np.clip(valid_recall.median() - tolerance, 0, 1)) if len(valid_recall) else None
        reasons = []
        if e_star is None or h_star is None:
            reasons.append("no_complete_validation_reference")
        if r_star is None:
            reasons.append("no_positive_validation_reference")
        complete = not reasons
        output["folds"][str(fold)] = {
            "fold_id": int(fold), "E_star": e_star, "H_star": h_star, "R_star": r_star,
            "status": "complete" if complete else "quality_target_incomplete",
            "complete": complete, "missing_reasons": reasons,
            "validation_sites": site_means.index.astype(str).tolist(),
            "validation_sites_with_recall": valid_recall.index.astype(str).tolist(),
            "n_validation_sites": len(site_means), "excluded_sites": excluded_sites,
            "seed_counts": {str(k): int(v) for k, v in source.groupby("site_id").seed.nunique().items()},
            "reference": reference,
        }
    return output


def _aubc(group, column):
    x = group.budget_n.to_numpy(float)
    y = group[column].to_numpy(float)
    if len(x) < 2:
        return None, "fewer_than_two_budgets"
    if not np.isfinite(y).all():
        return None, "missing_metric_at_budget"
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x) / 2) / (x[-1] - x[0])), None


def _target_for(targets, fold):
    rows = targets.get("folds", targets)
    if isinstance(rows, list):
        return next((r for r in rows if str(r.get("fold_id")) == str(fold)), {})
    return rows.get(str(fold), rows.get(fold, {}))


def _attainment(metrics, targets, config):
    rows = []
    expected = config.get("replay", {}).get("budgets", [5, 7, 9, 11, 13])
    for key, group in metrics.groupby(CURVE_KEYS, sort=True, dropna=False):
        group = group.sort_values("budget_n")
        if group.budget_n.duplicated().any():
            raise ValueError("Duplicate budgets within a replay curve")
        row = dict(zip(CURVE_KEYS, key))
        target = _target_for(targets, row["fold_id"])
        complete = bool(target.get("complete", target.get("status") == "complete"))
        complete = complete and all(target.get(name) is not None and np.isfinite(target[name]) for name in ["E_star", "H_star", "R_star"])
        budgets = group.budget_n.to_numpy(int)
        row.update(last_budget=int(budgets[-1]), n_first_hit=None, n_stable=None,
                   n_budgets=len(budgets), budgets=json.dumps(budgets.tolist()),
                   budget_curve_complete=(budgets.tolist() == list(expected)),
                   missing_budgets=json.dumps([int(n) for n in expected if n not in budgets]),
                   n_first_hit_scope="observed_budget_points")
        for metric, dest in [("log_mae", "aubc_log_mae"), ("h_point_mae_m", "aubc_h_point_mae")]:
            row[dest], row[f"{dest}_reason"] = _aubc(group, metric)
        if not complete:
            row.update(status="quality_target_incomplete", attainment_reason="validation_quality_target_unavailable", hit_sequence=None)
        else:
            positives = (group.n_true_positive.to_numpy(float) > 0 if "n_true_positive" in group
                         else group.has_positive_labels.to_numpy(bool))
            e, h, r = (group[col].to_numpy(float) for col in ["log_mae", "h_point_mae_m", "recall"])
            valid = np.isfinite(e) & np.isfinite(h) & (~positives | np.isfinite(r))
            hits = valid & (e <= target["E_star"]) & (h <= target["H_star"]) & (~positives | (r >= target["R_star"]))
            if hits.any():
                row["n_first_hit"] = int(budgets[np.flatnonzero(hits)[0]])
            if not row["budget_curve_complete"]:
                row["status"] = "incomplete_budget_curve"
                row["attainment_reason"] = "configured_budget_points_not_fully_observed"
            else:
                stable = np.logical_and.accumulate(hits[::-1])[::-1]
                if stable.any():
                    row["n_stable"] = int(budgets[np.flatnonzero(stable)[0]])
                row["status"] = "reached" if row["n_stable"] is not None else "not_reached_by_budget"
                row["attainment_reason"] = None if valid.all() else "missing_quality_metric_at_budget"
            row["hit_sequence"] = json.dumps([bool(h) for h in hits])
        rows.append(row)
    return pd.DataFrame(rows)


def _comparisons(metrics, config):
    quality = config.get("quality", {})
    ref_model, ref_policy = quality.get("reference_model", "gp"), quality.get("reference_policy", "maximin")
    values = [c for c in METRIC_COLUMNS if c in metrics]
    pair_keys = ["fold_id", "site_id", "seed", "budget_n"]
    results = []
    for (model, policy), method in metrics.groupby(["model_id", "policy_id"], sort=True):
        references = [(ref_model, ref_policy, "reference_baseline")]
        if policy != "maximin" and (model, "maximin") != (ref_model, ref_policy):
            references.append((model, "maximin", "strategy_effect"))
        for rm, rp, scope in references:
            if (model, policy) == (rm, rp):
                continue
            reference = metrics[(metrics.model_id == rm) & (metrics.policy_id == rp)]
            joined = method[pair_keys + values].merge(reference[pair_keys + values], on=pair_keys, how="inner", suffixes=("_method", "_reference"), validate="one_to_one")
            for metric in values:
                base = joined[pair_keys].copy()
                base["method_value"] = joined[f"{metric}_method"]
                base["reference_value"] = joined[f"{metric}_reference"]
                base["difference"] = base.method_value - base.reference_value
                base["metric"] = metric
                base["model_id"], base["policy_id"] = model, policy
                base["reference_model"], base["reference_policy"] = rm, rp
                base["comparison_scope"] = scope
                base["level"] = "seed"
                base["n_sites"] = base.difference.notna().astype(int)
                base["n_pairs"] = base.difference.notna().astype(int)
                base["missing_reason"] = np.where(base.difference.isna(), "paired_metric_undefined", None)
                results.append(base)
                # A shared validity mask prevents different denominators across the pair.
                valid = base[base.difference.notna()]
                if valid.empty:
                    continue
                cols = ["model_id", "policy_id", "reference_model", "reference_policy", "comparison_scope", "metric", "budget_n"]
                site = valid.groupby(cols + ["fold_id", "site_id"], as_index=False).agg(
                    method_value=("method_value", "mean"), reference_value=("reference_value", "mean"),
                    difference=("difference", "mean"), n_pairs=("difference", "count"))
                site["level"], site["n_sites"] = "site", 1
                results.append(site)
                macro = site.groupby(cols, as_index=False).agg(
                    method_value=("method_value", "mean"), reference_value=("reference_value", "mean"),
                    difference=("difference", "mean"), n_sites=("site_id", "size"), n_pairs=("n_pairs", "sum"))
                macro["level"] = "macro"
                results.append(macro)
    result = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    if not result.empty:
        result["delta"] = result.difference
    return result


def _aggregate(metrics, attainment):
    frames = []
    for source, metric_columns, budget_cols in [(metrics, METRIC_COLUMNS, ["budget_n"]),
            (attainment, ["aubc_log_mae", "aubc_h_point_mae"], [])]:
        for metric in metric_columns:
            if metric not in source:
                continue
            source = source.copy()
            source[metric] = pd.to_numeric(source[metric], errors="coerce")
            keys = ["model_id", "policy_id"] + budget_cols
            sites = source.groupby(keys + ["fold_id", "site_id"], as_index=False).agg(value=(metric, "mean"), n_valid_seeds=(metric, "count"))
            sites["metric"], sites["level"] = metric, "site"
            sites["n_sites"] = sites.value.notna().astype(int)
            frames.append(sites)
            macro = sites.groupby(keys, as_index=False).agg(value=("value", "mean"), n_sites=("value", "count"), n_total_sites=("value", "size"))
            macro["metric"], macro["level"] = metric, "macro"
            frames.append(macro)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _costs(attainment, config):
    if attainment.empty:
        return pd.DataFrame()
    q = config.get("quality", {})
    rm, rp = q.get("reference_model", "gp"), q.get("reference_policy", "maximin")
    reference = attainment[(attainment.model_id == rm) & (attainment.policy_id == rp)]
    if reference.empty:
        return pd.DataFrame()
    keys = ["fold_id", "site_id", "seed"]
    reference = reference[keys + ["n_stable", "last_budget", "status"]].rename(
        columns={c: f"baseline_{c}" for c in ["n_stable", "last_budget", "status"]})
    initial = int(config.get("replay", {}).get("initial_locations", 5))
    econ = config.get("economics", {})
    required_costs = ["marginal_cost_per_cpt", "extra_analysis_cost", "extra_mobilization_cost"]
    money_ready = bool(econ.get("currency")) and all(econ.get(c) is not None for c in required_costs)
    if money_ready and any(not np.isfinite(float(econ[c])) or float(econ[c]) < 0 for c in required_costs):
        raise ValueError("Cost scenarios must be finite and nonnegative")
    rows = []
    for (model, policy), method in attainment.groupby(["model_id", "policy_id"]):
        if (model, policy) == (rm, rp):
            continue
        paired = method.merge(reference, on=keys, how="left", validate="many_to_one")
        for _, item in paired.iterrows():
            row = {k: item[k] for k in CURVE_KEYS}
            row.update(reference_model=rm, reference_policy=rp, n_stable=item.n_stable,
                       baseline_n_stable=item.baseline_n_stable, last_budget=item.last_budget,
                       baseline_last_budget=item.baseline_last_budget, saved_cpt=None,
                       saved_fraction_of_baseline_extra=None, savings_lower_bound_exclusive=None,
                       savings_upper_bound_exclusive=None, max_affordable_extra_cost_in_cpt_units=None,
                       break_even_cost_per_cpt=None, net_savings=None, currency=econ.get("currency"),
                       monetary_status="cost_scenario_missing" if not money_ready else "attainment_pair_required")
            mr, br = pd.notna(item.n_stable), pd.notna(item.baseline_n_stable)
            if pd.isna(item.baseline_status):
                row["pair_status"] = "baseline_pair_missing"
            elif item.status == "quality_target_incomplete" or item.baseline_status == "quality_target_incomplete":
                row["pair_status"] = "quality_target_incomplete"
            elif item.status == "incomplete_budget_curve" or item.baseline_status == "incomplete_budget_curve":
                row["pair_status"] = "incomplete_budget_curve"
            elif mr and br:
                row["pair_status"] = "both_reached"
                saved = int(item.baseline_n_stable - item.n_stable)
                extra_b, extra_m = max(int(item.baseline_n_stable) - initial, 0), max(int(item.n_stable) - initial, 0)
                row.update(saved_cpt=saved, baseline_extra_cpt=extra_b, method_extra_cpt=extra_m,
                           saved_fraction_of_baseline_extra=(saved / extra_b if extra_b else None),
                           max_affordable_extra_cost_in_cpt_units=saved)
                if econ.get("extra_analysis_cost") is not None and econ.get("extra_mobilization_cost") is not None and saved > 0:
                    row["break_even_cost_per_cpt"] = (float(econ["extra_analysis_cost"]) + float(econ["extra_mobilization_cost"])) / saved
                if money_ready:
                    row["net_savings"] = saved * float(econ["marginal_cost_per_cpt"]) - float(econ["extra_analysis_cost"]) - float(econ["extra_mobilization_cost"])
                    row["monetary_status"] = "conditional_scenario"
            elif mr:
                row["pair_status"] = "only_method_reached"
                row["savings_lower_bound_exclusive"] = float(item.baseline_last_budget - item.n_stable)
            elif br:
                row["pair_status"] = "only_baseline_reached"
                row["savings_upper_bound_exclusive"] = float(item.baseline_n_stable - item.last_budget)
            else:
                row["pair_status"] = "neither_reached"
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        group = out.groupby(["model_id", "policy_id"])
        out["n_all_instances"] = group.pair_status.transform("size")
        out["n_both_reached"] = group.pair_status.transform(lambda s: (s == "both_reached").sum())
        out["both_reached_fraction"] = out.n_both_reached / out.n_all_instances
    return out


def summarize_results(metrics, quality_targets, config):
    """Pair seeds first, then average seeds within sites, then sites equally."""
    summary_role = "test"
    if "role" in metrics:
        if not (metrics.role == "test").any() and (metrics.role == "pilot").any():
            summary_role = "pilot"
            quality_targets = {"folds": {}}
        metrics = metrics.loc[metrics.role == summary_role].copy()
    else:
        metrics = metrics.copy()
    if metrics.empty:
        return {name: pd.DataFrame() for name in ["attainment", "comparisons", "calibration", "costs", "aggregate"]}
    if metrics.duplicated(CURVE_KEYS + ["budget_n"]).any():
        raise ValueError("Duplicate experiment keys cannot be summarized")
    attainment = _attainment(metrics, quality_targets, config)
    calibration = []
    for level in [50, 80, 90]:
        if f"coverage_{level}" not in metrics:
            continue
        frame = metrics[CURVE_KEYS + ["budget_n", "n_eval_locations", f"coverage_{level}", f"width_{level}_log"]].copy()
        frame = frame.rename(columns={f"coverage_{level}": "empirical_coverage", f"width_{level}_log": "mean_width_log"})
        frame["nominal_coverage"] = level / 100
        frame["missing_reason"] = np.where(frame.empirical_coverage.isna(), "predictive_interval_unavailable", None)
        calibration.append(frame)
    result = {"attainment": attainment, "comparisons": _comparisons(metrics, config),
              "calibration": pd.concat(calibration, ignore_index=True) if calibration else pd.DataFrame(),
              "costs": _costs(attainment, config), "aggregate": _aggregate(metrics, attainment)}
    for frame in result.values():
        frame["summary_role"] = summary_role
    return result
