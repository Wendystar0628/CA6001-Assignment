"""Trajectory-level counterfactual economics for adaptive supplementary CPTs.

The experiment supplies counts and hidden-audit outcomes. This module supplies
explicit cost/schedule scenarios; it never selects a stopping count from truth.
Owner financial and shipping-user effects are deliberately separate outputs.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


IDENTIFIERS = ["run_id", "fold_id", "role", "site_id", "seed", "model_id", "policy_id"]
SENSITIVITY_PARAMETERS = ["cost_per_cpt", "cpt_per_rig_workday",
    "batch_wait_calendar_days", "project_cost_per_calendar_day",
    "opening_net_per_calendar_day", "ai_incremental_cost", "mobilization_cost",
    "repair_wait_calendar_days"]


def load_economics_config(path: str | Path) -> dict:
    """Read a case YAML; each parameter retains its value and provenance."""
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _value(config, name, default=None):
    entry = config.get("parameters", {}).get(name, default)
    return entry.get("value", default) if isinstance(entry, dict) else entry


def _float(config, name, default=0.0):
    value = float(_value(config, name, default))
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"Economic parameter {name} must be finite and nonnegative")
    return value


def _present(value):
    return value is not None and value is not pd.NA and not (
        isinstance(value, (float, np.floating)) and np.isnan(value))


def _boolean(value):
    if not _present(value):
        return None
    if isinstance(value, str):
        lowered = value.lower().strip()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
        raise ValueError(f"Unrecognized audit/scope boolean: {value!r}")
    return bool(value)


def _sequence(value):
    if not _present(value):
        return None
    return json.loads(value) if isinstance(value, str) else list(value)


def _count(value, name):
    value = float(value)
    if not np.isfinite(value) or value < 0 or not value.is_integer():
        raise ValueError(f"{name} must be a finite nonnegative integer")
    return int(value)


def _batches(row, trajectories, n_stop):
    sizes = _sequence(row.get("batch_sizes"))
    if sizes is not None:
        sizes = [_count(x, "batch_size") for x in sizes]
        if sum(sizes) != n_stop:
            raise ValueError("batch_sizes must sum to n_stop, including initial CPTs")
        return sum(size > 0 for size in sizes)
    if _present(row.get("n_batches")):
        count = _count(row["n_batches"], "n_batches")
        if n_stop and count == 0:
            raise ValueError("Positive n_stop requires at least one batch")
        return count
    if trajectories is not None and not trajectories.empty:
        matches = trajectories
        for column in IDENTIFIERS:
            if column in matches and column in row:
                matches = matches.loc[matches[column] == row[column]]
        if "budget_n" in matches:
            return int(matches.loc[matches.budget_n <= n_stop, "budget_n"].nunique())
    return 1 if n_stop else 0


def _field_days(n, config):
    capacity = _float(config, "cpt_per_rig_workday") * _float(config, "parallel_rigs", 1)
    if capacity <= 0:
        raise ValueError("CPT productivity and parallel rigs must be positive")
    return int(math.ceil(n / capacity))


def _calendar_days(workdays, config):
    schedule = config.get("schedule", {})
    method = schedule.get("calendar_conversion", "seven_over_five")
    if method == "calendar_equals_workday":
        return float(workdays)
    if method != "seven_over_five":
        raise ValueError("Supported calendar_conversion: seven_over_five or calendar_equals_workday")
    workweek = float(schedule.get("workdays_per_week", 5))
    if not 0 < workweek <= 7:
        raise ValueError("workdays_per_week must lie in (0, 7]")
    return float(workdays) * 7.0 / workweek


def _penetration(row, config, n_full, n_stop):
    """Depths describe complete planned soundings, never just modelled cells."""
    full = _sequence(row.get("full_depths_m"))
    measured = _sequence(row.get("measured_depths_m"))
    default = _float(config, "plan_depth_m")
    if full is None:
        full = [default] * n_full
        depth_basis = "uniform_full_depth_scenario"
    else:
        depth_basis = "external_full_planned_depths"
    if len(full) != n_full or not np.isfinite(full).all() or any(d <= 0 for d in full):
        raise ValueError("full_depths_m must contain one positive planned depth per full-plan CPT")
    if measured is None:
        if len(set(full)) > 1 and n_stop != n_full:
            raise ValueError("Variable full depths require measured_depths_m for the selected CPTs")
        measured = list(full) if n_stop == n_full else [default if not full else full[0]] * n_stop
    if len(measured) != n_stop or not np.isfinite(measured).all() or any(d <= 0 for d in measured):
        raise ValueError("measured_depths_m must contain one positive complete depth per measured CPT")
    return float(sum(full)), float(sum(measured)), depth_basis


def _invoice(n, metres, visits, config):
    mode = config.get("case", {}).get("pricing_mode", "all_in_per_cpt")
    common = visits * _float(config, "mobilization_cost")
    if mode == "all_in_per_cpt":
        # Rig/crew/processing are inside this unit price. Duration does not
        # generate another rig-day invoice, including during repair.
        return common + n * _float(config, "cost_per_cpt")
    if mode == "itemized_per_metre":
        return (common + n * (_float(config, "setup_cost_per_cpt") +
                _float(config, "processing_cost_per_cpt")) +
                metres * _float(config, "penetration_cost_per_metre"))
    raise ValueError(f"Unsupported CPT pricing_mode: {mode}")


def _evaluate_one(row, config, alpha, trajectories=None):
    n_full, n_stop = (_count(row[name], name) for name in ("n_full", "n_stop"))
    if n_stop > n_full:
        raise ValueError("n_stop cannot exceed n_full")
    n_initial = _count(row.get("n_initial", 0), "n_initial")
    if n_initial > n_stop:
        raise ValueError("n_initial cannot exceed n_stop")
    policy = str(row.get("policy_id", "P_adapt"))
    is_full = policy == "B_full"
    if is_full and n_stop != n_full:
        raise ValueError("B_full must complete the full planned CPT count")
    batches = 1 if is_full and n_full else _batches(row, trajectories, n_stop)
    early = n_stop < n_full
    passed = _boolean(row.get("audit_pass"))
    replaceable = _boolean(row.get("scope_replaceable", True))
    # An unverified claim earns no early-stop credit. A failed/missing audit or
    # nonreplaceable deep task is repaired by measuring the remaining plan.
    repair = early and (passed is not True or replaceable is not True)
    repair_n = n_full - n_stop if repair else 0
    completed_n = n_stop + repair_n
    full_metres, measured_metres, depth_basis = _penetration(row, config, n_full, n_stop)
    completed_metres = full_metres if repair else measured_metres
    extra_visits = _count(row.get("additional_visits", 0), "additional_visits")
    repair_visits = 1 if repair else 0
    full_field = _field_days(n_full, config)
    method_field = _field_days(n_stop, config) + (_field_days(repair_n, config) if repair else 0)
    batch_wait = max(0, batches - 1) * _float(config, "batch_wait_calendar_days")
    review = (0 if is_full else batches) * _float(config, "review_workdays_per_batch")
    repair_wait = _float(config, "repair_wait_calendar_days") if repair else 0.0
    visit_wait = float(row.get("additional_visit_wait_calendar_days", 0.0))
    if not np.isfinite(visit_wait) or visit_wait < 0:
        raise ValueError("additional_visit_wait_calendar_days must be finite and nonnegative")
    full_duration = _calendar_days(full_field, config)
    method_duration = _calendar_days(method_field + review, config) + batch_wait + repair_wait + visit_wait
    delta_inv = full_duration - method_duration
    delta_project = float(alpha) * delta_inv  # Negative delay remains negative.
    full_cost = _invoice(n_full, full_metres, 1, config)
    method_cost = _invoice(completed_n, completed_metres, 1 + extra_visits + repair_visits, config)
    direct = full_cost - method_cost
    time = delta_project * _float(config, "project_cost_per_calendar_day")
    opening = delta_project * _float(config, "opening_net_per_calendar_day")
    ai_cost = 0.0 if is_full else _float(config, "ai_incremental_cost")
    social_daily = (_float(config, "social_affected_vessels_per_day") *
        _float(config, "social_hours_saved_per_vessel") * _float(config, "social_vessel_hour_cost"))
    if repair:
        status = "failed_stop_repaired" if passed is False else "unverified_stop_completed"
        if replaceable is not True:
            status = "nonreplaceable_scope_completed"
    elif early:
        status = "audited_early_stop_conditional_scope"
    else:
        status = "full_plan_completed" if is_full else "no_early_stop"
    out = {name: row.get(name) for name in IDENTIFIERS if name in row}
    out.update(case_id=config.get("case", {}).get("case_id"),
        currency=config.get("case", {}).get("currency", "EUR"),
        price_year=config.get("case", {}).get("price_year"), perspective="owner_financial",
        monetary_basis="counterfactual_scenario_not_observed_project_saving",
        economic_status=status, alpha=float(alpha), n_full=n_full, n_stop=n_stop,
        n_initial=n_initial, n_batches=batches, audit_pass=passed,
        is_early_stop=early, claimed_saved_cpt=n_full - n_stop,
        repaired_cpt=repair_n, charged_cpt=completed_n, saved_cpt=n_full - completed_n,
        false_early_stop=bool(early and passed is False),
        scope_replaceable_assumption=replaceable, scope_note=config.get("case", {}).get("scope_note"),
        full_planned_metres=full_metres, measured_metres_at_stop=measured_metres,
        charged_metres=completed_metres, depth_basis=depth_basis,
        full_field_workdays=full_field, method_field_workdays=method_field,
        additional_review_workdays=review, batch_wait_calendar_days=batch_wait,
        repair_wait_calendar_days=repair_wait, additional_visit_wait_calendar_days=visit_wait,
        additional_visits=extra_visits,
        repair_visits=repair_visits, full_investigation_calendar_days=full_duration,
        method_investigation_calendar_days=method_duration,
        investigation_days_saved=delta_inv, project_days_saved=delta_project,
        full_investigation_cost=full_cost, method_investigation_cost=method_cost,
        direct_saving=direct, project_time_saving=time, early_opening_benefit=opening,
        ai_incremental_cost=ai_cost, net_benefit=direct + time + opening - ai_cost,
        social_user_benefit=delta_project * social_daily,
        social_daily_user_benefit=social_daily,
        reliable_early_stop=bool(early and passed is True and replaceable is True))
    for column in ("stop_reason", "quality_target_id", "n_audit", "counts_source"):
        if column in row:
            out[column] = row[column]
    return out


def parameter_table(config: dict) -> pd.DataFrame:
    """Flatten values with source type, year, range, and overlap notes for reports."""
    case = config.get("case", {})
    rows = []
    for name, item in config.get("parameters", {}).items():
        entry = dict(item) if isinstance(item, dict) else {"value": item}
        rows.append({"parameter": name, "currency": case.get("currency"),
            "price_year": entry.pop("price_year", case.get("price_year")),
            "region": entry.pop("region", case.get("region")), **entry})
    rows.append({"parameter": "alpha", "value": case.get("primary_alpha", 1.0),
        "unit": "project_calendar_day/investigation_calendar_day",
        "source_type": "team_specified_scenario_assumption", "lower": 0.5, "upper": 1.0,
        "source": "Execution plan section 9.1: alpha=1 primary, 0.5/0.75/1 sensitivity; no historical critical-path verification.",
        "region": case.get("region"), "price_year": case.get("price_year")})
    return pd.DataFrame(rows)


def evaluate_economic_case(decisions: pd.DataFrame, config: dict,
                           trajectories: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Evaluate actual stop decisions against traditional completion of all CPTs.

    Required columns: n_full, n_stop, audit_pass. Preserve trajectory identifiers
    (run_id/fold_id/role/site_id/seed/model_id/policy_id) when present. n_batches
    includes the initial field batch. A JSON batch_sizes list can supply it.
    Unknown audit outcomes trigger completion, never a saving credit.

    Returns the primary alpha=1 owner table, alpha plus one-at-a-time parameter
    sensitivity, a separately labelled shipping-user table and provenance. No
    social effect is added to net_benefit. Monetary values remain scenarios.
    """
    required = {"n_full", "n_stop", "audit_pass"}
    if not required.issubset(decisions):
        raise ValueError(f"Economic decisions missing columns: {sorted(required - set(decisions))}")
    if config.get("case", {}).get("perspective", "owner_financial") != "owner_financial":
        raise ValueError("This version fixes owner_financial as primary; social user effects are separate")
    primary_alpha = float(config.get("case", {}).get("primary_alpha", 1.0))
    alphas = config.get("case", {}).get("alpha_values", [0.5, 0.75, 1.0])
    if primary_alpha != 1.0 or any(not 0 <= float(a) <= 1 for a in alphas):
        raise ValueError("Primary alpha must be 1; sensitivity alpha must lie in [0, 1]")
    keys = [c for c in IDENTIFIERS if c in decisions]
    if keys and decisions.duplicated(keys).any():
        raise ValueError("Expected one final economic decision per trajectory")
    economics, sensitivity = [], []
    for row in decisions.to_dict("records"):
        economics.append(_evaluate_one(row, config, primary_alpha, trajectories))
        for alpha in alphas:
            sensitivity.append({**_evaluate_one(row, config, float(alpha), trajectories),
                "sensitivity_kind": "alpha", "varied_parameter": "alpha", "parameter_value": float(alpha)})
        # This small, explicitly labelled grid describes economic assumptions;
        # it is not used to select a model, policy or stopping decision.
        opening_entry = config.get("parameters", {}).get("opening_net_per_calendar_day", {})
        if isinstance(opening_entry, dict):
            opening_values = sorted({float(opening_entry[key]) for key in ("lower", "value", "upper")
                if opening_entry.get(key) is not None})
        else:
            opening_values = [float(opening_entry)]
        for alpha in alphas:
            for daily_net in opening_values:
                varied = deepcopy(config)
                varied["parameters"]["opening_net_per_calendar_day"] = {"value": daily_net}
                sensitivity.append({**_evaluate_one(row, varied, float(alpha), trajectories),
                    "sensitivity_kind": "alpha_opening_grid",
                    "varied_parameter": "alpha_x_opening_net",
                    "parameter_value": daily_net, "daily_net_benefit": daily_net,
                    "opening_net_per_calendar_day": daily_net})
        for name in SENSITIVITY_PARAMETERS:
            item = config.get("parameters", {}).get(name, {})
            if not isinstance(item, dict):
                continue
            for bound in ("lower", "upper"):
                value = item.get(bound)
                if value is None or float(value) == float(item["value"]):
                    continue
                varied = deepcopy(config)
                varied["parameters"][name]["value"] = value
                sensitivity.append({**_evaluate_one(row, varied, primary_alpha, trajectories),
                    "sensitivity_kind": "one_at_a_time", "varied_parameter": name,
                    "parameter_value": float(value), "parameter_bound": bound})
    main = pd.DataFrame(economics)
    social_columns = [c for c in IDENTIFIERS if c in main] + ["case_id", "currency", "price_year",
        "economic_status", "alpha", "project_days_saved", "social_daily_user_benefit", "social_user_benefit"]
    social = main[social_columns].copy() if not main.empty else pd.DataFrame()
    if not social.empty:
        social["perspective"] = "social_user_time_only"
        social["included_in_owner_net_benefit"] = False
        social["monetary_basis"] = "separate_scenario_not_a_complete_social_CBA"
    return {"economics": main, "sensitivity": pd.DataFrame(sensitivity),
        "social_user": social, "parameters": parameter_table(config)}
