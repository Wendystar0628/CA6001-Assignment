"""Online, whole-group CPT replay without access to hidden measurements.

Audit scoring belongs to the experiment pipeline after this function returns.
The fixed grid G is the full planned geometry, including already measured CPTs.
"""
from __future__ import annotations

import json
from time import perf_counter

import numpy as np
import pandas as pd

from .adaptive_data import AdaptiveDataView, validate_split
from .adaptive_models import AdaptiveModelSuite, make_predictor
from .adaptive_policy import (
    StopRule, StopSignals, StopState, compute_change, coverage_fraction,
    select_batch, support_leaveout_groups, update_stop,
)


def _variant(policy_id, weights):
    if policy_id == "B_full":
        return "full"
    if policy_id == "B_seq":
        return "maximin"
    names = "".join(name for name, weight in zip("UDS", weights) if weight > 0)
    standard = np.allclose(weights, (.5, .3, .2)) or np.allclose(weights, (.7, 0., .3))
    return names if standard else names + "_" + "_".join(f"{w:g}" for w in weights)


def _frame(bundle):
    return bundle.frame if hasattr(bundle, "frame") else bundle


def _support_error(observed, groups, predictor, query):
    """Location-balanced leave-near-group-out error using revealed CPTs only."""
    errors, used = [], 0
    for members in groups.values():
        hidden = observed.location_id.isin(members)
        context = observed.loc[~hidden].reset_index(drop=True)
        held_out = observed.loc[hidden, ["location_id", "depth_index", "log_qc"]]
        if context.empty or held_out.empty:
            continue
        predictor.condition(context)
        prediction = _frame(predictor.predict(query(members)))
        matched = held_out.merge(prediction[["location_id", "depth_index", "pred_log_qc"]],
                                 on=["location_id", "depth_index"], validate="one_to_one")
        if len(matched) != len(held_out):
            raise ValueError("Support predictions must cover the whole held-out near group")
        matched["absolute_error"] = np.abs(matched.log_qc - matched.pred_log_qc)
        errors.extend(matched.groupby("location_id").absolute_error.mean().tolist())
        used += 1
    return (float(np.mean(errors)) if errors else np.nan), used


def run_trajectory(dataset, split, model_id, policy_id, weights=(.5, .3, .2),
                   stop_rule=None, *, diagnostics=True, suite=None,
                   support_predictor=None, log=print) -> dict:
    """Run one online trajectory and return four data frames plus metadata.

    A supplied suite is a source-weight cache owned by the caller. B_full must
    receive its own suite instance, never the live online decision suite. No
    audit measurements, full-plan predictions or audit scores enter a decision.
    A supplied single-model support predictor may also share initialized source
    weights; every leaveout explicitly overwrites its observed context. With no
    stop rule, every candidate is eventually revealed. Diagnostic group
    keys are frozen from the initial observed set and never selected by error.
    """
    if policy_id not in {"B_full", "B_seq", "P_adapt"}:
        raise ValueError("policy_id must be B_full, B_seq or P_adapt")
    validate_split(split)
    if len(split["initial_ids"]) < 5:
        raise ValueError("Adaptive replay requires at least five initial locations")
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (3,) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("weights must contain three finite nonnegative values with positive sum")
    weights = tuple((weights / weights.sum()).tolist())
    if isinstance(stop_rule, dict):
        stop_rule = StopRule(**stop_rule)
    variant = _variant(policy_id, weights)
    meta = {"run_id": f"{split['site_id']}__s{split['seed']}__{model_id}__{policy_id}__{variant}",
            "site_id": split["site_id"], "seed": int(split["seed"]),
            "fold_id": int(split["fold_id"]), "role": split["role"],
            "model_id": model_id, "policy_id": policy_id, "variant": variant}
    start = perf_counter()
    view = AdaptiveDataView(dataset, split)
    geometry = view.geometry()
    plan = set(split["full_ids"])
    audit = set(split["audit_ids"])
    union_query = view.query([*split["full_ids"], *split["audit_ids"]])
    full_mask = union_query.location_id.isin(plan)
    audit_mask = union_query.location_id.isin(audit)
    if set(union_query.location_id) != plan | audit:
        raise ValueError("Prediction grid must cover every plan and audit position")
    helper_id = "tabpfn_joint" if model_id == "xgb" else "xgb"
    needed_ids = tuple(dict.fromkeys((model_id, "gp", helper_id)))
    # A full-plan reference does not need acquisition auxiliaries or support CV.
    if policy_id == "B_full":
        needed_ids = (model_id,)
    initialize_s = 0.
    if suite is None:
        initialize_start = perf_counter()
        suite = AdaptiveModelSuite(split["fold_id"], needed_ids, device="cuda",
                                   role=split["role"], geometry=union_query)
        suite.initialize()
        initialize_s = perf_counter() - initialize_start
    if support_predictor is not None and getattr(support_predictor, "model_id", model_id) != model_id:
        raise ValueError("Cached support predictor must use the trajectory primary model")
    groups = support_leaveout_groups(split["initial_ids"], split["near_groups"], max_groups=5)
    trajectories, predictions, selections, runtime = [], [], [], []
    previous_grid, stop_state = None, StopState()
    selected = list(split["initial_ids"])
    selected_scores = None
    if policy_id == "B_full":
        view.reveal(split["candidate_ids"])
        selected = list(split["full_ids"])
    round_index = 0
    order = 0
    while True:
        round_start = perf_counter()
        known = view.revealed_ids
        remaining = sorted(plan - set(known))
        observed = view.observed()
        condition_start = perf_counter()
        suite.condition(observed)
        condition_s = perf_counter() - condition_start
        predict_start = perf_counter()
        bundles = suite.predict(union_query, needed_ids)
        predict_s = perf_counter() - predict_start
        main = _frame(bundles[model_id]).reset_index(drop=True)
        if len(main) != len(union_query) or not main[["location_id", "depth_index"]].equals(
                union_query[["location_id", "depth_index"]].reset_index(drop=True)):
            raise ValueError("Suite predictions must preserve the fixed query keys and order")
        grid = main.loc[full_mask].reset_index(drop=True)
        change = {"delta_q90": np.nan, "delta_h90": np.nan} if previous_grid is None else compute_change(previous_grid, grid)
        uncertainty, candidate_signals = np.nan, None
        if policy_id != "B_full":
            gp, helper = (_frame(bundles[m]).reset_index(drop=True) for m in ("gp", helper_id))
            if "std_log_qc" not in gp:
                raise ValueError("GP predictions must expose native std_log_qc")
            uncertainty = float(np.quantile(gp.loc[full_mask, "std_log_qc"], .9))
            candidate_signals = gp[["location_id", "std_log_qc"]].copy()
            candidate_signals["disagreement"] = (main.pred_log_qc.to_numpy() < 0) != (helper.pred_log_qc.to_numpy() < 0)
            candidate_signals = candidate_signals.groupby("location_id", sort=False).agg(
                uncertainty=("std_log_qc", "mean"), disagreement=("disagreement", "mean"))
        support_start = perf_counter()
        support_initialize_s = 0.
        support_error, n_support_groups = np.nan, 0
        if diagnostics and policy_id != "B_full" and len(known) >= 9 and remaining:
            if support_predictor is None:
                support_initialize_start = perf_counter()
                support_predictor = make_predictor(model_id, split["fold_id"], device="cuda",
                                                   role=split["role"], geometry=union_query)
                support_predictor.initialize()
                support_initialize_s = perf_counter() - support_initialize_start
            support_error, n_support_groups = _support_error(observed, groups, support_predictor, view.query)
        support_s = perf_counter() - support_start
        coverage = coverage_fraction(geometry, known, split["full_ids"], split.get("mandatory_ids", ()))
        signals = StopSignals(change["delta_q90"], change["delta_h90"], support_error,
                              uncertainty, bool(coverage >= 1. - 1e-12))
        if stop_rule is not None and policy_id != "B_full":
            stop_state = update_stop(stop_rule, stop_state, signals, len(known), len(remaining))
            stopped, reason = stop_state.stopped, stop_state.stop_reason
        else:
            stopped, reason = not remaining, "full_plan_complete" if not remaining else None
        record = {**meta, "round": round_index, "n_observed": len(known), "n_full": len(plan),
                  "known_ids": json.dumps(known), "revealed_ids": json.dumps(selected),
                  "selected_ids": json.dumps([]),
                  **change, "support_log_mae": support_error, "uncertainty_q90": uncertainty,
                  "coverage_fraction": coverage, "coverage_ok": signals.coverage_ok,
                  "stopped": stopped, "stop_reason": reason,
                  "n_support_audit_groups": n_support_groups,
                  "n_support_fixed_groups": len(groups), "pass_streak": stop_state.pass_streak}
        # Prediction files contain audit predictions and geometry, never labels.
        audit_prediction = main.loc[audit_mask].copy().reset_index(drop=True)
        leaked = {"qc_mpa", "log_qc", "low_1", "truth", "true_log_qc"} & set(audit_prediction)
        if leaked:
            raise ValueError(f"Audit prediction output contains measurement columns: {sorted(leaked)}")
        for key, value in {**meta, "round": round_index, "n_observed": len(known)}.items():
            audit_prediction[key] = value
        predictions.append(audit_prediction)
        for loc in selected:
            order += 1
            score = np.nan
            if selected_scores is not None:
                score = float(selected_scores.loc[loc, "selection_score"])
            selections.append({**meta, "round": round_index, "location_id": loc,
                               "score": score, "selection_order": order})
        selection_start = perf_counter()
        next_selected, next_scores = [], None
        if not stopped:
            next_selected, next_scores = select_batch(
                geometry, known, remaining, split["near_groups"],
                uncertainty=candidate_signals.uncertainty.to_dict(),
                disagreement=candidate_signals.disagreement.to_dict(), weights=weights,
                batch_size=2, policy="adaptive" if policy_id == "P_adapt" else "maximin")
            if not next_selected:
                raise ValueError("Acquisition did not select any remaining candidate group")
            next_scores = next_scores.set_index("location_id")
        select_s = perf_counter() - selection_start
        record["selected_ids"] = json.dumps(next_selected)
        elapsed = perf_counter() - round_start
        record["runtime_s"] = elapsed
        trajectories.append(record)
        runtime.append({**meta, "round": round_index, "n_observed": len(known),
                        "source_initialize_s": initialize_s if round_index == 0 else 0.,
                        "support_initialize_s": support_initialize_s,
                        "condition_s": condition_s, "predict_s": predict_s,
                        "support_diagnostic_s": support_s, "select_s": select_s,
                        "runtime_s": elapsed, "elapsed_trajectory_s": perf_counter() - start})
        if log is not None:
            message = f"{meta['run_id']} round={round_index} n={len(known)}/{len(plan)} stop={reason or '-'} ({elapsed:.2f}s)"
            if log is print:
                print(message, flush=True)
            else:
                log(message)
        if stopped:
            break
        # Complete the batch before any label is revealed or model conditioned.
        view.reveal(next_selected)
        previous_grid = grid
        selected, selected_scores = next_selected, next_scores
        round_index += 1
    metadata = {**meta, "weights": list(weights), "n_full": len(plan),
                "n_audit": len(audit), "initial_ids": list(split["initial_ids"]),
                "support_fixed_group_ids": list(groups), "grid_locations": len(plan),
                "grid_cells": int(full_mask.sum()), "audit_cells": int(audit_mask.sum()),
                "auxiliary_model_id": helper_id if policy_id != "B_full" else None,
                "diagnostics": bool(diagnostics), "runtime_s": perf_counter() - start,
                "source_initialize_s": initialize_s,
                "source_role": split["role"], "audit_labels_read": False}
    return {"trajectories": pd.DataFrame(trajectories),
            "predictions": pd.concat(predictions, ignore_index=True),
            "selections": pd.DataFrame(selections), "runtime": pd.DataFrame(runtime),
            "metadata": metadata}
