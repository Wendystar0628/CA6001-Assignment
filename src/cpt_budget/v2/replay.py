"""v0.2 fixed-holdout replay with separate prediction and selection models."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..acquisition import select_batch
from ..evaluate import evaluate, join_evaluation
from ..replay import _state_id
from ..types import PredictionBundle
from .acquisition import POLICIES, select_gp_batch


def run_replay_v2(dataset, split, predictor, config, *, policy, fold_id=0,
                  proxy=None, calibrator=None, proxy_calibrator=None):
    """Return v1's five tables while retaining original point metrics.

    A calibrator is frozen before this function is called; absent calibration
    is explicitly marked and may be cross-fitted by the validation pipeline.
    Evaluation truth is passed only to metric functions, never to acquisition.
    """
    if policy not in ("maximin", *POLICIES):
        raise ValueError(f"Unknown v2 replay policy: {policy}")
    from .risk import apply_risk, evaluate_risk, transform_probability

    site_id, seed = split["site_id"], int(split["seed"])
    evaluation_ids = list(map(str, split["evaluation_ids"]))
    pool_ids = list(map(str, split["pool_ids"]))
    observed_ids = list(map(str, split["initial_ids"]))
    budgets = [int(n) for n in split.get("budgets", config.get("replay", {}).get("budgets", [5, 7, 9, 11, 13]))]
    batch_size = int(config.get("replay", {}).get("batch_size", 2))
    if batch_size != 2:
        raise ValueError("v0.2 replay requires complete two-profile batches")
    if not evaluation_ids or len(evaluation_ids) != len(set(evaluation_ids)):
        raise ValueError("Replay requires a unique nonempty fixed evaluation set")
    if len(pool_ids) != len(set(pool_ids)) or len(observed_ids) != len(set(observed_ids)):
        raise ValueError("Pool and initial location IDs must be unique")
    if set(evaluation_ids) & set(pool_ids) or not set(observed_ids) <= set(pool_ids):
        raise ValueError("Holdout and action pool must be disjoint; initial points belong to pool")
    if not budgets or budgets != sorted(set(budgets)) or budgets[0] != len(observed_ids):
        raise ValueError("Budgets must increase from the actual observation count")
    if budgets[-1] > len(pool_ids) or any(b - a != 2 for a, b in zip(budgets, budgets[1:])):
        raise ValueError("Every budget increment must reveal one complete batch within pool")
    locations = dataset.locations(site_id)
    known = set(locations.location_id.astype(str))
    if not (set(pool_ids) | set(evaluation_ids)) <= known:
        raise ValueError("Replay IDs must belong to the requested site")
    if "near_group_id" in locations:
        eval_groups = set(locations.loc[locations.location_id.isin(evaluation_ids), "near_group_id"].dropna())
        pool_groups = set(locations.loc[locations.location_id.isin(pool_ids), "near_group_id"].dropna())
        if eval_groups & pool_groups:
            raise ValueError("Holdout and action pool share a near-neighbor group")
    if policy != "maximin" and proxy is None:
        if getattr(predictor, "model_id", None) == "gp":
            proxy = predictor
        else:
            raise ValueError("GP acquisition for another predictor requires a source-trained GP proxy")
    transform = None
    if proxy_calibrator is not None:
        transform = lambda probabilities: transform_probability(probabilities, proxy_calibrator)
    threshold = float(config.get("v2", {}).get("risk", {}).get("decision_threshold",
                      config.get("risk", {}).get("threshold", 0.2)))
    coords = locations[["location_id", "x_m", "y_m"]].copy()
    coords["location_id"] = coords.location_id.astype(str)
    groups = locations.get("investigation_group", pd.Series(dtype=str)).dropna().unique()
    if len(groups) > 1:
        raise ValueError("A replay site must belong to exactly one investigation group")
    investigation_group = str(groups[0]) if len(groups) else str(site_id)
    spacing = float(split["min_spacing_m"])
    if not np.isfinite(spacing) or spacing < 0:
        raise ValueError("Replay spacing must be finite and nonnegative")
    saved = {name: [] for name in ("predictions", "metrics", "recommendations", "acquisition_scores", "states")}
    role = split.get("role", "test")
    for round_id, budget in enumerate(budgets):
        remaining_ids = sorted(set(pool_ids) - set(observed_ids))
        observed = dataset.observed(observed_ids)
        predictor.condition(observed)
        eval_bundle = predictor.predict(dataset.query(evaluation_ids))
        if calibrator is not None:
            eval_bundle = PredictionBundle(apply_risk(eval_bundle.frame, calibrator, threshold=threshold), eval_bundle.model_id)
        common = dict(fold_id=fold_id, site_id=site_id, seed=seed, model_id=eval_bundle.model_id,
                      policy_id=policy, budget_n=budget, investigation_group=investigation_group,
                      selection_model_id="geometry" if policy == "maximin" else "gp",
                      study_version="exploratory_v2", target_version="v2_pending_joint_target",
                      risk_calibration_status="frozen_validation" if calibrator is not None else "not_applied")
        observed_set_id = _state_id(observed_ids)
        saved["states"].append({**common, "round_id": round_id, "observed_set_id": observed_set_id,
                                "observed_ids": observed_ids.copy(), "remaining_ids": remaining_ids.copy(),
                                "evaluation_ids": evaluation_ids.copy(), "role": role})
        truth = dataset.truth(evaluation_ids)
        metrics = {**evaluate(eval_bundle, truth, config), **common, "round_id": round_id,
                   "n_observed_locations": len(observed_ids), "role": role}
        if calibrator is not None:
            metrics.update(evaluate_risk(eval_bundle.frame, truth, config))
        saved["metrics"].append(metrics)
        eval_frame = join_evaluation(eval_bundle, truth)
        saved["predictions"].append(eval_frame.assign(**common, round_id=round_id, role="evaluation", split_role=role))
        if round_id == len(budgets) - 1:
            continue
        candidate_query = dataset.query(remaining_ids)
        candidate_bundle = predictor.predict(candidate_query)
        keys = ["location_id", "depth_index"]
        key_check = candidate_query[keys].merge(candidate_bundle.frame[keys], on=keys,
                                               how="outer", validate="one_to_one", indicator=True)
        if not (key_check._merge == "both").all():
            raise ValueError("Every candidate depth cell needs a prediction")
        candidate_frame = candidate_bundle.frame.drop(columns=["truth_qc_mpa", "truth_log_qc", "qc_mpa", "log_qc"], errors="ignore")
        if calibrator is not None:
            candidate_frame = apply_risk(candidate_frame, calibrator, threshold=threshold)
        saved["predictions"].append(candidate_frame.assign(**common, round_id=round_id, role="candidate", split_role=role))
        observed_coordinates = coords.loc[coords.location_id.isin(observed_ids)]
        if policy == "maximin":
            selected, scores = select_batch(observed_coordinates, coords.loc[coords.location_id.isin(remaining_ids)],
                                            None, config, policy="maximin", min_spacing_m=spacing, batch_size=2)
        else:
            if proxy is not predictor:
                proxy.condition(observed)
            selected, scores = select_gp_batch(proxy, observed_coordinates, candidate_query, config,
                                                policy=policy, min_spacing_m=spacing,
                                                probability_transform=transform)
        scores = scores.assign(**common, round_id=round_id, observed_set_id=observed_set_id, role=role)
        saved["acquisition_scores"].append(scores)
        recommendations = scores.loc[scores.selected].sort_values("selection_order").copy()
        recommendations["rank_in_batch"] = recommendations.selection_order
        recommendations["score"] = recommendations.selection_score
        recommendations["next_budget_n"] = budgets[round_id + 1]
        saved["recommendations"].append(recommendations)
        # Only this full-batch mutation changes which truth can be revealed.
        observed_ids = observed_ids + selected
        if len(observed_ids) != budgets[round_id + 1]:
            raise RuntimeError("Selection failed to produce a complete batch")
    return {name: (pd.concat(rows, ignore_index=True) if rows and isinstance(rows[0], pd.DataFrame)
                   else pd.DataFrame(rows)) for name, rows in saved.items()}
