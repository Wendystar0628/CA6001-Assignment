"""Fixed-holdout, batch-reveal replay. Selection has no access to hidden truth."""
from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd

from .acquisition import select_batch
from .evaluate import evaluate, join_evaluation


def _state_id(ids):
    return hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()[:16]


def run_replay(dataset, split, predictor, config, *, policy, fold_id=0):
    site_id, seed = split["site_id"], int(split["seed"])
    evaluation_ids = list(map(str, split["evaluation_ids"]))
    pool_ids = list(map(str, split["pool_ids"]))
    observed_ids = list(map(str, split["initial_ids"]))
    budgets = [int(n) for n in split.get("budgets", config.get("replay", {}).get("budgets", [5, 7, 9, 11, 13]))]
    batch_size = int(config.get("replay", {}).get("batch_size", 2))
    if not evaluation_ids or len(evaluation_ids) != len(set(evaluation_ids)):
        raise ValueError("Replay requires a nonempty unique fixed evaluation set")
    if len(pool_ids) != len(set(pool_ids)) or len(observed_ids) != len(set(observed_ids)):
        raise ValueError("Pool and initial location IDs must be unique")
    if set(evaluation_ids) & set(pool_ids) or not set(observed_ids) <= set(pool_ids):
        raise ValueError("Holdout and action pool must be disjoint; initial points belong to the pool")
    if budgets != sorted(set(budgets)) or not budgets or budgets[0] != len(observed_ids):
        raise ValueError("Budgets must increase from the actual initial observation count")
    if budgets[-1] > len(pool_ids) or any(b - a != batch_size for a, b in zip(budgets, budgets[1:])):
        raise ValueError("Every budget increment must be one complete batch within the pool")
    locations = dataset.locations(site_id)
    known = set(locations.location_id.astype(str))
    if not (set(pool_ids) | set(evaluation_ids)) <= known:
        raise ValueError("Replay IDs must all belong to the requested site")
    if "near_group_id" in locations:
        eval_groups = set(locations.loc[locations.location_id.isin(evaluation_ids), "near_group_id"].dropna())
        pool_groups = set(locations.loc[locations.location_id.isin(pool_ids), "near_group_id"].dropna())
        if eval_groups & pool_groups:
            raise ValueError("A near-neighbor group was split between holdout and action pool")
    coords = locations[["location_id", "x_m", "y_m"]].copy()
    coords["location_id"] = coords.location_id.astype(str)
    spacing = float(split["min_spacing_m"])
    saved = {name: [] for name in ("predictions", "metrics", "recommendations", "acquisition_scores", "states")}
    role = split.get("role", "test")
    for round_id, budget in enumerate(budgets):
        remaining_ids = sorted(set(pool_ids) - set(observed_ids))
        # Only this call reveals the already selected full batch, never one point mid-selection.
        predictor.condition(dataset.observed(observed_ids))
        eval_bundle = predictor.predict(dataset.query(evaluation_ids))
        model_id = eval_bundle.model_id
        common = dict(fold_id=fold_id, site_id=site_id, seed=seed, model_id=model_id,
                      policy_id=policy, budget_n=budget)
        observed_set_id = _state_id(observed_ids)
        state = {**common, "round_id": round_id, "observed_set_id": observed_set_id,
                 "observed_ids": observed_ids.copy(), "remaining_ids": remaining_ids.copy(),
                 "evaluation_ids": evaluation_ids.copy(), "role": role}
        saved["states"].append(state)
        truth = dataset.truth(evaluation_ids)
        metrics = {**evaluate(eval_bundle, truth, config), **common, "round_id": round_id,
                   "n_observed_locations": len(observed_ids), "role": role}
        saved["metrics"].append(metrics)
        eval_frame = join_evaluation(eval_bundle, truth)
        saved["predictions"].append(eval_frame.assign(**common, round_id=round_id, role="evaluation", split_role=role))
        if round_id == len(budgets) - 1:
            continue
        candidate_query = dataset.query(remaining_ids)
        candidate_bundle = predictor.predict(candidate_query)
        candidate_keys = ["location_id", "depth_index"]
        predicted_keys = candidate_bundle.frame[candidate_keys]
        key_check = candidate_query[candidate_keys].merge(predicted_keys, on=candidate_keys,
                                                         how="outer", validate="one_to_one", indicator=True)
        if not (key_check._merge == "both").all():
            raise ValueError("Every candidate depth cell needs a prediction before scoring")
        candidate_frame = candidate_bundle.frame.drop(
            columns=["truth_qc_mpa", "truth_log_qc", "qc_mpa", "log_qc"], errors="ignore")
        saved["predictions"].append(candidate_frame.assign(**common, round_id=round_id, role="candidate", split_role=role))
        selected, scores = select_batch(
            coords.loc[coords.location_id.isin(observed_ids)],
            coords.loc[coords.location_id.isin(remaining_ids)], candidate_bundle, config,
            policy=policy, min_spacing_m=spacing, batch_size=batch_size)
        scores = scores.assign(**common, round_id=round_id, observed_set_id=observed_set_id, role=role)
        saved["acquisition_scores"].append(scores)
        recommendations = scores.loc[scores.selected].sort_values("selection_order").copy()
        recommendations["rank_in_batch"] = recommendations.selection_order
        recommendations["score"] = recommendations.selection_score
        recommendations["next_budget_n"] = budgets[round_id + 1]
        saved["recommendations"].append(recommendations)
        # This mutation happens only after both recommendations have been saved.
        observed_ids = observed_ids + selected
        if len(observed_ids) != budgets[round_id + 1]:
            raise RuntimeError("Selection failed to produce the promised complete batch")
    return {name: (pd.concat(rows, ignore_index=True) if rows and isinstance(rows[0], pd.DataFrame)
                   else pd.DataFrame(rows)) for name, rows in saved.items()}
