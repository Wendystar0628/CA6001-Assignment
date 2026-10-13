"""Coordinate-only and predictive batch acquisition with no truth access."""
from __future__ import annotations

import numpy as np
import pandas as pd


POLICIES = ("maximin", "uncertainty", "threshold_uncertainty")


def _coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["location_id", "x_m", "y_m"]
    if not set(required).issubset(frame.columns):
        raise ValueError(f"Coordinates require {required}")
    out = frame[required].copy()
    out["location_id"] = out["location_id"].astype(str)
    out = out.drop_duplicates()
    if out.location_id.duplicated().any():
        raise ValueError("A location has inconsistent coordinates")
    if not np.isfinite(out[["x_m", "y_m"]].to_numpy(float)).all():
        raise ValueError("Acquisition requires finite metric coordinates")
    return out.sort_values("location_id", kind="stable").reset_index(drop=True)


def _nearest(points: np.ndarray, observed: np.ndarray) -> np.ndarray:
    if not len(observed):
        raise ValueError("Acquisition requires at least one observed location")
    return np.sqrt(((points[:, None] - observed[None, :]) ** 2).sum(axis=2)).min(axis=1)


def select_batch(observed_coordinates, candidate_coordinates, predictions, config,
                 *, policy, min_spacing_m, batch_size=2):
    """Select a complete batch before either point can be revealed.

    ``score`` describes the initial state; ``selection_score`` records the
    actual greedy score when chosen (which changes for maximin's second point).
    All ties are resolved by the stable string location ID.
    """
    if policy not in POLICIES:
        raise ValueError(f"Unknown acquisition policy: {policy}")
    if batch_size not in (1, 2):
        raise ValueError("The configured replay supports batches of one or two")
    if not np.isfinite(min_spacing_m) or min_spacing_m < 0:
        raise ValueError("min_spacing_m must be finite and nonnegative")
    observed = _coordinates(observed_coordinates)
    candidates = _coordinates(candidate_coordinates)
    if set(observed.location_id) & set(candidates.location_id):
        raise ValueError("Already observed locations cannot be selected")
    if len(candidates) < batch_size:
        raise ValueError("Insufficient remaining candidates for the requested batch")
    xy = candidates[["x_m", "y_m"]].to_numpy(float)
    distance = _nearest(xy, observed[["x_m", "y_m"]].to_numpy(float))
    scores = candidates.copy()
    scores["distance_to_observed_m"] = distance
    scores["threshold_bonus"] = 0.0
    scores["crossing_fraction"] = np.nan
    if policy == "maximin":
        scores["base_score"] = distance
        scores["uncertainty_term"] = 0.0
    else:
        if predictions is None:
            raise ValueError(f"{policy} requires a predictive distribution")
        frame = predictions.frame if hasattr(predictions, "frame") else predictions
        if not {"location_id", "q05_log", "q95_log"}.issubset(frame.columns):
            raise ValueError("Uncertainty policies require genuine q05_log/q95_log")
        frame = frame.copy()
        frame["location_id"] = frame.location_id.astype(str)
        if set(frame.location_id) != set(candidates.location_id):
            raise ValueError("Prediction locations must exactly match candidates")
        if "depth_index" in frame and frame.duplicated(["location_id", "depth_index"]).any():
            raise ValueError("Duplicate candidate depth predictions")
        bounds = frame[["q05_log", "q95_log"]].to_numpy(float)
        if not np.isfinite(bounds).all() or (bounds[:, 1] < bounds[:, 0]).any():
            raise ValueError("Prediction intervals must be finite and ordered")
        step = float(config.get("data", {}).get("depth_step_m", 0.25))
        tau = float(config.get("data", {}).get("tau_mpa", 1.0))
        weight = float(config.get("replay", {}).get("threshold_weight", 1.0))
        if step <= 0 or tau <= 0 or weight < 0:
            raise ValueError("Depth step/tau must be positive and threshold weight nonnegative")
        frame["width_term"] = step * (bounds[:, 1] - bounds[:, 0])
        frame["crosses"] = (bounds[:, 0] < np.log(tau)) & (np.log(tau) < bounds[:, 1])
        frame["bonus"] = frame.width_term * frame.crosses * (weight if policy == "threshold_uncertainty" else 0.0)
        grouped = frame.groupby("location_id", sort=False).agg(
            base_score=("width_term", "sum"), bonus=("bonus", "sum"),
            crossing_fraction=("crosses", "mean"))
        scores["base_score"] = scores.location_id.map(grouped.base_score)
        scores["threshold_bonus"] = scores.location_id.map(grouped.bonus)
        scores["crossing_fraction"] = scores.location_id.map(grouped.crossing_fraction)
        scores["uncertainty_term"] = scores.base_score
    scores["score"] = scores.base_score + scores.threshold_bonus
    scores["threshold_term"] = scores.threshold_bonus
    scores["total_score"] = scores.score
    scores["selected"] = False
    scores["selection_order"] = pd.Series(pd.NA, index=scores.index, dtype="Int64")
    scores["selection_score"] = np.nan
    scores["second_stage_score"] = np.nan
    scores["spacing_fallback"] = False
    scores["eligible_for_second"] = False
    scores["distance_to_first_m"] = np.nan
    first = int(scores.sort_values(["score", "location_id"], ascending=[False, True], kind="stable").index[0])
    selected = [first]
    scores.loc[first, ["selected", "selection_order", "selection_score"]] = [True, 1, scores.loc[first, "score"]]
    if batch_size == 2:
        d_first = np.sqrt(((xy - xy[first]) ** 2).sum(axis=1))
        second_scores = np.minimum(distance, d_first) if policy == "maximin" else scores.score.to_numpy(float)
        scores["distance_to_first_m"] = d_first
        scores["second_stage_score"] = second_scores
        scores["eligible_for_second"] = (d_first >= min_spacing_m)
        scores.loc[first, "eligible_for_second"] = False
        remaining = scores.drop(index=first)
        eligible = remaining[remaining.eligible_for_second]
        fallback = eligible.empty
        if fallback:
            eligible = remaining.sort_values(["distance_to_first_m", "second_stage_score", "location_id"], ascending=[False, False, True], kind="stable")
        else:
            eligible = eligible.sort_values(["second_stage_score", "location_id"], ascending=[False, True], kind="stable")
        second = int(eligible.index[0])
        selected.append(second)
        scores.loc[second, ["selected", "selection_order", "selection_score", "spacing_fallback"]] = [True, 2, second_scores[second], fallback]
    scores["selected_rank"] = scores.selection_order
    scores["spacing_relaxed"] = scores.spacing_fallback
    return scores.loc[selected, "location_id"].tolist(), scores
