"""Label-free, group-atomic acquisition and online stopping decisions.

Only coordinates, predictions and already observed support diagnostics enter
this module.  The replay runner owns all curve revelation and audit scoring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping

import numpy as np
import pandas as pd


def rank01(values) -> np.ndarray:
    """Ascending average ranks on [0, 1]; all ties (including n=1) are 0.5."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Ranks require a finite one-dimensional array")
    if len(values) < 2 or np.ptp(values) == 0:
        return np.full(len(values), 0.5)
    return (pd.Series(values).rank(method="average").to_numpy() - 1) / (len(values) - 1)


def _geometry(geometry: pd.DataFrame) -> pd.DataFrame:
    required = ["location_id", "x_m", "y_m"]
    if not set(required).issubset(geometry):
        raise ValueError(f"Geometry requires {required}")
    frame = geometry[required].copy()
    frame["location_id"] = frame.location_id.astype(str)
    frame = frame.drop_duplicates()
    if frame.location_id.duplicated().any():
        raise ValueError("A location has inconsistent coordinates")
    if not np.isfinite(frame[["x_m", "y_m"]].to_numpy(float)).all():
        raise ValueError("Geometry must use finite metric coordinates")
    return frame.set_index("location_id").sort_index()


def _groups(near_groups, geometry: pd.DataFrame) -> dict[str, str]:
    """Accept group->members, location->group, or a location/group DataFrame."""
    if near_groups is None:
        if "near_group_id" not in geometry:
            return {str(loc): str(loc) for loc in geometry.location_id}
        near_groups = geometry[["location_id", "near_group_id"]]
    if isinstance(near_groups, pd.DataFrame):
        frame = near_groups[["location_id", "near_group_id"]].astype(str).drop_duplicates()
        if frame.location_id.duplicated().any():
            raise ValueError("A location belongs to multiple near groups")
        return dict(zip(frame.location_id, frame.near_group_id))
    result = {}
    for key, value in near_groups.items():
        if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
            for location in value:
                location = str(location)
                if location in result and result[location] != str(key):
                    raise ValueError("A location belongs to multiple near groups")
                result[location] = str(key)
        else:
            result[str(key)] = str(value)
    return result


def _ids(values, label: str) -> list[str]:
    result = [str(value) for value in values]
    if len(result) != len(set(result)):
        raise ValueError(f"{label} location IDs must be unique")
    return result


def _auxiliary(values, ids, name: str, required: bool) -> np.ndarray:
    if not required:
        return np.zeros(len(ids))
    if values is None:
        raise ValueError(f"Adaptive acquisition requires {name}")
    if isinstance(values, (Mapping, pd.Series)):
        by_id = {str(key): value for key, value in values.items()}
        missing = set(ids) - set(by_id)
        if missing:
            raise ValueError(f"Missing {name} for candidates: {sorted(missing)}")
        array = np.array([by_id[loc] for loc in ids], dtype=float)
    else:
        array = np.asarray(values, dtype=float)
    if array.shape != (len(ids),) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite and align with candidate_ids")
    if (array < 0).any():
        raise ValueError(f"{name} must be nonnegative")
    if name == "disagreement" and (array > 1).any():
        raise ValueError("Low-value disagreement fractions must be at most one")
    return array


def select_batch(
    geometry: pd.DataFrame,
    known_ids,
    candidate_ids,
    near_groups=None,
    uncertainty=None,
    disagreement=None,
    weights=(0.5, 0.3, 0.2),
    batch_size: int = 2,
    *,
    policy: str = "adaptive",
):
    """Return ``(selected_ids, per_location_scores)`` without reading curves.

    Groups are indivisible.  Their U/D are member means and S is the minimum
    member distance to an observed point.  A nominal two-location batch can
    exceed two when a selected near group has multiple locations. U/D and
    their ranks stay fixed throughout the batch; only distances and S-ranks
    change after each group selection. Ties use group ID, then location ID.
    ``maximin`` uses the same atomic groups and raw S as its score.
    """
    if policy not in {"adaptive", "maximin"}:
        raise ValueError("Policy must be adaptive or maximin")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (3,) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("weights must contain three finite nonnegative values with positive sum")
    weights = weights / weights.sum()
    coords = _geometry(geometry)
    known = _ids(known_ids, "Known")
    candidates = _ids(candidate_ids, "Candidate")
    if not known:
        raise ValueError("Acquisition requires observed locations")
    if set(known) & set(candidates):
        raise ValueError("Observed locations cannot be candidates")
    if not set(known + candidates).issubset(coords.index):
        raise ValueError("Known/candidate IDs must have coordinates")
    group_map = _groups(near_groups, geometry)
    if not set(coords.index).issubset(group_map):
        raise ValueError("Every geometry location requires a near group")
    pools = {"known": set(known), "candidate": set(candidates)}
    for label, pool in pools.items():
        for group in {group_map[loc] for loc in pool}:
            members = {loc for loc in coords.index if group_map[loc] == group}
            if not members.issubset(pool):
                raise ValueError(f"The {label} set splits near group {group}")
    columns = ["location_id", "near_group_id", "group_size", "uncertainty", "disagreement",
               "rank_uncertainty", "rank_disagreement", "distance_to_known_m", "rank_distance",
               "score", "selected", "selection_order", "selection_score", "selection_distance_m"]
    if not candidates:
        return [], pd.DataFrame(columns=columns)
    values = pd.DataFrame({"location_id": candidates,
                           "near_group_id": [group_map[loc] for loc in candidates],
                           "uncertainty": _auxiliary(uncertainty, candidates, "uncertainty", policy == "adaptive" and weights[0] > 0),
                           "disagreement": _auxiliary(disagreement, candidates, "disagreement", policy == "adaptive" and weights[1] > 0)})
    group = values.groupby("near_group_id", sort=True).agg(
        uncertainty=("uncertainty", "mean"), disagreement=("disagreement", "mean"),
        group_size=("location_id", "size"), first_location_id=("location_id", "min"))
    group["rank_uncertainty"] = rank01(group.uncertainty)
    group["rank_disagreement"] = rank01(group.disagreement)
    group["selected"] = False
    group["selection_order"] = pd.Series(pd.NA, index=group.index, dtype="Int64")
    group["selection_score"] = np.nan
    group["selection_distance_m"] = np.nan
    observed_xy = coords.loc[known, ["x_m", "y_m"]].to_numpy(float)
    remaining = list(group.index)
    chosen = []
    chosen_count = 0
    first_round = True
    while remaining and chosen_count < batch_size:
        distances = []
        for group_id in remaining:
            ids = values.loc[values.near_group_id == group_id, "location_id"]
            xy = coords.loc[ids, ["x_m", "y_m"]].to_numpy(float)
            distances.append(float(np.sqrt(((xy[:, None] - observed_xy[None, :]) ** 2).sum(axis=2)).min()))
        ranked_distance = rank01(distances)
        work = group.loc[remaining].copy()
        work["current_distance"] = distances
        work["current_rank_distance"] = ranked_distance
        work["current_score"] = (weights[0] * work.rank_uncertainty + weights[1] * work.rank_disagreement
                                 + weights[2] * ranked_distance) if policy == "adaptive" else distances
        if first_round:
            group.loc[remaining, "distance_to_known_m"] = distances
            group.loc[remaining, "rank_distance"] = ranked_distance
            group.loc[remaining, "score"] = work.current_score
            first_round = False
        selected_group = work.reset_index().sort_values(
            ["current_score", "near_group_id", "first_location_id"],
            ascending=[False, True, True], kind="stable").iloc[0]
        gid = selected_group.near_group_id
        members = sorted(values.loc[values.near_group_id == gid, "location_id"])
        chosen.extend(members)
        chosen_count += len(members)
        group.loc[gid, ["selected", "selection_order", "selection_score", "selection_distance_m"]] = (
            True, len(group.loc[group.selected]) + 1,
            float(selected_group.current_score), float(selected_group.current_distance))
        # Coordinate update only: no conditioning or label callback is accepted.
        observed_xy = np.vstack([observed_xy, coords.loc[members, ["x_m", "y_m"]].to_numpy(float)])
        remaining.remove(gid)
    scores = values[["location_id", "near_group_id"]].merge(
        group.drop(columns="first_location_id"), left_on="near_group_id", right_index=True, validate="many_to_one")
    return chosen, scores[columns].sort_values("location_id", kind="stable").reset_index(drop=True)


def _prediction_frame(predictions) -> pd.DataFrame:
    frame = predictions.frame if hasattr(predictions, "frame") else predictions
    required = ["location_id", "depth_index", "pred_log_qc"]
    if not set(required).issubset(frame):
        raise ValueError(f"Change computation requires {required}")
    out = frame[required].copy()
    out["location_id"] = out.location_id.astype(str)
    if out.duplicated(["location_id", "depth_index"]).any():
        raise ValueError("Fixed-grid predictions must have unique keys")
    if not np.isfinite(out.pred_log_qc.to_numpy(float)).all():
        raise ValueError("Fixed-grid predictions must be finite")
    return out.set_index(["location_id", "depth_index"]).sort_index()


def compute_change(previous_predictions, current_predictions, depth_step=0.25, tau=1.0) -> dict[str, float]:
    """Prediction-only q90 changes on exactly the same fixed spatial/depth G."""
    if not np.isfinite([depth_step, tau]).all() or depth_step <= 0 or tau <= 0:
        raise ValueError("Depth step and low-qc threshold must be positive")
    previous = _prediction_frame(previous_predictions)
    current = _prediction_frame(current_predictions)
    if previous.empty or not previous.index.equals(current.index):
        raise ValueError("Consecutive predictions must use the same nonempty fixed grid G")
    delta_q = np.abs(current.pred_log_qc.to_numpy() - previous.pred_log_qc.to_numpy())
    previous_h = (previous.pred_log_qc < np.log(tau)).groupby(level="location_id").sum() * depth_step
    current_h = (current.pred_log_qc < np.log(tau)).groupby(level="location_id").sum() * depth_step
    return {"delta_q90": float(np.quantile(delta_q, 0.9)),
            "delta_h90": float(np.quantile(np.abs(current_h - previous_h), 0.9))}


def coverage_fraction(geometry, known_ids, full_ids, mandatory_ids=()) -> float:
    """Fraction of populated full-plan quadrants and mandatory points covered.

    Quadrants use the full-plan bounding-box midpoint, fixed for the replay.
    Degenerate coordinate axes naturally leave some quadrants unpopulated.
    """
    coords = _geometry(geometry)
    known, full, mandatory = set(_ids(known_ids, "Known")), set(_ids(full_ids, "Full")), set(_ids(mandatory_ids, "Mandatory"))
    if not full or not known.issubset(full) or not mandatory.issubset(full) or not full.issubset(coords.index):
        raise ValueError("Coverage requires a nonempty full plan containing known and mandatory IDs")
    xy = coords.loc[sorted(full), ["x_m", "y_m"]]
    midpoint = (xy.min() + xy.max()) / 2
    quadrants = ((xy.x_m >= midpoint.x_m).astype(int) + 2 * (xy.y_m >= midpoint.y_m).astype(int))
    required = set(quadrants)
    covered = set(quadrants.loc[sorted(known)])
    return (len(required & covered) + len(mandatory & known)) / (len(required) + len(mandatory))


def support_leaveout_groups(known_ids, near_groups, *, max_groups=5, fixed_group_ids=None) -> dict[str, list[str]]:
    """Choose at most five whole observed groups for online LGO diagnostics.

    Freeze the keys returned at the initial state and pass them back through
    ``fixed_group_ids`` on later rounds. No candidate or audit curves enter.
    """
    known = _ids(known_ids, "Known")
    if not isinstance(max_groups, (int, np.integer)) or max_groups < 1:
        raise ValueError("max_groups must be positive")
    mapping = _groups(near_groups, pd.DataFrame({"location_id": known}))
    if not set(known).issubset(mapping):
        raise ValueError("Observed locations require near-group membership")
    for group in {mapping[loc] for loc in known}:
        if any(loc not in known for loc, gid in mapping.items() if gid == group):
            raise ValueError(f"The known set splits near group {group}")
    observed = {}
    for loc in known:
        observed.setdefault(mapping[loc], []).append(loc)
    keys = sorted(observed) if fixed_group_ids is None else [str(group) for group in fixed_group_ids]
    if fixed_group_ids is not None and not set(keys).issubset(observed):
        raise ValueError("Fixed diagnostic groups must already be fully observed")
    keys = [group for group in keys if group in observed][:max_groups]
    return {group: sorted(observed[group]) for group in keys}


@dataclass(frozen=True)
class StopRule:
    """Thresholds must be frozen on development data, never audit labels."""
    delta_q90_max: float
    delta_h90_max: float
    support_log_mae_max: float
    uncertainty_q90_max: float
    min_observed: int = 9
    consecutive_rounds: int = 2

    def __post_init__(self):
        limits = [self.delta_q90_max, self.delta_h90_max, self.support_log_mae_max, self.uncertainty_q90_max]
        if not np.isfinite(limits).all() or min(limits) < 0:
            raise ValueError("Stop thresholds must be finite and nonnegative")
        if self.min_observed < 1 or self.consecutive_rounds < 1:
            raise ValueError("Minimum observations and consecutive rounds must be positive")


@dataclass(frozen=True)
class StopSignals:
    delta_q90: float
    delta_h90: float
    support_log_mae: float
    uncertainty_q90: float
    coverage_ok: bool


@dataclass(frozen=True)
class StopState:
    pass_streak: int = 0
    stopped: bool = False
    stop_reason: str | None = None
    n_observed: int = 0
    round_index: int = -1
    checks: dict[str, bool] = field(default_factory=dict)


def update_stop(rule: StopRule, state: StopState, signals: StopSignals | Mapping, n_observed: int, remaining_n: int) -> StopState:
    """Advance one completed observation round using online signals only.

    A full plan has its own reason even if stability thresholds also pass.
    Missing/nonfinite numerical diagnostics fail their check and reset streak.
    Calling this again after a terminal state preserves that decision.
    """
    if state.stopped:
        return state
    if n_observed < 0 or remaining_n < 0 or n_observed < state.n_observed:
        raise ValueError("Observed counts cannot decrease and counts cannot be negative")
    if state.round_index >= 0 and n_observed == state.n_observed:
        raise ValueError("A new stopping round requires newly observed locations")
    if isinstance(signals, Mapping):
        signals = StopSignals(**signals)
    checks = {"minimum_observed": n_observed >= rule.min_observed,
              "coverage": bool(signals.coverage_ok)}
    for name in ("delta_q90", "delta_h90", "support_log_mae", "uncertainty_q90"):
        value = getattr(signals, name)
        checks[name] = bool(np.isfinite(value) and value >= 0 and value <= getattr(rule, name + "_max"))
    streak = state.pass_streak + 1 if all(checks.values()) else 0
    reason = ("full_plan_complete" if remaining_n == 0 else
              "online_stability" if streak >= rule.consecutive_rounds else None)
    return StopState(streak, reason is not None, reason, n_observed, state.round_index + 1, checks)
