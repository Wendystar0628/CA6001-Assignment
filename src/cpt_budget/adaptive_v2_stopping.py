"""Online stopping diagnostics that only use revealed measurements.

This module changes the *measurement* of stopping signals, not the registered
audit quality targets.  A stable prediction is not evidence of low error.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .adaptive_policy import _geometry, _groups, _ids, compute_change


def remaining_grid_signals(previous, current, remaining_ids, *, depth_step=.25, tau=1.):
    """Measure change and uncertainty on positions still unobserved *now*.

    Both predictions are restricted to the same remaining set.  Newly revealed
    positions are excluded from both sides.  Audit positions are never required
    and should not be passed in ``remaining_ids``. The GP's native uncertainty
    may be attached as ``std_log_qc`` on ``current`` by the caller.
    """
    ids = set(_ids(remaining_ids, "Remaining"))
    frame = current.frame if hasattr(current, "frame") else current
    subset = frame.loc[frame.location_id.astype(str).isin(ids)].copy()
    if not ids:
        return {"delta_q90": np.nan, "delta_h90": np.nan,
                "uncertainty_q90": np.nan, "remaining_locations": 0}
    if set(subset.location_id.astype(str)) != ids:
        raise ValueError("Current predictions must cover every remaining position")
    signals = {"delta_q90": np.nan, "delta_h90": np.nan}
    if previous is not None:
        old = previous.frame if hasattr(previous, "frame") else previous
        old = old.loc[old.location_id.astype(str).isin(ids)]
        signals = compute_change(old, subset, depth_step=depth_step, tau=tau)
    uncertainty = np.nan
    if "std_log_qc" in subset:
        values = subset.std_log_qc.to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Native uncertainty must be finite and nonnegative")
        uncertainty = float(np.quantile(values, .9))
    return {**signals, "uncertainty_q90": uncertainty, "remaining_locations": len(ids)}


def current_support_groups(geometry, known_ids, near_groups, *, max_groups=5):
    """Deterministic spatially spread LGO groups from the *current* support.

    ``max_groups=None`` audits every observed group. A finite budget starts at
    the group nearest the observed centroid, then picks farthest centroids.
    Selection uses coordinates only and cannot select a group by its error.
    """
    known = _ids(known_ids, "Known")
    coords = _geometry(geometry)
    if not set(known).issubset(coords.index):
        raise ValueError("Known positions require coordinates")
    mapping = _groups(near_groups, geometry)
    groups = {}
    for loc in known:
        if loc not in mapping:
            raise ValueError("Known positions require near-group membership")
        groups.setdefault(mapping[loc], []).append(loc)
    for group, members in groups.items():
        if {loc for loc, key in mapping.items() if key == group} != set(members):
            raise ValueError("Support diagnostics must leave out whole near groups")
    if max_groups is not None and (not isinstance(max_groups, int) or max_groups < 1):
        raise ValueError("max_groups must be a positive integer or None")
    keys = sorted(groups)
    if max_groups is None or len(keys) <= max_groups:
        return {key: sorted(groups[key]) for key in keys}
    centers = np.array([coords.loc[groups[key], ["x_m", "y_m"]].mean().to_numpy(float) for key in keys])
    chosen = [int(np.argmin(np.linalg.norm(centers - centers.mean(axis=0), axis=1)))]
    while len(chosen) < max_groups:
        distance = np.linalg.norm(centers[:, None] - centers[chosen][None, :], axis=2).min(axis=1)
        distance[chosen] = -1.
        chosen.append(int(np.argmax(distance)))
    return {keys[index]: sorted(groups[keys[index]]) for index in chosen}


def support_quality_signals(observed, geometry, near_groups, predictor, query, *,
                            remaining_ids=(), max_groups=5, depth_step=.25, tau=1.):
    """LGO log/thickness error and geometric representativeness of known data.

    ``predictor`` is separate from the live replay suite. All input curves must
    already be revealed. ``query`` accepts a list of location IDs and returns
    geometry only. Returns ``(signals, per_location_errors)``. Candidate risk is
    estimated by matching candidate distance-to-known with held-out-group
    distance-to-rest. Extrapolation beyond audited distance support is explicit.
    """
    if not np.isfinite([depth_step, tau]).all() or depth_step <= 0 or tau <= 0:
        raise ValueError("Positive depth step and tau are required")
    known = sorted(observed.location_id.astype(str).unique())
    remaining = _ids(remaining_ids, "Remaining")
    if set(known) & set(remaining):
        raise ValueError("Remaining positions must be unobserved")
    coords = _geometry(geometry)
    if not set(remaining).issubset(coords.index):
        raise ValueError("Remaining positions require coordinates")
    groups = current_support_groups(geometry, known, near_groups, max_groups=max_groups)
    records = []
    for group, members in groups.items():
        mask = observed.location_id.astype(str).isin(members)
        context = observed.loc[~mask].reset_index(drop=True)
        if context.empty:
            continue
        truth = observed.loc[mask, ["location_id", "depth_index", "log_qc"]].copy()
        request = query(members)
        if {"log_qc", "qc_mpa", "truth", "true_log_qc"} & set(request):
            raise ValueError("Support query must contain geometry only")
        predictor.condition(context)
        prediction = predictor.predict(request)
        prediction = prediction.frame if hasattr(prediction, "frame") else prediction
        merged = truth.merge(prediction[["location_id", "depth_index", "pred_log_qc"]],
                             on=["location_id", "depth_index"], validate="one_to_one")
        if len(merged) != len(truth) or not np.isfinite(merged.pred_log_qc.to_numpy(float)).all():
            raise ValueError("LGO predictions must cover every held-out depth")
        rest = coords.loc[sorted(set(known) - set(members)), ["x_m", "y_m"]].to_numpy(float)
        for loc, rows in merged.groupby("location_id", sort=True):
            point = coords.loc[str(loc), ["x_m", "y_m"]].to_numpy(float)
            records.append({"near_group_id": group, "location_id": str(loc),
                            "log_mae": float(abs(rows.log_qc - rows.pred_log_qc).mean()),
                            "thickness_mae_m": float(abs((rows.log_qc < np.log(tau)).sum()
                                                         - (rows.pred_log_qc < np.log(tau)).sum()) * depth_step),
                            "leaveout_distance_m": float(np.linalg.norm(rest - point, axis=1).min())})
    detail = pd.DataFrame(records, columns=["near_group_id", "location_id", "log_mae",
                                          "thickness_mae_m", "leaveout_distance_m"])
    result = {"support_log_mae": np.nan, "support_thickness_mae_m": np.nan,
              "matched_support_log_mae": np.nan, "matched_support_thickness_mae_m": np.nan,
              "support_groups": int(detail.near_group_id.nunique()),
              "observed_groups": len(current_support_groups(geometry, known, near_groups, max_groups=None)),
              "candidate_distance_q90_m": np.nan, "support_distance_max_m": np.nan,
              "candidate_geometry_supported": False}
    if not detail.empty:
        result.update(support_log_mae=float(detail.log_mae.mean()),
                      support_thickness_mae_m=float(detail.thickness_mae_m.mean()),
                      support_distance_max_m=float(detail.leaveout_distance_m.max()))
        if remaining:
            xy = coords.loc[remaining, ["x_m", "y_m"]].to_numpy(float)
            support_xy = coords.loc[known, ["x_m", "y_m"]].to_numpy(float)
            distances = np.linalg.norm(xy[:, None] - support_xy[None, :], axis=2).min(axis=1)
            # Nearest distance-domain neighbour; this is an error estimate, not
            # a confidence guarantee. A fixed development correction is needed.
            nearest = np.abs(np.log1p(distances)[:, None]
                             - np.log1p(detail.leaveout_distance_m.to_numpy())[None, :]).argmin(axis=1)
            matched = detail.iloc[nearest]
            result.update(matched_support_log_mae=float(matched.log_mae.mean()),
                          matched_support_thickness_mae_m=float(matched.thickness_mae_m.mean()),
                          candidate_distance_q90_m=float(np.quantile(distances, .9)),
                          candidate_geometry_supported=bool(distances.max() <= detail.leaveout_distance_m.max() + 1e-9))
    return result, detail


@dataclass(frozen=True)
class DevelopmentErrorMargin:
    """Empirical development-site residual margin, not a safety guarantee."""
    log_margin: float
    thickness_margin_m: float
    development_sites: tuple[str, ...]
    estimator: str = "matched_support"

    def upper_estimates(self, signals):
        prefix = "matched_support" if self.estimator == "matched_support" else "support"
        return {"estimated_log_mae_upper": signals[f"{prefix}_log_mae"] + self.log_margin,
                "estimated_thickness_mae_upper_m": signals[f"{prefix}_thickness_mae_m"] + self.thickness_margin_m,
                "empirical_development_margin_only": True}


def calibrate_development_margin(records, *, estimator="matched_support"):
    """Freeze conservative worst-site error offsets on development rows only.

    Required columns: role, site_id, audit_log_mae, audit_thickness_mae_m and
    the chosen support-error estimate. The correction takes the maximum
    underestimation in each development site and then the maximum across sites.
    This deliberately performs no evaluation-dependent quantile search.
    """
    if records.empty or not records.role.eq("development").all():
        raise ValueError("Calibration accepts nonempty development rows only")
    if estimator not in {"support", "matched_support"}:
        raise ValueError("Unknown support estimator")
    needed = ["audit_log_mae", "audit_thickness_mae_m", f"{estimator}_log_mae",
              f"{estimator}_thickness_mae_m"]
    finite = records.loc[np.isfinite(records[needed].to_numpy(float)).all(axis=1)].copy()
    if finite.empty:
        raise ValueError("Calibration requires finite support and audit errors")
    finite["log_gap"] = finite.audit_log_mae - finite[f"{estimator}_log_mae"]
    finite["h_gap"] = finite.audit_thickness_mae_m - finite[f"{estimator}_thickness_mae_m"]
    maxima = finite.groupby("site_id")[["log_gap", "h_gap"]].max()
    return DevelopmentErrorMargin(max(0., float(maxima.log_gap.max())),
                                  max(0., float(maxima.h_gap.max())),
                                  tuple(sorted(finite.site_id.astype(str).unique())), estimator)
