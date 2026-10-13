"""Geometry-only source episodes that follow the real maximin replay trajectory.

An episode is one budget on a trajectory, not a new independent site.  The
held-out query profiles and their entire near groups stay fixed along that
trajectory.  No measurement value is read when generating the plan.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..acquisition import select_batch
from ..data.normalize import geometry_scale, stable_id
from ..data.splits import _farthest_initial, derived_seed, select_group_subset


TRAIN_SCENARIO_SEEDS = (107, 211, 307, 401, 503, 601, 701, 809)
VALIDATION_SCENARIO_SEEDS = (907, 1009)
BUDGETS = (5, 7, 9, 11, 13)
PLAN_VERSION = "maximin_trajectory_v4"
_COLUMNS = [
    "fold_id", "role", "site_id", "investigation_group", "episode_id",
    "trajectory_id", "scenario_seed", "seed", "budget", "support_location_ids",
    "query_location_ids", "pool_location_ids", "withheld_support_side_ids",
    "support_groups", "query_near_groups", "query_profiles_per_batch",
    "min_spacing_m", "spacing_fallback_count", "evaluation_fraction_actual",
]


def _site_geometry(dataset, fold: dict, role: str) -> pd.DataFrame:
    """Read identifiers and coordinates; enforce the already-frozen role lists."""
    if role not in ("train", "validation"):
        raise ValueError("Trajectory episodes only support train or validation; test is forbidden")
    role_key = f"{role}_sites"
    if role_key not in fold:
        raise ValueError(f"Frozen fold is missing {role_key}")
    ids = [str(value) for value in fold[role_key]]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate sites in {role_key}")
    for other in ("train", "validation", "test"):
        if other != role and set(ids) & set(map(str, fold.get(f"{other}_sites", []))):
            raise ValueError(f"Frozen {role} and {other} sites overlap")
        if other != role and set(fold.get(f"{role}_groups", [])) & set(fold.get(f"{other}_groups", [])):
            raise ValueError(f"Frozen {role} and {other} investigation groups overlap")

    required = ["site_id", "location_id", "near_group_id", "x_m", "y_m"]
    source = dataset.locations()
    if not set(required).issubset(source):
        raise ValueError(f"Prepared location geometry requires {required}")
    geometry = source.loc[source.site_id.astype(str).isin(ids), required].copy()
    for name in ("site_id", "location_id", "near_group_id"):
        if geometry[name].isna().any():
            raise ValueError(f"Prepared geometry has missing {name}")
        geometry[name] = geometry[name].astype(str)
    if geometry.location_id.duplicated().any():
        raise ValueError("Prepared location IDs must be unique")
    if not np.isfinite(geometry[["x_m", "y_m"]].to_numpy(float)).all():
        raise ValueError("Trajectory geometry must have finite metric coordinates")
    sites = dataset.sites[["site_id", "investigation_group"]].copy()
    sites["site_id"] = sites.site_id.astype(str)
    sites["investigation_group"] = sites.investigation_group.astype(str)
    if sites.site_id.duplicated().any():
        raise ValueError("Prepared site IDs must be unique")
    known = sites.set_index("site_id")
    missing = sorted(set(ids) - set(known.index))
    if missing:
        raise ValueError(f"Frozen role contains unknown sites: {missing}")
    if f"{role}_groups" in fold:
        allowed_groups = set(map(str, fold[f"{role}_groups"]))
        actual = set(known.loc[ids, "investigation_group"])
        if not actual <= allowed_groups:
            raise ValueError("Frozen site/group role assignments disagree")
    return geometry.merge(sites, on="site_id", validate="many_to_one").sort_values(
        ["site_id", "location_id"], kind="stable").reset_index(drop=True)


def build_trajectory_episodes(dataset, fold: dict, config: dict, *, role: str) -> pd.DataFrame:
    """Build source-only whole-profile plans using the registered fixed seeds.

    The largest feasible registered budget is chosen using only group counts.
    Smaller source sites remain useful when they can leave at least one entire
    near group as query and five support locations.  Truncated budgets and all
    skipped site/scenario cases are disclosed in ``attrs['provenance']``.
    """
    geometry = _site_geometry(dataset, fold, role)
    replay = config.get("replay", {})
    if tuple(map(int, replay.get("budgets", BUDGETS))) != BUDGETS:
        raise ValueError("v4 trajectory budgets are frozen to 5/7/9/11/13")
    if int(replay.get("initial_locations", 5)) != 5 or int(replay.get("batch_size", 2)) != 2:
        raise ValueError("v4 trajectory requires five initial locations and batches of two")
    fraction = float(replay.get("evaluation_fraction", 0.2))
    if not 0 < fraction < 1:
        raise ValueError("Evaluation fraction must lie strictly between zero and one")
    seeds = TRAIN_SCENARIO_SEEDS if role == "train" else VALIDATION_SCENARIO_SEEDS
    episode_options = config.get("v4", {}).get("episodes", {})
    if tuple(map(int, episode_options.get("budgets", BUDGETS))) != BUDGETS:
        raise ValueError("v4 episode budgets are frozen to 5/7/9/11/13")
    configured_seeds = episode_options.get(f"{role}_seeds", seeds)
    if tuple(map(int, configured_seeds)) != seeds:
        raise ValueError(f"v4 {role} scenario seeds are frozen to {list(seeds)}")
    query_batch = int(episode_options.get("query_profiles_per_batch",
                     config.get("episodes", {}).get("query_profiles_per_batch", 8)))
    if query_batch < 1:
        raise ValueError("Query profiles per batch must be positive")
    fid = int(fold["fold_id"])
    rows, skipped, trajectories = [], [], []
    sites = sorted(map(str, fold[f"{role}_sites"]))
    for site_id in sites:
        site = geometry.loc[geometry.site_id == site_id].reset_index(drop=True)
        n_locations = len(site)
        groups = {str(key): frame.location_id.tolist()
                  for key, frame in site.groupby("near_group_id", sort=True)}
        # The minimum complete query group determines feasibility, never qc.
        min_query_size = min(map(len, groups.values()), default=n_locations + 1)
        feasible = [budget for budget in BUDGETS if n_locations - min_query_size >= budget]
        if not feasible:
            for scenario_seed in seeds:
                skipped.append({"site_id": site_id, "scenario_seed": scenario_seed,
                                "n_locations": n_locations, "missing_budgets": list(BUDGETS),
                                "reason": "cannot_leave_whole_query_group_and_five_support_locations"})
            continue
        maximum = max(feasible)
        for scenario_seed in seeds:
            # Same geometry-only seed derivation and first-point mechanism as
            # replay; the scenario seeds are disjoint from the evaluation seeds.
            seed = derived_seed(scenario_seed, "replay", site_id)
            rng = np.random.default_rng(seed)
            target = math.ceil(fraction * n_locations)
            query_groups = select_group_subset(groups, target, n_locations - maximum, rng)
            if not query_groups:
                raise RuntimeError("Feasible whole-group query unexpectedly disappeared")
            query = sorted(location for group in query_groups for location in groups[group])
            pool = site.loc[~site.location_id.isin(query)].copy()
            pool_ids = sorted(pool.location_id.tolist())
            min_spacing = geometry_scale(pool[["x_m", "y_m"]].to_numpy())
            support = _farthest_initial(pool, 5, rng)
            group_lookup = site.set_index("location_id").near_group_id
            trajectory_id = stable_id("trajectory_v4", fid, role, site_id, scenario_seed)
            fallback_count = 0
            selected_batches = []
            for budget in feasible:
                while len(support) < budget:
                    batch, scores = select_batch(
                        pool.loc[pool.location_id.isin(support)],
                        pool.loc[~pool.location_id.isin(support)], None, config,
                        policy="maximin", min_spacing_m=min_spacing, batch_size=2)
                    fallback_count += int(scores.spacing_fallback.sum())
                    selected_batches.append(batch)
                    support.extend(batch)
                rows.append({
                    "fold_id": fid, "role": role, "site_id": site_id,
                    "investigation_group": str(site.investigation_group.iloc[0]),
                    "episode_id": stable_id("episode_v4", fid, role, site_id, scenario_seed, budget),
                    "trajectory_id": trajectory_id, "scenario_seed": scenario_seed,
                    "seed": seed, "budget": budget, "support_location_ids": support.copy(),
                    "query_location_ids": query.copy(), "pool_location_ids": pool_ids.copy(),
                    "withheld_support_side_ids": sorted(set(pool_ids) - set(support)),
                    "support_groups": sorted(group_lookup.loc[support].unique().tolist()),
                    "query_near_groups": sorted(query_groups),
                    "query_profiles_per_batch": query_batch, "min_spacing_m": min_spacing,
                    "spacing_fallback_count": fallback_count,
                    "evaluation_fraction_actual": len(query) / n_locations,
                })
            unavailable = [budget for budget in BUDGETS if budget not in feasible]
            trajectories.append({
                "site_id": site_id, "scenario_seed": scenario_seed,
                "trajectory_id": trajectory_id, "n_locations": n_locations,
                "n_query_locations": len(query), "n_pool_locations": len(pool),
                "budgets": feasible, "missing_budgets": unavailable,
                "min_spacing_m": min_spacing, "spacing_fallback_count": fallback_count,
                "selected_batches": selected_batches,
                "evaluation_fraction_actual": len(query) / n_locations,
            })
            if unavailable:
                skipped.append({"site_id": site_id, "scenario_seed": scenario_seed,
                                "n_locations": n_locations, "missing_budgets": unavailable,
                                "reason": "insufficient_pool_after_whole_query_group_holdout"})
    result = pd.DataFrame(rows, columns=_COLUMNS)
    coverage = []
    for site_id in sites:
        subset = result.loc[result.site_id == site_id]
        supports = {location for ids in subset.support_location_ids for location in ids}
        queries = {location for ids in subset.query_location_ids for location in ids}
        coverage.append({"site_id": site_id, "n_locations": int((geometry.site_id == site_id).sum()),
                         "n_episodes": len(subset), "n_trajectories": int(subset.trajectory_id.nunique()),
                         "budgets": sorted(map(int, subset.budget.unique())),
                         "n_unique_support_locations": len(supports),
                         "n_unique_query_locations": len(queries),
                         "n_distinct_query_masks": len({tuple(ids) for ids in subset.query_location_ids}),
                         "n_distinct_support_masks": len({tuple(sorted(ids)) for ids in subset.support_location_ids})})
    geometry_json = geometry.to_json(orient="records", double_precision=15)
    result.attrs["provenance"] = {
        "plan_version": PLAN_VERSION, "fold_id": fid, "role": role,
        "scenario_seeds": list(seeds), "budgets": list(BUDGETS),
        "geometry_sha256": hashlib.sha256(geometry_json.encode("utf-8")).hexdigest(),
        "geometry_columns_used": list(geometry.columns), "labels_read": False,
        "n_episodes": len(result), "n_trajectories": len(trajectories),
        "n_included_sites": int(result.site_id.nunique()), "n_requested_sites": len(sites),
        "evaluation_fraction_target": fraction,
        "small_site_rule": "Largest registered budget leaving one whole query near group; never filter by labels.",
        "separation": "Whole near groups held out from support for each trajectory; masks may differ across trajectories.",
        "dependence": "Budgets within a trajectory and trajectories within a site are correlated, not independent samples.",
        "site_coverage": coverage, "trajectories": trajectories, "skipped": skipped,
    }
    return result


def save_episode_plan(episodes: pd.DataFrame, path: str | Path) -> dict:
    """Save the plan and its explicit coverage/skip record beside it."""
    if "provenance" not in episodes.attrs:
        raise ValueError("Episode plan is missing geometry and coverage provenance")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(path, index=False)
    metadata_path = path.with_suffix(".provenance.json")
    metadata_path.write_text(json.dumps(episodes.attrs["provenance"], ensure_ascii=False,
                                        indent=2, allow_nan=False), encoding="utf-8")
    return {"episodes_path": str(path), "provenance_path": str(metadata_path),
            "n_episodes": len(episodes), "n_trajectories": episodes.attrs["provenance"]["n_trajectories"]}
