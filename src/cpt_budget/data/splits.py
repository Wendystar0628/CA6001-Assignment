"""Freeze folds, episodic masks and replay pools from geometry and counts only."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .normalize import geometry_scale, stable_id


def derived_seed(seed: int, *parts: object) -> int:
    text = json.dumps([seed, *parts], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def select_group_subset(groups: dict[str, list[str]], target: int, max_count: int, rng: np.random.Generator) -> list[str]:
    """Closest nonempty whole-group subset, with seeded ties and a hard pool limit."""
    ordered = list(sorted(groups))
    rng.shuffle(ordered)
    reachable: dict[int, list[str]] = {0: []}
    for group in ordered:
        size = len(groups[group])
        for count, chosen in list(reachable.items()):
            next_count = count + size
            if next_count <= max_count and next_count not in reachable:
                reachable[next_count] = chosen + [group]
    candidates = [count for count in reachable if count > 0]
    if not candidates:
        return []
    tie_order = {count: float(rng.random()) for count in candidates}
    best = min(candidates, key=lambda count: (abs(count - target), tie_order[count]))
    return reachable[best]


def _validation_groups(source: pd.DataFrame, fraction: float, rng: np.random.Generator) -> tuple[list[str], dict]:
    if len(source) < 2:
        raise ValueError("At least two source investigation groups are needed for train/validation separation")
    ordered = source.index.tolist()
    rng.shuffle(ordered)
    # At most two replay-capable groups are counted; retain full groups throughout.
    reachable = {(0, 0): []}
    for group in ordered:
        size = int(source.loc[group, "n_locations"])
        replay = int(source.loc[group, "n_replay_sites"] > 0)
        for (count, replay_count), chosen in list(reachable.items()):
            key = (count + size, min(2, replay_count + replay))
            reachable.setdefault(key, chosen + [group])
    total = int(source.n_locations.sum())
    target = fraction * total
    candidates = [(key, chosen) for key, chosen in reachable.items() if 0 < key[0] < total and len(chosen) < len(source)]
    near_fraction = [(key, chosen) for key, chosen in candidates if fraction <= key[0] / total <= max(.25, fraction)]
    pool = near_fraction or candidates
    tie = {tuple(chosen): float(rng.random()) for _, chosen in pool}
    # Within the requested band, prefer two complete replay groups. If the band is
    # impossible, prioritize proximity and record that group indivisibility limits it.
    if near_fraction:
        key, chosen = min(pool, key=lambda item: (-item[0][1], abs(item[0][0] - target), tie[tuple(item[1])]))
    else:
        key, chosen = min(pool, key=lambda item: (abs(item[0][0] - target), -item[0][1], tie[tuple(item[1])]))
    return sorted(chosen), {"n_validation_locations": key[0], "validation_location_fraction": key[0] / total,
                            "n_validation_replay_groups": key[1], "validation_fraction_target": fraction,
                            "validation_note": "" if key[1] >= 2 else "Fewer than two replay groups fit the fixed group/fraction allocation."}


def make_folds(dataset, config: dict) -> list[dict]:
    sites = dataset.sites.copy()
    options = config.get("split", {})
    # Strict mode is an explicit configuration decision, never a performance filter.
    if options.get("exclude_unverified_groups", False):
        bad = set(sites.loc[~sites.grouping_verified.astype(bool), "investigation_group"])
        sites = sites[~sites.investigation_group.isin(bad)]
    sites = sites[sites.n_locations > 0]
    groups = sites.groupby("investigation_group").agg(n_locations=("n_locations", "sum"), n_replay_sites=("replay_eligible", "sum"))
    n_folds, seed = int(options.get("outer_folds", 3)), int(options.get("seed", 20261013))
    if len(groups) < n_folds + 1:
        raise ValueError("Too few investigation groups for the requested outer folds and source validation")
    rng = np.random.default_rng(seed)
    tie = {key: float(rng.random()) for key in sorted(groups.index)}
    ordered = sorted(groups.index, key=lambda key: (-groups.loc[key, "n_locations"], -groups.loc[key, "n_replay_sites"], tie[key]))
    assignment, counts, replay_counts = [[] for _ in range(n_folds)], [0] * n_folds, [0] * n_folds
    for group in ordered:
        fold_tie = rng.random(n_folds)
        dest = min(range(n_folds), key=lambda k: (counts[k], replay_counts[k], fold_tie[k]))
        assignment[dest].append(group)
        counts[dest] += int(groups.loc[group, "n_locations"])
        replay_counts[dest] += int(groups.loc[group, "n_replay_sites"])
    folds = []
    for fold_id, test_groups in enumerate(assignment):
        source_groups = groups.loc[~groups.index.isin(test_groups)]
        validation, notes = _validation_groups(source_groups, float(options.get("validation_fraction", .2)), np.random.default_rng(derived_seed(seed, "validation", fold_id)))
        train = sorted(set(source_groups.index) - set(validation))
        fold = {"fold_id": fold_id, "train_groups": train, "validation_groups": validation, "test_groups": sorted(test_groups), **notes}
        for role in ("train", "validation", "test"):
            selected = sites[sites.investigation_group.isin(fold[f"{role}_groups"])]
            fold[f"{role}_sites"] = sorted(selected.site_id.tolist())
            fold[f"{role}_n_locations"] = int(selected.n_locations.sum())
            fold[f"{role}_unverified_groups"] = sorted(selected.loc[~selected.grouping_verified.astype(bool), "investigation_group"].unique().tolist())
        folds.append(fold)
    return folds


def make_episodes(dataset, folds: list[dict], config: dict) -> tuple[pd.DataFrame, list[dict]]:
    options = config.get("episodes", {})
    base_seed = int(config.get("split", {}).get("seed", 20261013))
    rows, skipped = [], []
    locations = dataset.locations()
    for fold in folds:
        for role in ("train", "validation", "test"):
            for site_id in fold[f"{role}_sites"]:
                site = locations[locations.site_id == site_id].sort_values("location_id")
                group_members = {key: frame.location_id.tolist() for key, frame in site.groupby("near_group_id", sort=True)}
                support_sizes = sorted({min(int(size), len(site) - 1) for size in options.get("support_sizes", [5, 9, 13])})[:int(options.get("masks_per_site", 3))]
                for support_size in support_sizes:
                    if support_size < 5:
                        skipped.append({"site_id": site_id, "fold_id": fold["fold_id"], "role": role, "support_size": support_size, "reason": "fewer_than_five_support_locations"})
                        continue
                    seed = derived_seed(base_seed, "episode", site_id, support_size)
                    rng = np.random.default_rng(seed)
                    order = list(group_members)
                    rng.shuffle(order)
                    support_groups, support_candidates = [], []
                    for group in order:
                        if len(support_candidates) >= support_size:
                            break
                        support_groups.append(group)
                        support_candidates.extend(group_members[group])
                    query = sorted(loc for group, members in group_members.items() if group not in support_groups for loc in members)
                    if not query or len(support_candidates) < support_size:
                        skipped.append({"site_id": site_id, "fold_id": fold["fold_id"], "role": role, "support_size": support_size,
                                        "reason": "fixed_near_group_partition_cannot_leave_query_and_required_support"})
                        continue
                    support = sorted(rng.choice(support_candidates, size=support_size, replace=False).tolist())
                    withheld = sorted(set(support_candidates) - set(support))
                    rows.append({"fold_id": fold["fold_id"], "role": role, "site_id": site_id,
                                 "episode_id": stable_id("episode", fold["fold_id"], role, site_id, support_size),
                                 "support_location_ids": support, "query_location_ids": query,
                                 "withheld_support_side_ids": withheld, "support_groups": sorted(support_groups),
                                 "query_profiles_per_batch": int(options.get("query_profiles_per_batch", 8)), "seed": seed})
    columns = ["fold_id", "role", "site_id", "episode_id", "support_location_ids", "query_location_ids", "withheld_support_side_ids", "support_groups", "query_profiles_per_batch", "seed"]
    return pd.DataFrame(rows, columns=columns), skipped


def _farthest_initial(pool: pd.DataFrame, n_initial: int, rng: np.random.Generator) -> list[str]:
    pool = pool.sort_values("location_id").reset_index(drop=True)
    points, ids = pool[["x_m", "y_m"]].to_numpy(), pool.location_id.tolist()
    chosen = [int(rng.integers(len(pool)))]
    while len(chosen) < n_initial:
        distance = np.linalg.norm(points[:, None, :] - points[chosen][None, :, :], axis=2).min(axis=1)
        distance[chosen] = -np.inf
        chosen.append(int(np.argmax(distance)))
    return [ids[index] for index in chosen]


def make_replay_splits(dataset, config: dict) -> tuple[list[dict], list[dict]]:
    options = config.get("replay", {})
    budgets = [int(v) for v in options.get("budgets", [5, 7, 9, 11, 13])]
    maximum = max(budgets)
    initial_count = int(options.get("initial_locations", 5))
    if initial_count != min(budgets):
        raise ValueError("The initial observed count must equal the first budget")
    splits, skipped = [], []
    for site_id, site in dataset.locations().groupby("site_id", sort=True):
        if len(site) < int(options.get("min_site_locations", 20)):
            skipped.append({"site_id": site_id, "reason": "fewer_than_minimum_replay_locations", "n_locations": len(site)})
            continue
        groups = {key: frame.location_id.tolist() for key, frame in site.groupby("near_group_id", sort=True)}
        for seed in options.get("seeds", [17, 29, 43, 71, 101]):
            rng = np.random.default_rng(derived_seed(int(seed), "replay", site_id))
            target = math.ceil(float(options.get("evaluation_fraction", .2)) * len(site))
            chosen_groups = select_group_subset(groups, target, len(site) - maximum, rng)
            if not chosen_groups:
                skipped.append({"site_id": site_id, "seed": int(seed), "reason": "whole_near_groups_cannot_leave_evaluation_and_maximum_budget", "n_locations": len(site)})
                continue
            evaluation_ids = sorted(loc for group in chosen_groups for loc in groups[group])
            pool = site[~site.location_id.isin(evaluation_ids)].copy()
            initial = _farthest_initial(pool, initial_count, rng)
            evaluation = site[site.location_id.isin(evaluation_ids)]
            initial_xy = pool.set_index("location_id").loc[initial, ["x_m", "y_m"]].to_numpy()
            evaluation_distance = np.linalg.norm(evaluation[["x_m", "y_m"]].to_numpy()[:, None] - initial_xy[None, :], axis=2).min(axis=1)
            site_row = dataset.sites.set_index("site_id").loc[site_id]
            splits.append({"site_id": site_id, "seed": int(seed), "evaluation_ids": evaluation_ids,
                           "pool_ids": sorted(pool.location_id.tolist()), "initial_ids": initial, "budgets": budgets,
                           "min_spacing_m": geometry_scale(pool[["x_m", "y_m"]].to_numpy()),
                           "evaluation_near_groups": sorted(chosen_groups), "n_evaluation": len(evaluation_ids),
                           "evaluation_fraction_actual": len(evaluation_ids) / len(site),
                           "evaluation_initial_nn_min_m": float(evaluation_distance.min()),
                           "evaluation_initial_nn_median_m": float(np.median(evaluation_distance)),
                           "evaluation_initial_nn_max_m": float(evaluation_distance.max()),
                           "investigation_group": site_row.investigation_group,
                           "grouping_verified": bool(site_row.grouping_verified)})
    return splits, skipped


def make_splits(dataset, config: dict) -> dict:
    prepared_dir = Path(config["_root"]) / config.get("project", {}).get("prepared_dir", "data_processed")
    prepared_dir.mkdir(parents=True, exist_ok=True)
    folds = make_folds(dataset, config)
    episodes, skipped_episodes = make_episodes(dataset, folds, config)
    replay, skipped_replay = make_replay_splits(dataset, config)
    # Strict source exclusion must also remove replay entries outside the saved folds.
    included_sites = {site for fold in folds for site in fold["test_sites"]}
    if config.get("split", {}).get("exclude_unverified_groups", False):
        replay = [split for split in replay if split["site_id"] in included_sites]
    fold_document = {"folds": folds, "seed": int(config.get("split", {}).get("seed", 20261013)),
                     "grouping_radius_m": float(config.get("split", {}).get("grouping_radius_m", 250)),
                     "exclude_unverified_groups": bool(config.get("split", {}).get("exclude_unverified_groups", False)),
                     "unverified_grouping_sites": dataset.sites.loc[~dataset.sites.grouping_verified.astype(bool), "site_id"].tolist()}
    for name, document in (("folds.json", fold_document), ("replay_splits.json", replay),
                           ("split_preparation.json", {"n_episodes": len(episodes), "n_replay_splits": len(replay),
                                                       "skipped_episodes": skipped_episodes, "skipped_replay": skipped_replay})):
        (prepared_dir / name).write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    episodes.to_parquet(prepared_dir / "episodes.parquet", index=False)
    return {"folds": folds, "episodes": episodes, "replay_splits": replay,
            "skipped_episodes": skipped_episodes, "skipped_replay": skipped_replay}
