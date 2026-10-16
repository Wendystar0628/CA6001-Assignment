"""Frozen geometry partitions and offline representation diagnostics for adaptive CPT.

The online view reveals whole location groups only. The old test archive is an
already inspected historical result; this experiment is a limited-data
re-evaluation, with each evaluation site excluded from its own source artifacts.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, load_config
from .data.normalize import midpoint_values, stable_id
from .data.splits import derived_seed, select_group_subset
from .types import PreparedDataset, QUERY_COLUMNS, load_dataset

DEVELOPMENT_FOLDS = {
    "lower_hutt_town_hall": 0, "tauranga_farmers_building": 0,
    "christchurch_metro_sports_facility": 2, "marshland": 2,
}
EVALUATION_FOLDS = {
    "christchurch_stadium": 0, "wellington_airport": 0,
    "newlock": 1, "hollywood": 2,
}
PRESET_SEEDS = (17, 29, 43)
GEOMETRY_COLUMNS = [
    "site_id", "profile_id", "location_id", "near_group_id", "x_m", "y_m", "geometry_scale_m",
]


def _dataset(dataset=None):
    return load_dataset(load_config()) if dataset is None else dataset


def _farthest_groups(pool, groups, target, rng):
    """Choose at least target true positions, adding an entire group each time."""
    indexed = pool.set_index("location_id")
    ordered = sorted(indexed.index)
    first = ordered[int(rng.integers(len(ordered)))]
    group_for = {loc: group for group, ids in groups.items() for loc in ids}
    chosen = list(groups[group_for[first]])
    while len(chosen) < target:
        remaining = [loc for loc in ordered if loc not in chosen]
        if not remaining:
            break
        xy = indexed.loc[remaining, ["x_m", "y_m"]].to_numpy(float)
        observed = indexed.loc[chosen, ["x_m", "y_m"]].to_numpy(float)
        distances = np.linalg.norm(xy[:, None, :] - observed[None, :, :], axis=2).min(axis=1)
        next_id = remaining[int(np.argmax(distances))]
        chosen.extend(groups[group_for[next_id]])
    return chosen


def build_splits(dataset=None, seeds=PRESET_SEEDS, site_ids=None, initial_locations=5,
                 audit_fraction=.2, minimum_plan_locations=9):
    """Freeze S0/C/A from coordinates and existing near-group IDs, without qc.

    ``budgets`` are observation checkpoints, not permission to split a group.
    Actual adaptive batches may overshoot a checkpoint. ``n_full`` counts only
    S0 union C, excludes audit CPTs, and is always an explicit final endpoint.
    """
    dataset = _dataset(dataset)
    selected_sites = list(site_ids) if site_ids is not None else sorted({**DEVELOPMENT_FOLDS, **EVALUATION_FOLDS})
    if not 0 < audit_fraction < 1 or initial_locations < 1:
        raise ValueError("Need a positive initial size and an audit fraction strictly between zero and one")
    minimum_plan_locations = max(int(minimum_plan_locations), int(initial_locations))
    locations = dataset.locations()
    splits = []
    for site_id in selected_sites:
        site = locations.loc[locations.site_id.eq(site_id), GEOMETRY_COLUMNS].sort_values("location_id")
        if site.empty:
            raise ValueError(f"No usable positions at {site_id}")
        groups = {str(key): sorted(frame.location_id.tolist()) for key, frame in site.groupby("near_group_id", sort=True)}
        if site.near_group_id.isna().any() or site.location_id.duplicated().any():
            raise ValueError(f"Invalid position registry: {site_id}")
        role = "development" if site_id in DEVELOPMENT_FOLDS else "evaluation"
        fold_id = {**DEVELOPMENT_FOLDS, **EVALUATION_FOLDS}.get(site_id)
        if fold_id is None:
            raise ValueError(f"Site {site_id} has no frozen adaptive source-fold role")
        for seed in seeds:
            rng = np.random.default_rng(derived_seed(int(seed), "adaptive_v1", site_id))
            audit_groups = select_group_subset(groups, math.ceil(len(site) * audit_fraction),
                                               len(site) - minimum_plan_locations, rng)
            if not audit_groups:
                raise ValueError(f"{site_id}: whole groups cannot leave an audit and the minimum plan")
            audit_ids = sorted(loc for group in audit_groups for loc in groups[group])
            full = site.loc[~site.location_id.isin(audit_ids)]
            plan_groups = {group: ids for group, ids in groups.items() if group not in audit_groups}
            initial = _farthest_groups(full, plan_groups, int(initial_locations), rng)
            full_ids = sorted(full.location_id.tolist())
            candidate_ids = sorted(set(full_ids) - set(initial))
            n_full, n_initial = len(full_ids), len(initial)
            budgets = list(range(n_initial, n_full + 1, 2))
            if budgets[-1] != n_full:
                budgets.append(n_full)
            split = {
                "split_id": stable_id("adaptive", site_id, int(seed), audit_ids, initial),
                "site_id": site_id, "seed": int(seed), "fold_id": int(fold_id), "role": role,
                "initial_ids": initial, "candidate_ids": candidate_ids, "audit_ids": audit_ids,
                "full_ids": full_ids, "near_groups": groups, "audit_near_groups": sorted(audit_groups),
                "n_full": n_full, "n_initial": n_initial, "n_candidates": len(candidate_ids),
                "n_audit": len(audit_ids), "n_site_positions": len(site), "budgets": budgets,
                "audit_fraction_requested": float(audit_fraction), "audit_fraction_actual": len(audit_ids) / len(site),
                "initial_locations_requested": int(initial_locations), "batch_locations_requested": 2,
                "geometry_scale_m": float(site.geometry_scale_m.iloc[0]),
                "plan_definition": "all representative usable positions except fixed hidden audit groups",
                "evaluation_scope": "limited-data re-evaluation after historical test inspection",
            }
            validate_split(split)
            splits.append(split)
    return splits


def validate_split(split):
    initial, candidate, audit, full = (set(split[k]) for k in ("initial_ids", "candidate_ids", "audit_ids", "full_ids"))
    if initial & candidate or initial & audit or candidate & audit or initial | candidate != full:
        raise ValueError("S0, C and A must be disjoint and full must equal S0 union C")
    if len(full) != split["n_full"] or split["budgets"][-1] != len(full):
        raise ValueError("The full-plan endpoint must count every planned position exactly once")
    all_ids = set()
    for group, ids in split["near_groups"].items():
        members = set(ids)
        if all_ids & members or not members:
            raise ValueError(f"Invalid near group {group}")
        all_ids |= members
        if sum(bool(members & side) for side in (initial, candidate, audit)) != 1:
            raise ValueError(f"Near group {group} crosses an information boundary")
    if all_ids != full | audit:
        raise ValueError("All registered positions must belong to exactly one side")


class AdaptiveDataView:
    """Online data boundary; no API reveals audit or unrevealed candidate qc."""

    def __init__(self, dataset, split):
        validate_split(split)
        self.__dataset = dataset
        self.split = json.loads(json.dumps(split))
        self.__revealed = list(split["initial_ids"])
        ids = set(split["full_ids"]) | set(split["audit_ids"])
        self.__geometry = dataset.locations().loc[lambda x: x.location_id.isin(ids), GEOMETRY_COLUMNS].copy()
        geometry_grid = dataset.grid[[c for c in QUERY_COLUMNS if c in dataset.grid]].copy()
        self.__query_dataset = PreparedDataset(dataset.profiles, geometry_grid, dataset.sites, dataset.root)
        if set(self.__geometry.location_id) != ids:
            raise ValueError("A split position is missing from the geometry registry")

    @property
    def revealed_ids(self):
        return list(self.__revealed)

    def geometry(self):
        return self.__geometry.copy()

    def query(self, ids):
        ids = list(ids)
        if set(ids) - set(self.__geometry.location_id):
            raise KeyError("Unknown position in online query")
        return self.__query_dataset.query(ids)

    def observed(self):
        return self.__dataset.observed(self.__revealed)

    def reveal(self, ids):
        ids = list(ids)
        requested = set(ids)
        allowed = set(self.split["candidate_ids"]) - set(self.__revealed)
        if len(ids) != len(requested) or not requested <= allowed:
            raise ValueError("Only new, non-audit candidate positions may be revealed")
        for group, members in self.split["near_groups"].items():
            if requested & set(members) and not set(members) <= requested:
                raise ValueError(f"Revelation must include the whole near group: {group}")
        self.__revealed.extend(ids)
        return self.observed()


def audit_truth(dataset, split):
    """Offline evaluator entry point; never pass its return to the online view."""
    return dataset.truth(split["audit_ids"])


def fixed_query_grid(dataset, split):
    """Fixed planned-location/depth G; coordinates were known before replay.

    This is not a continuous-space accuracy claim. Query rows stay unchanged as
    some plan positions are gradually revealed. No audit measurements enter G.
    """
    geometry_grid = dataset.grid[[c for c in QUERY_COLUMNS if c in dataset.grid]].copy()
    geometry_dataset = PreparedDataset(dataset.profiles, geometry_grid, dataset.sites, dataset.root)
    return geometry_dataset.query(split["full_ids"])


def raw_low_thickness(array, threshold=1., start=2., end=8., max_gap=.5):
    """Integrate a piecewise-linear raw qc curve; skip uncovered or large gaps."""
    values = np.asarray(array, float)
    values = values[np.isfinite(values).all(axis=1) & (values[:, 0] >= 0) & (values[:, 1] > 0)]
    values = values[np.argsort(values[:, 0], kind="stable")]
    _, first = np.unique(values[:, 0], return_index=True)
    values = values[first]
    low_intervals, covered = [], 0.
    for (z0, q0), (z1, q1) in zip(values[:-1], values[1:]):
        lo, hi = max(float(z0), start), min(float(z1), end)
        if hi <= lo or z1 - z0 > max_gap + 1e-9:
            continue
        covered += hi - lo
        left = q0 + (q1 - q0) * (lo - z0) / (z1 - z0)
        right = q0 + (q1 - q0) * (hi - z0) / (z1 - z0)
        if left < threshold and right < threshold:
            low_intervals.append((lo, hi))
        elif (left < threshold) != (right < threshold):
            crossing = lo + (threshold - left) * (hi - lo) / (right - left)
            low_intervals.append((lo, crossing) if left < threshold else (crossing, hi))
    merged = []
    for lo, hi in low_intervals:
        if merged and lo <= merged[-1][1] + 1e-9:
            merged[-1] = (merged[-1][0], hi)
        else:
            merged.append((lo, hi))
    return {"raw_low_thickness_m": float(sum(hi - lo for lo, hi in merged)),
            "raw_covered_depth_m": float(covered), "low_intervals": merged}


def representation_diagnostics(dataset=None, config=None):
    """Offline raw-vs-midpoint check only; does not alter training labels."""
    dataset, config = _dataset(dataset), config or load_config()
    options = config["data"]
    start, end, step = (float(options[k]) for k in ("depth_start_m", "depth_end_m", "depth_step_m"))
    threshold = float(options.get("tau_mpa", 1))
    max_gap = float(options.get("max_interpolation_gap_m", .5))
    rows = []
    with np.load(dataset.root / config["project"]["input_dir"] / options["arrays"], allow_pickle=False) as arrays:
        for profile in dataset.locations().itertuples(index=False):
            raw = arrays[str(profile.array_key)]
            valid_raw = raw[np.isfinite(raw).all(axis=1) & (raw[:, 0] >= 0) & (raw[:, 1] > 0)]
            grid = dataset.grid.loc[dataset.grid.location_id.eq(profile.location_id)].sort_values("depth_index")
            integral = raw_low_thickness(raw, threshold, start, end, max_gap)
            h_grid = float((grid.qc_mpa < threshold).sum() * step)
            centers = grid.z_center_m.to_numpy()
            reproduced = midpoint_values(raw, centers, max_gap)
            intervals = integral.pop("low_intervals")
            thin = [(lo, hi) for lo, hi in intervals if hi - lo < step]
            missed = [(lo, hi) for lo, hi in thin if not np.any((centers >= lo) & (centers <= hi))]
            # 0.1 m is diagnostic only: existing models stay on the frozen 0.25 m target.
            fine_centers = start + (np.arange(int(round((end - start) / .1))) + .5) * .1
            fine = midpoint_values(raw, fine_centers, max_gap)
            h_fine = float((fine < threshold).sum() * .1) if np.isfinite(fine).all() else np.nan
            complete = bool(np.isclose(integral["raw_covered_depth_m"], end - start, atol=1e-6))
            rows.append({"site_id": profile.site_id, "profile_id": profile.profile_id,
                         "location_id": profile.location_id, "source_file": profile.source_file,
                         "array_key": profile.array_key, "threshold_mpa": threshold, **integral,
                         "raw_depth_min_m": float(valid_raw[:, 0].min()),
                         "raw_depth_max_m": float(valid_raw[:, 0].max()),
                         "raw_complete_window": complete, "grid_025_thickness_m": h_grid,
                         "grid_025_signed_error_m": h_grid - integral["raw_low_thickness_m"],
                         "grid_025_abs_error_m": abs(h_grid - integral["raw_low_thickness_m"]),
                         "grid_010_diagnostic_thickness_m": h_fine,
                         "grid_010_diagnostic_abs_error_m": abs(h_fine - integral["raw_low_thickness_m"]),
                         "raw_low_intervals": len(intervals), "raw_thin_intervals_lt_025": len(thin),
                         "thin_intervals_without_025_midpoint": len(missed),
                         "thin_thickness_without_025_midpoint_m": float(sum(hi - lo for lo, hi in missed)),
                         "grid_reproduction_max_abs_qc_mpa": float(np.max(np.abs(reproduced - grid.qc_mpa.to_numpy())))})
    return pd.DataFrame(rows)


def curve_duplicate_audit(dataset=None, max_log_difference=.001):
    """Offline hygiene report; no target-derived groups are changed during replay."""
    dataset = _dataset(dataset)
    locations = dataset.locations().set_index("location_id")
    matrix = dataset.grid.pivot(index="location_id", columns="depth_index", values="log_qc")
    rows = []
    ids = sorted(locations.index)
    values = matrix.loc[ids].to_numpy(float)
    for i, a in enumerate(ids[:-1]):
        differences = np.max(np.abs(values[i + 1:] - values[i]), axis=1)
        for offset in np.flatnonzero(differences <= max_log_difference):
            b = ids[i + 1 + int(offset)]
            site_a, site_b = locations.loc[a, "site_id"], locations.loc[b, "site_id"]
            rows.append({"site_id": site_a, "site_b": site_b, "location_a": a, "location_b": b,
                         "near_group_a": locations.loc[a, "near_group_id"], "near_group_b": locations.loc[b, "near_group_id"],
                         "max_abs_log_difference": float(differences[offset]), "exact_grid_duplicate": differences[offset] == 0,
                         "same_near_group": locations.loc[a, "near_group_id"] == locations.loc[b, "near_group_id"],
                         "cross_site": site_a != site_b,
                         "adaptive_replay_site": bool({site_a, site_b} & set({**DEVELOPMENT_FOLDS, **EVALUATION_FOLDS}))})
    return pd.DataFrame(rows, columns=["site_id", "site_b", "location_a", "location_b", "near_group_a", "near_group_b",
                                       "max_abs_log_difference", "exact_grid_duplicate", "same_near_group", "cross_site", "adaptive_replay_site"])


def source_training_manifest(dataset=None):
    """Trace actual retained sources, including cross-fold historical exposure."""
    dataset = _dataset(dataset)
    folds = json.loads((dataset.root / "artifacts/evaluation/folds.json").read_text(encoding="utf-8"))["folds"]
    rows = []
    for fold in folds:
        folder = dataset.root / "artifacts/models" / f"fold_{fold['fold_id']}"
        for component, filename in (("tabpfn", "tabpfn_training.json"), ("xgb", "xgb.json"),
                                    ("cnp", "cnp_training.json"), ("fpca_basis", "basis_manifest.json"),
                                    ("gp_trend", "gp_trend.json")):
            path = folder / filename
            record = json.loads(path.read_text(encoding="utf-8"))
            training = record.get("train_sites", record.get("source_site_ids", fold["train_sites"]))
            validation = record.get("validation_sites", [])
            for site in dataset.sites.site_id:
                rows.append({"fold_id": fold["fold_id"], "component": component, "site_id": site,
                             "source_training": site in training, "source_checkpoint_selection": site in validation,
                             "source_role": "train" if site in training else "validation" if site in validation else "excluded",
                             "artifact_metadata": path.relative_to(dataset.root).as_posix(),
                             "evidence": "fold_train_sites; prior leakage_audit numerically checked GP trend" if component == "gp_trend" else "retained component metadata"})
    return pd.DataFrame(rows)


def newlock_source_mapping(dataset=None):
    """Map all 98 archive profiles to the 80 screened shallow-task profiles.

    This is a provenance check, not an extension of the frozen input data. Some
    rejected curves can cover every midpoint yet fail full 2-8 m raw coverage.
    """
    from openpyxl import load_workbook
    from .data.import_existing import _read_profile

    dataset = _dataset(dataset)
    prepared = dataset.profiles.loc[dataset.profiles.site_id.eq("newlock")].set_index("source_file")
    rows = []
    with zipfile.ZipFile(dataset.root / "Data/downloads/Global_CPT1196.zip") as archive:
        workbook = load_workbook(io.BytesIO(archive.read("CPT_3_1196/@Newlock/locations.xlsx")),
                                 read_only=True, data_only=True)
        try:
            positions = {str(row[0]).strip().lower(): (float(row[1]), float(row[2]))
                         for row in list(workbook.worksheets[0].values)[1:] if row and row[0] is not None}
        finally:
            workbook.close()
        names = sorted(name for name in archive.namelist() if "/@Newlock/" in name and name.lower().endswith(".xls"))
        centers = 2 + (np.arange(24) + .5) * .25
        for name in names:
            raw = _read_profile(archive.read(name), name)
            source_id = Path(name).stem.split("_")[0]
            xy = positions.get(source_id.lower())
            included = name in prepared.index
            minimum, maximum = float(raw[:, 0].min()), float(raw[:, 0].max())
            reasons = []
            if minimum > 2 + 1e-6:
                reasons.append("raw_curve_starts_below_2m_window_top")
            if maximum < 8 - 1e-6:
                reasons.append("raw_curve_ends_before_8m_window_bottom")
            if not included and not reasons:
                reasons.append("not_in_frozen_screened_manifest_other_reason")
            rows.append({"source_file": name, "source_workbook_id": source_id,
                         "archive_coordinate_match": xy is not None,
                         "archive_x_m": xy[0] if xy else np.nan, "archive_y_m": xy[1] if xy else np.nan,
                         "screened_2_8m": included,
                         "profile_id": prepared.loc[name, "profile_id"] if included else "",
                         "location_id": prepared.loc[name, "location_id"] if included else "",
                         "raw_depth_min_m": minimum, "raw_depth_max_m": maximum,
                         "n_valid_depth_samples": len(raw),
                         "midpoint_025_complete": bool(np.isfinite(midpoint_values(raw, centers, .5)).all()),
                         "strict_2_8m_raw_coverage": minimum <= 2 + 1e-6 and maximum >= 8 - 1e-6,
                         "exclusion_reason": ";".join(reasons) if not included else ""})
    return pd.DataFrame(rows)


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare_adaptive_data(output=None, seeds=PRESET_SEEDS, dataset=None):
    """Write new versioned manifests and diagnostics; never rewrite v0.9.1 files."""
    dataset = _dataset(dataset)
    output = Path(output) if output else dataset.root / "results/adaptive_v1"
    output.mkdir(parents=True, exist_ok=True)
    splits = build_splits(dataset, seeds=seeds)
    duplicates = curve_duplicate_audit(dataset)
    if len(duplicates) and (duplicates.adaptive_replay_site & ~duplicates.same_near_group).any():
        raise ValueError("Cross-group near-duplicate curves at a replay site need a frozen hygiene grouping before replay")
    sources = source_training_manifest(dataset)
    representation = representation_diagnostics(dataset)
    raw_depths = representation.set_index("location_id")
    for split in splits:
        own = sources.loc[sources.fold_id.eq(split["fold_id"]) & sources.site_id.eq(split["site_id"])]
        if own.source_training.any() or (split["role"] == "evaluation" and own.source_checkpoint_selection.any()):
            raise ValueError(f"{split['site_id']} is exposed in its own retained source artifacts")
    rows = []
    for split in splits:
        by_id = dataset.locations(split["site_id"]).set_index("location_id")
        for side, key in (("S0", "initial_ids"), ("C", "candidate_ids"), ("A", "audit_ids")):
            for loc in split[key]:
                profile = by_id.loc[loc]
                rows.append({"split_id": split["split_id"], "site_id": split["site_id"], "seed": split["seed"],
                             "fold_id": split["fold_id"], "role": split["role"], "location_id": loc,
                             "near_group_id": profile.near_group_id, "profile_id": profile.profile_id,
                             "source_file": profile.source_file, "x_m": profile.x_m, "y_m": profile.y_m,
                             "partition": side, "in_full_plan": side != "A", "n_full": split["n_full"],
                             "n_initial": split["n_initial"], "grouping_verified": bool(dataset.sites.set_index("site_id").loc[split["site_id"], "grouping_verified"]),
                             "raw_depth_max_m": float(raw_depths.loc[loc, "raw_depth_max_m"]),
                             "research_window_start_m": 2., "research_window_end_m": 8.,
                             "evaluation_scope": split["evaluation_scope"]})
    pd.DataFrame(rows).to_csv(output / "data_manifest.csv", index=False)
    dataset.profiles.to_csv(output / "profile_manifest.csv", index=False)
    sites = dataset.sites.copy()
    sites["adaptive_role"] = sites.site_id.map({**{s: "development" for s in DEVELOPMENT_FOLDS}, **{s: "evaluation" for s in EVALUATION_FOLDS}}).fillna("source_only")
    sites["adaptive_fold_id"] = sites.site_id.map({**DEVELOPMENT_FOLDS, **EVALUATION_FOLDS}).astype("Int64")
    sites["historical_test_already_unblinded"] = sites.site_id.isin(EVALUATION_FOLDS)
    sites.to_csv(output / "site_manifest.csv", index=False)
    sources.to_csv(output / "source_training_manifest.csv", index=False)
    duplicates.to_csv(output / "curve_duplicate_diagnostics.csv", index=False)
    newlock = newlock_source_mapping(dataset)
    newlock.to_csv(output / "newlock_source_mapping.csv", index=False)
    _write_json(output / "splits.json", splits)
    representation.to_csv(output / "representation_profiles.csv", index=False)
    summaries = []
    for site, group in [("ALL", representation), *list(representation.groupby("site_id", sort=True))]:
        complete = group.loc[group.raw_complete_window]
        summaries.append({"site_id": site, "n_profiles": len(group), "n_complete_raw_windows": len(complete),
                          "grid_025_mae_m": float(complete.grid_025_abs_error_m.mean()),
                          "grid_025_p90_abs_error_m": float(complete.grid_025_abs_error_m.quantile(.9)),
                          "grid_025_max_abs_error_m": float(complete.grid_025_abs_error_m.max()),
                          "grid_025_mean_signed_error_m": float(complete.grid_025_signed_error_m.mean()),
                          "grid_010_diagnostic_mae_m": float(complete.grid_010_diagnostic_abs_error_m.mean()),
                          "raw_thin_intervals_lt_025": int(group.raw_thin_intervals_lt_025.sum()),
                          "thin_intervals_without_025_midpoint": int(group.thin_intervals_without_025_midpoint.sum()),
                          "thin_thickness_without_025_midpoint_m": float(group.thin_thickness_without_025_midpoint_m.sum())})
    pd.DataFrame(summaries).to_csv(output / "representation_summary.csv", index=False)
    log_path = dataset.root / "results/final_test/unblind_log.jsonl"
    unblind = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = {"version": "adaptive_v1", "seeds": list(seeds), "n_splits": len(splits),
               "n_source_profiles": len(dataset.profiles), "n_planned_positions": int(dataset.profiles.location_id.nunique()),
               "n_usable_positions": len(dataset.locations()), "n_grid_rows": len(dataset.grid), "n_sites": len(dataset.sites),
               "source_profile_exclusion_counts": dataset.profiles.exclusion_reason.value_counts().rename(index={"": "retained_representative"}).to_dict(),
               "n_position_near_groups": int(dataset.locations().near_group_id.nunique()),
               "near_group_size_counts": {str(int(size)): int(count) for size, count in dataset.locations().groupby("near_group_id").size().value_counts().sort_index().items()},
               "development_sites": DEVELOPMENT_FOLDS, "evaluation_sites": EVALUATION_FOLDS,
               "evaluation_scope": "limited-data re-evaluation; no new independent blind-test claim",
               "historical_unblind_events": unblind, "no_replay_cross_group_curve_duplicates": True,
               "near_duplicate_grid_pairs": len(duplicates),
               "cross_site_near_duplicate_grid_pairs": int(duplicates.cross_site.sum()) if len(duplicates) else 0,
               "duplicate_curve_rule": "offline maximum absolute log-qc difference <= 0.001 over all 24 cells; exact-grid screening, not full-curve identity proof",
               "whole_group_rule": "all near-group members stay together in S0, C, A and later batches",
               "source_exclusion_check_passed": True,
               "newlock_source_mapping": {"archive_profiles": len(newlock),
                                           "coordinate_matched": int(newlock.archive_coordinate_match.sum()),
                                           "screened_profiles": int(newlock.screened_2_8m.sum()),
                                           "excluded_profiles": int((~newlock.screened_2_8m).sum()),
                                           "exclusion_reasons": newlock.loc[~newlock.screened_2_8m, "exclusion_reason"].value_counts().to_dict()},
               "representation_reference": "piecewise-linear raw positive qc, exact threshold crossings, no gaps > 0.5 m, 2-8 m",
               "representation_complete_only_summary": True, "runtime_depth_step_m": .25,
               "grid_010_status": "diagnostic only; no model retraining or claim of a shared 0.1 m evaluation",
               "representation_overall": summaries[0],
               "frozen_inputs_sha256": {p.relative_to(dataset.root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in (dataset.root / "artifacts/evaluation/folds.json",
                                                    dataset.root / "data_processed_v2/profiles.parquet",
                                                    dataset.root / "data_processed_v2/grid.parquet")}}
    _write_json(output / "data_diagnostics.json", summary)
    return summary
