"""Convert the screened registry into complete midpoint CPT profiles.

No labels are used for representative selection, geometry grouping or scale.
The raw NPZ remains the source for displaying original sampling intervals.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from openpyxl import load_workbook


def stable_id(prefix: str, *parts: object) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return prefix + "_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def site_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def connected_components(ids: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    parent = {key: key for key in ids}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for a, b in edges:
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    groups: dict[str, list[str]] = {}
    for key in sorted(ids):
        groups.setdefault(find(key), []).append(key)
    return sorted(groups.values())


def geometry_scale(coordinates: np.ndarray) -> float:
    """Median nearest-neighbor spacing from the fixed, distinct planned points."""
    points = np.unique(np.asarray(coordinates, dtype=float), axis=0)
    if len(points) < 2:
        return 1.0  # Explicit singleton fallback, never estimated from measurements.
    distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    return float(np.median(distance.min(axis=1)))


def midpoint_values(array: np.ndarray, centers: np.ndarray, max_gap: float) -> np.ndarray:
    """Linear interpolation without extrapolation or bridging a large gap."""
    values = np.asarray(array, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Each NPZ entry must have exactly depth and qc columns")
    values = values[np.isfinite(values).all(axis=1) & (values[:, 0] >= 0) & (values[:, 1] > 0)]
    if not len(values):
        return np.full(len(centers), np.nan)
    values = values[np.argsort(values[:, 0], kind="stable")]
    # Keep the first valid measurement at a repeated depth, matching the source parser.
    _, unique = np.unique(values[:, 0], return_index=True)
    depth, qc = values[unique].T
    result = np.interp(centers, depth, qc, left=np.nan, right=np.nan)
    right = np.searchsorted(depth, centers, side="left")
    for k, (center, j) in enumerate(zip(centers, right)):
        if j < len(depth) and np.isclose(depth[j], center, rtol=0, atol=1e-9):
            result[k] = qc[j]
        elif 0 < j < len(depth) and depth[j] - depth[j - 1] > max_gap + 1e-9:
            result[k] = np.nan
    return result


def _normal_id(value: object) -> str:
    if isinstance(value, (int, float)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip().lower()


def recover_global_coordinates(archive_path: Path, source_sites: set[str]) -> tuple[dict, dict]:
    """Read only location workbooks; never guess a CRS from coordinate magnitude."""
    coordinates, provenance = {}, {}
    if not archive_path.exists():
        return coordinates, {site: {"status": "archive_missing"} for site in source_sites}
    with zipfile.ZipFile(archive_path) as archive:
        for name in sorted(archive.namelist()):
            parts = name.split("/")
            if len(parts) < 3 or parts[1] not in source_sites or "locat" not in name.lower() or not name.endswith(".xlsx"):
                continue
            site = parts[1]
            workbook = load_workbook(io.BytesIO(archive.read(name)), read_only=True, data_only=True)
            rows = iter(workbook.worksheets[0].values)
            header = [str(v or "").strip().lower() for v in next(rows)]
            longitude_index = next((i for i, h in enumerate(header) if h in {"x(degree)", "longitude", "longitude(degree)"}), None)
            latitude_index = next((i for i, h in enumerate(header) if h in {"y(degree)", "latitude", "latitude(degree)"}), None)
            properties = {key: str(getattr(workbook.properties, key) or "") for key in ("title", "subject", "description", "keywords")}
            provenance[site] = {"coordinate_file": name, "header": header, "workbook_properties": properties,
                                "status": "geographic_columns" if longitude_index is not None and latitude_index is not None else "global_coordinates_unavailable"}
            for row in rows:
                if not row or row[0] is None or longitude_index is None or latitude_index is None:
                    continue
                try:
                    longitude, latitude = float(row[longitude_index]), float(row[latitude_index])
                except (ValueError, TypeError, IndexError):
                    continue
                if np.isfinite(longitude) and np.isfinite(latitude) and abs(longitude) <= 180 and abs(latitude) <= 90:
                    coordinates[(site, _normal_id(row[0]))] = (longitude, latitude, name)
            workbook.close()
    return coordinates, provenance


def _minimum_haversine(a: np.ndarray, b: np.ndarray) -> float:
    # Longitude/latitude in degrees; avoid comparing unrelated local origins.
    a, b = np.deg2rad(a), np.deg2rad(b)
    delta = a[:, None, :] - b[None, :, :]
    h = np.sin(delta[:, :, 1] / 2) ** 2 + np.cos(a[:, None, 1]) * np.cos(b[None, :, 1]) * np.sin(delta[:, :, 0] / 2) ** 2
    return float((2 * 6371008.8 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))).min())


def _investigation_groups(profiles: pd.DataFrame, definitions: dict, radius: float) -> tuple[dict, list[dict]]:
    ids = sorted(profiles.site_id.unique())
    edges, evidence = [], []
    coordinates = {sid: frame[["longitude", "latitude"]].dropna().drop_duplicates().to_numpy() for sid, frame in profiles.groupby("site_id")}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            manual_a = definitions.get(a, {}).get("investigation_group")
            manual_b = definitions.get(b, {}).get("investigation_group")
            if manual_a and manual_a == manual_b:
                edges.append((a, b))
                evidence.append({"site_a": a, "site_b": b, "reason": "fixed_sites_yaml_group", "group": manual_a})
            if len(coordinates[a]) and len(coordinates[b]):
                distance = _minimum_haversine(coordinates[a], coordinates[b])
                if distance <= radius:
                    edges.append((a, b))
                    evidence.append({"site_a": a, "site_b": b, "reason": "geographic_nearest_pair", "distance_m": distance})
    mapping = {}
    for component in connected_components(ids, edges):
        configured = {definitions.get(sid, {}).get("investigation_group") for sid in component}
        group_id = next(iter(configured)) if len(configured) == 1 and None not in configured else "investigation_" + "__".join(component)
        mapping.update({sid: group_id for sid in component})
    return mapping, evidence


def prepare_data(config: dict):
    from cpt_budget.types import PreparedDataset

    root = Path(config["_root"])
    project, options = config.get("project", {}), config.get("data", {})
    input_dir = root / project.get("input_dir", "Data")
    prepared_dir = root / project.get("prepared_dir", "data_processed")
    manifest_path = input_dir / options.get("manifest", "cpt_screened_spatial_profiles.json")
    arrays_path = input_dir / options.get("arrays", "cpt_screened_depth_qc.npz")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    definitions_path = root / config.get("split", {}).get("sites_config", "configs/sites.yaml")
    site_document = yaml.safe_load(definitions_path.read_text(encoding="utf-8")) if definitions_path.exists() else {}
    definitions = (site_document or {}).get("sites", {})
    source_definition = {entry.get("source_site", sid): (sid, entry) for sid, entry in definitions.items()}
    start, end, step = (float(options.get(name, default)) for name, default in (("depth_start_m", 2), ("depth_end_m", 8), ("depth_step_m", .25)))
    bins = int(round((end - start) / step))
    if bins <= 0 or not np.isclose(start + bins * step, end):
        raise ValueError("Depth window must contain a positive integer number of cells")
    centers = start + (np.arange(bins) + .5) * step
    coordinate_map, coordinate_provenance = recover_global_coordinates(input_dir / "downloads/Global_CPT1196.zip", {r["site"] for r in manifest})
    records, interpolated = [], {}
    with np.load(arrays_path, allow_pickle=False) as arrays:
        for record in manifest:
            source_site, source_file = record["site"], record["file"]
            site_id, definition = source_definition.get(source_site, (site_slug(source_site), {}))
            key = str(record["array_key"])
            if key not in arrays:
                raise ValueError(f"Missing explicit array_key {key!r} for {source_file}")
            profile_id = stable_id("profile", source_site, source_file)
            xy = record.get("xy")
            if xy is None or len(xy) != 2 or not np.isfinite(xy).all():
                raise ValueError(f"Screened profile lacks finite local coordinates: {source_file}")
            x, y = (0.0 if float(v) == 0 else float(v) for v in xy)
            location_id = stable_id("location", site_id, x, y)
            header = record.get("header", [])
            if len(header) < 2 or "qc" not in str(header[1]).lower() or "mpa" not in str(header[1]).lower():
                raise ValueError(f"qc/MPa provenance missing: {source_file}")
            measurements = arrays[key]
            values = midpoint_values(measurements, centers, float(options.get("max_interpolation_gap_m", .5)))
            interpolated[profile_id] = values
            # New Lock's profile filenames include a run suffix; location workbook IDs do not.
            source_id = Path(source_file).stem.split("_")[0] if source_site == "@Newlock" else Path(source_file).stem
            longitude, latitude, coordinate_file = coordinate_map.get((source_site, _normal_id(source_id)), (np.nan, np.nan, coordinate_provenance.get(source_site, {}).get("coordinate_file", "")))
            valid_depth = measurements[np.isfinite(measurements).all(axis=1) & (measurements[:, 1] > 0), 0]
            intervals = np.diff(np.unique(valid_depth))
            usable = bool(np.isfinite(values).all())
            records.append({"profile_id": profile_id, "site_id": site_id, "source_site": source_site,
                            "location_id": location_id, "source_file": source_file, "array_key": key,
                            "x_m": x, "y_m": y, "longitude": longitude, "latitude": latitude,
                            "coordinate_file": coordinate_file, "coordinate_reference": definition.get("coordinate_reference", "source_local_xy_m"),
                            "depth_reference": "recorded_depth_m", "usable": usable, "is_representative": False,
                            "exclusion_reason": "" if usable else "incomplete_midpoint_grid_or_interpolation_gap",
                            "raw_sampling_median_m": float(np.median(intervals)) if len(intervals) else np.nan,
                            "raw_sampling_min_m": float(intervals.min()) if len(intervals) else np.nan,
                            "raw_sampling_max_m": float(intervals.max()) if len(intervals) else np.nan})
    profiles = pd.DataFrame(records).sort_values(["site_id", "source_file"]).reset_index(drop=True)
    if profiles.profile_id.duplicated().any():
        raise ValueError("Manifest repeats a source profile; refusing an ambiguous registry")
    for _, frame in profiles.groupby("location_id", sort=True):
        eligible = frame[frame.usable]
        if len(eligible):
            representative = eligible.sort_values("source_file").index[0]
            profiles.loc[representative, "is_representative"] = True
            profiles.loc[eligible.index.difference([representative]), "exclusion_reason"] = "same_coordinate_nonrepresentative"
    near_radius = float(config.get("split", {}).get("within_site_near_radius_m", 2))
    for site_id, frame in profiles.groupby("site_id", sort=True):
        locations = frame.drop_duplicates("location_id").sort_values("location_id")
        points, ids = locations[["x_m", "y_m"]].to_numpy(), locations.location_id.tolist()
        distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        edges = [(ids[i], ids[j]) for i, j in zip(*np.where(np.triu(distances < near_radius, 1)))]
        membership = {loc: stable_id("near", site_id, component) for component in connected_components(ids, edges) for loc in component}
        profiles.loc[frame.index, "near_group_id"] = frame.location_id.map(membership)
        profiles.loc[frame.index, "geometry_scale_m"] = geometry_scale(points)
    groups, group_edges = _investigation_groups(profiles, definitions, float(config.get("split", {}).get("grouping_radius_m", 250)))
    profiles["investigation_group"] = profiles.site_id.map(groups)
    sites, grid_parts = [], []
    for site_id, frame in profiles.groupby("site_id", sort=True):
        selected = frame[frame.is_representative & frame.usable]
        definition = definitions.get(site_id, {})
        global_complete = bool(frame[["longitude", "latitude"]].notna().all(axis=1).all())
        verified = bool(global_complete or definition.get("grouping_verified", False))
        sites.append({"site_id": site_id, "source_site": frame.source_site.iloc[0],
                      "display_name": definition.get("display_name", frame.source_site.iloc[0].lstrip("@")),
                      "investigation_group": groups[site_id], "grouping_verified": verified,
                      "global_position_verified": global_complete,
                      "grouping_evidence": definition.get("grouping_evidence", "source geographic coordinate columns" if global_complete else "original archive site folder only; cross-site independence unverified"),
                      "coordinate_reference": frame.coordinate_reference.iloc[0],
                      "longitude": float(frame.longitude.mean()), "latitude": float(frame.latitude.mean()),
                      "geometry_scale_m": float(frame.geometry_scale_m.iloc[0]),
                      "geometry_scale_source": "all_screened_planned_distinct_positions" if frame.location_id.nunique() > 1 else "singleton_1m_fallback",
                      "n_profiles": len(frame), "n_planned_locations": frame.location_id.nunique(),
                      "n_locations": len(selected), "n_near_groups": selected.near_group_id.nunique(),
                      "replay_eligible": len(selected) >= int(config.get("replay", {}).get("min_site_locations", 20))})
        for row in selected.itertuples():
            qc = interpolated[row.profile_id]
            block = pd.DataFrame({"profile_id": row.profile_id, "site_id": site_id, "location_id": row.location_id,
                                  "near_group_id": row.near_group_id, "x_m": row.x_m, "y_m": row.y_m,
                                  "geometry_scale_m": row.geometry_scale_m, "depth_index": np.arange(bins),
                                  "z_center_m": centers, "qc_mpa": qc, "log_qc": np.log(qc),
                                  "depth_reference": "recorded_depth_m"})
            for threshold, suffix in ((.5, "0_5"), (1., "1"), (2., "2")):
                block[f"low_{suffix}"] = qc < threshold
                block[f"thickness_{suffix}_m"] = float((qc < threshold).sum() * step)
            grid_parts.append(block)
    if not grid_parts:
        raise ValueError("No complete representative CPT grids remain")
    grid, sites_frame = pd.concat(grid_parts, ignore_index=True), pd.DataFrame(sites)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("profiles", profiles), ("grid", grid), ("sites", sites_frame)):
        frame.to_parquet(prepared_dir / f"{name}.parquet", index=False)
    summary = {"manifest": str(manifest_path.relative_to(root)), "arrays": str(arrays_path.relative_to(root)),
               "n_candidate_profiles": len(profiles), "n_distinct_planned_locations": int(profiles.location_id.nunique()),
               "n_usable_representative_locations": int(grid.location_id.nunique()), "n_grid_rows": len(grid),
               "depth_cells": bins, "depth_centers_m": centers.tolist(), "n_sites": len(sites_frame),
               "n_replay_sites_by_size": int(sites_frame.replay_eligible.sum()), "n_investigation_groups": int(sites_frame.investigation_group.nunique()),
               "excluded_profile_reasons": profiles.loc[profiles.exclusion_reason != "", "exclusion_reason"].value_counts().to_dict(),
               "coordinate_provenance": coordinate_provenance, "investigation_group_edges": group_edges,
               "unverified_grouping_sites": sites_frame.loc[~sites_frame.grouping_verified, "site_id"].tolist(),
               "limitations": ["Missing global positions are not reconstructed from local coordinates; provisional groups do not prove cross-site independence.",
                               "Global coordinate columns are used as geographic degrees; their original datum is not declared in the workbook.",
                               "Recorded depth is retained; elevation and inclination corrections are not mixed into the main task."]}
    (prepared_dir / "preparation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return PreparedDataset(profiles, grid, sites_frame, root)
