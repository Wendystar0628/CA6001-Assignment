"""Offline New Lock metadata and spatial predictability diagnostics.

All QC curves, including future/audit curves, are used ONLY for diagnostics.
This script does not implement an acquisition policy or alter prepared inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay
from scipy.stats import spearmanr
import xlrd
from openpyxl import load_workbook


def _finite(value):
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path('results/adaptive_v2_diagnostics'))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    profiles = pd.read_parquet(root / 'data_processed_v2/profiles.parquet')
    profiles = profiles.loc[profiles.site_id.eq('newlock') & profiles.usable & profiles.is_representative].copy()
    grid = pd.read_parquet(root / 'data_processed_v2/grid.parquet')
    grid = grid.loc[grid.location_id.isin(profiles.location_id)].copy()
    profiles['source_id'] = profiles.source_file.map(lambda s: Path(s).stem.split('_')[0])
    profiles['survey_family'] = profiles.source_id.str.extract(r'^([A-Za-z]+)', expand=False)
    metadata_rows, archive_summary = [], {}
    independent_path = root / 'Data/downloads/Terneuzen_CPT98_excel.zip'
    with zipfile.ZipFile(root / 'Data/downloads/Global_CPT1196.zip') as archive:
        members = sorted(n for n in archive.namelist() if '/@Newlock/' in n and not n.endswith('/'))
        workbook = load_workbook(io.BytesIO(archive.read('CPT_3_1196/@Newlock/locations.xlsx')),
                                 read_only=True, data_only=True)
        sheets = list(workbook.worksheets)
        position_rows = list(sheets[0].values)
        positions = {str(r[0]).strip().lower(): (float(r[1]), float(r[2])) for r in position_rows[1:] if r[0] is not None}
        archive_summary['locations'] = dict(
            sheet_count=len(sheets), row_count=len(position_rows) - 1,
            columns=list(position_rows[0]),
            nonempty_properties={k: str(getattr(workbook.properties, k)) for k in
                                 ('title', 'subject', 'description', 'keywords') if getattr(workbook.properties, k)},
            has_ground_elevation=False, has_explicit_crs=False,
        )
        workbook.close()
        hashes = {}
        for name in members:
            content = archive.read(name)
            hashes[Path(name).name] = hashlib.sha256(content).hexdigest()
            if not name.lower().endswith('.xls'):
                continue
            book = xlrd.open_workbook(file_contents=content)
            try:
                sh = book.sheet_by_index(0)
                values = np.asarray([sh.row_values(i)[:4] for i in range(1, sh.nrows)], dtype=float)
                valid = np.isfinite(values[:, :2]).all(axis=1) & (values[:, 0] >= 0) & (values[:, 1] > 0)
                nonincreasing = np.flatnonzero(np.diff(values[:, 0]) <= 0)
                source_id = Path(name).stem.split('_')[0]
                xy = positions.get(source_id.lower())
                p = profiles.loc[profiles.source_file.eq(name)]
                metadata_rows.append(dict(
                    source_file=name, source_id=source_id,
                    survey_family=re.match(r'[A-Za-z]+', source_id).group(0),
                    sheet_count=book.nsheets, column_count=sh.ncols,
                    headers=json.dumps(sh.row_values(0)), n_raw_rows=len(values),
                    invalid_or_nonpositive_qc_n=int((~valid).sum()),
                    depth_min_all=float(np.nanmin(values[:, 0])), depth_max_all=float(np.nanmax(values[:, 0])),
                    depth_min_valid=float(values[valid, 0].min()), depth_max_valid=float(values[valid, 0].max()),
                    depth_strictly_increasing=bool((np.diff(values[:, 0]) > 0).all()),
                    nonincreasing_depth_events=json.dumps([
                        {'previous_depth_m': float(values[k, 0]), 'next_depth_m': float(values[k + 1, 0])}
                        for k in nonincreasing]),
                    nonincreasing_in_2_8m_n=int(sum(2 <= values[k, 0] <= 8 or 2 <= values[k + 1, 0] <= 8 for k in nonincreasing)),
                    coordinate_match=xy is not None, prepared=not p.empty,
                    coordinate_prepared_max_abs_difference_m=(
                        float(np.max(np.abs(p[['x_m', 'y_m']].iloc[0].to_numpy(float) - xy))) if not p.empty else np.nan),
                    fs_valid_fraction=float(np.mean(np.isfinite(values[:, 2]) & (values[:, 2] > -9999))),
                    u2_valid_fraction=float(np.mean(np.isfinite(values[:, 3]) & (values[:, 3] > -9999))),
                ))
            finally:
                book.release_resources()
        archive_summary['global_members'] = members
        if independent_path.exists():
            with zipfile.ZipFile(independent_path) as original:
                independent_members = [n for n in original.namelist() if not n.endswith('/')]
                original_hashes = {Path(n).name: hashlib.sha256(original.read(n)).hexdigest() for n in independent_members}
                archive_summary['independent_archive'] = dict(
                    members=independent_members,
                    same_filename_count=len(set(hashes) & set(original_hashes)),
                    byte_identical_member_count=sum(hashes[n] == original_hashes[n] for n in hashes if n in original_hashes),
                    missing_from_global=sorted(set(original_hashes) - set(hashes)),
                )
                # Byte differences can be workbook container metadata. Compare
                # every changed numeric sheet instead of implying data drift.
                changed_checks = []
                original_names = {Path(n).name: n for n in independent_members}
                for name in members:
                    base = Path(name).name
                    if base not in original_hashes or hashes[base] == original_hashes[base]:
                        continue
                    first = xlrd.open_workbook(file_contents=archive.read(name))
                    second = xlrd.open_workbook(file_contents=original.read(original_names[base]))
                    try:
                        a, b = first.sheet_by_index(0), second.sheet_by_index(0)
                        av = np.asarray([a.row_values(i) for i in range(1, a.nrows)], dtype=float)
                        bv = np.asarray([b.row_values(i) for i in range(1, b.nrows)], dtype=float)
                        changed_checks.append(dict(source_filename=base,
                            numeric_data_identical=bool(av.shape == bv.shape and np.array_equal(av, bv, equal_nan=True))))
                    finally:
                        first.release_resources()
                        second.release_resources()
                archive_summary['independent_archive']['changed_container_numeric_checks'] = changed_checks

    raw_metadata = pd.DataFrame(metadata_rows)
    raw_metadata.to_csv(output / 'metadata_source_inventory.csv', index=False)
    ids = profiles.sort_values('location_id').location_id.tolist()
    p = profiles.set_index('location_id').loc[ids]
    curves = grid.pivot(index='location_id', columns='depth_index', values='log_qc').loc[ids].to_numpy(float)
    xy = p[['x_m', 'y_m']].to_numpy(float)
    distance = cdist(xy, xy)
    h = .25 * np.sum(curves < 0, axis=1)
    family = p.survey_family.to_numpy()
    groups = p.near_group_id.to_numpy()
    pair_rows = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            # Fixed central target depth interval for every shift. The best shift
            # is an oracle diagnostic; it is never a legal unrevealed feature.
            central = curves[i, 4:20]
            shift_errors = [np.mean(np.abs(central - curves[j, 4+s:20+s])) for s in range(-4, 5)]
            best = int(np.argmin(shift_errors))
            dx, dy = np.abs(xy[i] - xy[j])
            angle = float(np.degrees(np.arctan2(dy, dx)))
            pair_rows.append(dict(
                location_a=ids[i], location_b=ids[j], family_a=family[i], family_b=family[j],
                same_family=family[i] == family[j], same_near_group=groups[i] == groups[j],
                distance_m=float(distance[i, j]), abs_dx_m=float(dx), abs_dy_m=float(dy),
                direction_degrees=angle,
                curve_log_mae=float(np.mean(np.abs(curves[i] - curves[j]))),
                curve_semivariance=float(.5 * np.mean((curves[i] - curves[j]) ** 2)),
                thickness_abs_diff_m=float(abs(h[i] - h[j])),
                low_label_disagreement=float(np.mean((curves[i] < 0) != (curves[j] < 0))),
                central_unshifted_log_mae=float(shift_errors[4]),
                oracle_best_shift_m=.25 * (best - 4),
                oracle_shifted_log_mae=float(shift_errors[best]),
                oracle_shift_improvement=float(shift_errors[4] - shift_errors[best]),
            ))
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(output / 'metadata_pair_diagnostics.csv', index=False)

    location_rows = []
    for i, loc in enumerate(ids):
        eligible = np.flatnonzero(groups != groups[i])
        nearest = eligible[np.argsort(distance[i, eligible], kind='stable')]
        one, local = int(nearest[0]), nearest[:5]
        same = nearest[family[nearest] == family[i]]
        different = nearest[family[nearest] != family[i]]
        location_rows.append(dict(
            location_id=loc, source_id=p.loc[loc, 'source_id'], survey_family=family[i],
            x_m=float(xy[i, 0]), y_m=float(xy[i, 1]), low_thickness_m=float(h[i]),
            mean_log_qc=float(curves[i].mean()),
            nearest_outside_group_id=ids[one], nearest_distance_m=float(distance[i, one]),
            nearest_same_family=bool(family[i] == family[one]),
            nearest_curve_log_mae=float(np.mean(np.abs(curves[i] - curves[one]))),
            nearest_thickness_abs_diff_m=float(abs(h[i] - h[one])),
            five_neighbor_low_thickness_std_m=float(h[local].std()),
            five_neighbor_low_thickness_range_m=float(np.ptp(h[local])),
            nearest_same_family_distance_m=float(distance[i, same[0]]) if len(same) else np.nan,
            nearest_same_family_curve_log_mae=float(np.mean(np.abs(curves[i] - curves[same[0]]))) if len(same) else np.nan,
            nearest_other_family_distance_m=float(distance[i, different[0]]) if len(different) else np.nan,
            nearest_other_family_curve_log_mae=float(np.mean(np.abs(curves[i] - curves[different[0]]))) if len(different) else np.nan,
        ))
    locations = pd.DataFrame(location_rows)
    locations.to_csv(output / 'metadata_location_diagnostics.csv', index=False)
    family_summary = locations.groupby('survey_family', as_index=False).agg(
        n_locations=('location_id', 'size'), mean_log_qc=('mean_log_qc', 'mean'),
        mean_low_thickness_m=('low_thickness_m', 'mean'),
        std_low_thickness_m=('low_thickness_m', 'std'),
        mean_nearest_log_mae=('nearest_curve_log_mae', 'mean'),
        mean_nearest_h_abs_diff_m=('nearest_thickness_abs_diff_m', 'mean'),
    )
    family_summary.to_csv(output / 'metadata_family_summary.csv', index=False)

    grouped_pairs = pairs.loc[~pairs.same_near_group].copy()
    grouped_pairs['distance_bin_m'] = pd.cut(grouped_pairs.distance_m,
        [0, 10, 25, 50, 100, 200, 400, np.inf], right=False).astype(str)
    grouped_pairs['direction'] = np.where(grouped_pairs.direction_degrees <= 22.5, 'near_x_axis',
        np.where(grouped_pairs.direction_degrees >= 67.5, 'near_y_axis', 'diagonal'))
    spatial_summary = grouped_pairs.groupby(['distance_bin_m', 'same_family'], observed=True, as_index=False).agg(
        pair_n=('curve_log_mae', 'size'), mean_distance_m=('distance_m', 'mean'),
        mean_curve_log_mae=('curve_log_mae', 'mean'), median_curve_log_mae=('curve_log_mae', 'median'),
        mean_h_abs_diff_m=('thickness_abs_diff_m', 'mean'),
        mean_low_label_disagreement=('low_label_disagreement', 'mean'),
        mean_oracle_shift_improvement=('oracle_shift_improvement', 'mean'),
    )
    spatial_summary.to_csv(output / 'metadata_spatial_summary.csv', index=False)
    directional = grouped_pairs.groupby(['distance_bin_m', 'direction'], observed=True, as_index=False).agg(
        pair_n=('curve_log_mae', 'size'), mean_curve_log_mae=('curve_log_mae', 'mean'),
        mean_semivariance=('curve_semivariance', 'mean'))
    directional.to_csv(output / 'metadata_directional_summary.csv', index=False)

    audit_geometry = []
    split_path = root / 'results/adaptive_v1/splits.json'
    if split_path.exists():
        id_index = {loc: k for k, loc in enumerate(ids)}
        for split in json.loads(split_path.read_text(encoding='utf-8')):
            if split['site_id'] != 'newlock':
                continue
            train = np.asarray([id_index[loc] for loc in split['full_ids']])
            audit = np.asarray([id_index[loc] for loc in split['audit_ids']])
            inside = Delaunay(xy[train]).find_simplex(xy[audit]) >= 0
            for i, in_hull in zip(audit, inside):
                j = train[np.argmin(distance[i, train])]
                audit_geometry.append(dict(split_id=split['split_id'], seed=split['seed'],
                    audit_location_id=ids[i], audit_source_id=p.iloc[i].source_id,
                    inside_full_plan_convex_hull=bool(in_hull),
                    nearest_full_location_id=ids[j], nearest_full_distance_m=float(distance[i, j]),
                    nearest_full_curve_log_mae=float(np.mean(np.abs(curves[i] - curves[j]))),
                    nearest_full_h_abs_diff_m=float(abs(h[i] - h[j])),
                    nearest_full_same_family=bool(family[i] == family[j])))
        pd.DataFrame(audit_geometry).to_csv(output / 'metadata_audit_geometry.csv', index=False)

    # Leave out complete near groups. These are offline diagnostics of usable
    # geometric/survey metadata, not replay savings or held-out model selection.
    loo_rows = []
    for group in sorted(set(groups)):
        test = np.flatnonzero(groups == group)
        train = np.flatnonzero(groups != group)
        for i in test:
            near = train[np.argsort(distance[i, train], kind='stable')[:5]]
            same_train = train[family[train] == family[i]]
            predictions = {'all_depth_median': np.median(curves[train], axis=0),
                           'five_nearest_depth_median': np.median(curves[near], axis=0)}
            if len(same_train) >= 3:
                predictions['same_family_depth_median'] = np.median(curves[same_train], axis=0)
                same_near = same_train[np.argsort(distance[i, same_train], kind='stable')[:5]]
                predictions['same_family_five_nearest_depth_median'] = np.median(curves[same_near], axis=0)
            for method, pred in predictions.items():
                lo, tl = pred < 0, curves[i] < 0
                loo_rows.append(dict(location_id=ids[i], near_group_id=group, survey_family=family[i],
                    diagnostic_method=method, log_mae=float(np.mean(np.abs(pred - curves[i]))),
                    thickness_abs_error_m=float(abs(.25 * lo.sum() - h[i])),
                    tp=int((lo & tl).sum()), fn=int((~lo & tl).sum()),
                    fp=int((lo & ~tl).sum()), tn=int((~lo & ~tl).sum())))
    loo = pd.DataFrame(loo_rows)
    loo.to_csv(output / 'metadata_group_leaveout.csv', index=False)
    loo_summary = []
    for method, rows in loo.groupby('diagnostic_method'):
        tp, fn, fp, tn = rows[['tp', 'fn', 'fp', 'tn']].sum()
        loo_summary.append(dict(method=method, n_locations=len(rows), log_mae=rows.log_mae.mean(),
            thickness_mae_m=rows.thickness_abs_error_m.mean(), recall=tp/(tp+fn), fpr=fp/(fp+tn)))

    same_group = pairs.loc[pairs.same_near_group]
    nearest_ids = locations.set_index('location_id').nearest_outside_group_id.to_dict()
    nearest_pairs = pairs.loc[pairs.apply(lambda row:
        nearest_ids[row.location_a] == row.location_b or nearest_ids[row.location_b] == row.location_a, axis=1)]
    summary = dict(
        experiment_kind='post_unblinding_offline_metadata_diagnostic',
        metadata=archive_summary,
        n_prepared_locations=len(profiles), n_near_groups=profiles.near_group_id.nunique(),
        prepared_depth_reference=profiles.depth_reference.unique().tolist(),
        raw_header_counts=raw_metadata.headers.value_counts().to_dict(),
        raw_sheet_count_counts=raw_metadata.sheet_count.value_counts().to_dict(),
        source_prepared_coordinate_max_abs_difference_m=raw_metadata.coordinate_prepared_max_abs_difference_m.max(),
        raw_depth_nonincreasing_profile_n=int((~raw_metadata.depth_strictly_increasing).sum()),
        raw_depth_nonincreasing_2_8m_event_n=int(raw_metadata.nonincreasing_in_2_8m_n.sum()),
        spatial_spearman=dict(
            distance_vs_curve_mae=spearmanr(grouped_pairs.distance_m, grouped_pairs.curve_log_mae).statistic,
            distance_vs_h_difference=spearmanr(grouped_pairs.distance_m, grouped_pairs.thickness_abs_diff_m).statistic,
            abs_dx_vs_curve_mae=spearmanr(grouped_pairs.abs_dx_m, grouped_pairs.curve_log_mae).statistic,
            abs_dy_vs_curve_mae=spearmanr(grouped_pairs.abs_dy_m, grouped_pairs.curve_log_mae).statistic),
        nearest_outside_group=dict(mean_distance_m=locations.nearest_distance_m.mean(),
            median_distance_m=locations.nearest_distance_m.median(),
            mean_curve_log_mae=locations.nearest_curve_log_mae.mean(),
            median_curve_log_mae=locations.nearest_curve_log_mae.median(),
            mean_h_abs_diff_m=locations.nearest_thickness_abs_diff_m.mean(),
            five_neighbor_mean_h_std_m=locations.five_neighbor_low_thickness_std_m.mean()),
        within_near_group=dict(pair_n=len(same_group), mean_curve_log_mae=same_group.curve_log_mae.mean(),
            mean_h_abs_diff_m=same_group.thickness_abs_diff_m.mean()),
        nearest_oracle_vertical_alignment=dict(unique_nearest_pair_n=len(nearest_pairs),
            central_unshifted_log_mae=nearest_pairs.central_unshifted_log_mae.mean(),
            central_oracle_shifted_log_mae=nearest_pairs.oracle_shifted_log_mae.mean(),
            fraction_selected_nonzero_shift=(nearest_pairs.oracle_best_shift_m != 0).mean(),
            meaning='Diagnostic only; selecting the shift requires both true curves.'),
        audit_geometry_summary=pd.DataFrame(audit_geometry).groupby('seed').agg(
            n_audit=('audit_location_id', 'size'),
            inside_hull_fraction=('inside_full_plan_convex_hull', 'mean'),
            mean_nearest_distance_m=('nearest_full_distance_m', 'mean'),
            mean_nearest_curve_log_mae=('nearest_full_curve_log_mae', 'mean'),
            mean_nearest_h_difference_m=('nearest_full_h_abs_diff_m', 'mean')).reset_index().to_dict('records')
            if audit_geometry else [],
        grouped_leaveout=loo_summary,
        oracle_alignment_warning='QC-based shifts use both complete curves and cannot be used as unrevealed/audit geometry. No elevations or datum offsets are supplied in either archive.',
        limitations=['Depth header identifies a depth axis in metres but does not establish a common ground/elevation datum.',
            'Survey identifier family is known from workbook names; its role as a geologic or elevation covariate is unverified.',
            'Pair statistics reuse locations and are descriptive; no independent-pair confidence claims are made.',
            'Known/future/audit labels are all read for offline diagnostics; no saving or online generalisation claim is made.',
            'fs and u2 are measured CPT responses and cannot serve as features before a location is investigated.'],
        mutations='only new diagnostic script and CSV/JSON outputs', timing_compared=False,
    )
    (output / 'metadata_summary.json').write_text(json.dumps(_finite(summary), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(_finite({k: v for k, v in summary.items() if k != 'metadata'}), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
