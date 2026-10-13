"""Restore the screened numeric inputs from their original CPT workbooks.

This replays the existing parser for explicitly registered profiles; it neither
re-screens the full database nor changes the screened manifest or raw archive.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np


def _read_profile(content: bytes, source_file: str) -> np.ndarray:
    suffix = Path(source_file).suffix.lower()
    if suffix == ".xlsx":
        from openpyxl import load_workbook

        workbook = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
        try:
            rows = iter(workbook.worksheets[0].values)
            header = next(rows)
            pairs = []
            for row in rows:
                try:
                    depth, qc = float(row[0]), float(row[1])
                except (ValueError, TypeError, IndexError):
                    continue
                if np.isfinite(depth) and np.isfinite(qc) and depth >= 0 and qc > 0:
                    pairs.append((depth, qc))
            array = np.asarray(pairs, dtype=float).reshape(-1, 2)
        finally:
            workbook.close()
    elif suffix == ".xls":
        import xlrd

        workbook = xlrd.open_workbook(file_contents=content)
        try:
            sheet = workbook.sheet_by_index(0)
            header = sheet.row_values(0)
            # Match the original New Lock audit: all body cells in these two
            # columns are numeric. A source revision with text fails explicitly.
            array = np.asarray([sheet.row_values(i)[:2] for i in range(1, sheet.nrows)], dtype=float).reshape(-1, 2)
            array = array[np.isfinite(array).all(axis=1) & (array[:, 0] >= 0) & (array[:, 1] > 0)]
        finally:
            workbook.release_resources()
    else:
        raise ValueError(f"Unsupported registered source format: {source_file}")
    if len(header) < 2 or "qc" not in str(header[1]).lower() or "mpa" not in str(header[1]).lower():
        raise ValueError(f"Source header does not confirm qc in MPa: {source_file}")
    if len(array) < 2:
        raise ValueError(f"Fewer than two valid depth/qc rows: {source_file}")
    array = array[np.argsort(array[:, 0], kind="stable")]
    _, keep = np.unique(array[:, 0], return_index=True)
    return array[keep].astype(np.float32)


def restore_arrays(config: dict) -> dict:
    """Create the configured NPZ only when absent and return a JSON-safe summary.

    Existing files are left entirely untouched, including in a concurrent race.
    Their counts are returned as None because this operation does not inspect them.
    """
    root = Path(config["_root"])
    input_dir = root / config.get("project", {}).get("input_dir", "Data")
    options = config.get("data", {})
    manifest_path = input_dir / options.get("manifest", "cpt_screened_spatial_profiles.json")
    archive_path = input_dir / options.get("source_archive", "downloads/Global_CPT1196.zip")
    arrays_path = input_dir / options.get("arrays", "cpt_screened_depth_qc.npz")
    result = {"status": "already_exists", "arrays_path": str(arrays_path.resolve()),
              "manifest_path": str(manifest_path.resolve()), "archive_path": str(archive_path.resolve()),
              "n_arrays": None, "n_rows": None, "source_formats": None}
    if arrays_path.exists():
        return result
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Missing source archive: {archive_path}. Download Global_CPT_3_1196.zip "
            "from https://tc304.org/attachments/database/cpt/global_cpt/Global_CPT_3_1196.zip "
            "and save it as Data/downloads/Global_CPT1196.zip (see Data/README.md)."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("Screened manifest must be a nonempty list")
    arrays, formats = {}, Counter()
    with zipfile.ZipFile(archive_path) as archive:
        for record in manifest:
            source_file, array_key = record["file"], str(record["array_key"])
            if array_key in arrays:
                raise ValueError(f"Screened manifest repeats array_key {array_key!r}")
            # Zip members are read by exact name; nothing is extracted to disk.
            arrays[array_key] = _read_profile(archive.read(source_file), source_file)
            formats[Path(source_file).suffix.lower().lstrip(".")] += 1
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".cpt_restore_", suffix=".npz", dir=arrays_path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(temporary, **arrays)
        # A same-directory hard link publishes the complete file atomically and
        # fails if another process has already created the destination. Unlike
        # replace(), it cannot overwrite an existing NPZ.
        try:
            os.link(temporary_path, arrays_path)
        except FileExistsError:
            return result
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    result.update(status="restored", n_arrays=len(arrays), n_rows=sum(len(array) for array in arrays.values()), source_formats=dict(formats))
    return result
