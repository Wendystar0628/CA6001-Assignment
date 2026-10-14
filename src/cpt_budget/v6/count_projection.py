"""Frozen source count, minimum-change point ordering, independent GP risk head.

This fourth-round exploration keeps every profile's learned half-count fixed.
It changes only the depth ordering used to allocate those low-resistance cells.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cpt_budget.config import write_json
from .common import RUN, REFERENCE, BEST, KEYS, load_frozen, emit_candidate
from .thickness_head import minimally_relabel

PROFILE_KEYS = KEYS[:-1]
BASES = {
    "v6_count_project_fpca_guard_gp": ("functional", "v6_fpca_residual_sign_guard"),
    "v6_count_project_graph_union_gp": ("graph", "v6_graph_residual_low_union"),
    "v6_count_project_support_stack_gp": ("adaptive", "support_stack_structural"),
}
COUNT_COLUMNS = [*PROFILE_KEYS, "half_count_decision", "predicted_continuous_count", "residual_point_count"]


def load_frozen_counts(path=None):
    """Explicit allowlist: query truth and unrelated fields never enter the loader."""
    counts = pd.read_csv(path or RUN / "thickness_head/profile_decisions.csv", usecols=COUNT_COLUMNS)
    if counts.duplicated(PROFILE_KEYS).any():
        raise ValueError("Duplicate profile count decisions")
    numeric = counts[["half_count_decision", "predicted_continuous_count", "residual_point_count"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("Nonfinite source-head count decisions")
    k, predicted, existing = numeric.T
    if (not np.equal(k, np.floor(k)).all() or ((k < 0) | (k > 24)).any()
            or not np.equal(existing, np.floor(existing)).all() or ((existing < 0) | (existing > 24)).any()):
        raise ValueError("Frozen count is not a valid 24-cell integer")
    expected = np.clip(np.floor(.5*(predicted+existing)+.5), 0, 24).astype(int)
    if not np.array_equal(k.astype(int), expected):
        raise ValueError("Frozen half count disagrees with predeclared source-count formula")
    counts["half_count_decision"] = k.astype(int)
    return counts


def project_count(base, count, margin=.05):
    """For a fixed count, use the base's lowest cells and modify only label flips.

Under this fixed crossing rule, ascending base order minimizes total absolute
point change. Ties preserve the depth order supplied by the caller.
"""
    base = np.asarray(base, float)
    if base.ndim != 1 or not np.isfinite(base).all():
        raise ValueError("One finite complete point profile required")
    count = float(count)
    if not np.isfinite(count) or count != int(count) or not 0 <= count <= len(base):
        raise ValueError("Requested count must be an in-range integer")
    desired = np.zeros(len(base), dtype=bool)
    desired[np.argsort(base, kind="stable")[:int(count)]] = True
    return minimally_relabel(base, desired, margin=margin)


def _align_prediction(wide, path, column, model_id=None):
    columns = [*KEYS, column] + (["model_id"] if model_id is not None else [])
    frame = pd.read_parquet(path, columns=columns)
    if model_id is not None:
        frame = frame[frame.model_id.eq(model_id)].drop(columns="model_id")
    aligned = wide[KEYS].merge(frame, on=KEYS, how="left", validate="one_to_one")
    value = aligned[column].to_numpy(float)
    if len(aligned) != len(wide) or not np.isfinite(value).all():
        raise ValueError("Frozen point or risk head does not cover every cell")
    return value


def project_frame(wide, base, counts):
    """Only geometry, frozen prediction values and source-count decisions enter."""
    if wide.duplicated(KEYS).any():
        raise ValueError("Duplicate frozen cell keys")
    base = np.asarray(base, float)
    if base.shape != (len(wide),):
        raise ValueError("Base predictions do not align with frozen rows")
    # Drop unrelated fields even if a caller passed a larger table.
    counted = wide[KEYS].merge(counts[[*PROFILE_KEYS, "half_count_decision"]],
        on=PROFILE_KEYS, how="left", validate="many_to_one")
    if counted.half_count_decision.isna().any():
        raise ValueError("Missing frozen profile count")
    predicted = np.empty(len(wide), float)
    records = []
    for key, block in counted.groupby(PROFILE_KEYS, sort=False):
        block = block.sort_values("depth_index")
        if len(block) != 24 or block.depth_index.tolist() != list(range(24)):
            raise ValueError("Projection requires the complete 24-cell profile")
        index = block.index.to_numpy()
        k = int(block.half_count_decision.iloc[0])
        values = project_count(base[index], k)
        predicted[index] = values
        actual = int((values < 0.).sum())
        if actual != k:
            raise RuntimeError("Final point thickness differs from frozen count")
        records.append({**dict(zip(PROFILE_KEYS, key)), "frozen_count": k,
            "actual_point_count": actual, "base_count": int((base[index] < 0.).sum()),
            "flipped_cells": int(((base[index] < 0.) != (values < 0.)).sum()),
            "absolute_point_change": float(np.abs(values-base[index]).sum())})
    return predicted, pd.DataFrame(records)


def protocol():
    return {
        "round": 4, "trigger": "Round 3 learned half-count improved thickness; event ranking changed point magnitudes.",
        "candidates": BASES, "frozen_count_file": "thickness_head/profile_decisions.csv",
        "count_allowlist": COUNT_COLUMNS, "count_column": "half_count_decision",
        "count_formula": "floor((source_predicted_count + original_residual_point_count)/2 + .5)",
        "count_fitting": "Unchanged round-3 source-only count head; no refit or new parameter selection.",
        "point_projection": "Ascending base logqc, stable depth tie-break, choose exactly k; only flipped cells set to +/- .05.",
        "risk_head": "Frozen gp p_low_1; independent task head, not inferred from projected point values.",
        "risk_intervals": "None transferred. GP event probability is not a native CDF of the projected point model.",
        "query_truth_read_by_generator": False, "outer_test_used": False,
        "all_candidates_and_parameters_frozen_before_new_scores": True,
    }


def run(output=None):
    output = Path(output or RUN / "count_projection")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "method_protocol.json", protocol())
    wide = load_frozen()
    counts = load_frozen_counts()
    original_counts = wide.assign(_low=wide[BEST] < 0.).groupby(PROFILE_KEYS)._low.sum().rename("actual_residual_count").reset_index()
    check = counts.merge(original_counts, on=PROFILE_KEYS, validate="one_to_one")
    if len(check) != len(counts) or not check.residual_point_count.eq(check.actual_residual_count).all():
        raise ValueError("Stored count provenance differs from frozen original residual")
    gp_probability = _align_prediction(wide, REFERENCE / "predictions_oof.parquet", "p_low_1", "gp")
    original_half = _align_prediction(wide, RUN / "thickness_head/v6_count_head_half_residual/predictions.parquet", "pred_log_qc")
    existing_half_counts = wide.assign(_low=original_half < 0.).groupby(PROFILE_KEYS)._low.sum().rename("actual_half_count").reset_index()
    check = counts.merge(existing_half_counts, on=PROFILE_KEYS, validate="one_to_one")
    if len(check) != len(counts) or not check.half_count_decision.eq(check.actual_half_count).all():
        raise ValueError("Stored half count differs from third-round final point profile")
    diagnostics = []
    for model_id, (family, base_id) in BASES.items():
        base = _align_prediction(wide, RUN / family / base_id / "predictions.parquet", "pred_log_qc")
        prediction, records = project_frame(wide, base, counts)
        records["model_id"] = model_id
        diagnostics.append(records)
        emit_candidate(model_id, prediction, wide=wide, output_root=output,
            probability=gp_probability, metadata={"round": 4, "base_model": base_id,
                "count_model": "v6_count_head_half_residual", "risk_model": "gp",
                "risk_head_independent_of_point_projection": True, "point_margin": .05})
    diagnostics = pd.concat(diagnostics, ignore_index=True)
    diagnostics.to_csv(output / "projection_diagnostics.csv", index=False)
    write_json(output / "execution_summary.json", {"models": list(BASES), "states_per_model": 60,
        "prediction_rows_per_model": len(wide), "profiles_per_model": len(counts),
        "all_final_profile_counts_equal_round3_half_count": bool(diagnostics.frozen_count.eq(diagnostics.actual_point_count).all()),
        "native_gp_probabilities_unchanged": True, "parameters_frozen_before_scores": True})
    return diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.output)
