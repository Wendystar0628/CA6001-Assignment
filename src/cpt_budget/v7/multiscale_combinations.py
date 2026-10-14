"""Combine predicted thickness counts or fixed Haar bands without query truth."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from ..config import write_json
from .multiscale import haar_forward, haar_inverse
from .common import RUN, KEYS, emit_candidate, load_frozen, load_native, read_candidate

PROFILE_KEYS = KEYS[:-1]
BASES = {"simplex": "v7_simplex_linear", "fpca": "v6_fpca_residual_w050"}
DETAILS = {"nearest": "v7_haar_krr_nearest", "median": "v7_haar_krr_median"}
ALPHAS = (0.5, 1.0)


def transfer_low_count(base_profiles, detail_profiles, alpha, *, margin=0.05):
    """Preserve base amplitudes except the minimal count-changing sign flips.

    Columns must already be sorted by depth. Stable sorting uses increasing
    depth for any equal base predictions. Half counts are rounded upward.
    """
    base, detail = np.asarray(base_profiles, float), np.asarray(detail_profiles, float)
    if base.ndim != 2 or detail.shape != base.shape or not np.isfinite(base).all() or not np.isfinite(detail).all():
        raise ValueError("Finite aligned profile matrices are required")
    if not 0 <= alpha <= 1 or margin <= 0:
        raise ValueError("Invalid count blend or sign-flip margin")
    base_count, detail_count = (base < 0).sum(axis=1), (detail < 0).sum(axis=1)
    target_count = np.floor((1-alpha)*base_count + alpha*detail_count + 0.5).astype(int)
    order = np.argsort(base, axis=1, kind="stable")
    target_low = np.zeros_like(base, dtype=bool)
    for row, count in enumerate(target_count):
        target_low[row, order[row, :count]] = True
    output = base.copy()
    output[target_low & (base >= 0)] = -margin
    output[(~target_low) & (base < 0)] = margin
    return output, target_count


def combine_haar_bands(base_profiles, detail_profiles):
    """Use exactly 3 coarse base coefficients and 21 detail-member coefficients."""
    base, detail = np.asarray(base_profiles, float), np.asarray(detail_profiles, float)
    if base.ndim != 2 or base.shape != detail.shape or base.shape[1] != 24:
        raise ValueError("Expected two aligned 24-depth profile matrices")
    if not np.isfinite(base).all() or not np.isfinite(detail).all():
        raise ValueError("Profiles must be finite")
    coefficients = haar_forward(detail)
    coefficients[:, :3] = haar_forward(base)[:, :3]
    return haar_inverse(coefficients)


def _matrix(wide, values):
    frame = wide[KEYS].copy()
    frame["value"] = np.asarray(values, float)
    matrix = frame.pivot(index=PROFILE_KEYS, columns="depth_index", values="value").sort_index()
    if matrix.shape[1] != 24 or not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("Complete 24-depth frozen profiles required")
    return matrix


def _aligned(wide, matrix, values):
    frame = pd.DataFrame(values, index=matrix.index, columns=matrix.columns)
    frame = frame.stack().rename("value").reset_index()
    return wide[KEYS].merge(frame, on=KEYS, how="left", validate="one_to_one").value.to_numpy(float)


def run(output_root=None):
    output = Path(output_root or RUN / "multiscale_combinations")
    candidate_ids = [f"v7_hcount_{base}_{detail}_a{int(alpha*100):03d}_gp"
                     for base in BASES for detail in DETAILS for alpha in ALPHAS]
    candidate_ids += [f"v7_hcoeff_simplex_{detail}_gp" for detail in DETAILS]
    metadata = {
        "scope": "exploratory_validation_only", "family": "multiscale_combinations",
        "parameters_selected_before_evaluation": True, "candidates": candidate_ids,
        "base_models": BASES, "detail_models": DETAILS, "count_mix_alphas": ALPHAS,
        "count_rule": "floor((1-alpha)*base_low_count + alpha*detail_low_count + 0.5)",
        "rank_rule": "base point prediction ascending, depth ascending on ties",
        "sign_flips": "selected nonnegative values -> -0.05; deselected negative values -> +0.05; preserve others",
        "haar_bands": "orthonormal 3-level Haar: first 3 coefficients from simplex, remaining 21 from Haar member",
        "probability_source": "unchanged GP native p_low_1, independent auxiliary event-risk head",
        "probability_point_consistency": "not a CDF for the combined point prediction; no joint distribution claim",
        "information_boundary": "only frozen predictions and keys are read; no query truth or calibrated OOF risk inputs",
        "limitations": ["Transferred predicted count can be wrong; count agreement is not truth agreement.",
                        "Threshold sign edits can increase continuous error or false positives.",
                        "Independent GP risk probabilities do not describe uncertainty of these edited point predictions."],
    }
    write_json(output / "method_protocol.json", metadata)
    wide = load_frozen()
    gp_probability = load_native("gp")
    bases = {
        "simplex": _matrix(wide, read_candidate("local_geometry", BASES["simplex"], wide=wide)),
        "fpca": _matrix(wide, wide[BASES["fpca"]]),
    }
    details = {key: _matrix(wide, read_candidate("multiscale", model, wide=wide))
               for key, model in DETAILS.items()}
    diagnostics, paths = [], {}
    for base_key, base in bases.items():
        for detail_key, detail in details.items():
            if not base.index.equals(detail.index) or not base.columns.equals(detail.columns):
                raise ValueError("Member profiles are not aligned")
            for alpha in ALPHAS:
                model_id = f"v7_hcount_{base_key}_{detail_key}_a{int(alpha*100):03d}_gp"
                values, count = transfer_low_count(base.to_numpy(), detail.to_numpy(), alpha)
                actual = (values < 0).sum(axis=1)
                if not np.array_equal(count, actual):
                    raise AssertionError("Count correction failed")
                row = base.index.to_frame(index=False)
                row["model_id"] = model_id
                row["base_count"], row["detail_count"] = (base.to_numpy()<0).sum(axis=1), (detail.to_numpy()<0).sum(axis=1)
                row["target_count"], row["actual_count"] = count, actual
                row["changed_cells"] = ((base.to_numpy()<0) != (values<0)).sum(axis=1)
                diagnostics.append(row)
                paths[model_id] = str(emit_candidate(model_id, _aligned(wide, base, values), wide=wide,
                    probability=gp_probability, output_root=output,
                    metadata={"family": "multiscale_combinations", "base": BASES[base_key],
                              "detail_member": DETAILS[detail_key], "count_alpha": alpha,
                              "probability_source": metadata["probability_source"],
                              "probability_point_consistency": metadata["probability_point_consistency"]}))
                print(model_id, flush=True)
    base = bases["simplex"]
    for detail_key, detail in details.items():
        model_id = f"v7_hcoeff_simplex_{detail_key}_gp"
        values = combine_haar_bands(base.to_numpy(), detail.to_numpy())
        paths[model_id] = str(emit_candidate(model_id, _aligned(wide, base, values), wide=wide,
            probability=gp_probability, output_root=output,
            metadata={"family": "multiscale_combinations", "base": BASES["simplex"],
                      "detail_member": DETAILS[detail_key], "haar_coarse_count": 3,
                      "probability_source": metadata["probability_source"],
                      "probability_point_consistency": metadata["probability_point_consistency"]}))
        print(model_id, flush=True)
    pd.concat(diagnostics, ignore_index=True).to_csv(output / "count_diagnostics.csv", index=False)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.output)
