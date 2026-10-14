"""Frozen FPCA/residual combinations, including a source-subspace transfer.

No query labels or calibrated probabilities enter these point estimators.
The subspace candidate transfers the low-rank functional part while retaining
the residual model's orthogonal detail; it is not another scalar weight sweep.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cpt_budget.config import PROJECT_ROOT, write_json

RESIDUAL = "residual_ft_crps_guarded_a050"
FUNCTIONAL = "functional_fpca_krr"
SHAPE = "registered_profile_shape"
PROFILE_KEYS = ["fold_id", "site_id", "seed", "budget_n", "location_id"]
ROW_KEYS = [*PROFILE_KEYS, "depth_index"]
MODEL_IDS = (
    "v6_fpca_residual_w025",
    "v6_fpca_residual_w050",
    "v6_fpca_residual_w075",
    "v6_fpca_residual_sign_guard",
    "v6_fpca_shape_residual_equal",
    "v6_fpca_subspace_residual_detail",
)


def protocol():
    return {
        "study_version": "0.6-combinations",
        "family": "functional_combinations",
        "parameter_freeze": "Six candidates sent to the parent before computing any new scores.",
        "source_models": [RESIDUAL, FUNCTIONAL, SHAPE],
        "candidates": {
            MODEL_IDS[0]: "0.75 R + 0.25 F",
            MODEL_IDS[1]: "0.50 R + 0.50 F",
            MODEL_IDS[2]: "0.25 R + 0.75 F",
            MODEL_IDS[3]: "0.50 R + 0.50 F clipped to the same side of log(qc)=0 as R",
            MODEL_IDS[4]: "(R + F + shape) / 3",
            MODEL_IDS[5]: "R + (F - R) P, P = source_components.T @ source_components",
        },
        "basis": "Frozen v5 fold-specific source-training FPCA basis; rank cap 8, no refit.",
        "point_guard": "Preserves R point labels and hence its thickness; does not protect calibrated risk labels.",
        "subspace_assumption": "Transfer source-supported joint profile variation from F, preserve R's orthogonal detail.",
        "subspace_limit": "A source FPCA subspace is not a physical frequency filter or proof of thin-layer preservation.",
        "labels_in_combination": False,
        "outer_test_used": False,
        "native_probabilities_or_intervals": False,
    }


def load_bases(root=None):
    root = Path(root or PROJECT_ROOT / "runs/v5-exploration-20261014/functional/bases")
    components, depths = {}, {}
    for fold_id in (0, 1, 2):
        with np.load(root / f"fold_{fold_id}" / "basis.npz") as artifact:
            components[fold_id] = artifact["components"].copy()
            depths[fold_id] = artifact["depth_indices"].copy()
    return components, depths


def combine_predictions(wide: pd.DataFrame, components_by_fold, depth_indices_by_fold):
    """Return arrays in input row order; consume only keys and frozen predictions."""
    required = [*ROW_KEYS, RESIDUAL, FUNCTIONAL, SHAPE]
    if not set(required).issubset(wide.columns):
        raise ValueError("Missing frozen prediction or profile key columns")
    if wide.duplicated(ROW_KEYS).any():
        raise ValueError("Duplicate frozen profile/depth rows")
    values = wide[[RESIDUAL, FUNCTIONAL, SHAPE]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Frozen component predictions must be finite")
    residual, functional, shape = values.T
    outputs = {MODEL_IDS[i]: (1.0 - weight) * residual + weight * functional
               for i, weight in enumerate((0.25, 0.50, 0.75))}
    mixture = outputs[MODEL_IDS[1]]
    outputs[MODEL_IDS[3]] = np.where(residual < 0.0,
                                    np.minimum(mixture, -1e-9), np.maximum(mixture, 0.0))
    outputs[MODEL_IDS[4]] = (residual + functional + shape) / 3.0
    transferred = np.empty(len(wide), dtype=float)
    keys = wide[ROW_KEYS].reset_index(drop=True)
    for fold_id, fold_rows in keys.groupby("fold_id", sort=False):
        fold_id = int(fold_id)
        if fold_id not in components_by_fold or fold_id not in depth_indices_by_fold:
            raise ValueError("Missing fold-specific source FPCA basis")
        component = np.asarray(components_by_fold[fold_id], dtype=float)
        depth_order = np.asarray(depth_indices_by_fold[fold_id])
        if (component.ndim != 2 or component.shape[1] != len(depth_order)
                or not np.isfinite(component).all()
                or len(np.unique(depth_order)) != len(depth_order)
                or not np.allclose(component @ component.T, np.eye(len(component)), atol=1e-10)):
            raise ValueError("Source FPCA basis must be finite and orthonormal on its depth grid")
        for _, profile in fold_rows.groupby(PROFILE_KEYS, sort=False):
            if len(profile) != len(depth_order) or set(profile.depth_index) != set(depth_order):
                raise ValueError("Source-subspace transfer requires every depth of each profile")
            positions = profile.set_index("depth_index").loc[depth_order].index
            # Reindex original positional row IDs by the stored source depth order.
            row_positions = pd.Series(profile.index.to_numpy(), index=profile.depth_index).loc[positions].to_numpy()
            correction = functional[row_positions] - residual[row_positions]
            transferred[row_positions] = residual[row_positions] + (correction @ component.T) @ component
    if not np.isfinite(transferred).all():
        raise ValueError("Source-subspace transfer produced nonfinite predictions")
    outputs[MODEL_IDS[5]] = transferred
    return outputs


def run(output=None):
    from .common import load_frozen, emit_candidate

    output = Path(output or PROJECT_ROOT / "runs/v6-combinations-20261014/functional")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "method_protocol.json", protocol())
    wide = load_frozen()
    components, depths = load_bases()
    candidates = combine_predictions(wide, components, depths)
    for model_id, prediction in candidates.items():
        emit_candidate(model_id, prediction, wide=wide, output_root=output,
                       metadata={"family": "functional_combinations",
                                 "formula": protocol()["candidates"][model_id]})
    write_json(output / "execution_summary.json", {
        "models": list(candidates), "prediction_rows_per_model": len(wide),
        "states_per_model": len(wide[PROFILE_KEYS[:-1]].drop_duplicates()),
        "row_alignment": "All profile transfers restored to original row order.",
        "source_basis_ranks": {str(fid): len(basis) for fid, basis in components.items()},
        "query_labels_consumed": False,
    })
    return candidates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.output)
