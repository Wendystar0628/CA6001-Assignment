"""Round-two simplex combinations with a separate, frozen GP risk head."""
from __future__ import annotations

import numpy as np

from cpt_budget.config import write_json
from cpt_budget.v7.common import BEST, RUN, emit_candidate, load_frozen, load_native, read_candidate

MEMBERS = {
    "residual": BEST,
    "fpca50": "v6_fpca_residual_w050",
    "graph_union": "v6_graph_residual_low_union",
    "structural": "support_stack_structural",
}
SETTINGS = {
    "round": 2,
    "selection_timing": "Fixed after first-round local geometry results, before combination results",
    "scope": "exploratory_validation_only",
    "simplex": "v7_simplex_linear",
    "members": MEMBERS,
    "simplex_weights": [0.5, 0.75],
    "median_members": ["v7_simplex_linear", MEMBERS["fpca50"], MEMBERS["graph_union"]],
    "point_control": "unchanged v7_simplex_linear",
    "risk_head": "frozen GP native p_low_1, never OOF risk_probability",
    "interpretation": "Separate point and risk heads; GP probability is not the CDF of the combined point predictor",
    "extra_training_or_parameter_search": "none",
}


def combine_points(simplex, members):
    """Ten deterministic point-head constructions; no labels or risk input."""
    simplex = np.asarray(simplex, float)
    if simplex.ndim != 1 or not np.isfinite(simplex).all():
        raise ValueError("Simplex must be a finite vector")
    if set(members) != set(MEMBERS):
        raise ValueError("Exactly the four frozen members are required")
    checked = {}
    for name, vector in members.items():
        value = np.asarray(vector, float)
        if value.shape != simplex.shape or not np.isfinite(value).all():
            raise ValueError("All point members must match simplex rows and be finite")
        checked[name] = value
    result = {}
    for name in MEMBERS:
        for weight in (0.5, 0.75):
            key = f"v7_simplex_{name}_s{int(weight*100):03d}_gp"
            result[key] = weight * simplex + (1. - weight) * checked[name]
    result["v7_simplex_median_fpca_graph_gp"] = np.median(
        np.stack([simplex, checked["fpca50"], checked["graph_union"]]), axis=0)
    result["v7_simplex_point_gp"] = simplex.copy()
    return result


def run():
    output = RUN / "simplex_combinations"
    # Record fixed designs before reading any newly computed combination results.
    write_json(output / "protocol.json", SETTINGS)
    wide = load_frozen()
    simplex = read_candidate("local_geometry", "v7_simplex_linear", wide=wide)
    members = {name: wide[model].to_numpy(float) for name, model in MEMBERS.items()}
    probability = load_native("gp")
    candidates = combine_points(simplex, members)
    for model_id, values in candidates.items():
        emit_candidate(model_id, values, wide=wide, output_root=output, probability=probability,
                       metadata={**SETTINGS, "model_id": model_id})
    print(f"Completed {len(candidates)} simplex combinations; each has 60 states / {len(wide)} rows.")


if __name__ == "__main__":
    run()
