"""Round-two fixed structural blends with a separate native GP risk head."""
from __future__ import annotations

import numpy as np

from ..config import write_json
from .common import BEST, RUN, emit_candidate, load_frozen, load_native, read_candidate


MEMBERS = {
    "cokfull": ("cokriging", "v7_cokriging_full"),
    "coklmc": ("cokriging", "v7_cokriging_lmc"),
    "biharmonic": ("graph_completion", "v7_graph_biharmonic"),
}
SETTINGS = {
    "round": 2,
    "scope": "Exploratory combinations designed after round-one results; not preregistered confirmation",
    "base_models": {"simplex": "v7_simplex_linear", "residual": BEST},
    "structural_models": {key: value[1] for key, value in MEMBERS.items()},
    "point_rule": "0.75 base log-qc + 0.25 structural-member log-qc",
    "weight_search": False,
    "risk_head": "unchanged frozen GP native p_low_1; no cokriging probabilities or borrowed intervals",
    "truth_boundary": "Only frozen model outputs are passed into the combiner",
    "reason": "Test thickness-oriented covariance/graph structure while retaining stronger continuous predictions",
    "limitations": ["Point and event heads are separate; do not imply a coherent joint predictive distribution",
                    "Cokriging native intervals were undercovered in round one and are not copied"],
}


def combine_points(bases, members):
    if set(bases) != {"simplex", "residual"} or set(members) != set(MEMBERS):
        raise ValueError("Exactly two registered bases and three structural members are required")
    checked = {}
    shape = None
    for key, value in {**bases, **members}.items():
        array = np.asarray(value, dtype=float)
        if array.ndim != 1 or not np.isfinite(array).all():
            raise ValueError("Point predictions must be finite one-dimensional arrays")
        shape = array.shape if shape is None else shape
        if array.shape != shape:
            raise ValueError("Point members must align to the same frozen rows")
        checked[key] = array
    return {f"v7_{base}_{member}_b075_gp": 0.75 * checked[base] + 0.25 * checked[member]
            for base in bases for member in members}


def main():
    output = RUN / "covariance_combinations"
    wide = load_frozen()
    bases = {"simplex": read_candidate("local_geometry", "v7_simplex_linear", wide=wide),
             "residual": wide[BEST].to_numpy(float)}
    members = {key: read_candidate(family, model, wide=wide)
               for key, (family, model) in MEMBERS.items()}
    probability = load_native("gp")
    candidates = combine_points(bases, members)
    write_json(output / "method_protocol.json", {**SETTINGS, "candidates": list(candidates)})
    for model_id, values in candidates.items():
        emit_candidate(model_id, values, wide=wide, output_root=output,
                       probability=probability, metadata=SETTINGS)
    write_json(output / "run_summary.json", {"candidate_count": len(candidates),
               "states_per_candidate": 60, "rows_per_candidate": len(wide),
               "risk_head": "frozen native gp p_low_1", "run_id": "v7-discovery-20261014",
               "round": 2, "runtime_errors": []})
    print(f"Completed {len(candidates)} covariance/graph combinations; 60 states / {len(wide)} rows each")


if __name__ == "__main__":
    main()
