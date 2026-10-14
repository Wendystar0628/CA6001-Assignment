"""Keep revealed-target-adapted thickness while exchanging amplitude/risk heads.

This is a second exploratory round nominated after the first-round results.
Only candidate point predictions and native probabilities enter this combiner.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cpt_budget.config import write_json
from cpt_budget.v6.count_projection import project_frame
from .common import RUN, BEST, KEYS, load_frozen, load_native, read_candidate, emit_candidate

COUNT_MODEL = "v7_local_half_count_stack"
BASE_NAMES = ("simplex", "simplex_residual", "fpca50")
RISK_NAMES = ("localrisk", "gprisk")
MODELS = {f"v7_transfer_{base}_{risk}": (base, risk)
          for base in BASE_NAMES for risk in RISK_NAMES}
PROFILE_KEYS = KEYS[:-1]


def protocol():
    return {
        "round": 2,
        "family": "local_count_with_alternative_amplitude_and_risk_heads",
        "candidate_nomination": "Selected after inspecting v7 first-round thickness results; exploratory reuse, not independent confirmation.",
        "parameters_frozen_before_second_round_results": True,
        "count_source": COUNT_MODEL,
        "count_source_column": "pred_log_qc only; count actual negative cells per complete profile",
        "count_refit": False,
        "candidates": MODELS,
        "bases": {"simplex": "v7_simplex_linear",
                  "simplex_residual": "0.5*v7_simplex_linear + 0.5*residual_ft_crps_guarded_a050",
                  "fpca50": "v6_fpca_residual_w050"},
        "projection": "Exactly preserve source profile count; ascending base logqc, stable depth ties, flip only disagreements to +/- .05.",
        "risk_heads": {"localrisk": f"Unchanged native p_low_1 from {COUNT_MODEL}",
                       "gprisk": "Unchanged native p_low_1 from frozen GP"},
        "risk_semantics": "Independent native event head; neither a CDF nor an interval distribution for projected points.",
        "query_truth_read": False, "old_oof_risk_read": False,
        "outer_test_used": False,
    }


def frozen_profile_counts(wide, source_point):
    """Recover count solely from source predictions, rejecting incomplete profiles."""
    source_point = np.asarray(source_point, float)
    if source_point.shape != (len(wide),) or not np.isfinite(source_point).all():
        raise ValueError("One finite source point value is required for every frozen cell")
    if wide.duplicated(KEYS).any():
        raise ValueError("Frozen cells must be unique")
    frame = wide[KEYS].copy()
    frame["_source_low"] = source_point < 0.
    rows = []
    for key, block in frame.groupby(PROFILE_KEYS, sort=False):
        if len(block) != 24 or set(block.depth_index) != set(range(24)):
            raise ValueError("Count source requires complete 24-depth profiles")
        rows.append({**dict(zip(PROFILE_KEYS, key)),
                     "half_count_decision": int(block._source_low.sum())})
    return pd.DataFrame(rows)


def run(output=None):
    output = Path(output or RUN / "transfer_combinations")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "method_protocol.json", protocol())
    wide = load_frozen()
    source_points = read_candidate("local_transfer", COUNT_MODEL, wide=wide)
    counts = frozen_profile_counts(wide, source_points)
    simplex = read_candidate("local_geometry", "v7_simplex_linear", wide=wide)
    bases = {"simplex": simplex,
             "simplex_residual": .5*(simplex + wide[BEST].to_numpy(float)),
             "fpca50": wide.v6_fpca_residual_w050.to_numpy(float)}
    risks = {"localrisk": read_candidate("local_transfer", COUNT_MODEL, column="p_low_1", wide=wide),
             "gprisk": load_native("gp")}
    all_diagnostics = []
    for short, base in bases.items():
        point, diagnostic = project_frame(wide, base, counts)
        if not diagnostic.frozen_count.eq(diagnostic.actual_point_count).all():
            raise RuntimeError("Projection changed the fixed source profile thickness")
        diagnostic["base"] = short
        all_diagnostics.append(diagnostic)
        for risk in RISK_NAMES:
            model = f"v7_transfer_{short}_{risk}"
            emit_candidate(model, point, wide=wide, output_root=output, probability=risks[risk],
                           metadata={**protocol(), "base": short, "risk_head": risk})
            print(f"{model}: {len(counts)} fixed profile counts / {len(wide)} prediction rows", flush=True)
    counts.to_csv(output / "frozen_profile_counts.csv", index=False)
    diagnostics = pd.concat(all_diagnostics, ignore_index=True)
    diagnostics.to_csv(output / "projection_diagnostics.csv", index=False)
    write_json(output / "execution_summary.json", {"models": list(MODELS), "states_per_model": 60,
        "rows_per_model": len(wide), "profiles_per_model": len(counts),
        "all_source_counts_exactly_preserved": bool(diagnostics.frozen_count.eq(diagnostics.actual_point_count).all()),
        "risk_heads_unchanged": True, "source_count_refit": False,
        "truth_read_by_combiner": False, "old_oof_risk_read": False})
    return diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.output)
