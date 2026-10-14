"""Frozen complementary mixtures with separate GP risk/event heads.

Round four follows the third exploratory round. Three distinct point
constructions are frozen without sweeping weights; each receives either the
existing native GP event head or the same previously frozen structured decoder.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

from cpt_budget.config import write_json
from cpt_budget.diagnostics.common import iter_validation_states, STATE_KEYS
from .common import RUN, REFERENCE, BEST, load_frozen, emit_candidate
from .decoupled import align_column
from .indicator import IndicatorLayerPredictor, structured_event_profile


POINT_NAMES = ("adaptive_graph", "fpca_graph", "median_four")
MODELS = tuple(f"v6_complement_{name}_{head}" for name in POINT_NAMES for head in ("gp", "event"))
SETTINGS = {
    "round": 4,
    "selection_scope": "exploratory follow-up after rounds 1-3; no independent confirmation",
    "point_heads": {
        "adaptive_graph": "0.5 support_residual_stack_structural + 0.5 v6_graph_residual_low_union",
        "fpca_graph": "0.5 v6_fpca_residual_w050 + 0.5 v6_graph_residual_low_union",
        "median_four": "cellwise median(BEST, functional_fpca_krr, graph_huber, registered_profile_shape)",
    },
    "mixture_weights": "fixed 0.5; no alpha sweep",
    "even_median": "arithmetic mean of the two middle ordered values",
    "risk_head": "existing GP native p_low_1 only; never OOF-calibrated risk_probability",
    "event_decoder": {"tau": 0.5, "potts": 0.25, "cardinality": 0.25, "margin": 0.05},
    "count_prior": "3NN IDW2 weighted median currently revealed low-resistance cell count",
    "probability_semantics": {
        "gp": "separate GP event head; not the CDF of the combined point estimate",
        "event": "exact marginal of the finite event Gibbs model; not a calibrated geological probability",
    },
    "thickness_semantics": "H_point is computed from the actual final amplitude threshold, not a substituted count",
}


def complementary_points(best, fpca, graph, shape, adaptive_residual, fpca_residual, graph_union):
    arrays = [np.asarray(x, float) for x in (best, fpca, graph, shape, adaptive_residual, fpca_residual, graph_union)]
    if any(a.ndim != 1 or a.shape != arrays[0].shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError("Complementary point inputs must be aligned finite vectors")
    best, fpca, graph, shape, adaptive_residual, fpca_residual, graph_union = arrays
    return {"adaptive_graph": 0.5 * adaptive_residual + 0.5 * graph_union,
            "fpca_graph": 0.5 * fpca_residual + 0.5 * graph_union,
            "median_four": np.median(np.column_stack([best, fpca, graph, shape]), axis=1)}


def decode_state(observed, query, point_heads, gp_probability):
    """Condition the count prior only on support and decode complete query profiles."""
    query = query[["location_id", "depth_index", "x_m", "y_m"]].reset_index(drop=True)
    gp_probability = np.asarray(gp_probability, float)
    if gp_probability.shape != (len(query),):
        raise ValueError("One aligned GP probability per query cell required")
    arrays = {name: np.asarray(value, float) for name, value in point_heads.items()}
    if any(value.shape != (len(query),) for value in arrays.values()):
        raise ValueError("One aligned point prediction per query cell required")
    event = IndicatorLayerPredictor(mode="cardinality")
    event.condition(observed)
    values = {name: np.full(len(query), np.nan) for name in arrays}
    marginals = {name: np.full(len(query), np.nan) for name in arrays}
    records = []
    for location, profile in query.groupby("location_id", sort=False):
        profile = profile.sort_values("depth_index")
        if not np.array_equal(profile.depth_index.to_numpy(), event.depth_ids):
            raise ValueError("Decoder requires the full observed depth grid per query profile")
        xy = profile[["x_m", "y_m"]].drop_duplicates()
        if len(xy) != 1:
            raise ValueError("Query profile requires one coordinate")
        _, _, target_count = event._spatial(xy.iloc[0].to_numpy(float))
        positions = profile.index.to_numpy()
        for name, base in arrays.items():
            predicted, probability, labels = structured_event_profile(
                base[positions], gp_probability[positions], target_count, **SETTINGS["event_decoder"])
            values[name][positions], marginals[name][positions] = predicted, probability
            records.append({"base": name, "location_id": location, "support_count_prior": target_count,
                            "predicted_count": int(labels.sum()), "expected_count": float(probability.sum()),
                            "changed_cells": int(np.sum((base[positions] < 0) != labels))})
    return values, marginals, records


def run(output_root=None):
    output = Path(output_root or RUN / "complementary")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "protocol.json", {"models": MODELS, **SETTINGS})
    started = time.perf_counter()
    wide = load_frozen()
    gp = align_column(wide, REFERENCE / "predictions_oof.parquet", "p_low_1", "gp")
    cached = lambda folder, name: align_column(wide, RUN / folder / name / "predictions.parquet", "pred_log_qc")
    point_heads = complementary_points(wide[BEST], wide.functional_fpca_krr, wide.graph_huber,
        wide.registered_profile_shape,
        cached("adaptive", "support_residual_stack_structural"),
        cached("functional", "v6_fpca_residual_w050"),
        cached("graph", "v6_graph_residual_low_union"))
    for name, value in point_heads.items():
        emit_candidate(f"v6_complement_{name}_gp", value, wide=wide, probability=gp,
                       output_root=output, metadata=SETTINGS)
    decoded = {name: np.full(len(wide), np.nan) for name in POINT_NAMES}
    probabilities = {name: np.full(len(wide), np.nan) for name in POINT_NAMES}
    diagnostics, state_count = [], 0
    for state in iter_validation_states():
        state_started = time.perf_counter()
        mask = np.ones(len(wide), dtype=bool)
        for key in STATE_KEYS:
            mask &= wide[key].eq(state.metadata[key]).to_numpy()
        positions = np.flatnonzero(mask)
        query = wide.loc[mask, ["location_id", "depth_index", "x_m", "y_m"]]
        values, marginal, records = decode_state(state.observed, query,
            {name: value[positions] for name, value in point_heads.items()}, gp[positions])
        for name in POINT_NAMES:
            decoded[name][positions], probabilities[name][positions] = values[name], marginal[name]
        diagnostics.extend({**state.metadata, **row} for row in records)
        write_json(output / "contexts" / f"{state.key}.json", {
            "observed_location_ids": state.observed.location_id.drop_duplicates().tolist(),
            "observed_set_id": state.metadata["observed_set_id"],
            "query_information": "geometry plus existing legal point and native GP probability heads",
            "elapsed_seconds": time.perf_counter() - state_started})
        state_count += 1
        print(f"complementary: {state.key}", flush=True)
    for name in POINT_NAMES:
        emit_candidate(f"v6_complement_{name}_event", decoded[name], wide=wide,
                       probability=probabilities[name], output_root=output, metadata=SETTINGS)
    pd.DataFrame(diagnostics).to_csv(output / "decoding_diagnostics.csv", index=False)
    write_json(output / "completion.json", {"models": MODELS, "states": state_count,
               "rows_per_model": len(wide), "elapsed_seconds": time.perf_counter() - started,
               "query_truth_used": False, "oof_calibrated_risk_used": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    run(parser.parse_args().output)
