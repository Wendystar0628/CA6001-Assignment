"""Frozen residual-amplitude and registered-shape combinations.

These constructions consume point predictions only. In particular, a shape
profile is a prediction from revealed neighbours, never the hidden query's
shape. No threshold, mixture weight, or transformation is fitted on query truth.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


RESIDUAL_MODEL = "residual_ft_crps_guarded_a050"
SHAPE_MODEL = "registered_profile_shape"
MODELS = {
    "v6_shape_blend025": "blend025",
    "v6_shape_blend050": "blend050",
    "v6_shape_blend075": "blend075",
    "v6_shape_residual_amplitude": "amplitude",
    "v6_shape_residual_trend": "trend",
    "v6_shape_residual_softmin": "softmin",
}
FROZEN_SETTINGS = {
    "members": [RESIDUAL_MODEL, SHAPE_MODEL],
    "shape_blend_weights": [0.25, 0.5, 0.75],
    "amplitude_scale_ratio_clip": [0.5, 2.0],
    "amplitude_scale_numerical_floor": 1e-12,
    "trend_smoothing_bins": 5,
    "trend_smoothing_boundary": "numpy_reflect_excluding_endpoint",
    "softmin_temperature_log_qc": 0.25,
    "selection": "fixed before running new combinations; no query-truth fitting",
    "uncertainty": "point-only; inherited risk and distribution outputs discarded",
}


def _smooth(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values, ((0, 0), (2, 2)), mode="reflect")
    return np.apply_along_axis(lambda row: np.convolve(row, np.ones(5)/5, mode="valid"), 1, padded)


def combine_profiles(residual: np.ndarray, shape: np.ndarray, mode: str) -> np.ndarray:
    """Combine equal [profile, depth] arrays without labels or model scores.

    ``amplitude`` retains residual mean/scale but takes depthwise morphology
    from the registered profile. ``trend`` transfers registered high-frequency
    deviations to a smoothed residual trend. Neither identifies soil classes.
    ``softmin`` favours low values; its larger possible false-positive cost is
    evaluated explicitly, rather than treating sensitivity as a free benefit.
    """
    residual, shape = np.asarray(residual, float), np.asarray(shape, float)
    if residual.ndim != 2 or residual.shape != shape.shape or residual.shape[1] < 5:
        raise ValueError("Equal profile-by-depth arrays with at least five depths required")
    if not np.isfinite(residual).all() or not np.isfinite(shape).all():
        raise ValueError("Both member predictions must be finite")
    if mode in {"blend025", "blend050", "blend075"}:
        weight = {"blend025": .25, "blend050": .5, "blend075": .75}[mode]
        return (1-weight)*residual + weight*shape
    if mode == "amplitude":
        ratio = residual.std(axis=1, keepdims=True) / np.maximum(shape.std(axis=1, keepdims=True), 1e-12)
        ratio = np.clip(ratio, .5, 2.)
        return residual.mean(axis=1, keepdims=True) + ratio*(shape-shape.mean(axis=1, keepdims=True))
    if mode == "trend":
        return _smooth(residual) + shape - _smooth(shape)
    if mode == "softmin":
        return -.25*(logsumexp(np.stack([-residual/.25, -shape/.25]), axis=0) - np.log(2.))
    raise ValueError(f"Unknown combination mode: {mode}")


def main():
    # Shared frozen-table boundary is deliberately kept outside transformation.
    from cpt_budget.config import write_json
    from cpt_budget.v6.common import KEYS, RUN, emit_candidate, load_frozen
    wide = load_frozen()
    depths = wide.groupby(KEYS[:-1], sort=False).depth_index.agg(tuple)
    if not depths.map(lambda values: values == tuple(range(24))).all():
        raise ValueError("Every frozen query profile must contain the ordered 24-depth grid")
    output = RUN / "alignment"
    write_json(output / "frozen_protocol.json", {
        **FROZEN_SETTINGS, "models": MODELS,
        "physical_scope": "Depthwise resistance morphology, not interpreted soil classes or layer labels.",
        "transfer_scope": "Residual amplitudes and registered predicted event locations are complementary hypotheses, not observed query information.",
    })
    residual = wide[RESIDUAL_MODEL].to_numpy().reshape(-1, 24)
    shape = wide[SHAPE_MODEL].to_numpy().reshape(-1, 24)
    diagnostics = {}
    for model_id, mode in MODELS.items():
        values = combine_profiles(residual, shape, mode)
        emit_candidate(model_id, values.ravel(), wide=wide, output_root=output,
                       metadata={**FROZEN_SETTINGS, "combination_mode": mode})
        diagnostics[model_id] = {
            "query_profiles": len(values), "states": 60,
            "predicted_low_resistance_cells": int((values < 0).sum()),
            "min_prediction_log_qc": float(values.min()),
            "max_prediction_log_qc": float(values.max()),
        }
    write_json(output / "prediction_diagnostics.json", diagnostics)
    print(f"Exported {len(MODELS)} frozen shape/residual combinations, 60 states each.")


if __name__ == "__main__":
    main()
