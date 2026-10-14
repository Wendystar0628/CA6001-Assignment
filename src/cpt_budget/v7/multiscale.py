"""Fixed vertical multiscale representations with scale-specific spatial rules.

Haar and DCT are complete invertible representations, not a source-fitted FPCA
subspace. Coarse profiles interpolate smoothly; thin-layer detail is transferred
locally instead of being averaged with the same weights at every depth scale.
Only revealed complete CPT profiles are used by the candidate generators.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.fft import dct, idct

from ..config import write_json
from ..models.base import validate_observed, validate_query_site
from ..models.distribution import point_prediction_frame
from ..types import PredictionBundle
from ..v5.functional import _location_geometry, _profile_matrix


VARIANTS = (
    "haar_krr_nearest", "haar_krr_median", "haar_multiband",
    "dct_krr_detail", "dct_krr_nearest",
)


def haar_forward(values, levels=3):
    """Orthonormal Haar: [coarsest approximation, coarse ... fine details]."""
    current = np.asarray(values, dtype=float)
    if current.shape[-1] % (2 ** levels) or levels < 1:
        raise ValueError("Profile length must be divisible by 2**levels")
    details = []
    for _ in range(levels):
        left, right = current[..., ::2], current[..., 1::2]
        details.append((left - right) / np.sqrt(2.0))
        current = (left + right) / np.sqrt(2.0)
    return np.concatenate([current, *reversed(details)], axis=-1)


def haar_inverse(coefficients, levels=3):
    coefficients = np.asarray(coefficients, dtype=float)
    n = coefficients.shape[-1]
    if n % (2 ** levels) or levels < 1:
        raise ValueError("Profile length must be divisible by 2**levels")
    size = n // (2 ** levels)
    current, offset = coefficients[..., :size], size
    for _ in range(levels):
        detail = coefficients[..., offset:offset + size]
        reconstructed = np.empty((*current.shape[:-1], 2 * size), dtype=float)
        reconstructed[..., ::2] = (current + detail) / np.sqrt(2.0)
        reconstructed[..., 1::2] = (current - detail) / np.sqrt(2.0)
        current, offset, size = reconstructed, offset + size, size * 2
    return current


def local_weights(distances, *, power=2.0, neighbours=None):
    distances = np.asarray(distances, dtype=float)
    keep = np.ones_like(distances, dtype=bool)
    if neighbours is not None:
        order = np.argsort(distances, axis=1, kind="stable")
        keep[:] = False
        np.put_along_axis(keep, order[:, :min(neighbours, distances.shape[1])], True, axis=1)
    weights = np.where(keep, np.maximum(distances, 1e-8) ** (-power), 0)
    zero = distances <= 1e-8
    weights = np.where(zero.any(axis=1, keepdims=True), zero.astype(float), weights)
    return weights / weights.sum(axis=1, keepdims=True)


def weighted_median(values, weights):
    """Coordinatewise weighted median, with deterministic lowest-middle tie."""
    values, weights = np.asarray(values), np.asarray(weights)
    order = np.argsort(values, axis=0, kind="stable")
    sorted_values = np.take_along_axis(values, order, axis=0)
    sorted_weights = weights[:, order]
    index = (np.cumsum(sorted_weights, axis=1) < 0.5).sum(axis=1)
    index = np.minimum(index, len(values) - 1)
    return sorted_values[index, np.arange(values.shape[1])[None, :]]


class MultiscaleProfilePredictor:
    def __init__(self, variant):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown multiscale variant: {variant}")
        self.variant = variant
        self.model_id = f"v7_{variant}"
        self.observed = None

    def condition(self, observed):
        self.observed = validate_observed(observed)
        profiles = _profile_matrix(self.observed)
        if profiles.shape[1] != 24:
            raise ValueError("These frozen candidates require the project's complete 24-depth grid")
        self.depth_indices = profiles.columns.to_numpy()
        self.support_locations = list(profiles.index.get_level_values("location_id"))
        self.xy = _location_geometry(self.observed, self.support_locations)
        self.profiles = profiles.to_numpy(dtype=float)
        self.is_haar = self.variant.startswith("haar")
        self.coefficients = (haar_forward(self.profiles) if self.is_haar else
                             dct(self.profiles, type=2, norm="ortho", axis=1))
        self.coarse_count = 3 if self.is_haar else 4
        self.coarse_mean = self.coefficients[:, :self.coarse_count].mean(axis=0)
        scores = self.coefficients[:, :self.coarse_count] - self.coarse_mean
        distance = np.linalg.norm(self.xy[:, None] - self.xy[None, :], axis=2)
        pairwise = distance[np.triu_indices(len(self.xy), k=1)]
        nonzero = pairwise[pairwise > 1e-8]
        self.length_scale = float(np.median(nonzero)) if len(nonzero) else 1.0
        kernel = np.exp(-0.5 * (distance / self.length_scale) ** 2)
        self.coarse_fit = np.linalg.solve(kernel + 0.1 * np.eye(len(self.xy)), scores)
        self.context_manifest = dict(
            model_id=self.model_id, observed_location_ids=list(map(str, self.support_locations)),
            representation="Haar 3 levels" if self.is_haar else "orthonormal DCT-II",
            coarse_count=self.coarse_count, full_count=24, krr_ridge=0.1,
            length_scale_m=self.length_scale, source_training=False,
            context_selection="all revealed profiles; no support LOO or query-label selection",
        )

    def predict(self, query_grid):
        validate_query_site(query_grid, self.observed)
        locations = query_grid.location_id.drop_duplicates().tolist()
        xy = _location_geometry(query_grid, locations)
        if not np.isfinite(xy).all():
            raise ValueError("Query geometry must be finite")
        if not np.isin(query_grid.depth_index, self.depth_indices).all():
            raise ValueError("Query depth outside conditioned grid")
        distances = np.linalg.norm(xy[:, None] - self.xy[None, :], axis=2)
        coarse = self.coarse_mean + np.exp(-0.5 * (distances / self.length_scale) ** 2) @ self.coarse_fit
        detail = self.coefficients[:, self.coarse_count:]
        if self.variant.endswith("nearest"):
            details = detail[np.argmin(distances, axis=1)]
        elif self.variant == "haar_multiband":
            # Distinct horizontal locality for coarse, middle and fine details.
            bands = []
            for start, end, power in ((3, 6, 1.5), (6, 12, 2.5), (12, 24, 4.0)):
                bands.append(local_weights(distances, power=power) @ self.coefficients[:, start:end])
            details = np.concatenate(bands, axis=1)
        else:
            details = weighted_median(detail, local_weights(distances, power=2.0, neighbours=3))
        coefficients = np.concatenate([coarse, details], axis=1)
        predicted = haar_inverse(coefficients) if self.is_haar else idct(coefficients, type=2, norm="ortho", axis=1)
        # A query at an already revealed coordinate has no interpolation uncertainty.
        zero = distances <= 1e-8
        for i in np.flatnonzero(zero.any(axis=1)):
            predicted[i] = self.profiles[zero[i]].mean(axis=0)
        location_index = {location: i for i, location in enumerate(locations)}
        depth_index = {depth: i for i, depth in enumerate(self.depth_indices)}
        values = predicted[query_grid.location_id.map(location_index).to_numpy(),
                           query_grid.depth_index.map(depth_index).to_numpy()]
        return PredictionBundle(point_prediction_frame(query_grid, values), self.model_id)


def run(output_root=None):
    from ..diagnostics.common import iter_validation_states, STATE_KEYS
    from .common import RUN, KEYS, emit_candidate, load_frozen
    output = Path(output_root or RUN / "multiscale")
    wide = load_frozen()
    write_json(output / "method_protocol.json", {
        "family": "vertical_multiscale", "scope": "exploratory_validation_only",
        "variants": VARIANTS, "source_training": False,
        "transform": {"haar_levels": 3, "haar_coarse_coefficients": 3, "dct_type": 2,
                      "dct_coarse_coefficients": 4, "full_profile_depths": 24,
                      "normalization": "orthonormal; all 24 coefficients are reconstructed"},
        "coarse_regression": {"method": "centered RBF KRR", "ridge": 0.1,
                              "scale": "median nonzero pairwise revealed distance"},
        "detail_regression": {"nearest": "one geometry-nearest whole profile's coefficients",
                              "median": "3 nearest revealed profiles, inverse-square weights, coefficient median",
                              "multiband": "Haar detail powers 1.5, 2.5, 4.0 from coarse to fine"},
        "parameters_selected_before_evaluation": True,
        "truth_boundary": "Transforms and interpolation see revealed profiles and query geometry only.",
        "limitations": ["Invertible bases preserve supplied profiles; they do not guarantee prediction of unseen thin layers.",
                        "Transferred fine layers need not exist at the query position; nearest-donor detail also transfers noise.",
                        "Haar is shift-sensitive; DCT can ring around sharp layers.",
                        "Point-only outputs have no native probability interpretation."],
        "references": ["https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html",
                       "https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.idct.html",
                       "https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html"],
    })
    parts = {variant: [] for variant in VARIANTS}
    contexts = []
    for state in iter_validation_states():
        for variant in VARIANTS:
            model = MultiscaleProfilePredictor(variant)
            model.condition(state.observed)
            frame = model.predict(state.query).frame
            for key in STATE_KEYS:
                frame[key] = state.metadata[key]
            parts[variant].append(frame[KEYS + ["pred_log_qc"]])
            contexts.append({**{key: state.metadata[key] for key in STATE_KEYS}, **model.context_manifest})
        print(f"multiscale: {state.key}", flush=True)
    paths = {}
    for variant, frames in parts.items():
        predictions = pd.concat(frames, ignore_index=True)
        aligned = wide[KEYS].merge(predictions, on=KEYS, how="left", validate="one_to_one")
        model_id = f"v7_{variant}"
        paths[model_id] = str(emit_candidate(model_id, aligned.pred_log_qc, wide=wide, output_root=output,
            metadata={"family": "vertical_multiscale", "variant": variant,
                      "frozen_parameters": "../method_protocol.json", "native_uncertainty": False}))
    write_json(output / "contexts.json", contexts)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.output)
