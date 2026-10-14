"""Bounded profile registration followed by geometry-only spatial interpolation.

This is an experimental registration/interpolation construction inspired by DTW
and functional registration, not a reproduction of a stratigraphic classifier.
Only revealed profiles are aligned. A hidden query's shape is never available.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cpt_budget.models.base import validate_observed, validate_query_site
from cpt_budget.models.distribution import point_prediction_frame
from cpt_budget.types import PredictionBundle


FROZEN_SETTINGS = {
    "neighbors": 3,
    "distance_power": 2,
    "max_displacement_m": 0.5,
    "warp_grid_step_bins": 0.5,
    "local_thickness_ratio": [0.5, 1.5],
    "shift_penalty": 0.05,
    "slope_penalty": 0.05,
    "scale_floor_log_qc": 0.25,
    "reference": "unweighted_same_depth_profile_medoid_of_geometric_neighbors",
    "endpoints": "fixed_first_and_last_observed_depth_centers",
    "hyperparameter_selection": "none; frozen before validation",
}


def _matching_values(profiles: np.ndarray, mode: str) -> np.ndarray:
    if mode == "shape":
        return (profiles - profiles.mean(axis=1, keepdims=True)) / np.maximum(
            profiles.std(axis=1, keepdims=True), 0.25)
    if mode == "log":
        return profiles / max(float(profiles.std()), 0.25)
    raise ValueError("Matching mode must be log or shape")


def bounded_warp(reference: np.ndarray, source: np.ndarray, *, band_bins: float = 2.) -> np.ndarray:
    """Map every reference bin to source depth on a half-bin monotone lattice.

    Strict slopes in {0.5, 1, 1.5} prevent layer collapse. Both endpoint centers
    are fixed. Penalize displacement and nonidentity slope in addition to the
    squared discrepancy. This differs from unconstrained classical DTW paths.
    """
    reference, source = np.asarray(reference, float), np.asarray(source, float)
    if reference.ndim != 1 or reference.shape != source.shape or len(source) < 2:
        raise ValueError("Profiles must be equally sized one-dimensional arrays")
    if not np.isfinite(reference).all() or not np.isfinite(source).all() or band_bins <= 0:
        raise ValueError("Finite profiles and positive registration band required")
    n = len(reference)
    nodes = np.arange(2 * n - 1, dtype=float) / 2
    values = np.interp(nodes, np.arange(n), source)
    cost = np.full((n, len(nodes)), np.inf)
    back = np.full((n, len(nodes)), -1, dtype=int)
    cost[0, 0] = (reference[0] - source[0]) ** 2
    # Identity first provides deterministic, least-distorting tie breaking.
    deltas = (2, 1, 3)
    for j in range(1, n):
        eligible = np.flatnonzero(np.abs(nodes - j) <= band_bins + 1e-12)
        if j == n - 1:
            eligible = np.array([len(nodes) - 1])
        for k in eligible:
            local = (reference[j] - values[k]) ** 2 + 0.05 * ((nodes[k] - j) / band_bins) ** 2
            for step in deltas:
                previous = k - step
                if previous < 0:
                    continue
                candidate = cost[j - 1, previous] + local + 0.05 * (step / 2 - 1) ** 2
                if candidate < cost[j, k]:
                    cost[j, k], back[j, k] = candidate, previous
    if not np.isfinite(cost[-1, -1]):
        raise RuntimeError("No admissible registration path")
    mapping = np.empty(n)
    k = len(nodes) - 1
    for j in range(n - 1, -1, -1):
        mapping[j] = nodes[k]
        k = back[j, k]
    return mapping


class LayerAlignmentPredictor:
    """Three-neighbor registered barycentric profile interpolation.

    For support map phi_i(t), interpolate both amplitudes y_i(phi_i(t)) and
    physical depth maps phi_i(t) with geometric weights. Invert the resulting
    depth map before reporting at real query depths. Averaging aligned values
    at the reference depths without this inverse step would misplace layers.
    """

    def __init__(self, config: dict | None = None, *, mode: str = "log"):
        if mode not in {"log", "shape"}:
            raise ValueError("Matching mode must be log or shape")
        self.config = config or {}
        self.mode = mode
        self.model_id = f"registered_profile_{mode}"
        self.observed = None

    def condition(self, observed: pd.DataFrame) -> None:
        self.observed = validate_observed(observed)
        depths = observed[["depth_index", "z_center_m"]].drop_duplicates().sort_values("depth_index")
        if depths.depth_index.duplicated().any() or len(depths) < 2:
            raise ValueError("A common complete depth grid is required")
        self.depth_ids = depths.depth_index.to_numpy()
        self.z = depths.z_center_m.to_numpy(float)
        step = np.diff(self.z)
        if np.any(step <= 0) or not np.allclose(step, step[0]):
            raise ValueError("Registration requires a regular increasing depth grid")
        self.band_bins = 0.5 / step[0]
        locations = observed[["location_id", "x_m", "y_m"]].drop_duplicates().sort_values("location_id")
        if locations.location_id.duplicated().any():
            raise ValueError("Each support location needs one coordinate")
        self.ids = locations.location_id.to_numpy()
        self.xy = locations[["x_m", "y_m"]].to_numpy(float)
        profiles = observed.pivot(index="location_id", columns="depth_index", values="log_qc")
        self.profiles = profiles.reindex(index=self.ids, columns=self.depth_ids).to_numpy(float)
        if not np.isfinite(self.profiles).all():
            raise ValueError("Each revealed CPT must have the complete depth profile")
        self._cache = {}
        self.context_manifest = {
            "model_id": self.model_id, "observed_location_ids": self.ids.tolist(),
            "settings": FROZEN_SETTINGS, "matching_mode": self.mode,
            "query_information": "coordinates and depth only",
            "uncertainty": "point predictions only; no native probability or interval",
        }

    def _registered_neighbors(self, nearest):
        key = tuple(sorted(int(i) for i in nearest))
        if key not in self._cache:
            profiles = self.profiles[list(key)]
            matching = _matching_values(profiles, self.mode)
            pairwise = ((matching[:, None] - matching[None, :]) ** 2).mean(axis=2)
            medoid = int(np.argmin(pairwise.sum(axis=1)))
            identity = np.arange(len(self.z), dtype=float)
            maps = np.array([identity if i == medoid else bounded_warp(
                matching[medoid], row, band_bins=self.band_bins) for i, row in enumerate(matching)])
            amplitude = np.array([np.interp(mapping, identity, profile)
                                  for mapping, profile in zip(maps, profiles)])
            self._cache[key] = (maps, amplitude)
        return key, self._cache[key]

    def predict(self, query_grid: pd.DataFrame) -> PredictionBundle:
        validate_query_site(query_grid, self.observed)
        if not np.isfinite(query_grid[["x_m", "y_m"]].to_numpy(float)).all():
            raise ValueError("Finite query coordinates required")
        if not query_grid.depth_index.isin(self.depth_ids).all():
            raise ValueError("Query depth is outside the conditioned grid")
        prediction = np.empty(len(query_grid))
        displacement = np.zeros(len(query_grid))
        query_displacement = np.zeros(len(query_grid))
        registration_status = np.empty(len(query_grid), dtype=object)
        work = query_grid.reset_index(drop=True)
        identity = np.arange(len(self.z), dtype=float)
        for _, rows in work.groupby("location_id", sort=False):
            coordinates = rows[["x_m", "y_m"]].drop_duplicates()
            if len(coordinates) != 1:
                raise ValueError("Each query location needs one coordinate")
            distance = np.linalg.norm(self.xy - coordinates.to_numpy(float)[0], axis=1)
            nearest = np.argsort(distance, kind="stable")[:3]
            if distance[nearest[0]] < 1e-8:
                profile = self.profiles[nearest[0]]
                status, maximum_shift, query_shift = "colocated_exact", 0., 0.
            else:
                key, (maps, amplitude) = self._registered_neighbors(nearest)
                weights = distance[list(key)] ** -2
                weights /= weights.sum()
                query_map = weights @ maps
                aligned_value = weights @ amplitude
                profile = np.interp(identity, query_map, aligned_value)
                maximum_shift = float(np.max(np.abs(maps - identity))) * (self.z[1] - self.z[0])
                query_shift = float(np.max(np.abs(query_map - identity))) * (self.z[1] - self.z[0])
                status = "identity_registration" if maximum_shift < 1e-10 else "registered"
            indices = pd.Index(self.depth_ids).get_indexer(rows.depth_index)
            prediction[rows.index] = profile[indices]
            displacement[rows.index], query_displacement[rows.index] = maximum_shift, query_shift
            registration_status[rows.index] = status
        frame = point_prediction_frame(query_grid, prediction)
        frame["registration_max_support_shift_m"] = displacement
        frame["registration_query_shift_m"] = query_displacement
        frame["registration_status"] = registration_status
        return PredictionBundle(frame, self.model_id)


def main():
    from cpt_budget.config import PROJECT_ROOT, write_json
    from cpt_budget.diagnostics.common import diagnostic_config, run_predictor
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs/v5-exploration-20261014/layer_alignment")
    args = parser.parse_args()
    config = diagnostic_config()
    write_json(args.output / "frozen_settings.json", {
        **FROZEN_SETTINGS, "models": ["registered_profile_log", "registered_profile_shape"],
        "sources": ["https://www.francois-petitjean.com/Research/DBA.php",
                    "https://doi.org/10.1029/2024PA005082"],
        "limitation": "24 depth cells cannot recover unresolved thin layers; registration is not geological labeling.",
    })
    for mode in ("log", "shape"):
        output = args.output / f"registered_profile_{mode}"
        result = run_predictor(lambda fold, mode=mode: LayerAlignmentPredictor(config, mode=mode),
                               output, config=config)
        # The immutable shared v3 evaluator owns the truth boundary, but these
        # outputs belong to the new exploratory study. Relabel without changing
        # any predictions, metric values, split, or observed state identity.
        for name, frame in result.items():
            frame["run_id"] = "v5-exploration-20261014"
            frame["study_version"] = "0.5-exploration"
            if name == "predictions":
                frame.to_parquet(output / "predictions.parquet", index=False)
            else:
                frame.to_csv(output / "metrics.csv", index=False)
        for path in (output / "states").glob("*.parquet"):
            frame = pd.read_parquet(path)
            frame["run_id"], frame["study_version"] = "v5-exploration-20261014", "0.5-exploration"
            frame.to_parquet(path, index=False)
        frame = result["predictions"].drop_duplicates(["fold_id", "site_id", "seed", "budget_n", "location_id"])
        write_json(output / "registration_diagnostics.json", {
            "status_counts_query_profiles": frame.registration_status.value_counts().to_dict(),
            "max_support_shift_m": float(frame.registration_max_support_shift_m.max()),
            "mean_max_support_shift_m": float(frame.registration_max_support_shift_m.mean()),
            "mean_query_shift_m": float(frame.registration_query_shift_m.mean()),
            "numerical_failures": 0,
            "failure_policy": "hard error; no exception-driven fallback",
            "zero_deformation_policy": "identity warp exactly reduces to same-depth 3-neighbor IDW",
        })


if __name__ == "__main__":
    main()
