"""Exact spatial residual GP with an immutable source-training depth trend."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from ..features import depth_values
from ..types import PredictionBundle
from .base import validate_observed, validate_query_site
from .distribution import gaussian_prediction_frame


def fit_gp_trend(dataset, train_sites) -> pd.Series:
    """Average locations within each train site, then give each site equal weight."""
    train_sites = list(dict.fromkeys(train_sites))
    if not train_sites:
        raise ValueError("A formal GP trend requires source training sites.")
    train = dataset.grid.loc[dataset.grid["site_id"].isin(train_sites)]
    absent = set(train_sites).difference(train["site_id"].unique())
    if absent:
        raise ValueError(f"No prepared training observations for {sorted(absent)}")
    by_site = train.groupby(["site_id", "depth_index"])["log_qc"].mean()
    trend = by_site.groupby("depth_index").mean().sort_index().rename("train_site_equal_log_qc")
    trend.attrs["train_sites"] = train_sites
    trend.attrs["weighting"] = "site_equal_after_within_site_location_mean"
    return trend


class GPPredictor:
    model_id = "gp"

    def __init__(self, config: dict | None = None, trend=None, artifact=None):
        self.config = config or {}
        self.settings = self.config.get("gp", {})
        if trend is None and self.config.get("run", {}).get("mode") != "pilot":
            raise ValueError("Formal GP requires a source-training trend; zero trend is pilot-only.")
        self.trend = None if trend is None else pd.Series(trend, dtype=float).copy()
        if self.trend is not None and (not self.trend.index.is_unique or not np.isfinite(self.trend).all()):
            raise ValueError("Trend needs one finite value per depth index.")
        self.observed = None
        self.regressor = None
        self.origin = None

    def _trend_at(self, frame: pd.DataFrame) -> np.ndarray:
        if self.trend is None:
            return np.zeros(len(frame))
        values = frame["depth_index"].map(self.trend).to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Source trend does not cover every requested depth index.")
        return values

    def _coordinates(self, frame: pd.DataFrame) -> np.ndarray:
        xy = frame[["x_m", "y_m"]].to_numpy(dtype=float) - self.origin
        return np.column_stack((xy, depth_values(frame)))

    def condition(self, observed: pd.DataFrame) -> None:
        self.observed = validate_observed(observed)
        self.origin = self.observed[["x_m", "y_m"]].mean().to_numpy()
        length_scale = self.settings.get("length_scale_m", [20.0, 20.0, 1.0])
        if len(length_scale) != 3:
            raise ValueError("GP requires independent x, y, z length scales in metres.")
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=length_scale,
            length_scale_bounds=self.settings.get("length_scale_bounds_m", [(0.1, 10000.0), (0.1, 10000.0), (0.05, 100.0)]),
            nu=1.5,
        ) + WhiteKernel(float(self.settings.get("noise_level", 0.05)), (1e-6, 10.0))
        optimize = bool(self.settings.get("optimize", True))
        self.regressor = GaussianProcessRegressor(
            kernel=kernel, alpha=float(self.settings.get("jitter", 1e-8)), normalize_y=False,
            optimizer="fmin_l_bfgs_b" if optimize else None,
            n_restarts_optimizer=min(2, max(0, int(self.settings.get("n_restarts_optimizer", 0)))) if optimize else 0,
            random_state=int(self.config.get("split", {}).get("seed", 20261013)),
        )
        self.regressor.fit(self._coordinates(self.observed), self.observed["log_qc"].to_numpy() - self._trend_at(self.observed))

    def predict(self, query_grid: pd.DataFrame) -> PredictionBundle:
        validate_query_site(query_grid, self.observed)
        if query_grid.empty:
            return PredictionBundle(gaussian_prediction_frame(query_grid, [], []), self.model_id)
        mean, std = self.regressor.predict(self._coordinates(query_grid), return_std=True)
        # WhiteKernel stays in the predictive covariance: intervals concern observations.
        frame = gaussian_prediction_frame(query_grid, mean + self._trend_at(query_grid), std)
        return PredictionBundle(frame, self.model_id)
