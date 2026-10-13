"""Three-neighbour inverse-distance interpolation of revealed log resistance."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import build_features, idw_value
from ..types import PredictionBundle
from .base import validate_observed, validate_query_site
from .distribution import point_prediction_frame


class IDWPredictor:
    model_id = "idw"

    def __init__(self, config: dict | None = None, artifact=None):
        self.config = config or {}
        settings = self.config.get("idw", {})
        self.neighbors = int(settings.get("nearest_neighbors", 3))
        self.power = float(settings.get("power", 2.0))
        if self.neighbors < 1 or self.power <= 0:
            raise ValueError("IDW needs a positive neighbour count and distance power.")
        self.observed = None

    def condition(self, observed: pd.DataFrame) -> None:
        self.observed = validate_observed(observed)

    def predict(self, query_grid: pd.DataFrame) -> PredictionBundle:
        validate_query_site(query_grid, self.observed)
        if self.neighbors == 3 and self.power == 2.0:
            prediction = build_features(self.observed, query_grid)["idw_log_qc"].to_numpy()
        else:
            groups = {
                key: frame.sort_values("location_id", kind="stable")
                for key, frame in self.observed.groupby(["site_id", "depth_index"], sort=False)
            }
            prediction = []
            for row in query_grid.itertuples(index=False):
                support = groups.get((row.site_id, row.depth_index))
                if support is None:
                    raise ValueError(f"No observed resistance at depth_index={row.depth_index}")
                distance = np.linalg.norm(support[["x_m", "y_m"]].to_numpy() - [row.x_m, row.y_m], axis=1)
                nearest = np.argsort(distance, kind="stable")[:self.neighbors]
                prediction.append(idw_value(support["log_qc"].to_numpy()[nearest], distance[nearest], power=self.power))
        return PredictionBundle(point_prediction_frame(query_grid, prediction), self.model_id)
