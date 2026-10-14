"""Retained inference component, consolidated from cpt_budget.v2.residual (v0.8-engineering)."""
from __future__ import annotations
import json
from pathlib import Path
from xgboost import XGBRegressor
from cpt_budget.models.base import validate_observed, validate_query_site
from cpt_budget.models.distribution import point_prediction_frame
from cpt_budget.models.idw import IDWPredictor
from cpt_budget.types import PredictionBundle
from cpt_budget.components.spatial_features import FEATURE_COLUMNS_V2, build_features_v2
ALPHA_CANDIDATES = (0.0, 0.25, 0.5, 1.0)

def _baseline(features, observed, query, config):
    settings = config.get('idw', {})
    if int(settings.get('nearest_neighbors', 3)) == 3 and float(settings.get('power', 2.0)) == 2.0:
        return features.idw_log_qc.to_numpy(float)
    predictor = IDWPredictor(config)
    predictor.condition(observed)
    return predictor.predict(query).frame.pred_log_qc.to_numpy(float)

class ResidualPredictor:
    model_id = 'idw_residual'

    def __init__(self, config: dict | None=None, artifact: Path | None=None):
        self.config = config or {}
        if artifact is None:
            raise ValueError('ResidualPredictor requires a source-trained fold artifact.')
        artifact = Path(artifact)
        self.metadata = json.loads(artifact.read_text(encoding='utf-8'))
        if self.metadata.get('feature_columns') != FEATURE_COLUMNS_V2:
            raise ValueError('Residual artifact has incompatible feature names/order.')
        self.alpha = float(self.metadata['alpha'])
        if self.alpha not in ALPHA_CANDIDATES:
            raise ValueError('Residual alpha is not in the registered candidate set.')
        stored = self.metadata.get('idw_settings', {})
        current = self.config.get('idw', {})
        for name, default in (('nearest_neighbors', 3), ('power', 2.0)):
            if stored.get(name, default) != current.get(name, default):
                raise ValueError('Residual artifact and inference IDW settings must match.')
        self.regressor = None
        if self.alpha:
            self.regressor = XGBRegressor(n_jobs=int(self.config.get('v2', {}).get('residual', {}).get('n_jobs', 4)), device='cpu')
            self.regressor.load_model(artifact.parent / self.metadata['model_filename'])
            if self.regressor.get_booster().feature_names != FEATURE_COLUMNS_V2:
                raise ValueError('Residual trees have incompatible feature names/order.')
        self.idw = IDWPredictor(self.config)
        self.observed = None

    def condition(self, observed: pd.DataFrame) -> None:
        self.observed = validate_observed(observed)
        self.idw.condition(observed)

    def predict(self, query_grid: pd.DataFrame) -> PredictionBundle:
        validate_query_site(query_grid, self.observed)
        if self.alpha == 0 or query_grid.empty:
            frame = self.idw.predict(query_grid).frame
        else:
            features = build_features_v2(self.observed, query_grid)
            baseline = _baseline(features, self.observed, query_grid, self.config)
            frame = point_prediction_frame(query_grid, baseline + self.alpha * self.regressor.predict(features))
        return PredictionBundle(frame, self.model_id)
