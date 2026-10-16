"""Finite target-only tree alternatives; fitting sees revealed near groups only."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from .adaptive_v2_models import _cv_metrics, _folds, _length, _matern, _weighted_median
from .models.base import validate_observed, validate_query_site
from .models.distribution import point_prediction_frame
from .types import QUERY_COLUMNS


@dataclass(frozen=True)
class TreeCandidate:
    name: str
    family: str
    residual: bool = False
    rotate: bool = False
    count_mode: str = 'native'


CANDIDATES = (
    TreeCandidate('target_et_raw', 'extra'),
    TreeCandidate('target_et_residual', 'extra', True, True),
    TreeCandidate('target_rf_raw', 'forest'),
    TreeCandidate('target_rf_residual', 'forest', True),
    TreeCandidate('target_xgb_raw', 'xgb_absolute'),
    TreeCandidate('target_xgb_residual', 'xgb_square', True),
)
MEDIAN_COUNT_CANDIDATES = tuple(TreeCandidate('target_et_median_'+mode, 'median_extra', True, True, mode)
    for mode in ('native', 'half', 'full'))
TARGET_TREE_MODEL_IDS = ('target_tree_cv', 'target_tree_log_cv', 'target_et_median_count_cv', 'target_tree_augmented_cv') + tuple(
    c.name for c in CANDIDATES) + tuple(c.name for c in MEDIAN_COUNT_CANDIDATES)


def _estimator(candidate):
    common = dict(n_estimators=128, min_samples_leaf=2, random_state=6001, n_jobs=2)
    if candidate.family in ('extra', 'median_extra'):
        return ExtraTreesRegressor(**common, max_features=1., criterion=(
            'absolute_error' if candidate.family == 'median_extra' else 'squared_error'))
    if candidate.family == 'forest':
        return RandomForestRegressor(**common, criterion='absolute_error', max_features=1.)
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=200, max_depth=3, learning_rate=.05,
                        min_child_weight=2, reg_lambda=3., random_state=6001,
                        n_jobs=2, tree_method='hist', objective=(
                            'reg:absoluteerror' if candidate.family == 'xgb_absolute' else 'reg:squarederror'))


def _feature_coordinates(xy, depths, center, scale, rotation):
    coordinates = ((xy-center)/scale) @ rotation
    # Depth values are fixed indices, so the transform cannot inspect held-out labels.
    depth_scale = max(float(np.ptp(depths)), 1.)
    return np.column_stack((np.repeat(coordinates, len(depths), axis=0),
                            np.tile((depths-np.min(depths))/depth_scale, len(xy))))


def _fit(curves, xy, depths, candidate):
    center, scale = xy.mean(axis=0), np.maximum(xy.std(axis=0), 1.)
    rotation = np.eye(2)
    if candidate.rotate and len(xy) > 1:
        _, _, vt = np.linalg.svd((xy-center)/scale, full_matrices=False)
        rotation = vt.T
    trend = np.median(curves, axis=0) if candidate.residual else np.zeros(curves.shape[1])
    estimator = _estimator(candidate)
    estimator.fit(_feature_coordinates(xy, depths, center, scale, rotation), (curves-trend).ravel())
    return dict(estimator=estimator, center=center, scale=scale, rotation=rotation,
                trend=trend, depths=depths, candidate=candidate, curves=curves,
                xy=xy, length=_length(xy))


def _predict(fitted, xy):
    features = _feature_coordinates(xy, fitted['depths'], fitted['center'], fitted['scale'], fitted['rotation'])
    if fitted['candidate'].family == 'median_extra':
        prediction = np.median(np.stack([tree.predict(features) for tree in fitted['estimator'].estimators_]), axis=0)
    else:
        prediction = fitted['estimator'].predict(features)
    answer = prediction.reshape(len(xy), -1) + fitted['trend']
    mode = fitted['candidate'].count_mode
    if mode != 'native':
        weights = np.maximum(_matern(xy, fitted['xy'], fitted['length']), 1e-12)
        weights /= weights.sum(axis=1, keepdims=True)
        observed_counts = (fitted['curves'] < 0).sum(axis=1)
        counts = np.array([int(_weighted_median(observed_counts[:, None], w)[0]) for w in weights])
        if mode == 'half':
            counts = np.floor(.5*counts + .5*(answer < 0).sum(axis=1)+.5).astype(int)
        # The curve's own depth ranking chooses low cells; the count is learned
        # only from complete revealed profiles. Every downstream score uses this
        # one adjusted curve, including amplitude, H, recall and false positives.
        order = np.argsort(answer, axis=1, kind='stable')
        low = np.zeros_like(answer, dtype=bool)
        np.put_along_axis(low, order, np.arange(answer.shape[1])[None, :] < counts[:, None], axis=1)
        answer = np.where(low & (answer >= 0), -.025, np.where(~low & (answer < 0), .025, answer))
    return answer


def _joint_loss(metrics):
    # The same pre-existing limits determine the CV ranking. Audit labels never
    # participate. Deficits receive continuous penalties rather than hard rejection.
    loss = metrics['log_mae']/.626458169250689 + metrics['thickness_mae_m']/.88125
    if np.isfinite(metrics['recall']):
        loss += max(0., (.5-metrics['recall'])/.5)
    if np.isfinite(metrics['fpr']):
        loss += max(0., (metrics['fpr']-.1)/.1)
    return float(loss)


class TargetTreePredictor:
    def __init__(self, model_id='target_tree_cv', fold_id=1, **kwargs):
        if model_id not in TARGET_TREE_MODEL_IDS:
            raise ValueError(f'Unknown target tree model: {model_id}')
        self.model_id, self.fold_id = model_id, int(fold_id)
        self.observed = None

    def initialize(self):
        return self

    def condition(self, observed):
        self.observed = validate_observed(observed).reset_index(drop=True)
        matrix = self.observed.pivot(index='location_id', columns='depth_index', values='log_qc').sort_index()
        if matrix.isna().any().any():
            raise ValueError('Target trees require complete revealed depth profiles')
        if self.observed.groupby('location_id')[['x_m', 'y_m']].nunique().ne(1).any().any():
            raise ValueError('A location must have one coordinate pair')
        locations = self.observed.drop_duplicates('location_id').set_index('location_id').loc[matrix.index]
        self.locations, self.depths = matrix.index.to_numpy(), matrix.columns.to_numpy(int)
        self.xy, self.curves = locations[['x_m', 'y_m']].to_numpy(float), matrix.to_numpy(float)
        groups = locations.near_group_id.to_numpy() if 'near_group_id' in locations else self.locations
        folds = _folds(groups)
        if self.model_id in ('target_tree_cv', 'target_tree_log_cv'):
            choices = CANDIDATES
        elif self.model_id == 'target_tree_augmented_cv':
            choices = CANDIDATES+MEDIAN_COUNT_CANDIDATES
        elif self.model_id == 'target_et_median_count_cv':
            choices = MEDIAN_COUNT_CANDIDATES
        else:
            choices = tuple(c for c in CANDIDATES+MEDIAN_COUNT_CANDIDATES if c.name == self.model_id)
        records = []
        for candidate in choices:
            out = np.full_like(self.curves, np.nan)
            for mask in folds:
                fitted = _fit(self.curves[~mask], self.xy[~mask], self.depths, candidate)
                out[mask] = _predict(fitted, self.xy[mask])
            metrics = _cv_metrics(out, self.curves) if folds else dict(
                log_mae=np.nan, thickness_mae_m=np.nan, recall=np.nan, fpr=np.nan)
            records.append((candidate, metrics, out))
        if folds:
            loss = (lambda item: item[1]['log_mae']) if self.model_id == 'target_tree_log_cv' else (
                lambda item: _joint_loss(item[1]))
            selected = min(records, key=loss)
        else:
            selected = records[0]
        self.selected_candidate, self.cv_metrics, self.crossfit_predictions = selected
        self.cv_candidates = [{**asdict(c), **m, 'selection_loss': _joint_loss(m)} for c, m, _ in records]
        self.candidate_cv_predictions = {c.name: out for c, _, out in records}
        self.fitted = _fit(self.curves, self.xy, self.depths, self.selected_candidate)
        self.context_manifest = dict(model_id=self.model_id, fold_id=self.fold_id,
            observed_location_ids=self.locations.tolist(), candidate=asdict(self.selected_candidate),
            cv=self.cv_metrics, cv_candidates=self.cv_candidates,
            selection='revealed_near_group_5fold_cross_validation',
            source_weights_used=False, audit_labels_used=False)
        return self

    def predict(self, query):
        query = query[[c for c in QUERY_COLUMNS if c in query]].copy().reset_index(drop=True)
        validate_query_site(query, self.observed)
        if query.empty:
            return point_prediction_frame(query, [])
        if not query.depth_index.isin(self.depths).all():
            raise ValueError('Query depths must be covered by revealed profiles')
        geometry = query.drop_duplicates('location_id').set_index('location_id')
        values = _predict(self.fitted, geometry[['x_m', 'y_m']].to_numpy(float))
        loc = geometry.index.get_indexer(query.location_id)
        depth = pd.Index(self.depths).get_indexer(query.depth_index)
        result = values[loc, depth]
        known = pd.MultiIndex.from_frame(self.observed[['location_id', 'depth_index']])
        positions = known.get_indexer(pd.MultiIndex.from_frame(query[['location_id', 'depth_index']]))
        revealed = positions >= 0
        result[revealed] = self.observed.iloc[positions[revealed]].log_qc.to_numpy(float)
        frame = point_prediction_frame(query, result)
        frame['is_revealed'] = revealed
        return frame


def make_target_tree_predictor(model_id='target_tree_cv', fold_id=1, **kwargs):
    return TargetTreePredictor(model_id, fold_id, **kwargs)
