"""Revealed-group-only target adaptation for the adaptive v2 experiments.

Every candidate predicts a single log(qc) curve.  Profile, thickness and low-qc
scores are consequently computed from the same curve.  No audit labels, source
training rows, or unrevealed candidate curves are accepted by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .models.base import validate_observed, validate_query_site
from .models.distribution import point_prediction_frame
from .types import QUERY_COLUMNS


MAIN_V2_MODEL_IDS = ('target_trend_gp', 'target_fpca_gp', 'target_profile_cv', 'target_joint_cv')
CONTROL_V2_MODEL_IDS = ('target_scalar_median', 'target_depth_mean', 'target_depth_median')
V2_MODEL_IDS = MAIN_V2_MODEL_IDS + CONTROL_V2_MODEL_IDS


@dataclass(frozen=True)
class ProfileCandidate:
    kind: str
    length_ratio: float = 1.
    ridge: float = .1
    count_mode: str = 'native'


def _length(xy):
    if len(xy) < 2:
        return 1.
    d = cdist(xy, xy)
    d[d < 1e-9] = np.inf
    nearest = d.min(axis=1)
    return max(float(np.median(nearest[np.isfinite(nearest)])), 1.) if np.isfinite(nearest).any() else 1.


def _matern(x, y, length):
    r = np.sqrt(3.) * cdist(x, y) / length
    return (1. + r) * np.exp(-r)


def _weighted_median(values, weights):
    order = np.argsort(values, axis=0, kind='stable')
    sorted_values = np.take_along_axis(values, order, axis=0)
    w = np.take_along_axis(np.broadcast_to(weights[:, None], values.shape), order, axis=0)
    ix = np.argmax(np.cumsum(w, axis=0) >= .5 * w.sum(axis=0), axis=0)
    return sorted_values[ix, np.arange(values.shape[1])]


def _fit_candidate(curves, xy, candidate):
    mean = np.median(curves, axis=0)
    fitted = {'candidate': candidate, 'curves': curves, 'xy': xy, 'mean': mean,
              'length': _length(xy) * candidate.length_ratio}
    if candidate.kind in ('scalar', 'depth_mean', 'depth_median', 'local_median'):
        return fitted
    residuals = curves - mean
    if candidate.kind == 'fpca':
        _, singular, vt = np.linalg.svd(residuals, full_matrices=False)
        if np.sum(singular ** 2) > 1e-12:
            rank = min(8, len(curves) - 1, int(np.searchsorted(np.cumsum(singular ** 2) / np.sum(singular ** 2), .9)) + 1)
        else:
            rank = 0
        basis = vt[:rank]
        residuals = residuals @ basis.T
        fitted['basis'] = basis
    kernel = _matern(xy, xy, fitted['length']) + candidate.ridge * np.eye(len(xy))
    fitted['alpha'] = np.linalg.solve(kernel, residuals)
    return fitted


def _predict_candidate(fitted, query_xy):
    candidate, curves, xy = fitted['candidate'], fitted['curves'], fitted['xy']
    n, depth = len(query_xy), curves.shape[1]
    if candidate.kind == 'scalar':
        answer = np.full((n, depth), np.median(curves))
    elif candidate.kind == 'depth_mean':
        answer = np.broadcast_to(curves.mean(axis=0), (n, depth)).copy()
    elif candidate.kind == 'depth_median':
        answer = np.broadcast_to(fitted['mean'], (n, depth)).copy()
    elif candidate.kind == 'local_median':
        weights = _matern(query_xy, xy, fitted['length'])
        weights = np.maximum(weights, 1e-12)
        answer = np.stack([_weighted_median(curves, w) for w in weights])
    else:
        answer = _matern(query_xy, xy, fitted['length']) @ fitted['alpha']
        if candidate.kind == 'fpca':
            answer = answer @ fitted['basis']
        answer = answer + fitted['mean']
    if candidate.count_mode != 'native':
        # Complete-curve count decisions from observed profiles only.  This is a
        # point-curve correction, not a probability masquerading as amplitude.
        weights = _matern(query_xy, xy, fitted['length'])
        weights = np.maximum(weights, 1e-12)
        weights /= weights.sum(axis=1, keepdims=True)
        observed_low = curves < 0
        low_probability = weights @ observed_low
        observed_counts = observed_low.sum(axis=1)
        counts = np.array([int(_weighted_median(observed_counts[:, None], w)[0]) for w in weights])
        if candidate.count_mode == 'half':
            counts = np.floor(.5 * counts + .5 * (answer < 0).sum(axis=1) + .5).astype(int)
        rank = np.lexsort((answer, -low_probability), axis=1)
        low = np.zeros_like(answer, bool)
        np.put_along_axis(low, rank, np.arange(depth)[None, :] < counts[:, None], axis=1)
        answer = np.where(low & (answer >= 0), -.025, np.where(~low & (answer < 0), .025, answer))
    return answer


def _cv_metrics(predicted, truth):
    errors = abs(predicted - truth).mean(axis=1)
    h = .25 * abs((predicted < 0).sum(axis=1) - (truth < 0).sum(axis=1))
    low, chosen = truth < 0, predicted < 0
    recall = float((low & chosen).sum() / low.sum()) if low.any() else np.nan
    fpr = float((~low & chosen).sum() / (~low).sum()) if (~low).any() else np.nan
    return dict(log_mae=float(errors.mean()), thickness_mae_m=float(h.mean()), recall=recall, fpr=fpr)


def _folds(groups):
    """All near groups stay atomic; hash folds are invariant to input row order."""
    unique = sorted(set(groups), key=lambda g: hashlib.sha256(str(g).encode()).hexdigest())
    nfold = min(5, len(unique))
    return [np.isin(groups, unique[i::nfold]) for i in range(nfold)] if nfold > 1 else []


class TargetProfilePredictor:
    def __init__(self, model_id='target_profile_cv', fold_id=0, **kwargs):
        if model_id not in V2_MODEL_IDS:
            raise ValueError(f'Unknown v2 target model: {model_id}')
        self.model_id, self.fold_id = model_id, int(fold_id)
        self.observed = None

    def initialize(self):
        return self

    def _candidates(self):
        control = {'target_scalar_median': 'scalar', 'target_depth_mean': 'depth_mean',
                   'target_depth_median': 'depth_median'}
        if self.model_id in control:
            return [ProfileCandidate(control[self.model_id])]
        if self.model_id == 'target_joint_cv':
            return [ProfileCandidate(kind, ratio, .25, count)
                    for kind in ('gp', 'fpca') for ratio in (1., 2.)
                    for count in ('native', 'half', 'empirical')]
        kinds = {'target_trend_gp': ('gp',), 'target_fpca_gp': ('fpca',),
                 'target_profile_cv': ('gp', 'fpca', 'local_median')}[self.model_id]
        choices = [ProfileCandidate(kind, ratio, ridge)
                   for kind in kinds for ratio in (.5, 1., 2., 4.)
                   for ridge in ((.05, .25) if kind != 'local_median' else (.1,))]
        if self.model_id == 'target_profile_cv':
            choices += [ProfileCandidate('depth_mean'), ProfileCandidate('depth_median')]
        return choices

    def condition(self, observed):
        self.observed = validate_observed(observed).reset_index(drop=True)
        matrix = self.observed.pivot(index='location_id', columns='depth_index', values='log_qc').sort_index()
        if matrix.isna().any().any():
            raise ValueError('Target profile adaptation requires complete observed depth profiles')
        self.depths = matrix.columns.to_numpy(int)
        self.locations = matrix.index.to_numpy()
        self.curves = matrix.to_numpy(float)
        locations = self.observed.drop_duplicates('location_id').set_index('location_id').loc[matrix.index]
        if self.observed.groupby('location_id')[['x_m', 'y_m']].nunique().ne(1).any().any():
            raise ValueError('One location must have one coordinate pair')
        self.xy = locations[['x_m', 'y_m']].to_numpy(float)
        groups = locations.near_group_id.to_numpy() if 'near_group_id' in locations else self.locations
        folds, choices = _folds(groups), self._candidates()
        records = []
        for candidate in choices:
            out = np.full_like(self.curves, np.nan)
            for mask in folds:
                fitted = _fit_candidate(self.curves[~mask], self.xy[~mask], candidate)
                out[mask] = _predict_candidate(fitted, self.xy[mask])
            if folds:
                metrics = _cv_metrics(out, self.curves)
            else:
                metrics = dict(log_mae=np.nan, thickness_mae_m=np.nan, recall=np.nan, fpr=np.nan)
            records.append((candidate, metrics, out))
        if folds:
            if self.model_id in ('target_profile_cv', 'target_joint_cv'):
                # Targets fixed before these experiments.  The ranking balances
                # amplitude and integrated thickness; low-qc diagnostics remain
                # visible and are never selected using audit outcomes.
                def loss(item):
                    m = item[1]
                    return m['log_mae'] / .626458169250689 + m['thickness_mae_m'] / .88125
            else:
                def loss(item):
                    return item[1]['log_mae']
            best = min(records, key=loss)
        else:
            best = records[0]
        self.selected_candidate, self.cv_metrics, self.crossfit_predictions = best
        self.cv_candidates = [{**asdict(c), **m} for c, m, _ in records]
        self.fitted = _fit_candidate(self.curves, self.xy, self.selected_candidate)
        self.context_manifest = dict(model_id=self.model_id, fold_id=self.fold_id,
            observed_location_ids=self.locations.tolist(), candidate=asdict(self.selected_candidate),
            selection='revealed_near_group_5fold_cross_validation', cv=self.cv_metrics,
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
        values = _predict_candidate(self.fitted, geometry[['x_m', 'y_m']].to_numpy(float))
        loc = geometry.index.get_indexer(query.location_id)
        depth = pd.Index(self.depths).get_indexer(query.depth_index)
        result = values[loc, depth]
        known = pd.MultiIndex.from_frame(self.observed[['location_id', 'depth_index']])
        indices = known.get_indexer(pd.MultiIndex.from_frame(query[['location_id', 'depth_index']]))
        revealed = indices >= 0
        if revealed.any():
            known_xy = self.observed.iloc[indices[revealed]][['x_m', 'y_m']].to_numpy(float)
            if not np.allclose(query.loc[revealed, ['x_m', 'y_m']].to_numpy(float), known_xy, atol=1e-9, rtol=0):
                raise ValueError('Revealed query geometry disagrees with the observation')
        result[revealed] = self.observed.iloc[indices[revealed]].log_qc.to_numpy(float)
        frame = point_prediction_frame(query, result)
        frame['is_revealed'] = revealed
        return frame


def make_v2_predictor(model_id, fold_id=0, **kwargs):
    return TargetProfilePredictor(model_id, fold_id, **kwargs)
