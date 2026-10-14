"""Retained inference component, consolidated from cpt_budget.v8.joint (v0.8-engineering)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from cpt_budget.models.base import validate_observed, validate_query_site
from cpt_budget.types import QUERY_COLUMNS
N_DEPTHS = 24
MARGIN = 0.05

def _query_batch(query, base, observed, locations, xy):
    if set(query) - set(QUERY_COLUMNS):
        raise ValueError('Query must contain geometry only')
    validate_query_site(query, observed)
    if query.empty or query.duplicated(['location_id', 'depth_index']).any():
        raise ValueError('Unique complete query profiles required')
    if set(query.location_id) & set(locations):
        raise ValueError('Query profile must not be in revealed support')
    if 'near_group_id' in query and set(query.near_group_id) & set(observed.near_group_id):
        raise ValueError('Query near group must not overlap revealed support')
    if query.groupby('location_id')[['x_m', 'y_m']].nunique().ne(1).any().any():
        raise ValueError('Query geometry must be constant across depth')
    base = np.asarray(base, float)
    if base.shape != (len(query),) or not np.isfinite(base).all():
        raise ValueError('Finite base values aligned to query rows are required')
    ordered = query.reset_index(drop=True).sort_values(['location_id', 'depth_index'], kind='stable')
    order = ordered.index.to_numpy()
    groups = ordered.groupby('location_id', sort=False)
    if groups.size().ne(N_DEPTHS).any():
        raise ValueError('Complete 24-depth query profiles required')
    if not np.array_equal(ordered.depth_index.to_numpy(), np.tile(np.arange(N_DEPTHS), len(groups))):
        raise ValueError('Complete 24-depth query profiles required')
    geometry = ordered.drop_duplicates('location_id')
    query_xy = geometry[['x_m', 'y_m']].to_numpy(float)
    if not np.isfinite(query_xy).all():
        raise ValueError('Finite query geometry required')
    distance = cdist(query_xy, xy)
    if (distance <= 1e-09).any():
        raise ValueError('Query cannot be colocated with revealed support')
    return (ordered, order, geometry.location_id.tolist(), base[order].reshape(-1, N_DEPTHS), distance)

def _lower_weighted_quantile(values, weights, q=0.5):
    """Batched atom-axis lower empirical quantile; stable atom ordering."""
    values = np.asarray(values)
    weights = np.asarray(weights, float)
    if values.ndim == 2:
        values = values[:, :, None]
        squeeze = True
    else:
        squeeze = False
    indices = np.argsort(values, axis=1, kind='stable')
    ordered_values = np.take_along_axis(values, indices, axis=1)
    ordered_weights = np.take_along_axis(np.broadcast_to(weights[:, :, None], values.shape), indices, axis=1)
    passed = np.cumsum(ordered_weights, axis=1) >= q
    first = np.argmax(passed, axis=1)
    answer = np.take_along_axis(ordered_values, first[:, None, :], axis=1)[:, 0, :]
    return answer[:, 0] if squeeze else answer

def batch_thickness_decision(base, probability, count):
    """Vectorized version of the fixed-count event assignment and crossing rule."""
    base = np.asarray(base, float)
    probability = np.asarray(probability, float)
    count = np.asarray(count, int)
    if base.ndim != 2 or base.shape != probability.shape or count.shape != (len(base),) or (not np.isfinite(base).all()) or (not np.isfinite(probability).all()) or ((probability < 0) | (probability > 1)).any() or ((count < 0) | (count > base.shape[1])).any():
        raise ValueError('Finite aligned point/probability arrays and valid counts required')
    rank = np.lexsort((np.broadcast_to(np.arange(base.shape[1]), base.shape), base, -probability), axis=1)
    low = np.zeros_like(base, bool)
    np.put_along_axis(low, rank, np.arange(base.shape[1])[None, :] < count[:, None], axis=1)
    return np.where(low & (base >= 0), -MARGIN, np.where(~low & (base < 0), MARGIN, base))

def _finish(atoms, weights, base, order, locations, *, count_override=None):
    weights = weights / weights.sum(axis=1, keepdims=True)
    low = atoms < 0
    probability = np.clip(np.einsum('ns,nsd->nd', weights, low), 0.0, 1.0)
    atom_counts = low.sum(axis=2)
    count = _lower_weighted_quantile(atom_counts, weights).astype(int)
    effective_count = count if count_override is None else np.asarray(count_override, int)
    point = batch_thickness_decision(base, probability, effective_count)
    point_flat, probability_flat = (np.empty(len(order)), np.empty(len(order)))
    point_flat[order], probability_flat[order] = (point.ravel(), probability.ravel())
    return {'point': point_flat, 'probability': probability_flat, 'atoms': atoms, 'weights': weights, 'atom_counts': atom_counts, 'count_median': count, 'count_expected': (weights * atom_counts).sum(axis=1), 'base_count': (base < 0).sum(axis=1), 'decision_count': effective_count, 'locations': locations, 'order': order, 'base_matrix': base, 'probability_matrix': probability}

class FastJointAdapter:
    """Condition once, predict arbitrary batches without per-depth pandas work."""

    def condition(self, observed):
        frame = validate_observed(observed)
        if 'near_group_id' not in frame or frame.near_group_id.isna().any():
            raise ValueError('Known near groups are required')
        if 'z_center_m' not in frame or not np.isfinite(frame.z_center_m).all():
            raise ValueError('Finite support depth coordinates are required')
        if frame.groupby('location_id')[['x_m', 'y_m', 'near_group_id']].nunique().ne(1).any().any():
            raise ValueError('Support profile geometry and near groups must be constant')
        self.observed = frame
        self.locations = sorted(frame.location_id.unique())
        profile = frame.pivot(index='location_id', columns='depth_index', values='log_qc')
        if list(profile.columns) != list(range(N_DEPTHS)) or profile.isna().any().any():
            raise ValueError('Complete 24-depth revealed profiles are required')
        geometry = frame.drop_duplicates('location_id').set_index('location_id').loc[self.locations]
        self.curves = profile.loc[self.locations].to_numpy(float)
        self.xy = geometry[['x_m', 'y_m']].to_numpy(float)
        self.groups = geometry.near_group_id.to_numpy()
        self.group_size = geometry.near_group_id.map(geometry.near_group_id.value_counts()).to_numpy(float)
        distance = cdist(self.xy, self.xy)
        nonzero = pdist(self.xy)
        nonzero = nonzero[nonzero > 1e-09]
        if not len(nonzero):
            raise ValueError('At least two distinct support coordinates required')
        self.length = float(np.median(nonzero))
        allowed = (distance > 1e-09) & (self.groups[:, None] != self.groups[None, :])
        candidate_distance = np.where(allowed, distance, np.inf)
        nearest = np.argsort(candidate_distance, axis=1, kind='stable')[:, :min(3, len(self.xy))]
        closest = np.take_along_axis(candidate_distance, nearest, axis=1)
        reference_count = np.isfinite(closest).sum(axis=1)
        if not reference_count.all():
            raise ValueError('Empirical errors require another independent revealed CPT at each depth')
        weight = (closest[:, :1] / closest) ** 2
        weight /= weight.sum(axis=1, keepdims=True)
        self.error_curves = self.curves - np.einsum('nk,nkd->nd', weight, self.curves[nearest])
        self.loo_min_reference_count = int(reference_count.min())
        self._adaptation = None
        return self

    def batch_predict(self, query, base_values, kind='error', *, shrinkage=0.5, policy='reference'):
        if not np.isfinite(shrinkage) or shrinkage < 0:
            raise ValueError('Finite nonnegative shrinkage required')
        _, order, locations, base, distance = _query_batch(query, base_values, self.observed, self.locations, self.xy)
        if policy != 'reference':
            if self._adaptation is None:
                self._adaptation = self.fit_support_adaptation()
            if policy == 'support_alpha':
                shrinkage = self._adaptation['alpha']
            elif policy not in ('count_shrink', 'count_bias'):
                raise ValueError('Unknown support adaptation policy')
        log_weight = -0.5 * (distance / self.length) ** 2 - np.log(self.group_size)[None, :]
        weights = np.exp(log_weight - log_weight.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        if kind == 'error':
            deviation = np.broadcast_to(self.error_curves, (len(base), *self.error_curves.shape))
        elif kind == 'analog':
            mean = weights @ self.curves
            deviation = self.curves[None, :, :] - mean[:, None, :]
        else:
            raise ValueError('Unknown complete-curve mixture kind')
        atoms = base[:, None, :] + shrinkage * deviation
        result = _finish(atoms, weights, base, order, locations)
        if policy in ('count_shrink', 'count_bias'):
            if policy == 'count_shrink':
                mix = self._adaptation['count_weight']
                desired = np.floor(mix * result['count_median'] + (1 - mix) * result['base_count'] + 0.5).astype(int)
            else:
                desired = np.clip(result['count_median'] + self._adaptation['count_bias'], 0, N_DEPTHS)
            result = _finish(atoms, weights, base, order, locations, count_override=desired)
        return result

    def fit_support_adaptation(self):
        """Select modest count corrections using whole-group held-out support.

        The surrogate amplitude predictor is 3NN IDW, not the full structural
        ensemble. Its count transport may be inaccurate; this limitation is
        explicit in experiment metadata. Held groups never enter error atoms.
        """
        records = []
        for group in np.unique(self.groups):
            held = self.observed.near_group_id.eq(group)
            training = self.observed.loc[~held]
            if training.near_group_id.nunique() < 2:
                continue
            model = FastJointAdapter().condition(training)
            truth = self.observed.loc[held].sort_values(['location_id', 'depth_index'])
            query = truth[[c for c in QUERY_COLUMNS if c in truth]]
            geometry = truth.drop_duplicates('location_id')
            distance = cdist(geometry[['x_m', 'y_m']], model.xy)
            if (distance <= 1e-09).any():
                alias = np.any(cdist(model.xy, geometry[['x_m', 'y_m']]) <= 1e-09, axis=1)
                training = training.loc[~training.location_id.isin(np.asarray(model.locations)[alias])]
                if training.near_group_id.nunique() < 2:
                    continue
                model = FastJointAdapter().condition(training)
                distance = cdist(geometry[['x_m', 'y_m']], model.xy)
            nearest = np.argsort(distance, axis=1, kind='stable')[:, :3]
            closest = np.take_along_axis(distance, nearest, axis=1)
            weight = (closest[:, :1] / closest) ** 2
            weight /= weight.sum(axis=1, keepdims=True)
            base = np.einsum('nk,nkd->nd', weight, model.curves[nearest])
            actual = truth.log_qc.to_numpy(float).reshape(-1, N_DEPTHS)
            for alpha in (0.25, 0.5, 0.75):
                result = model.batch_predict(query, base.ravel(), shrinkage=alpha)
                for j, location in enumerate(result['locations']):
                    records.append({'group': str(group), 'location': location, 'alpha': alpha, 'actual_count': int((actual[j] < 0).sum()), 'predicted_count': int(result['count_median'][j]), 'base_count': int((base[j] < 0).sum()), 'group_weight': 1.0 / len(geometry)})
        if not records:
            return dict(alpha=0.5, count_weight=1.0, count_bias=0, held_profiles=0, losses=[])
        frame = pd.DataFrame(records)
        losses = []
        for alpha in (0.25, 0.5, 0.75):
            sample = frame.loc[frame.alpha.eq(alpha)]
            loss = np.average(np.abs(sample.predicted_count - sample.actual_count), weights=sample.group_weight)
            losses.append(dict(alpha=alpha, count_mae=float(loss)))
        selected_alpha = min(losses, key=lambda r: (r['count_mae'], abs(r['alpha'] - 0.5)))['alpha']
        sample = frame.loc[frame.alpha.eq(0.5)]
        shrink_loss = []
        for mixing in (0.0, 0.5, 1.0):
            prediction = np.floor(mixing * sample.predicted_count + (1 - mixing) * sample.base_count + 0.5)
            loss = np.average(np.abs(prediction - sample.actual_count), weights=sample.group_weight)
            shrink_loss.append((float(loss), -mixing))
        selected_weight = -min(shrink_loss)[1]
        bias_values = sample.actual_count.to_numpy() - sample.predicted_count.to_numpy()
        bias_weights = sample.group_weight.to_numpy(float)
        bias_order = np.argsort(bias_values, kind='stable')
        cut = np.searchsorted(np.cumsum(bias_weights[bias_order] / bias_weights.sum()), 0.5, side='left')
        bias = int(np.clip(bias_values[bias_order[min(cut, len(bias_order) - 1)]], -2, 2))
        return dict(alpha=float(selected_alpha), count_weight=float(selected_weight), count_bias=bias, held_profiles=len(sample), losses=losses)
