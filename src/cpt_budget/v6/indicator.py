"""Support-only indicator interpolation and exact structured event decoding.

The binary field is qc < 1 MPa. Its MAP labels constrain the actual reported
log(qc), so the usual H_point metric remains the thickness of that prediction.
The native probabilities are exact marginals of the stated finite Gibbs model,
not calibrated geological probabilities or a substituted thickness estimate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from cpt_budget.models.base import validate_observed, validate_query_site
from cpt_budget.models.distribution import point_prediction_frame
from cpt_budget.types import PredictionBundle


MODELS = {
    'v6_event_potts_residual': ('potts', 'residual_ft_crps_guarded_a050'),
    'v6_event_cardinality_residual': ('cardinality', 'residual_ft_crps_guarded_a050'),
    'v6_event_adaptive_residual': ('adaptive', 'residual_ft_crps_guarded_a050'),
    'v6_event_cardinality_idw': ('cardinality', 'idw'),
}
SETTINGS = {
    'indicator_neighbors': 3, 'indicator_distance_power': 2,
    'jeffreys_pseudocount_per_class': 0.5,
    'amplitude_tau_log_qc': 0.5, 'potts_cost': 0.25,
    'cardinality_cost': 0.25, 'sign_margin_log_qc': 0.05,
    'target_count': 'spatially weighted median of revealed profile event counts',
    'adaptive_tau_grid': [0.25, 0.5, 1.0],
    'adaptive_cardinality_grid': [0.0, 0.25, 1.0],
    'adaptive_objective': 'log_MAE + 0.25 * H_point_MAE_m / 0.25m + 0.5 * balanced_event_error',
    'adaptive_validation': 'leave one revealed location and its entire near group / colocated supports out; IDW amplitude proxy, not residual predictions',
    'adaptive_tie_break': 'lowest tau, then lowest cardinality cost',
    'query_information': 'geometry and frozen legitimate base predictions only',
    'sources': [
        'https://link.springer.com/article/10.1007/s11004-008-9146-8',
        'https://ascelibrary.org/doi/10.1061/(ASCE)GM.1943-5622.0002113',
    ],
    'source_scope': 'Indicator interpolation and structured CPT labeling motivate an original small Gibbs model; not reproductions of indicator kriging or HMRF papers.',
}


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind='stable')
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    return float(values[np.searchsorted(np.cumsum(weights), 0.5 * weights.sum(), side='left')])


def structured_event_profile(base, probability, target_count, *, tau=0.5,
                             potts=0.25, cardinality=0.25, margin=0.05):
    """Exact MAP and marginals over all 2**depth binary profiles by DP.

E(s) = sum[-log p(s) + (project(base,s)-base)^2/(2 tau^2)]
       + potts * number_of_transitions + cardinality*(sum(s)-target_count)^2.
Projection is the closest amplitude satisfying state 0 >= margin or state 1
<= -margin. Cardinality is a soft potential, never a metric substitution.
"""
    base, probability = np.asarray(base, float), np.asarray(probability, float)
    if base.ndim != 1 or base.shape != probability.shape or not len(base):
        raise ValueError('One probability per depth is required')
    if (not np.isfinite(base).all() or not np.isfinite(probability).all()
            or ((probability < 0) | (probability > 1)).any()):
        raise ValueError('Finite amplitudes and valid probabilities are required')
    if tau <= 0 or margin <= 0 or potts < 0 or cardinality < 0:
        raise ValueError('Positive scale/margin and nonnegative penalties required')
    n = len(base)
    if not 0 <= target_count <= n:
        raise ValueError('Count must be inside the profile length')
    p = np.clip(probability, 1e-9, 1 - 1e-9)
    amplitudes = np.column_stack([np.maximum(base, margin), np.minimum(base, -margin)])
    unary = -np.log(np.column_stack([1 - p, p])) + (amplitudes - base[:, None])**2 / (2 * tau**2)
    terminal = cardinality * (np.arange(n + 1) - target_count)**2
    minimum = np.full((n, n + 1, 2), np.inf)
    forward = np.full_like(minimum, -np.inf)
    back = np.full((n, n + 1, 2), -1, dtype=np.int8)
    for s in (0, 1):
        minimum[0, s, s] = unary[0, s]
        forward[0, s, s] = -unary[0, s]
    transitions = potts * (np.arange(2)[:, None] != np.arange(2)[None, :])
    for t in range(1, n):
        for s in (0, 1):
            counts = np.arange(s, t + 2)
            previous = counts - s
            costs = minimum[t - 1, previous, :] + transitions[:, s]
            choices = np.argmin(costs, axis=1)
            minimum[t, counts, s] = costs[np.arange(len(counts)), choices] + unary[t, s]
            back[t, counts, s] = choices
            forward[t, counts, s] = logsumexp(
                forward[t - 1, previous, :] - transitions[:, s], axis=1) - unary[t, s]
    count, state = np.unravel_index(np.argmin(minimum[-1] + terminal[:, None]), (n + 1, 2))
    labels = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        labels[t] = state
        previous_state = int(back[t, count, state])
        count -= state
        state = previous_state
    # Backward values condition on the accumulated prefix count, retaining the
    # nonlocal cardinality potential in every marginal.
    backward = np.full_like(forward, -np.inf)
    backward[-1, :, :] = -terminal[:, None]
    for t in range(n - 2, -1, -1):
        counts = np.arange(t + 2)
        for s in (0, 1):
            alternatives = np.column_stack([
                backward[t + 1, counts + nxt, nxt] - unary[t + 1, nxt] - transitions[s, nxt]
                for nxt in (0, 1)])
            backward[t, counts, s] = logsumexp(alternatives, axis=1)
    log_z = logsumexp(forward[-1] - terminal[:, None])
    marginal = np.exp(logsumexp(forward[:, :, 1] + backward[:, :, 1], axis=1) - log_z)
    marginal = np.clip(marginal, 0., 1.)
    prediction = amplitudes[np.arange(n), labels]
    return prediction, marginal, labels


class IndicatorLayerPredictor:
    """A support-only event field, optionally with a supplied legal base model."""
    def __init__(self, *, mode='cardinality', model_id=None):
        if mode not in {'potts', 'cardinality', 'adaptive'}:
            raise ValueError('Unknown indicator decoding mode')
        self.mode = mode
        self.model_id = model_id or f'event_{mode}'
        self.observed = None

    def condition(self, observed):
        self.observed = validate_observed(observed)
        depths = observed[['depth_index', 'z_center_m']].drop_duplicates().sort_values('depth_index')
        if depths.depth_index.duplicated().any() or len(depths) < 2:
            raise ValueError('A common complete depth grid is required')
        self.depth_ids = depths.depth_index.to_numpy()
        self.depths = depths.z_center_m.to_numpy(float)
        steps = np.diff(self.depths)
        if (steps <= 0).any() or not np.allclose(steps, steps[0]):
            raise ValueError('A regular depth grid is required')
        self.step_m = float(steps[0])
        locations = observed[['location_id', 'x_m', 'y_m']].drop_duplicates().sort_values('location_id')
        if locations.location_id.duplicated().any():
            raise ValueError('One coordinate per support profile is required')
        self.ids = locations.location_id.to_numpy()
        self.xy = locations[['x_m', 'y_m']].to_numpy(float)
        if 'near_group_id' in observed:
            groups = observed[['location_id', 'near_group_id']].drop_duplicates()
            if groups.location_id.duplicated().any() or groups.near_group_id.isna().any():
                raise ValueError('Every support location needs one known near group')
            self.near_groups = groups.set_index('location_id').reindex(self.ids).near_group_id.to_numpy()
        else:
            self.near_groups = self.ids.copy()
        self.profiles = observed.pivot(index='location_id', columns='depth_index', values='log_qc').reindex(
            index=self.ids, columns=self.depth_ids).to_numpy(float)
        if not np.isfinite(self.profiles).all():
            raise ValueError('Complete finite support profiles required')
        self.tau, self.cardinality = 0.5, 0. if self.mode == 'potts' else 0.25
        self.loo_scores = []
        if self.mode == 'adaptive' and len(self.ids) > 1:
            self._select_observed_only()
        self.context_manifest = {
            'observed_location_ids': self.ids.tolist(), 'settings': SETTINGS,
            'mode': self.mode, 'selected_tau': self.tau,
            'selected_cardinality': self.cardinality, 'loo_scores': self.loo_scores,
            'probability_interpretation': 'exact marginals of finite event Gibbs model',
            'amplitude_interpretation': 'MAP label-constrained minimum perturbation of base amplitude',
        }

    def _spatial(self, coordinates, *, omit=None):
        distances = np.linalg.norm(self.xy - coordinates, axis=1)
        if omit is not None:
            excluded = (self.near_groups == self.near_groups[omit]) | (np.linalg.norm(self.xy - self.xy[omit], axis=1) <= 1e-8)
            distances[excluded] = np.inf
        available = np.flatnonzero(np.isfinite(distances))
        if not len(available):
            raise ValueError('No independent support remains after near-group exclusion')
        nearest = available[np.argsort(distances[available], kind='stable')[:3]]
        if distances[nearest[0]] <= 1e-10:
            weights = (distances[nearest] <= 1e-10).astype(float)
        else:
            weights = distances[nearest]**-2
        weights /= weights.sum()
        profiles = self.profiles[nearest]
        labels = profiles < 0.
        effective_n = 1. / np.square(weights).sum()
        p = (effective_n * (weights @ labels) + 0.5) / (effective_n + 1.)
        count = weighted_median(labels.sum(axis=1), weights)
        return weights @ profiles, p, count

    def _select_observed_only(self):
        if any(np.all((self.near_groups == self.near_groups[i]) |
                      (np.linalg.norm(self.xy - self.xy[i], axis=1) <= 1e-8)) for i in range(len(self.ids))):
            # No independent validation can be formed. Keep the fixed prior
            # settings rather than leaking a colocated or same-group sample.
            return
        cases = [(*self._spatial(self.xy[i], omit=i), self.profiles[i]) for i in range(len(self.ids))]
        for tau in SETTINGS['adaptive_tau_grid']:
            for gamma in SETTINGS['adaptive_cardinality_grid']:
                scores = []
                for base, p, count, truth in cases:
                    predicted, _, labels = structured_event_profile(base, p, count, tau=tau, cardinality=gamma)
                    true_labels = truth < 0
                    class_errors = [float(np.mean(labels[true_labels] == 0))] if true_labels.any() else []
                    if (~true_labels).any():
                        class_errors.append(float(np.mean(labels[~true_labels] == 1)))
                    h_error = self.step_m * abs(int(labels.sum()) - int(true_labels.sum()))
                    scores.append(float(np.mean(abs(predicted - truth))) + 0.25 * h_error / 0.25 + 0.5 * np.mean(class_errors))
                self.loo_scores.append({'tau': tau, 'cardinality': gamma, 'score': float(np.mean(scores))})
        best = min(self.loo_scores, key=lambda row: (row['score'], row['tau'], row['cardinality']))
        self.tau, self.cardinality = best['tau'], best['cardinality']

    def predict(self, query_grid, *, base_values=None):
        validate_query_site(query_grid, self.observed)
        query = query_grid.reset_index(drop=True)
        if not np.isfinite(query[['x_m', 'y_m']].to_numpy(float)).all():
            raise ValueError('Finite query geometry required')
        base = None if base_values is None else np.asarray(base_values, float)
        if base is not None and (base.shape != (len(query),) or not np.isfinite(base).all()):
            raise ValueError('One finite base value per query row required')
        prediction, probability = np.empty(len(query)), np.empty(len(query))
        targets, changes = np.empty(len(query)), np.empty(len(query))
        for _, rows in query.groupby('location_id', sort=False):
            rows = rows.sort_values('depth_index')
            if not np.array_equal(rows.depth_index.to_numpy(), self.depth_ids):
                raise ValueError('Each query must contain the complete conditioned depth grid')
            if len(rows[['x_m', 'y_m']].drop_duplicates()) != 1:
                raise ValueError('One query coordinate per profile required')
            mean, p, count = self._spatial(rows[['x_m', 'y_m']].to_numpy(float)[0])
            amplitude = mean if base is None else base[rows.index]
            value, marginal, _ = structured_event_profile(amplitude, p, count,
                tau=self.tau, cardinality=self.cardinality)
            prediction[rows.index], probability[rows.index] = value, marginal
            targets[rows.index] = count
            changes[rows.index] = np.sum((value < 0) != (amplitude < 0))
        frame = point_prediction_frame(query, prediction)
        frame['p_low_1'] = probability
        frame['indicator_target_count'] = targets
        frame['indicator_changed_cells'] = changes
        return PredictionBundle(frame, self.model_id)


def main():
    from cpt_budget.config import write_json
    from cpt_budget.diagnostics.common import iter_validation_states, STATE_KEYS
    from cpt_budget.v6.common import RUN, load_frozen, emit_candidate

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RUN / 'indicator')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'frozen_protocol.json', {'models': MODELS, **SETTINGS})
    wide = load_frozen()
    values = {name: np.full(len(wide), np.nan) for name in MODELS}
    probabilities = {name: np.full(len(wide), np.nan) for name in MODELS}
    diagnostics = []
    for state in iter_validation_states():
        mask = np.ones(len(wide), dtype=bool)
        for key in STATE_KEYS:
            mask &= wide[key].eq(state.metadata[key]).to_numpy()
        positions = np.flatnonzero(mask)
        query = wide.loc[mask, ['site_id', 'location_id', 'depth_index', 'x_m', 'y_m', 'z_center_m']].reset_index(drop=True)
        for name, (mode, baseline) in MODELS.items():
            model = IndicatorLayerPredictor(mode=mode, model_id=name)
            model.condition(state.observed)
            bundle = model.predict(query, base_values=wide.loc[mask, baseline].to_numpy(float))
            values[name][positions] = bundle.frame.pred_log_qc
            probabilities[name][positions] = bundle.frame.p_low_1
            for loc, frame in bundle.frame.groupby('location_id'):
                diagnostics.append({**state.metadata, 'model_id': name, 'location_id': loc,
                    'selected_tau': model.tau, 'selected_cardinality': model.cardinality,
                    'target_count': float(frame.indicator_target_count.iloc[0]),
                    'decoded_count': int((frame.pred_log_qc < 0).sum()),
                    'expected_count': float(frame.p_low_1.sum()),
                    'changed_cells': int(frame.indicator_changed_cells.iloc[0])})
            if mode == 'adaptive':
                write_json(args.output / 'adaptive_contexts' / f'{state.key}.json', model.context_manifest)
        print(f'event decoding completed: {state.key}', flush=True)
    for name in MODELS:
        emit_candidate(name, values[name], wide=wide, probability=probabilities[name],
            output_root=args.output, metadata={'settings': SETTINGS, 'mode_and_base': MODELS[name],
                'H_point_semantics': 'threshold the actual predicted qc; expected thickness is separate',
                'probability_semantics': 'exact Gibbs marginal before external risk calibration'})
    diagnostic_frame = pd.DataFrame(diagnostics)
    diagnostic_frame['run_id'], diagnostic_frame['study_version'] = 'v6-combinations-20261014', '0.6-combinations'
    diagnostic_frame.to_csv(args.output / 'decoding_diagnostics.csv', index=False)


if __name__ == '__main__':
    main()
