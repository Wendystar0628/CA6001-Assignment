"""Label-free profile acquisition based on reducible GP covariance.

These policies accept the planned prediction geometry and already conditioned
models only. The caller reveals a complete batch after this function returns.
No audit geometry, candidate measurements, or hypothetical labels are needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from sklearn.gaussian_process.kernels import WhiteKernel

from .adaptive_policy import _geometry, _groups, _ids


def _white_variance(kernel):
    if isinstance(kernel, WhiteKernel):
        return float(kernel.noise_level)
    if hasattr(kernel, 'k1') and hasattr(kernel, 'k2'):
        # The project GP is a signal product plus an additive WhiteKernel.
        if kernel.__class__.__name__ == 'Sum':
            return _white_variance(kernel.k1) + _white_variance(kernel.k2)
        if _white_variance(kernel.k1) or _white_variance(kernel.k2):
            raise ValueError('Multiplicative WhiteKernel is not supported')
    return 0.


def latent_posterior(gp_predictor, query_grid):
    """Return latent covariance and independent observation noise in log units.

    Explicit X/Y kernel calls remove WhiteKernel from signal covariance while
    the fitted training Cholesky still includes its observation noise. This is
    different from subtracting noise from an already ranked standard deviation.
    """
    gp_predictor = getattr(gp_predictor, 'predictor', gp_predictor)
    regressor = gp_predictor.regressor
    if regressor is None:
        raise ValueError('GP must be conditioned on revealed observations')
    coordinates = gp_predictor._coordinates(query_grid)
    kernel = regressor.kernel_
    cross = kernel(coordinates, regressor.X_train_)
    solved = solve_triangular(regressor.L_, cross.T, lower=True, check_finite=False)
    covariance = kernel(coordinates, coordinates.copy()) - solved.T @ solved
    covariance = (covariance + covariance.T) * .5
    covariance[np.diag_indices_from(covariance)] = np.maximum(np.diag(covariance), 0.)
    # normalize_y=False in GPPredictor; preserve correct units for other callers.
    scale = float(np.asarray(getattr(regressor, '_y_train_std', 1.)).reshape(-1)[0]) ** 2
    covariance *= scale
    noise = (_white_variance(kernel) + float(np.max(np.asarray(regressor.alpha)))) * scale
    return covariance, max(noise, 1.e-10)


def _aligned_values(predictions, query_grid):
    frame = predictions.frame if hasattr(predictions, 'frame') else predictions
    if frame is None:
        raise ValueError('target_ivr requires primary and helper predictions')
    keys = ['location_id', 'depth_index']
    frame = frame[keys + ['pred_log_qc']].copy()
    if frame.duplicated(keys).any():
        raise ValueError('Prediction keys must be unique')
    values = query_grid[keys].merge(frame, on=keys, how='left', validate='one_to_one').pred_log_qc.to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError('Predictions must cover the planned query grid')
    return values


def _gain(covariance, cells, weights, noise):
    cross = covariance[:, cells]
    observation = covariance[np.ix_(cells, cells)] + np.eye(len(cells)) * noise
    factor = cho_factor(observation, lower=True, check_finite=False)
    solved = cho_solve(factor, cross.T, check_finite=False)
    reductions = np.einsum('ij,ji->i', cross, solved)
    gain = float(np.dot(weights, np.maximum(reductions, 0.)) / max(weights.sum(), 1.e-12))
    return gain, cross, solved


def _scale_positive(values):
    values = np.maximum(np.asarray(values, float), 0.)
    return values / values.max() if values.size and values.max() > 1.e-12 else np.zeros_like(values)


def select_batch(geometry, known_ids, candidate_ids, near_groups=None, *,
                 gp_predictor, query_grid, primary_predictions=None,
                 helper_predictions=None, batch_size=2, policy='latent_ivr',
                 coverage_weight=.15):
    """Select atomic groups by integrated variance reduction per new CPT.

    latent_ivr minimises average planned-node latent variance. target_ivr uses
    the same covariance objective with continuous primary/helper discrepancy
    weights, normalised only by revealed log-qc dispersion. Coverage contributes
    a fixed 15% by default. Scores retain signal magnitudes rather than turning
    arbitrarily small differences into full-range ranks. Neither policy is a
    calibrated error probability or a stopping criterion.
    """
    if policy not in {'latent_ivr', 'target_ivr'}:
        raise ValueError('Policy must be latent_ivr or target_ivr')
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    if not 0 <= coverage_weight <= 1:
        raise ValueError('coverage_weight must be between zero and one')
    gp_predictor = getattr(gp_predictor, 'predictor', gp_predictor)
    coordinates = _geometry(geometry)
    known, candidates = _ids(known_ids, 'Known'), _ids(candidate_ids, 'Candidate')
    if not known or set(known) & set(candidates):
        raise ValueError('Need nonempty known locations disjoint from candidates')
    if not set(known + candidates) <= set(coordinates.index):
        raise ValueError('Every plan location needs geometry')
    columns = ['location_id', 'near_group_id', 'group_size', 'integrated_reduction',
               'reduction_per_cpt', 'distance_to_known_m', 'score', 'selected',
               'selection_order', 'selection_score', 'selection_distance_m',
               'observation_noise_variance', 'policy']
    if not candidates:
        return [], pd.DataFrame(columns=columns)
    group_map = _groups(near_groups, geometry)
    for name, pool in [('known', set(known)), ('candidate', set(candidates))]:
        for gid in {group_map[loc] for loc in pool}:
            members = {loc for loc in coordinates.index if group_map[loc] == gid}
            if not members <= pool:
                raise ValueError(f'The {name} set splits near group {gid}')
    # Restrict to exactly the plan, rejecting accidental audit-grid inclusion.
    query = query_grid[['location_id', 'depth_index', 'x_m', 'y_m', 'z_center_m']].copy().reset_index(drop=True)
    query.location_id = query.location_id.astype(str)
    if set(query.location_id) != set(known + candidates) or query.duplicated(['location_id', 'depth_index']).any():
        raise ValueError('query_grid must contain exactly the fixed plan and unique depth keys')
    if set(gp_predictor.observed.location_id.astype(str)) != set(known):
        raise ValueError('GP context must contain exactly the revealed locations')
    covariance, noise = latent_posterior(gp_predictor, query)
    weights = np.ones(len(query))
    if policy == 'target_ivr':
        primary, helper = (_aligned_values(frame, query) for frame in (primary_predictions, helper_predictions))
        observed = gp_predictor.observed.log_qc.to_numpy(float)
        spread = max(float(np.median(np.abs(observed - np.median(observed)))), .1)
        weights += np.minimum(np.abs(primary - helper) / spread, 3.)
    weights[query.location_id.isin(known).to_numpy()] = 0.
    groups = {gid: sorted(loc for loc in candidates if group_map[loc] == gid)
              for gid in sorted({group_map[loc] for loc in candidates})}
    cells_by_group = {gid: np.flatnonzero(query.location_id.isin(members)) for gid, members in groups.items()}
    selected, records = [], {}
    observed_xy = coordinates.loc[known, ['x_m', 'y_m']].to_numpy(float)
    while groups and len(selected) < batch_size:
        candidates_this_round = []
        for gid, members in groups.items():
            cells = cells_by_group[gid]
            gain, _, _ = _gain(covariance, cells, weights, noise)
            xy = coordinates.loc[members, ['x_m', 'y_m']].to_numpy(float)
            distance = float(np.linalg.norm(xy[:, None] - observed_xy[None, :], axis=2).min())
            candidates_this_round.append((gid, gain, gain / len(members), distance))
        gains = _scale_positive([row[2] for row in candidates_this_round])
        distances = _scale_positive([row[3] for row in candidates_this_round])
        scores = (1 - coverage_weight) * gains + coverage_weight * distances
        for (gid, gain, per_cpt, distance), score in zip(candidates_this_round, scores):
            if gid not in records:
                records[gid] = {'near_group_id': gid, 'group_size': len(groups[gid]),
                                'integrated_reduction': gain, 'reduction_per_cpt': per_cpt,
                                'distance_to_known_m': distance, 'score': float(score), 'selected': False,
                                'selection_order': pd.NA, 'selection_score': np.nan,
                                'selection_distance_m': np.nan, 'observation_noise_variance': noise,
                                'policy': policy}
        winner = sorted(zip(candidates_this_round, scores), key=lambda row: (-row[1], row[0][0]))[0]
        (gid, _, _, distance), score = winner
        members, cells = groups.pop(gid), cells_by_group[gid]
        selected.extend(members)
        records[gid].update(selected=True, selection_order=sum(r['selected'] for r in records.values()) + 1,
                            selection_score=float(score), selection_distance_m=distance)
        # Hypothetical covariance update only; no candidate label is accepted.
        _, cross, solved = _gain(covariance, cells, weights, noise)
        covariance -= cross @ solved
        covariance = (covariance + covariance.T) * .5
        covariance[np.diag_indices_from(covariance)] = np.maximum(np.diag(covariance), 0.)
        weights[cells] = 0.
        observed_xy = np.vstack([observed_xy, coordinates.loc[members, ['x_m', 'y_m']].to_numpy(float)])
    rows = [{'location_id': loc, **records[group_map[loc]]} for loc in candidates]
    return selected, pd.DataFrame(rows)[columns].sort_values('location_id', kind='stable').reset_index(drop=True)
