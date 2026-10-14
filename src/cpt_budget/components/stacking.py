"""Retained inference component, consolidated from cpt_budget.v6.adaptive (v0.8-engineering)."""
from __future__ import annotations
import itertools
import json
import numpy as np
from scipy.special import expit
from cpt_budget.config import PROJECT_ROOT
from cpt_budget.components.functional import ProfileBasis
MEMBERS = ('idw', 'functional_fpca_krr', 'registered_profile_shape', 'spatial_weighted_median')

def simplex_candidates():
    return np.array([w for w in itertools.product(range(5), repeat=4) if sum(w) == 4], float) / 4

def fit_weights(observed, loo_predictions, *, objective='l1'):
    """Choose a coarse regularized stack using revealed labels only."""
    if objective not in {'l1', 'structural'}:
        raise ValueError('Unknown stacking objective')
    observed = observed.reset_index(drop=True)
    matrix = np.asarray(loo_predictions, float)
    if matrix.shape != (len(observed), len(MEMBERS)) or not np.isfinite(matrix).all():
        raise ValueError('Invalid LOO member matrix')
    weights = simplex_candidates()
    prediction = matrix @ weights.T
    truth = observed.log_qc.to_numpy(float)
    l1 = np.abs(prediction - truth[:, None]).mean(axis=0)
    regularization = 0.05 * np.square(weights - 0.25).sum(axis=1)
    probability = expit(-prediction / 0.25)
    labels = truth < 0
    event = np.mean(np.where(labels[:, None], 2 * (1 - probability) ** 2, probability ** 2), axis=0)
    thickness = []
    for _, rows in observed.groupby('location_id', sort=False):
        thickness.append(np.abs(0.25 * probability[rows.index].sum(axis=0) - 0.25 * labels[rows.index].sum()))
    thickness = np.mean(thickness, axis=0)
    loss = l1 + regularization
    if objective == 'structural':
        loss = loss + 0.5 * thickness + 0.5 * event
    best = int(np.argmin(loss))
    return (weights[best], {'objective': objective, 'loss': float(loss[best]), 'loo_log_mae': float(l1[best]), 'loo_soft_thickness_mae_m': float(thickness[best]), 'loo_weighted_event_brier': float(event[best]), 'regularization': float(regularization[best]), 'weights': dict(zip(MEMBERS, weights[best].tolist()))})

def load_basis(fold_id):
    path = PROJECT_ROOT / 'artifacts/models' / f'fold_{fold_id}'
    manifest = json.loads((path / 'basis_manifest.json').read_text(encoding='utf-8'))
    with np.load(path / 'basis.npz') as arrays:
        return ProfileBasis(arrays['depth_indices'].copy(), arrays['components'].copy(), arrays['eigenvalues'].copy(), manifest['retained_variance'], tuple(manifest['source_site_ids']), manifest['source_profile_count'])
