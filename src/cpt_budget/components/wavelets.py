"""Retained inference component, consolidated from cpt_budget.v7.multiscale (v0.8-engineering)."""
from __future__ import annotations
import numpy as np

def haar_forward(values, levels=3):
    """Orthonormal Haar: [coarsest approximation, coarse ... fine details]."""
    current = np.asarray(values, dtype=float)
    if current.shape[-1] % 2 ** levels or levels < 1:
        raise ValueError('Profile length must be divisible by 2**levels')
    details = []
    for _ in range(levels):
        left, right = (current[..., ::2], current[..., 1::2])
        details.append((left - right) / np.sqrt(2.0))
        current = (left + right) / np.sqrt(2.0)
    return np.concatenate([current, *reversed(details)], axis=-1)

def local_weights(distances, *, power=2.0, neighbours=None):
    distances = np.asarray(distances, dtype=float)
    keep = np.ones_like(distances, dtype=bool)
    if neighbours is not None:
        order = np.argsort(distances, axis=1, kind='stable')
        keep[:] = False
        np.put_along_axis(keep, order[:, :min(neighbours, distances.shape[1])], True, axis=1)
    weights = np.where(keep, np.maximum(distances, 1e-08) ** (-power), 0)
    zero = distances <= 1e-08
    weights = np.where(zero.any(axis=1, keepdims=True), zero.astype(float), weights)
    return weights / weights.sum(axis=1, keepdims=True)
