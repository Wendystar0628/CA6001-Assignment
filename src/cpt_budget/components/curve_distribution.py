"""Retained inference component, consolidated from cpt_budget.v7.profile_distribution (v0.8-engineering)."""
from __future__ import annotations
import numpy as np
MARGIN = 0.05

def weighted_quantile(values, weights, q=0.5):
    """Lower inverse empirical CDF, one column at a time, no interpolation."""
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    scalar = values.ndim == 1
    if scalar:
        values = values[:, None]
    if values.ndim != 2 or weights.shape != (len(values),) or (not np.isfinite(values).all()) or (not np.isfinite(weights).all()) or (weights < 0).any() or (weights.sum() <= 0) or (not 0 < q < 1):
        raise ValueError('Finite atom values, nonnegative mass and interior quantile required')
    weights = weights / weights.sum()
    result = []
    for column in values.T:
        order = np.argsort(column, kind='stable')
        index = min(int(np.searchsorted(np.cumsum(weights[order]), q, side='left')), len(order) - 1)
        result.append(column[order[index]])
    result = np.asarray(result)
    return float(result[0]) if scalar else result

def thickness_decision(base, probability, count):
    """Fixed-count minimum expected Hamming loss, with minimum-change ties.

Select the largest marginal event probabilities. Among equal probabilities,
ascending base logqc minimizes total absolute modification under the frozen
+/-0.05 crossing rule. This is not a global minimum-change point estimator.
Only cells with a changed event label are modified.
"""
    base, probability = (np.asarray(base, float), np.asarray(probability, float))
    if base.ndim != 1 or probability.shape != base.shape or (not np.isfinite(base).all()) or (not np.isfinite(probability).all()) or ((probability < 0) | (probability > 1)).any() or (int(count) != count) or (not 0 <= count <= len(base)):
        raise ValueError('Finite point/event profiles and an integer in-range count required')
    rank = np.lexsort((np.arange(len(base)), base, -probability))
    low = np.zeros(len(base), bool)
    low[rank[:int(count)]] = True
    result = base.copy()
    result[low & (base >= 0.0)] = -MARGIN
    result[~low & (base < 0.0)] = MARGIN
    return result
