"""Retained inference component, consolidated from cpt_budget.v7.local_geometry (v0.8-engineering)."""
from __future__ import annotations
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

def simplex_weights(xy, query):
    """Nonnegative partition-of-unity weights, with explicit outside-hull rules."""
    xy, query = (np.asarray(xy, float), np.asarray(query, float))
    n = len(xy)
    if n == 0 or xy.shape != (n, 2) or query.shape != (2,):
        raise ValueError('Expected nonempty 2-D support and one query')
    distance = np.linalg.norm(xy - query, axis=1)
    if distance.min() < 1e-10:
        weights = (distance < 1e-10).astype(float)
        return (weights / weights.sum(), 'coincident')
    centered = xy - xy.mean(axis=0)
    if n < 3 or np.linalg.matrix_rank(centered, tol=1e-08) < 2:
        if n == 1 or np.max(np.linalg.norm(centered, axis=1)) < 1e-10:
            return (np.full(n, 1.0 / n), 'constant_degenerate')
        axis = np.linalg.svd(centered, full_matrices=False)[2][0]
        coordinate, q = (centered @ axis, (query - xy.mean(axis=0)) @ axis)
        order = np.argsort(coordinate, kind='stable')
        coordinate = coordinate[order]
        weights = np.zeros(n)
        if q <= coordinate[0]:
            weights[order[0]] = 1.0
        elif q >= coordinate[-1]:
            weights[order[-1]] = 1.0
        else:
            high = np.searchsorted(coordinate, q, side='right')
            t = (q - coordinate[high - 1]) / (coordinate[high] - coordinate[high - 1])
            weights[order[high - 1]], weights[order[high]] = (1.0 - t, t)
        return (weights, 'collinear_segment')
    triangulation = Delaunay(xy)
    simplex = int(triangulation.find_simplex(query, tol=1e-10))
    weights = np.zeros(n)
    if simplex >= 0:
        transform = triangulation.transform[simplex]
        first = transform[:2] @ (query - transform[2])
        barycentric = np.maximum(np.r_[first, 1.0 - first.sum()], 0.0)
        barycentric /= barycentric.sum()
        weights[triangulation.simplices[simplex]] = barycentric
        return (weights, 'inside_simplex')
    best_distance, best = (np.inf, None)
    for a, b in ConvexHull(xy).simplices:
        direction = xy[b] - xy[a]
        t = np.clip(np.dot(query - xy[a], direction) / np.dot(direction, direction), 0.0, 1.0)
        squared = float(np.sum((query - (xy[a] + t * direction)) ** 2))
        if squared < best_distance:
            best_distance, best = (squared, (a, b, t))
    a, b, t = best
    weights[a], weights[b] = (1.0 - t, t)
    return (weights, 'projected_hull_edge')
