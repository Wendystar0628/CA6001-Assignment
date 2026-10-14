"""Retained inference component, consolidated from cpt_budget.v8.haar (v0.8-engineering)."""
from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import ConvexHull, Delaunay
from cpt_budget.components.wavelets import haar_forward, local_weights
HAAR_MATRIX = haar_forward(np.eye(24))

def project_counts(base, count, margin=0.05):
    """Batched minimum-change projection with stable depth tie-breaking."""
    base = np.asarray(base, dtype=float)
    count = np.asarray(count, dtype=float)
    if base.ndim != 2 or count.shape != (len(base),) or (not np.isfinite(base).all()) or (not np.isfinite(count).all()) or (margin <= 0):
        raise ValueError('Finite profile rows and one finite count per row required')
    count = np.clip(np.floor(count + 0.5), 0, base.shape[1]).astype(int)
    order = np.argsort(base, axis=1, kind='stable')
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(base.shape[1]), order.shape), axis=1)
    low = rank < count[:, None]
    return np.where(low & (base >= 0), -margin, np.where(~low & (base < 0), margin, base))

class _SimplexGeometry:
    """Cache triangulation/hull once per revealed context, preserving v7 rules."""

    def __init__(self, xy, profiles):
        self.xy, inverse = np.unique(xy, axis=0, return_inverse=True)
        self.profiles = np.asarray([profiles[inverse == i].mean(axis=0) for i in range(len(self.xy))])
        self.center = self.xy.mean(axis=0)
        centered = self.xy - self.center
        self.degenerate = len(self.xy) < 3 or np.linalg.matrix_rank(centered, tol=1e-08) < 2
        self.constant = len(self.xy) == 1 or np.max(np.linalg.norm(centered, axis=1)) < 1e-10
        if self.degenerate and (not self.constant):
            self.axis = np.linalg.svd(centered, full_matrices=False)[2][0]
            self.coordinate = centered @ self.axis
            self.order = np.argsort(self.coordinate, kind='stable')
            self.coordinate = self.coordinate[self.order]
        elif not self.degenerate:
            self.triangulation = Delaunay(self.xy)
            self.edges = ConvexHull(self.xy).simplices
            self.a = self.xy[self.edges[:, 0]]
            self.delta = self.xy[self.edges[:, 1]] - self.a
            self.norm = np.sum(self.delta ** 2, axis=1)

    def predict(self, query_xy):
        query_xy = np.asarray(query_xy, dtype=float)
        weights = np.zeros((len(query_xy), len(self.xy)))
        distance = np.linalg.norm(query_xy[:, None] - self.xy[None], axis=2)
        zero = distance < 1e-10
        coincident = zero.any(axis=1)
        weights[coincident] = zero[coincident] / zero[coincident].sum(axis=1, keepdims=True)
        active = np.flatnonzero(~coincident)
        if not len(active):
            return weights @ self.profiles
        q = query_xy[active]
        if self.degenerate:
            if self.constant:
                weights[active] = 1 / len(self.xy)
            else:
                coordinate = (q - self.center) @ self.axis
                for i, x in zip(active, coordinate):
                    if x <= self.coordinate[0]:
                        weights[i, self.order[0]] = 1
                    elif x >= self.coordinate[-1]:
                        weights[i, self.order[-1]] = 1
                    else:
                        j = np.searchsorted(self.coordinate, x, side='right')
                        t = (x - self.coordinate[j - 1]) / (self.coordinate[j] - self.coordinate[j - 1])
                        weights[i, self.order[j - 1]], weights[i, self.order[j]] = (1 - t, t)
        else:
            simplex = self.triangulation.find_simplex(q, tol=1e-10)
            inside = simplex >= 0
            if inside.any():
                transform = self.triangulation.transform[simplex[inside]]
                bary = np.einsum('nij,nj->ni', transform[:, :2], q[inside] - transform[:, 2])
                bary = np.maximum(np.column_stack((bary, 1 - bary.sum(axis=1))), 0)
                bary /= bary.sum(axis=1, keepdims=True)
                weights[active[inside, None], self.triangulation.simplices[simplex[inside]]] = bary
            if (~inside).any():
                outside = q[~inside]
                t = np.clip(np.einsum('qei,ei->qe', outside[:, None] - self.a, self.delta) / self.norm, 0, 1)
                square = np.sum((outside[:, None] - (self.a + t[..., None] * self.delta)) ** 2, axis=2)
                best = np.argmin(square, axis=1)
                selected = t[np.arange(len(outside)), best]
                rows = active[~inside]
                weights[rows, self.edges[best, 0]], weights[rows, self.edges[best, 1]] = (1 - selected, selected)
        return weights @ self.profiles

class _FastCore:
    """Simplex base plus Haar-transferred profile used only for the low count.

    ``detail='median'`` is v8_haar_fast / v7 haar_krr_median (3NN inverse-square
    weighted-median detail coefficients). ``detail='nearest'`` is v7
    haar_krr_nearest: the 21 detail coefficients of the single geometry-nearest
    revealed profile (first index on ties). The coarse KRR and the exact-coordinate
    override are shared.
    """

    DETAILS = ('median', 'nearest')

    def __init__(self, xy, profiles, detail='median'):
        if detail not in self.DETAILS:
            raise ValueError(f'Unknown Haar detail rule: {detail}')
        self.detail = detail
        self.xy, self.profiles = (np.asarray(xy, float), np.asarray(profiles, float))
        self.simplex = _SimplexGeometry(self.xy, self.profiles)
        coefficients = self.profiles @ HAAR_MATRIX
        self.coarse_mean = coefficients[:, :3].mean(axis=0)
        distance = np.linalg.norm(self.xy[:, None] - self.xy[None], axis=2)
        pair = distance[np.triu_indices(len(xy), k=1)]
        pair = pair[pair > 1e-08]
        self.length_scale = float(np.median(pair)) if len(pair) else 1.0
        kernel = np.exp(-0.5 * (distance / self.length_scale) ** 2)
        self.coarse_fit = cho_solve(cho_factor(kernel + 0.1 * np.eye(len(xy)), check_finite=False), coefficients[:, :3] - self.coarse_mean, check_finite=False)
        detail = coefficients[:, 3:]
        self.detail_coefficients = detail
        self.detail_order = np.argsort(detail, axis=0, kind='stable')
        self.sorted_detail = np.take_along_axis(detail, self.detail_order, axis=0)

    def predict(self, xy):
        base = self.simplex.predict(xy)
        distance = np.linalg.norm(np.asarray(xy)[:, None] - self.xy[None], axis=2)
        coarse = self.coarse_mean + np.exp(-0.5 * (distance / self.length_scale) ** 2) @ self.coarse_fit
        if self.detail == 'nearest':
            detail = self.detail_coefficients[np.argmin(distance, axis=1)]
        else:
            weight = local_weights(distance, power=2.0, neighbours=3)
            cumulative = np.cumsum(weight[:, self.detail_order], axis=1)
            index = np.minimum((cumulative < 0.5).sum(axis=1), len(self.xy) - 1)
            detail = self.sorted_detail[index, np.arange(21)[None]]
        haar = np.column_stack((coarse, detail)) @ HAAR_MATRIX.T
        zero = distance <= 1e-08
        exact = zero.any(axis=1)
        if exact.any():
            haar[exact] = zero[exact] / zero[exact].sum(axis=1, keepdims=True) @ self.profiles
        return (base, haar)
