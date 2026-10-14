"""Retained inference component, consolidated from cpt_budget.v8.structural (v0.8-engineering)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from cpt_budget.components.geometry import simplex_weights
MEMBERS = ('idw', 'functional_fpca_krr', 'registered_profile_shape', 'spatial_weighted_median', 'simplex_linear')

def fast_bounded_warp(reference, source, *, band_bins=2.0):
    """Same lattice, costs and tie order as v5 bounded_warp; vectorized states."""
    reference, source = (np.asarray(reference, float), np.asarray(source, float))
    if reference.ndim != 1 or reference.shape != source.shape or len(source) < 2:
        raise ValueError('Profiles must have identical one-dimensional shapes')
    if not np.isfinite(reference).all() or not np.isfinite(source).all() or band_bins <= 0:
        raise ValueError('Finite profiles and positive band required')
    n = len(reference)
    nodes = np.arange(2 * n - 1, dtype=float) / 2
    values = np.interp(nodes, np.arange(n), source)
    cost = np.full((n, len(nodes)), np.inf)
    back = np.full((n, len(nodes)), -1, dtype=int)
    cost[0, 0] = (reference[0] - source[0]) ** 2
    deltas = np.array([2, 1, 3])
    penalty = 0.05 * (deltas / 2 - 1) ** 2
    for j in range(1, n):
        eligible = np.flatnonzero(np.abs(nodes - j) <= band_bins + 1e-12)
        if j == n - 1:
            eligible = np.array([len(nodes) - 1])
        previous = eligible[:, None] - deltas
        local = (reference[j] - values[eligible]) ** 2 + 0.05 * ((nodes[eligible] - j) / band_bins) ** 2
        candidates = cost[j - 1, np.maximum(previous, 0)] + local[:, None] + penalty
        candidates[previous < 0] = np.inf
        choice = candidates.argmin(axis=1)
        cost[j, eligible] = candidates[np.arange(len(eligible)), choice]
        back[j, eligible] = previous[np.arange(len(eligible)), choice]
    if not np.isfinite(cost[-1, -1]):
        raise RuntimeError('No admissible registration path')
    mapping, k = (np.empty(n), len(nodes) - 1)
    for j in range(n - 1, -1, -1):
        mapping[j], k = (nodes[k], back[j, k])
    return mapping

class _SimplexContext:
    """Cache geometry but preserve v7 coincident/collinear/outside-hull rules."""

    def __init__(self, xy, values):
        from scipy.spatial import ConvexHull, Delaunay
        self.xy, inverse = np.unique(xy, axis=0, return_inverse=True)
        self.values = np.array([values[inverse == i].mean(axis=0) for i in range(len(self.xy))])
        centered = self.xy - self.xy.mean(axis=0)
        self.tri = self.hull = None
        if len(self.xy) >= 3 and np.linalg.matrix_rank(centered, tol=1e-08) >= 2:
            self.tri, self.hull = (Delaunay(self.xy), ConvexHull(self.xy).simplices)

    def predict(self, query):
        distance = np.linalg.norm(self.xy - query, axis=1)
        if self.tri is None or distance.min() < 1e-10:
            weights, _ = simplex_weights(self.xy, query)
        else:
            simplex = int(self.tri.find_simplex(query, tol=1e-10))
            weights = np.zeros(len(self.xy))
            if simplex >= 0:
                transform = self.tri.transform[simplex]
                first = transform[:2] @ (query - transform[2])
                barycentric = np.maximum(np.r_[first, 1.0 - first.sum()], 0.0)
                barycentric /= barycentric.sum()
                weights[self.tri.simplices[simplex]] = barycentric
            else:
                best_distance, best = (np.inf, None)
                for a, b in self.hull:
                    direction = self.xy[b] - self.xy[a]
                    t = np.clip(np.dot(query - self.xy[a], direction) / np.dot(direction, direction), 0.0, 1.0)
                    squared = float(np.sum((query - (self.xy[a] + t * direction)) ** 2))
                    if squared < best_distance:
                        best_distance, best = (squared, (a, b, t))
                a, b, t = best
                weights[a], weights[b] = (1.0 - t, t)
        return weights @ self.values

class ProfileEngine:
    """A conditioned support owns caches; no cache survives changed labels."""

    def __init__(self, observed, basis, *, quality_context=True):
        self.observed, self.basis = (observed, basis)
        self.quality_context = quality_context
        locations = observed[['location_id', 'x_m', 'y_m']].drop_duplicates().sort_values('location_id')
        if locations.location_id.duplicated().any():
            raise ValueError('Coordinates must be constant per location')
        self.ids = locations.location_id.to_numpy()
        self.xy = locations[['x_m', 'y_m']].to_numpy(float)
        self.depths = basis.depth_indices
        self.profiles = observed.pivot(index='location_id', columns='depth_index', values='log_qc').reindex(index=self.ids, columns=self.depths).to_numpy(float)
        if not np.isfinite(self.profiles).all():
            raise ValueError('Complete finite profiles on the source basis grid required')
        depth_rows = observed[['depth_index', 'z_center_m']].drop_duplicates().sort_values('depth_index')
        z = depth_rows.z_center_m.to_numpy(float)
        if len(z) != len(self.depths) or np.any(np.diff(z) <= 0) or (not np.allclose(np.diff(z), np.diff(z)[0])):
            raise ValueError('Common regular depth grid required')
        self.band_bins = 0.5 / np.diff(z)[0]
        self.distance = np.linalg.norm(self.xy[:, None] - self.xy[None, :], axis=2)
        self.matching = (self.profiles - self.profiles.mean(axis=1, keepdims=True)) / np.maximum(self.profiles.std(axis=1, keepdims=True), 0.25)
        self.match_distance = ((self.matching[:, None] - self.matching[None, :]) ** 2).mean(axis=2)
        self.warps, self.registered, self.contexts = ({}, {}, {})
        self.full = tuple(range(len(self.ids)))
        self.subsets, self.exclusions = ([], [])
        for i, location in enumerate(self.ids):
            excluded = self.distance[i] < 1e-08
            if 'near_group_id' in observed:
                groups = observed.loc[observed.location_id.eq(location), 'near_group_id'].dropna().unique()
                exclude_ids = observed.loc[observed.near_group_id.isin(groups), 'location_id'].unique()
                excluded |= np.isin(self.ids, exclude_ids)
            subset = tuple(np.flatnonzero(~excluded))
            self.subsets.append(subset)
            self.exclusions.append({'held_location_id': location, 'excluded_location_ids': self.ids[excluded].tolist(), 'n_remaining_locations': len(subset)})
        self._prepare_contexts(set(self.subsets) | {self.full})

    def _prepare_contexts(self, subsets):
        batches = {}
        for subset in sorted(subsets):
            if not subset:
                continue
            index = list(subset)
            values, xy = (self.profiles[index], self.xy[index])
            mean = values.mean(axis=0)
            scores = (values - mean) @ self.basis.components.T
            squared = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2)
            d = np.sqrt(squared[np.triu_indices(len(xy), k=1)])
            d = d[d > 1e-12]
            length = float(np.median(d)) if len(d) else 1.0
            matrix = np.exp(-squared / (2 * length ** 2)) + 0.1 * np.eye(len(xy))
            self.contexts[subset] = {'mean': mean, 'length': length, 'xy': xy, 'simplex': _SimplexContext(xy, values) if self.quality_context or subset == self.full else None}
            batches.setdefault(len(subset), []).append((subset, matrix, scores))
        for batch in batches.values():
            solved = np.linalg.solve(np.stack([x[1] for x in batch]), np.stack([x[2] for x in batch]))
            for (subset, _, _), coefficient in zip(batch, solved):
                self.contexts[subset]['coefficient'] = coefficient

    def _registered(self, nearest):
        key = tuple(sorted(map(int, nearest)))
        if key not in self.registered:
            medoid = key[int(np.argmin(self.match_distance[np.ix_(key, key)].sum(axis=1)))]
            identity = np.arange(len(self.depths), dtype=float)
            maps = []
            for index in key:
                pair = (medoid, index)
                if pair not in self.warps:
                    self.warps[pair] = identity if index == medoid else fast_bounded_warp(self.matching[medoid], self.matching[index], band_bins=self.band_bins)
                maps.append(self.warps[pair])
            maps = np.array(maps)
            amplitude = np.array([np.interp(mapping, identity, self.profiles[index]) for mapping, index in zip(maps, key)])
            self.registered[key] = (maps, amplitude)
        return (key, self.registered[key])

    def predict_one(self, subset, query, *, distance=None):
        if not subset:
            return np.zeros((len(self.depths), len(MEMBERS)))
        index = np.array(subset)
        context = self.contexts[subset]
        distance = np.linalg.norm(self.xy[index] - query, axis=1) if distance is None else distance[index]
        nearest = np.argsort(distance, kind='stable')[:3]
        near_indices = index[nearest]
        d, values = (distance[nearest], self.profiles[near_indices])
        coincident = d <= 1e-09
        if coincident.any():
            idw = values[coincident].mean(axis=0)
        else:
            weight = (d.min() / d) ** 2
            idw = weight @ values / weight.sum()
        squared = ((np.asarray(query)[None, :] - context['xy']) ** 2).sum(axis=1)
        scores = np.exp(-squared / (2 * context['length'] ** 2)) @ context['coefficient']
        functional = context['mean'] + scores @ self.basis.components
        if d[0] < 1e-08:
            registered = median = values[0]
        else:
            key, (maps, amplitude) = self._registered(near_indices)
            key_distance = np.linalg.norm(self.xy[list(key)] - query, axis=1)
            weight = key_distance ** (-2)
            weight /= weight.sum()
            registered = np.interp(np.arange(len(self.depths)), weight @ maps, weight @ amplitude)
            weight = d ** (-2)
            weight /= weight.sum()
            order = np.argsort(values, axis=0, kind='stable')
            sorted_values = np.take_along_axis(values, order, axis=0)
            selected = (np.cumsum(weight[order], axis=0) >= 0.5).argmax(axis=0)
            median = sorted_values[selected, np.arange(len(self.depths))]
        simplex = context['simplex'].predict(query) if context['simplex'] is not None else np.zeros(len(self.depths))
        return np.column_stack([idw, functional, registered, median, simplex])

    def loo(self):
        by_location = np.array([self.predict_one(subset, self.xy[i], distance=self.distance[i]) for i, subset in enumerate(self.subsets)])
        return self.to_rows(self.observed, by_location, self.ids)

    def query(self, query):
        geometry = query[['location_id', 'x_m', 'y_m']].drop_duplicates()
        if geometry.location_id.duplicated().any() or not np.isfinite(geometry[['x_m', 'y_m']].to_numpy(float)).all():
            raise ValueError('Each query needs one finite coordinate')
        predicted = np.array([self.predict_one(self.full, xy) for xy in geometry[['x_m', 'y_m']].to_numpy(float)])
        return self.to_rows(query, predicted, geometry.location_id.to_numpy())

    def to_rows(self, rows, profiles, ids):
        location_index = pd.Index(ids).get_indexer(rows.location_id)
        depth_index = pd.Index(self.depths).get_indexer(rows.depth_index)
        if (location_index < 0).any() or (depth_index < 0).any():
            raise ValueError('Unseen profile or depth index')
        return profiles[location_index, depth_index]
