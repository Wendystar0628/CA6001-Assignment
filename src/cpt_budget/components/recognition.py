"""Retained inference component, consolidated from cpt_budget.v8.recognition (v0.8-engineering)."""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist, pdist
from cpt_budget.models.base import validate_observed
from cpt_budget.components.curve_distribution import thickness_decision, weighted_quantile
MAX_LOCAL_WEIGHT = 0.35
PROJECTION_WEIGHT = 0.5
VARIANTS = ('localize', 'amplitude', 'recall60')

def depth_stencil(n=24):
    """Unit-sum three-cell kernel; no extrapolation at the two depth edges."""
    d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    weight = np.where(d == 0, 0.5, np.where(d == 1, 0.25, 0.0))
    return weight / weight.sum(axis=1, keepdims=True)

class LocalEventEvidence:
    """Gaussian geometry/depth event evidence with whole-group LOO reliability.

Reliability measures whether spatial weighting predicts revealed events better
than a group-balanced depth climatology. No held-out query observation enters
this computation. Negative skill yields zero local correction.
"""

    def __init__(self, observed):
        self.observed = validate_observed(observed)
        frame = self.observed
        if 'near_group_id' not in frame or frame.near_group_id.isna().any():
            raise ValueError('Known support near groups are required')
        self.locations = sorted(frame.location_id.unique())
        curve = frame.pivot(index='location_id', columns='depth_index', values='log_qc')
        if list(curve.columns) != list(range(24)) or curve.isna().any().any():
            raise ValueError('Complete 24-cell support profiles are required')
        self.curves = curve.loc[self.locations].to_numpy(float)
        geometry = frame.drop_duplicates('location_id').set_index('location_id').loc[self.locations]
        self.xy = geometry[['x_m', 'y_m']].to_numpy(float)
        self.groups = geometry.near_group_id.to_numpy()
        self.group_size = geometry.near_group_id.map(geometry.near_group_id.value_counts()).to_numpy(float)
        distance = pdist(self.xy)
        positive = distance[distance > 1e-09]
        if not len(positive):
            raise ValueError('Independent support positions are required')
        self.length = float(np.median(positive))
        self.stencil = depth_stencil()
        self.events = (self.curves < 0.0).astype(float)
        self.smooth_events = self.events @ self.stencil.T
        pairwise = cdist(self.xy, self.xy)
        allowed = (self.groups[:, None] != self.groups[None, :]) & (pairwise > 1e-09)
        if not allowed.any(axis=1).all():
            raise ValueError('Whole-group LOO requires another independent support')
        spatial = np.exp(-0.5 * (pairwise / self.length) ** 2) / self.group_size[None, :]
        spatial *= allowed
        spatial /= spatial.sum(axis=1, keepdims=True)
        climatology = allowed / self.group_size[None, :]
        climatology /= climatology.sum(axis=1, keepdims=True)
        self.loo_local_probability = spatial @ self.smooth_events
        self.loo_reference_probability = climatology @ self.smooth_events
        unit_weight = 1.0 / self.group_size
        unit_weight /= unit_weight.sum()
        local_error = unit_weight @ (self.loo_local_probability - self.events) ** 2
        reference_error = unit_weight @ (self.loo_reference_probability - self.events) ** 2
        local_error = self.stencil @ local_error
        reference_error = self.stencil @ reference_error
        self.skill = np.clip(1.0 - (local_error + 1e-08) / (reference_error + 1e-08), 0.0, 1.0)
        self.local_weight = MAX_LOCAL_WEIGHT * self.skill
        self.loo_min_reference_count = int(allowed.sum(axis=1).min())

    def blend(self, probability, weights):
        p, w = (np.asarray(probability, float), np.asarray(weights, float))
        if p.shape != (24,) or w.shape != (len(self.locations),):
            raise ValueError('A full event profile and aligned support weights are required')
        local = w @ self.smooth_events
        return np.clip((1.0 - self.local_weight) * p + self.local_weight * local, 0.0, 1.0)

    def project_crossings(self, base, decision, weights):
        """Repair only newly crossed amplitudes using observed same-class values."""
        base, decision = (np.asarray(base, float), np.asarray(decision, float))
        output = decision.copy()
        changed = (base < 0.0) != (decision < 0.0)
        for depth in np.flatnonzero(changed):
            desired_low = decision[depth] < 0.0
            weight = np.asarray(weights, float)[:, None] * self.stencil[depth][None, :]
            selected = ((self.curves < 0.0) == desired_low) & (weight > 0.0)
            if selected.any():
                target = weighted_quantile(self.curves[selected], weight[selected])
                output[depth] = (1.0 - PROJECTION_WEIGHT) * decision[depth] + PROJECTION_WEIGHT * target
        return output

class FastRecognition:
    """Exact batched operator, or one of three fixed support-only refinements."""

    def __init__(self, variant='exact'):
        if variant not in ('exact', *VARIANTS):
            raise ValueError('Unknown recognition variant')
        self.variant = variant

    def condition(self, observed):
        from cpt_budget.components.joint import FastJointAdapter
        self.adapter = FastJointAdapter()
        self.adapter.condition(observed)
        self.evidence = None if self.variant == 'exact' else LocalEventEvidence(observed)
        return self

    def predict(self, query, base):
        result = self.adapter.batch_predict(query, base, kind='analog', shrinkage=0.5)
        if self.variant == 'exact':
            return result
        order = np.asarray(result['order'], int)
        bases = np.asarray(base, float)[order].reshape(-1, 24)
        probabilities = np.asarray(result['probability'], float)[order].reshape(-1, 24)
        points, local_probabilities, counts = ([], [], [])
        for index, current in enumerate(bases):
            weights = result['weights'][index]
            p = self.evidence.blend(probabilities[index], weights)
            count = int(result['count_median'][index])
            if self.variant == 'recall60':
                atom_count = (result['atoms'][index] < 0.0).sum(axis=1)
                count = int(weighted_quantile(atom_count, weights, 0.6))
            point = thickness_decision(current, p, count)
            if self.variant == 'amplitude':
                point = self.evidence.project_crossings(current, point, weights)
            points.append(point)
            local_probabilities.append(p)
            counts.append(count)
        output = dict(result)
        output['point'] = np.empty(len(query), float)
        output['point'][order] = np.asarray(points).reshape(-1)
        output['localizer_probability'] = np.empty(len(query), float)
        output['localizer_probability'][order] = np.asarray(local_probabilities).reshape(-1)
        output['chosen_count'] = np.asarray(counts)
        output['local_weight_mean'] = float(self.evidence.local_weight.mean())
        output['local_weight_max'] = float(self.evidence.local_weight.max())
        return output
