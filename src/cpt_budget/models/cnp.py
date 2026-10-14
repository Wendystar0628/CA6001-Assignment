"""Retained inference component, consolidated from cpt_budget.v5.neural_process (v0.8-engineering)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch import nn
from cpt_budget.models.base import validate_observed, validate_query_site
from cpt_budget.models.distribution import point_prediction_frame
from cpt_budget.types import PredictionBundle
MODEL_ID = 'cnp_profile'

class ProfileCNP(nn.Module):
    """Query-relative set encoder and shared depth-conditioned decoder."""

    def __init__(self, n_depth: int=24, hidden_width: int=64, xy_scale_m: float=50.0):
        super().__init__()
        if n_depth < 1 or hidden_width < 1 or xy_scale_m <= 0:
            raise ValueError('Depth count, width and metric coordinate scale must be positive')
        self.n_depth = int(n_depth)
        self.hidden_width = int(hidden_width)
        self.xy_scale_m = float(xy_scale_m)
        h = self.hidden_width
        self.profile_encoder = nn.Sequential(nn.Linear(n_depth, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.relative_encoder = nn.Sequential(nn.Linear(h + 3, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(2 * h + 1, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, support_xy, support_profiles, query_xy, depth_positions):
        """Return [query profiles, depths], with no query measurement input."""
        if support_profiles.ndim != 2 or support_profiles.shape[1] != self.n_depth:
            raise ValueError('The context must contain complete fixed-grid profiles')
        if len(support_profiles) != len(support_xy) or not len(support_profiles):
            raise ValueError('Support coordinates and complete profiles must correspond')
        if len(depth_positions) != self.n_depth:
            raise ValueError('Depth coordinates must match the trained complete profile grid')
        if not len(query_xy):
            return support_profiles.new_empty((0, self.n_depth))
        encoded = self.profile_encoder(support_profiles)
        relative = (support_xy[None, :, :] - query_xy[:, None, :]) / self.xy_scale_m
        signed_xy = relative.sign() * relative.abs().log1p()
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True).log1p()
        inputs = torch.cat([encoded[None].expand(len(query_xy), -1, -1), signed_xy, distance], dim=-1)
        local = self.relative_encoder(inputs)
        pooled = torch.cat([local.mean(dim=1), local.amax(dim=1)], dim=-1)
        decoder_input = torch.cat([pooled[:, None, :].expand(-1, self.n_depth, -1), depth_positions[None, :, None].expand(len(query_xy), -1, -1)], dim=-1)
        return self.decoder(decoder_input).squeeze(-1)

def _profile_arrays(rows, depth_indices, *, measurements):
    """Read labels only from explicitly observed/source-training frames."""
    required = ['site_id', 'location_id', 'depth_index', 'x_m', 'y_m']
    if measurements:
        required.append('log_qc')
    missing = set(required) - set(rows)
    if missing:
        raise ValueError(f'Profile data missing {sorted(missing)}')
    if rows.empty or rows.site_id.nunique() != 1:
        raise ValueError('Profiles must belong to one nonempty site')
    if rows.duplicated(['location_id', 'depth_index']).any():
        raise ValueError('Duplicate profile/depth keys')
    ids = sorted(rows.location_id.unique().tolist())
    coordinates = []
    for location_id in ids:
        block = rows[rows.location_id.eq(location_id)]
        xy = block[['x_m', 'y_m']].to_numpy(float)
        if not np.isfinite(xy).all() or not np.all(xy == xy[0]):
            raise ValueError('Each profile must have one finite horizontal coordinate')
        coordinates.append(xy[0])
        if measurements and set(block.depth_index) != set(depth_indices):
            raise ValueError('Every observed/source profile must contain all trained depths')
    profiles = None
    if measurements:
        profiles = rows.pivot(index='location_id', columns='depth_index', values='log_qc').reindex(index=ids, columns=depth_indices).to_numpy(float)
        if not np.isfinite(profiles).all():
            raise ValueError('Complete profile measurements must be finite')
    return (ids, np.asarray(coordinates), profiles)

class NeuralProcessPredictor:
    model_id = MODEL_ID

    def __init__(self, config=None, artifact=None, *, device='cpu'):
        if artifact is None:
            raise ValueError('A source-trained CNP checkpoint is required')
        path = Path(artifact)
        if path.is_dir():
            path = path / 'best.pt'
        saved = torch.load(path, map_location=device, weights_only=False)
        self.metadata = saved['metadata']
        m = self.metadata
        self.device = device
        self.network = ProfileCNP(m['n_depth'], m['hidden_width'], m['xy_scale_m']).to(device)
        self.network.load_state_dict(saved['state_dict'])
        self.network.eval()
        self.depth_indices = m['depth_indices']
        self.z = torch.as_tensor(m['depth_positions'], dtype=torch.float32, device=device)
        self.observed = None

    def condition(self, observed):
        self.observed = validate_observed(observed)
        ids, xy, profiles = _profile_arrays(self.observed, self.depth_indices, measurements=True)
        self.origin = xy.mean(axis=0)
        self.support_xy = torch.as_tensor(xy - self.origin, dtype=torch.float32, device=self.device)
        self.support_profiles = torch.as_tensor((profiles - self.metadata['target_mean']) / self.metadata['target_scale'], dtype=torch.float32, device=self.device)
        self.context_manifest = {'model_id': self.model_id, 'observed_location_ids': ids, 'n_complete_profiles': len(ids), 'weights_updated_on_target': False, 'native_distribution': False}

    def predict(self, query_grid):
        validate_query_site(query_grid, self.observed)
        if query_grid.empty:
            return PredictionBundle(point_prediction_frame(query_grid, np.empty(0)), self.model_id)
        ids, xy, _ = _profile_arrays(query_grid, self.depth_indices, measurements=False)
        if not set(query_grid.depth_index) <= set(self.depth_indices):
            raise ValueError('Query depth is outside the trained grid')
        query_xy = torch.as_tensor(xy - self.origin, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            prediction = self.network(self.support_xy, self.support_profiles, query_xy, self.z).cpu().numpy()
        prediction = prediction * self.metadata['target_scale'] + self.metadata['target_mean']
        lookup = {(loc, depth): prediction[i, j] for i, loc in enumerate(ids) for j, depth in enumerate(self.depth_indices)}
        values = np.asarray([lookup[row.location_id, row.depth_index] for row in query_grid.itertuples(index=False)])
        return PredictionBundle(point_prediction_frame(query_grid, values), self.model_id)
