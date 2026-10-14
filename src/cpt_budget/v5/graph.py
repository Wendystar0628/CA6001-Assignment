"""Geometry-only transductive graph reconstruction of CPT profiles.

Two frozen hypotheses share the same product graph: quadratic harmonic
extension and smooth graph total variation (pseudo-Huber). Revealed full
profiles are exact boundary values, not noisy fit targets. Unrevealed
locations supply geometry only; no source or evaluation labels are used.

Theory: Zhu, Ghahramani & Lafferty, ICML 2003, Gaussian Fields and Harmonic
Functions; Jung et al., IEEE TSP 2019, graph total variation minimization.
The pseudo-Huber approximation and product-graph CPT construction here are
project adaptations, not claims to reproduce either paper verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import cdist

from ..config import PROJECT_ROOT, write_json
from ..models.base import validate_observed, validate_query_site
from ..models.distribution import point_prediction_frame
from ..types import PredictionBundle, QUERY_COLUMNS, load_dataset


PARAMETERS = dict(neighbors=4, vertical_weight=0.15, huber_delta=0.35,
                  max_iterations=40, tolerance=1e-7, minimum_edge_weight=1e-8)
SOURCES = [
    'https://mlg.eng.cam.ac.uk/zoubin/ssl.html',
    'https://arxiv.org/abs/1901.09838',
    'https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html',
]


def _geometry_only(frame):
    extra = set(frame) - set(QUERY_COLUMNS)
    if extra:
        raise ValueError(f'Graph input must be geometry-only; unexpected columns {sorted(extra)}')
    required = {'site_id', 'location_id', 'depth_index', 'z_center_m', 'x_m', 'y_m'}
    if not required.issubset(frame) or frame.empty:
        raise ValueError('Nonempty planned geometry with complete coordinates is required')
    if frame.duplicated(['site_id', 'location_id', 'depth_index']).any():
        raise ValueError('Duplicate planned location/depth rows')
    if not np.isfinite(frame[['x_m', 'y_m', 'z_center_m']].to_numpy(float)).all():
        raise ValueError('Planned coordinates must be finite')
    return frame.copy(deep=True)


def _horizontal_edges(xy, neighbors=4, minimum_weight=1e-8):
    """Symmetric kNN union Euclidean MST; ties resolved by sorted locations."""
    n = len(xy)
    if n < 2:
        return np.empty((0, 2), dtype=int), np.empty(0), 0.0
    distance = cdist(xy, xy)
    ranked = distance.copy()
    np.fill_diagonal(ranked, np.inf)
    edges = set()
    for i in range(n):
        for j in np.argsort(ranked[i], kind='stable')[:min(neighbors, n-1)]:
            edges.add(tuple(sorted((i, int(j)))))
    # Prim also preserves zero-length connections between coincident locations;
    # dense scipy MST would round tiny distances to zero (missing edges).
    inside = {0}
    while len(inside) < n:
        candidates = [(distance[i, j], i, j) for i in sorted(inside)
                      for j in range(n) if j not in inside]
        _, i, j = min(candidates)
        edges.add(tuple(sorted((i, j))))
        inside.add(j)
    edges = np.asarray(sorted(edges), dtype=int)
    length = distance[edges[:, 0], edges[:, 1]]
    nonzero = length[length > 0]
    scale = float(np.median(nonzero)) if len(nonzero) else 1.0
    weights = np.maximum(np.exp(-0.5*(length/scale)**2), minimum_weight)
    # Mean horizontal weighted degree = 1; gives vertical_weight a stable
    # interpretation when a site has a different number of planned positions.
    weights /= 2*weights.sum()/n
    return edges, weights, scale


@dataclass
class ProductGraph:
    grid: pd.DataFrame
    index: pd.MultiIndex
    edges: np.ndarray
    weights: np.ndarray
    horizontal_scale_m: float
    horizontal_edge_count: int


def _build_graph(plan, parameters):
    locations = plan[['location_id', 'x_m', 'y_m']].drop_duplicates()
    if locations.location_id.duplicated().any():
        raise ValueError('A planned location has inconsistent coordinates')
    locations = locations.sort_values('location_id', kind='stable')
    depths = plan[['depth_index', 'z_center_m']].drop_duplicates().sort_values('depth_index')
    if depths.depth_index.duplicated().any() or (np.diff(depths.z_center_m) <= 0).any():
        raise ValueError('Depth indices must correspond to consistent increasing depths')
    n, d = len(locations), len(depths)
    index = pd.MultiIndex.from_product([locations.location_id, depths.depth_index],
                                     names=['location_id', 'depth_index'])
    indexed = plan.set_index(['location_id', 'depth_index'])
    if set(index) != set(indexed.index):
        raise ValueError('All planned locations must use the same complete depth grid')
    grid = indexed.loc[index].reset_index()
    planar, pweights, scale = _horizontal_edges(locations[['x_m', 'y_m']].to_numpy(float),
                                               parameters['neighbors'],
                                               parameters['minimum_edge_weight'])
    edges, weights = [], []
    for (i, j), weight in zip(planar, pweights):
        for k in range(d):
            edges.append((i*d+k, j*d+k))
            weights.append(weight)
    spacing = np.diff(depths.z_center_m.to_numpy(float))
    reference_spacing = float(np.median(spacing)) if len(spacing) else 1.0
    for i in range(n):
        for k in range(d-1):
            edges.append((i*d+k, i*d+k+1))
            weights.append(parameters['vertical_weight']*reference_spacing/spacing[k])
    return ProductGraph(grid, index, np.asarray(edges, dtype=int).reshape(-1, 2),
                        np.asarray(weights), scale, len(planar))


def _laplacian(n, edges, weights):
    i, j = edges.T
    return coo_matrix((np.r_[weights, weights, -weights, -weights],
                       (np.r_[i, j, i, j], np.r_[i, j, j, i])), shape=(n, n)).tocsr()


def _boundary_solve(graph, known, values, edge_weights):
    unknown = np.setdiff1d(np.arange(len(graph.grid)), known)
    result = np.empty(len(graph.grid))
    result[known] = values
    if len(unknown):
        laplacian = _laplacian(len(graph.grid), graph.edges, edge_weights)
        result[unknown] = spsolve(laplacian[unknown][:, unknown].tocsc(),
                                 -(laplacian[unknown][:, known] @ values))
    if not np.isfinite(result).all():
        raise FloatingPointError('Graph boundary solve produced non-finite predictions')
    return result, unknown


class GraphProfilePredictor:
    """Shared predictor interface with an explicit transductive input boundary.

    planned_grid includes all allowed survey positions, including geometry of
    fixed evaluation locations. It must come from PreparedDataset.query(),
    never from the full grid. Labels are supplied exclusively by condition().
    """
    def __init__(self, planned_grid, *, variant='harmonic', config=None):
        if variant not in ('harmonic', 'huber'):
            raise ValueError('Expected harmonic or huber')
        self.model_id = f'graph_{variant}'
        self.variant = variant
        self.parameters = dict(PARAMETERS)
        self.plan = _geometry_only(planned_grid)
        self.observed = None
        self.graphs = {}
        self.context_manifest = None

    def condition(self, observed):
        observed = validate_observed(observed)
        site = observed.site_id.iloc[0]
        if site not in self.graphs:
            plan = self.plan.loc[self.plan.site_id.eq(site)]
            if plan.empty:
                raise ValueError('Observed site is absent from the explicit planned geometry')
            self.graphs[site] = _build_graph(plan, self.parameters)
        graph = self.graphs[site]
        keys = pd.MultiIndex.from_frame(observed[['location_id', 'depth_index']])
        known = graph.index.get_indexer(keys)
        if (known < 0).any():
            raise ValueError('Observed location/depth is absent from the plan')
        if not np.allclose(observed[['x_m', 'y_m', 'z_center_m']].to_numpy(float),
                           graph.grid.iloc[known][['x_m', 'y_m', 'z_center_m']].to_numpy(float),
                           atol=1e-9, rtol=0):
            raise ValueError('Observed coordinates disagree with planned geometry')
        counts = observed.groupby('location_id').depth_index.nunique()
        if not counts.eq(graph.grid.depth_index.nunique()).all():
            raise ValueError('The boundary must consist of complete revealed profiles')
        self.observed, self.graph, self.known = observed, graph, known
        self.values = observed.log_qc.to_numpy(float)
        self.context_manifest = {
            'model_id': self.model_id, 'parameters': self.parameters,
            'site_id': site, 'revealed_location_ids': sorted(observed.location_id.unique()),
            'planned_location_ids': sorted(graph.grid.location_id.unique()),
            'planned_input_columns': list(self.plan), 'planned_geometry_transductive': True,
            'label_source': 'Only condition(observed) log_qc; no source labels',
            'horizontal_edges': graph.horizontal_edge_count,
            'horizontal_scale_m': graph.horizontal_scale_m,
            'product_graph_nodes': len(graph.grid), 'product_graph_edges': len(graph.edges),
            'vertical_connections': 'Adjacent depth bins only; no inferred layer labels',
        }

    def predict(self, query_grid):
        _geometry_only(query_grid)
        validate_query_site(query_grid, self.observed)
        keys = pd.MultiIndex.from_frame(query_grid[['location_id', 'depth_index']])
        query_index = self.graph.index.get_indexer(keys)
        if (query_index < 0).any():
            raise ValueError('Query geometry must already belong to the explicit plan')
        if not np.allclose(query_grid[['x_m', 'y_m', 'z_center_m']].to_numpy(float),
                           self.graph.grid.iloc[query_index][['x_m', 'y_m', 'z_center_m']].to_numpy(float),
                           atol=1e-9, rtol=0):
            raise ValueError('Query coordinates disagree with planned geometry')
        g, p = self.graph, self.parameters
        values, unknown = _boundary_solve(g, self.known, self.values, g.weights)
        iterations, converged = 0, True
        objective_history = []
        if self.variant == 'huber':
            def energy(v):
                delta = v[g.edges[:, 0]] - v[g.edges[:, 1]]
                return float(np.sum(g.weights*p['huber_delta']**2 *
                                    (np.sqrt(1+(delta/p['huber_delta'])**2)-1)))
            objective_history = [energy(values)]
            converged = not len(unknown)
            for iteration in range(p['max_iterations'] if len(unknown) else 0):
                difference = values[g.edges[:, 0]] - values[g.edges[:, 1]]
                reweight = 1/np.sqrt(1+(difference/p['huber_delta'])**2)
                next_values, _ = _boundary_solve(g, self.known, self.values, g.weights*reweight)
                objective_history.append(energy(next_values))
                if objective_history[-1] > objective_history[-2] + 1e-9:
                    raise FloatingPointError('IRLS unexpectedly increased its convex objective')
                change = float(np.max(abs(next_values-values)))
                values, iterations = next_values, iteration+1
                if change <= p['tolerance']*max(1., float(np.max(abs(values)))):
                    converged = True
                    break
            difference = values[g.edges[:, 0]] - values[g.edges[:, 1]]
            final_weights = g.weights / np.sqrt(1+(difference/p['huber_delta'])**2)
        else:
            final_weights = g.weights
        gradient = _laplacian(len(g.grid), g.edges, final_weights) @ values
        self.solver_diagnostics = {
            'graph_iterations': iterations, 'graph_converged': converged,
            'graph_stationarity_max': float(np.max(abs(gradient[unknown]))) if len(unknown) else 0.,
            'graph_boundary_max_error': float(np.max(abs(values[self.known]-self.values))),
        }
        if objective_history:
            self.solver_diagnostics.update(graph_objective_start=objective_history[0],
                                           graph_objective_end=objective_history[-1])
        frame = point_prediction_frame(query_grid, values[query_index])
        for key, value in self.solver_diagnostics.items():
            frame[key] = value
        return PredictionBundle(frame, self.model_id)


def main():
    from ..diagnostics.common import diagnostic_config, source_folds, run_predictor
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(PROJECT_ROOT/'runs/v5-exploration-20261014/graph'))
    parser.add_argument('--variant', choices=['harmonic', 'huber', 'all'], default='all')
    args = parser.parse_args()
    config = diagnostic_config()
    config['run']['mode'] = 'exploratory_graph_reconstruction_v5'
    dataset = load_dataset(config)
    # Only validation-site geometry is made available to this experiment.
    # No outer test labels are accessed to construct the candidate or plan.
    validation_sites = sorted({site for fold in source_folds() for site in fold['validation_sites']})
    ids = dataset.locations().loc[lambda x: x.site_id.isin(validation_sites), 'location_id']
    planned_grid = dataset.query(ids)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'graph_protocol.json', {
        'parameters': PARAMETERS, 'sources': SOURCES,
        'candidates': ['graph_harmonic', 'graph_huber'],
        'selection': 'Fixed before validation; no graph/penalty search',
        'geometry_scope': 'All usable planned positions in validation sites, through query() only',
        'novelty': 'Global graph-boundary reconstruction with adjacent-depth edges',
        'limits': 'Geometry alone cannot reveal hidden geological interfaces; no native probability.',
    })
    for variant in ['harmonic', 'huber'] if args.variant == 'all' else [args.variant]:
        directory = output/f'graph_{variant}'
        result = run_predictor(lambda fid: GraphProfilePredictor(planned_grid, variant=variant),
                               directory, config=config)
        # The frozen-state executor retains its v3 defaults. Relabel only these
        # new artifacts; never change the executor or its old experiment files.
        for frame in result.values():
            frame['study_version'] = '0.5-exploration'
            frame['run_id'] = 'v5-exploration-20261014'
        result['predictions'].to_parquet(directory/'predictions.parquet', index=False)
        result['metrics'].to_csv(directory/'metrics.csv', index=False)
        for state_path in (directory/'states').glob('*.parquet'):
            frame = pd.read_parquet(state_path)
            frame['study_version'] = '0.5-exploration'
            frame['run_id'] = 'v5-exploration-20261014'
            frame.to_parquet(state_path, index=False)
        state_diagnostics = result['predictions'].groupby(
            ['fold_id', 'site_id', 'seed', 'budget_n'], as_index=False).first()
        columns = [c for c in state_diagnostics if c.startswith('graph_')]
        state_diagnostics[['fold_id', 'site_id', 'seed', 'budget_n', *columns]].to_csv(
            directory/'solver_diagnostics.csv', index=False)


if __name__ == '__main__':
    main()
