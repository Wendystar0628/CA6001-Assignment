"""Whole-profile graph matrix completion, graph curvature and local anomalies.

These candidates change the representation/regularizer rather than retuning the
v5 pseudo-Huber solver. Entire missing rows are not identifiable from low rank
alone: every method explicitly uses the geometry of all planned positions.
Only condition(observed) provides labels. Source papers motivate the operators;
the centered exact-boundary CPT objectives below are project adaptations.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from ..config import PROJECT_ROOT, write_json
from ..models.base import validate_query_site
from ..models.distribution import point_prediction_frame
from ..types import PredictionBundle, load_dataset
from ..v5.graph import GraphProfilePredictor, _geometry_only, _horizontal_edges, _laplacian

VARIANTS = ('nuclear', 'nuclear_gradient', 'biharmonic', 'lowrank_local')
PARAMETERS = dict(neighbors=4, nuclear_weight=.25, nuclear_vertical_weight=.05,
                  rho=1., max_iterations=1000, absolute_tolerance=1e-6,
                  relative_tolerance=1e-5, curvature_tension=.05, rank=3,
                  residual_neighbors=3, residual_idw_power=2.)
SOURCES = ['https://arxiv.org/abs/1408.1717',
           'https://arxiv.org/abs/0810.3286',
           'https://web.stanford.edu/~boyd/papers/admm_distr_stats.html']


def _difference(d):
    return np.diff(np.eye(d), axis=1)


def _svt(matrix, threshold):
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    return (u * np.maximum(s-threshold, 0.)) @ vt


def _extension(operator, known, boundary):
    missing = np.setdiff1d(np.arange(len(operator)), known)
    x = np.empty((len(operator), boundary.shape[1]))
    x[known] = boundary
    if len(missing):
        x[missing] = np.linalg.solve(operator[np.ix_(missing, missing)],
                                    -operator[np.ix_(missing, known)] @ boundary)
    return x


def nuclear_completion(laplacian, known, boundary, *, gradient=False, parameters=None):
    """Convex graph+nuclear objective solved by exact-boundary two-block ADMM.

    X is centered by the revealed depth-wise median before this function.
    Plain: .5 tr(X' L X) + .025 ||X D||_F^2 + .25 ||X||_*.
    Gradient: .5 tr(X' L X) + .25 ||X D||_*.
    Constraint X[known]=boundary. D is unscaled adjacent-bin difference.
    """
    p = PARAMETERS if parameters is None else parameters
    n, d = len(laplacian), boundary.shape[1]
    missing = np.setdiff1d(np.arange(n), known)
    difference = _difference(d)
    a = difference if gradient else np.eye(d)
    vertical = np.zeros((d, d)) if gradient else p['nuclear_vertical_weight']*(difference @ difference.T)
    rho = p['rho']
    x = _extension(laplacian, known, boundary)
    z, dual = x @ a, np.zeros((n, a.shape[1]))
    if not len(missing):
        return x, dict(iterations=0, converged=True, primal_residual=0., dual_residual=0.)
    # The row/column eigenvectors are fixed through ADMM; both blocks are tiny.
    le, lu = np.linalg.eigh(laplacian[np.ix_(missing, missing)])
    re, ru = np.linalg.eigh(vertical+rho*(a @ a.T))
    denominators = le[:, None]+re[None, :]
    if np.min(denominators) <= 0:
        raise ValueError('Revealed boundary must anchor the connected planned graph')
    boundary_rhs = -laplacian[np.ix_(missing, known)] @ boundary
    converged = False
    primal = dual_residual = np.inf
    for iteration in range(1, p['max_iterations']+1):
        rhs = boundary_rhs+rho*((z-dual) @ a.T)[missing]
        x[missing] = lu @ ((lu.T @ rhs @ ru)/denominators) @ ru.T
        x[known] = boundary
        ax = x @ a
        previous_z = z.copy()
        z = _svt(ax+dual, p['nuclear_weight']/rho)
        dual += ax-z
        primal = float(np.linalg.norm(ax-z))
        # Only unknown rows are optimization variables after elimination.
        dual_residual = float(rho*np.linalg.norm(((z-previous_z) @ a.T)[missing]))
        eps_primal = np.sqrt(ax.size)*p['absolute_tolerance']+p['relative_tolerance']*max(np.linalg.norm(ax), np.linalg.norm(z))
        eps_dual = np.sqrt(len(missing)*d)*p['absolute_tolerance']+p['relative_tolerance']*rho*np.linalg.norm((dual @ a.T)[missing])
        if primal <= eps_primal and dual_residual <= eps_dual:
            converged = True
            break
    objective = .5*np.sum(x*(laplacian @ x))+.5*np.sum((x @ vertical)*x)
    objective += p['nuclear_weight']*np.linalg.svd(x @ a, compute_uv=False).sum()
    return x, dict(iterations=iteration, converged=converged,
                   primal_residual=primal, dual_residual=dual_residual,
                   primal_tolerance=float(eps_primal), dual_tolerance=float(eps_dual),
                   objective=float(objective))


class GraphCompletionPredictor(GraphProfilePredictor):
    def __init__(self, planned_grid, *, variant='nuclear'):
        if variant not in VARIANTS:
            raise ValueError(f'Unknown graph completion variant: {variant}')
        super().__init__(planned_grid, variant='harmonic')
        self.completion_variant = variant
        self.model_id = f'v7_graph_{variant}'
        self.completion_parameters = dict(PARAMETERS)

    def condition(self, observed):
        super().condition(observed)
        g = self.graph
        locations = g.grid[['location_id', 'x_m', 'y_m']].drop_duplicates()
        depths = sorted(g.grid.depth_index.unique())
        ids = list(locations.location_id)
        known_ids = sorted(observed.location_id.unique())
        known = np.asarray([ids.index(i) for i in known_ids])
        values = observed.pivot(index='location_id', columns='depth_index', values='log_qc').loc[known_ids, depths].to_numpy(float)
        xy = locations[['x_m', 'y_m']].to_numpy(float)
        edges, weights, _ = _horizontal_edges(xy, neighbors=PARAMETERS['neighbors'])
        lap = _laplacian(len(ids), edges, weights).toarray()
        center = np.median(values, axis=0)
        y = values-center
        variant = self.completion_variant
        with threadpool_limits(limits=1):
            if variant.startswith('nuclear'):
                result, diagnostic = nuclear_completion(lap, known, y, gradient=variant == 'nuclear_gradient')
            else:
                operator = lap @ lap+PARAMETERS['curvature_tension']*lap
                if variant == 'biharmonic':
                    result = _extension(operator, known, y)
                    rank = int(np.linalg.matrix_rank(y))
                else:
                    u, s, vt = np.linalg.svd(y, full_matrices=False)
                    rank = min(PARAMETERS['rank'], len(s))
                    lowrank = (u[:, :rank]*s[:rank]) @ vt[:rank]
                    result = _extension(operator, known, lowrank)
                    distance = cdist(xy, xy[known])
                    weight = np.zeros_like(distance)
                    for row in range(len(xy)):
                        near = np.argsort(distance[row], kind='stable')[:min(PARAMETERS['residual_neighbors'], len(known))]
                        if distance[row, near[0]] <= 1e-12:
                            weight[row, near[0]] = 1.
                        else:
                            w = distance[row, near]**-PARAMETERS['residual_idw_power']
                            weight[row, near] = w/w.sum()
                    result += weight @ (y-lowrank)
                diagnostic = dict(iterations=1, converged=True, primal_residual=0., dual_residual=0., rank=rank)
            result += center
            result[known] = values
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite graph matrix completion')
        self.completed = result.reshape(-1)
        diagnostic['boundary_max_error'] = float(np.max(abs(result[known]-values)))
        diagnostic['matrix_rows'], diagnostic['matrix_columns'] = result.shape
        self.completion_diagnostics = {f'gmc_{key}': value for key, value in diagnostic.items()}
        self.context_manifest.update(parameters=PARAMETERS, variant=variant,
            center='Revealed-only depth median', label_source='Revealed target profiles only',
            full_missing_rows='Identified through planned spatial graph, not low rank alone')

    def predict(self, query_grid):
        _geometry_only(query_grid)
        validate_query_site(query_grid, self.observed)
        keys = pd.MultiIndex.from_frame(query_grid[['location_id', 'depth_index']])
        index = self.graph.index.get_indexer(keys)
        if (index < 0).any():
            raise ValueError('Query absent from planned geometry')
        if not np.allclose(query_grid[['x_m', 'y_m', 'z_center_m']].to_numpy(float),
                           self.graph.grid.iloc[index][['x_m', 'y_m', 'z_center_m']].to_numpy(float), atol=1e-9, rtol=0):
            raise ValueError('Query coordinates disagree with plan')
        frame = point_prediction_frame(query_grid, self.completed[index])
        for key, value in self.completion_diagnostics.items():
            frame[key] = value
        return PredictionBundle(frame, self.model_id)


def main():
    from ..diagnostics.common import diagnostic_config, source_folds, run_predictor
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(PROJECT_ROOT/'runs/v7-discovery-20261014/graph_completion'))
    parser.add_argument('--variant', choices=[*VARIANTS, 'all'], default='all')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = diagnostic_config()
    config['run']['mode'] = 'v7_graph_completion_exploration'
    dataset = load_dataset(config)
    sites = sorted({s for fold in source_folds() for s in fold['validation_sites']})
    ids = dataset.locations().loc[lambda x: x.site_id.isin(sites), 'location_id']
    plan = dataset.query(ids)
    write_json(output/'protocol.json', dict(parameters=PARAMETERS, candidates=[f'v7_graph_{v}' for v in VARIANTS],
        sources=SOURCES, scope='Exploratory frozen validation; no test labels or source fitting',
        objective_plain='.5 tr(X.T L X) + .025 ||X D||F^2 + .25 ||X||*',
        objective_gradient='.5 tr(X.T L X) + .25 ||X D||*',
        boundary='Exact revealed entire profiles; X centered by observed depth median',
        lowrank_local='Observed rank<=3 SVD: biharmonic lowrank extension plus 3-neighbor IDW residual',
        selection='Four mechanisms frozen before evaluation; no query truth parameter selection',
        limit='No native probability; posthoc calibration belongs to common evaluator'))
    for variant in VARIANTS if args.variant == 'all' else [args.variant]:
        directory = output/f'v7_graph_{variant}'
        result = run_predictor(lambda fid: GraphCompletionPredictor(plan, variant=variant), directory, config=config)
        for frame in result.values():
            frame['study_version'] = '0.7-discovery'
            frame['run_id'] = 'v7-discovery-20261014'
        result['predictions'].to_parquet(directory/'predictions.parquet', index=False)
        result['metrics'].to_csv(directory/'metrics.csv', index=False)
        diagnostics = result['predictions'].groupby(['fold_id', 'site_id', 'seed', 'budget_n'], as_index=False).first()
        columns = [c for c in diagnostics if c.startswith('gmc_')]
        diagnostics[['fold_id', 'site_id', 'seed', 'budget_n', *columns]].to_csv(directory/'solver_diagnostics.csv', index=False)
        for path in (directory/'states').glob('*.parquet'):
            frame = pd.read_parquet(path)
            frame['study_version'], frame['run_id'] = '0.7-discovery', 'v7-discovery-20261014'
            frame.to_parquet(path, index=False)


if __name__ == '__main__':
    main()
