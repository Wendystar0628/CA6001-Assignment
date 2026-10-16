"""Truth-free adapters for the adaptive investigation experiment.

Source weights are immutable; only the explicitly revealed target context changes.
The five main point paths deliberately do not invoke the historical shared risk
head. GP uncertainty and model disagreement are ranking signals, not calibrated
failure probabilities.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtri

from .experiment import ASSETS, diagnostic_config, model_asset, planned_geometry
from .models.base import validate_observed, validate_query_site
from .models.distribution import point_prediction_frame
from .types import PredictionBundle, QUERY_COLUMNS

MAIN_MODEL_IDS = ('gp', 'xgb', 'cnp', 'tabpfn_joint', 'tabpfn_graph')
ABLATION_MODEL_IDS = ('tabpfn_residual_raw', 'tabpfn_residual_guarded',
                      'tabpfn_residual_unfinetuned', 'tabpfn_joint_unfinetuned')


def local_base_checkpoint() -> Path | None:
    """Find the already installed v2 weights; never download another generation."""
    roots = [os.environ.get('APPDATA'), os.environ.get('LOCALAPPDATA')]
    candidates = [Path(root) / 'tabpfn' / 'tabpfn-v2-regressor.ckpt' for root in roots if root]
    candidates.append(Path.home() / '.cache' / 'tabpfn' / 'tabpfn-v2-regressor.ckpt')
    return next((p for p in candidates if p.is_file()), None)


def fold_source_report(fold_id: int) -> dict:
    """Read training/selection provenance without opening any evaluation labels."""
    folds = json.loads((ASSETS / 'evaluation/folds.json').read_text(encoding='utf-8'))['folds']
    fold = next(f for f in folds if int(f['fold_id']) == int(fold_id))
    report = {'fold_id': int(fold_id), 'train_sites': fold['train_sites'],
              'validation_sites': fold['validation_sites'], 'test_sites': fold['test_sites'], 'models': {}}
    report['models']['gp'] = {'training_sites': fold['train_sites'], 'selection_sites': [],
                              'artifact': str(model_asset(fold_id, 'gp_trend.json'))}
    for model, name in [('xgb', 'xgb.json'), ('cnp', 'cnp_training.json'), ('tabpfn', 'tabpfn_training.json')]:
        artifact = model_asset(fold_id, name)
        metadata = json.loads(artifact.read_text(encoding='utf-8'))
        exposed = metadata.get('exposure', {}).get('per_site', {})
        train = sorted(set(metadata['train_sites']) | set(metadata.get('source_normalizer_sites', [])) | set(exposed))
        selection = sorted(set(metadata.get('validation_sites', [])))
        if not set(train) <= set(fold['train_sites']) or not set(selection) <= set(fold['validation_sites']):
            raise ValueError(f'{model}: artifact sources disagree with fold {fold_id}')
        report['models'][model] = {'training_sites': train, 'selection_sites': selection, 'artifact': str(artifact)}
    return report


def audit_fold_sources(fold_id: int, target_site: str, *, role: str = 'evaluation',
                       model_ids=MAIN_MODEL_IDS) -> dict:
    """Evaluation excludes both weight-training and checkpoint-selection exposure.

    Development may use that fold's validation sites, explicitly acknowledging
    their earlier role in early stopping. It never permits source-training sites.
    """
    if role not in ('development', 'evaluation'):
        raise ValueError('role must be development or evaluation')
    source = fold_source_report(fold_id)
    wanted = set('tabpfn' if m.startswith('tabpfn') else m for m in model_ids)
    leaks = {}
    for model in wanted:
        item = source['models'][model]
        if target_site in item['training_sites']:
            leaks[f'{model}.training_sites'] = target_site
        if role == 'evaluation' and target_site in item['selection_sites']:
            leaks[f'{model}.selection_sites'] = target_site
    if role == 'evaluation' and target_site not in source['test_sites']:
        leaks['fold.test_sites'] = 'Target is not an outer test site of this fold'
    return {'fold_id': int(fold_id), 'target_site': target_site, 'role': role,
            'passed': not leaks, 'leaks': leaks, 'sources': source}


def _geometry(query: pd.DataFrame) -> pd.DataFrame:
    """Discard labels even when a caller accidentally supplies a full data frame."""
    return query[[c for c in QUERY_COLUMNS if c in query]].copy().reset_index(drop=True)


class AdaptiveModelSuite:
    """Cache model initialization while refreshing the context every replay round.

    ``geometry`` freezes the graph's complete plan, optionally including an
    arbitrary complete-depth display grid. Graph query nodes must belong to it.
    ``predict`` accepts a model subset and returns PredictionBundles. Cells that
    have already been revealed return their measured values, so a fixed query
    grid can include locations revealed later in the trajectory.
    """
    def __init__(self, fold_id=0, model_ids=MAIN_MODEL_IDS, *, device=None,
                 role='evaluation', geometry=None, config=None):
        self.fold_id = int(fold_id)
        self.model_ids = tuple(dict.fromkeys(model_ids))
        unknown = set(self.model_ids) - set(MAIN_MODEL_IDS + ABLATION_MODEL_IDS)
        if unknown:
            raise ValueError(f'Unknown adaptive model IDs: {sorted(unknown)}')
        self.role, self.device = role, device
        self.config = copy.deepcopy(config or diagnostic_config())
        if device is not None:
            self.config.setdefault('tabpfn', {})['device'] = device
        self.geometry = None if geometry is None else _geometry(geometry)
        self.models, self.residuals = {}, {}
        self.observed = None

    def _ensure(self, model_id):
        if model_id in self.models:
            return
        from .runtime import make_model
        if model_id in ('gp', 'xgb', 'cnp'):
            self.models[model_id] = make_model(model_id, self.fold_id, device=self.device)
        elif model_id == 'tabpfn_graph':
            from .components.graph import GraphProfilePredictor
            plan = self.geometry if self.geometry is not None else planned_geometry()
            self.models[model_id] = GraphProfilePredictor(plan, variant='huber')
        elif model_id in ('tabpfn_joint', 'tabpfn_joint_unfinetuned'):
            from .runtime import TabPFNJoint
            self.models[model_id] = TabPFNJoint(self.fold_id, own_residual=False)
        else:
            self.models[model_id] = None
        if model_id.startswith('tabpfn'):
            variant = 'unfinetuned' if 'unfinetuned' in model_id else 'trained'
            if variant not in self.residuals:
                from .models.tabpfn_residual import FastGuardedPredictor
                artifact = local_base_checkpoint() if variant == 'unfinetuned' else model_asset(self.fold_id, 'tabpfn.ckpt')
                if artifact is None:
                    raise FileNotFoundError('Matched TabPFN v2 base weights are not installed locally')
                self.residuals[variant] = FastGuardedPredictor(self.config, artifact=artifact)
                self.residuals[variant].initialize()
                if self.observed is not None:
                    self.residuals[variant].condition(self.observed)
        model = self.models[model_id]
        if model is not None and hasattr(model, 'initialize'):
            model.initialize()
        if self.observed is not None and model is not None:
            model.condition(self.observed)

    def initialize(self):
        for model_id in self.model_ids:
            self._ensure(model_id)
        return self

    def condition(self, observed):
        observed = validate_observed(observed).reset_index(drop=True)
        check = audit_fold_sources(self.fold_id, str(observed.site_id.iloc[0]), role=self.role,
                                   model_ids=self.model_ids)
        if not check['passed']:
            raise ValueError(f'Source/target contamination: {check["leaks"]}')
        self.observed, self.source_audit = observed, check
        # Models initialized before this call must all get the new context.
        for model in self.models.values():
            if model is not None:
                model.condition(observed)
        for residual in self.residuals.values():
            residual.condition(observed)
        return self

    def predict(self, query, model_ids=None):
        query = _geometry(query)
        validate_query_site(query, self.observed)
        selected = self.model_ids if model_ids is None else tuple(model_ids)
        if not set(selected) <= set(self.model_ids):
            raise ValueError('Requested models must belong to this suite')
        for model_id in selected:
            self._ensure(model_id)
        index = pd.MultiIndex.from_frame(self.observed[['location_id', 'depth_index']])
        known_index = index.get_indexer(pd.MultiIndex.from_frame(query[['location_id', 'depth_index']]))
        known = known_index >= 0
        if known.any():
            columns = ['x_m', 'y_m', 'z_center_m']
            if not np.allclose(query.loc[known, columns].to_numpy(float), self.observed.iloc[known_index[known]][columns].to_numpy(float), atol=1e-9, rtol=0):
                raise ValueError('Revealed query geometry disagrees with the observation')
        hidden = query.loc[~known].reset_index(drop=True)
        residual_frames, output = {}, {}
        for model_id in selected:
            model = self.models[model_id]
            if hidden.empty:
                frame = point_prediction_frame(hidden, [])
            elif model_id.startswith('tabpfn'):
                variant = 'unfinetuned' if 'unfinetuned' in model_id else 'trained'
                if variant not in residual_frames:
                    residual_frames[variant] = self.residuals[variant].predict(hidden).frame
                residual = residual_frames[variant]
                values = residual.pred_log_qc.to_numpy(float)
                if model_id.startswith('tabpfn_joint'):
                    frame = model.predict(hidden, residual=values).frame
                elif model_id == 'tabpfn_graph':
                    graph = model.predict(hidden).frame.pred_log_qc.to_numpy(float)
                    mean = .5 * (values + graph)
                    result = np.where((values < 0) | (graph < 0), np.minimum(mean, -1e-9), np.maximum(mean, 0.))
                    frame = point_prediction_frame(hidden, result)
                else:
                    values = values if model_id.endswith('guarded') else residual.unguarded_pred_log_qc.to_numpy(float)
                    frame = point_prediction_frame(hidden, values)
            else:
                frame = model.predict(hidden).frame
            values = np.empty(len(query))
            values[known] = self.observed.iloc[known_index[known]].log_qc.to_numpy(float)
            values[~known] = frame.pred_log_qc.to_numpy(float)
            full = point_prediction_frame(query, values)
            if model_id == 'gp':
                # Existing GP frame stores exact Gaussian quantiles, including noise.
                std = np.zeros(len(query))
                if not hidden.empty:
                    std[~known] = (frame.q95_log.to_numpy(float) - frame.q05_log.to_numpy(float)) / (2 * ndtri(.95))
                full['std_log_qc'] = std
            full['is_revealed'] = known
            output[model_id] = PredictionBundle(full, model_id)
        return output

    def predict_auxiliary(self, query, primary_id):
        auxiliary = 'tabpfn_joint' if primary_id == 'xgb' else 'xgb'
        ids = tuple(dict.fromkeys(('gp', primary_id, auxiliary)))
        predictions = self.predict(query, ids)
        gp, primary, helper = (predictions[m].frame for m in ('gp', primary_id, auxiliary))
        rows = gp[['location_id', 'std_log_qc']].copy()
        rows['disagreement'] = (primary.pred_log_qc.to_numpy() < 0) != (helper.pred_log_qc.to_numpy() < 0)
        result = rows.groupby('location_id', sort=False).agg(gp_std_log_qc=('std_log_qc', 'mean'),
                                                          disagreement=('disagreement', 'mean')).reset_index()
        result.attrs['auxiliary_model_id'] = auxiliary
        result.attrs['ranking_only'] = True
        return result


class AdaptivePredictor:
    """Single-model replay API: initialize once; condition and predict every round."""
    def __init__(self, model_id, fold_id=0, **kwargs):
        self.model_id = model_id
        self.suite = AdaptiveModelSuite(fold_id, (model_id,), **kwargs)

    def initialize(self):
        self.suite.initialize()
        return self

    def condition(self, observed):
        self.suite.condition(observed)
        return self

    def predict(self, query) -> pd.DataFrame:
        return self.suite.predict(query)[self.model_id].frame


def make_predictor(model_id, fold_id=0, **kwargs):
    return AdaptivePredictor(model_id, fold_id, **kwargs)


get_predictor = make_predictor
