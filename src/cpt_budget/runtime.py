"""The thirteen retained model entries. Components are shared, not extra models."""
import copy
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import ndtr
from .experiment import diagnostic_config,model_asset,planned_geometry,catalog
from .types import PredictionBundle,QUERY_COLUMNS
from .models.base import validate_observed,validate_query_site
from .models.distribution import point_prediction_frame
from .components.stacking import load_basis,fit_weights
from .components.structural import ProfileEngine
from .components.joint import FastJointAdapter
from .components.haar import _FastCore,_SimplexGeometry,project_counts
from .components.recognition import FastRecognition

# Entries whose p_low_1 comes from the separate shared empirical risk head.
SHARED_RISK_ENTRIES=('balanced','thickness','recognition','count_shrink','thickness_guarded','tabpfn_joint')
# Point kinds and their Haar detail rule (thickness: v7 nearest; thickness_guarded: v8 median).
HAAR_DETAIL={'thickness':'nearest','thickness_guarded':'median'}

def _tabpfn_config(device=None):
    """Explicit TabPFN device; None keeps configs/main.yaml (auto: CUDA when available)."""
    config=diagnostic_config()
    if device is not None:
        if device not in ('cpu','cuda'):raise ValueError('TabPFN device must be cpu or cuda')
        if device=='cuda':
            import torch
            if not torch.cuda.is_available():raise RuntimeError('CUDA was requested for TabPFN but is unavailable; CPU fallback is not silent.')
        config=copy.deepcopy(config);config.setdefault('tabpfn',{})['device']=device
    return config

def _residual(fold_id,device=None):
    from .models.tabpfn_residual import FastGuardedPredictor
    return FastGuardedPredictor(_tabpfn_config(device),artifact=model_asset(fold_id,'tabpfn.ckpt'))

def _residual_values(residual,query):
    return residual.predict(query).frame.pred_log_qc.to_numpy(float)

class StructuralBase:
    def __init__(self,fold_id):
        self.basis=load_basis(fold_id)

    def condition(self,observed):
        self.observed=validate_observed(observed).reset_index(drop=True)
        if self.observed.site_id.isin(self.basis.source_site_ids).any():
            raise ValueError('Source FPCA contains target site')
        self.engine=ProfileEngine(self.observed,self.basis,quality_context=False)
        self.weights,_=fit_weights(self.observed,self.engine.loo()[:,:4],objective='structural')

    def values(self,query,stack_only=False):
        validate_query_site(query,self.observed)
        matrix=self.engine.query(query)
        stack=matrix[:,:4]@self.weights
        return stack if stack_only else .5*(matrix[:,4]+stack)

class EmpiricalDistribution:
    def condition(self,observed):
        self.context=FastJointAdapter().condition(observed)
        self.z=observed[['depth_index','z_center_m']].drop_duplicates().sort_values('depth_index').z_center_m.to_numpy(float)

    def probability(self,query,point):
        c=self.context
        w=-.5*(cdist(query[['x_m','y_m']].to_numpy(float),c.xy)/c.length)**2
        w=np.exp(w-w.max(axis=1,keepdims=True));w/=w.sum(axis=1,keepdims=True)
        d=-.5*((query.z_center_m.to_numpy(float)[:,None]-self.z[None,:])/.5)**2
        d=np.exp(d-d.max(axis=1,keepdims=True));d/=d.sum(axis=1,keepdims=True)
        p=ndtr((-point[:,None,None]-c.error_curves[None,:,:])/.15)
        return np.clip(np.einsum('ns,nd,nsd->n',w,d,p),0,1)

class TabCombination:
    """v6_graph_residual_low_union: guarded TabPFN residual + graph-Huber, low-label union."""
    def __init__(self,fold_id,device=None):
        from .components.graph import GraphProfilePredictor
        self.residual=_residual(fold_id,device)
        self.partner=GraphProfilePredictor(planned_geometry(),variant='huber')

    def initialize(self):
        self.residual.initialize()

    def condition(self,observed):
        self.residual.condition(observed);self.partner.condition(observed)

    def predict(self,query,residual=None):
        r=_residual_values(self.residual,query) if residual is None else residual
        p=self.partner.predict(query).frame.pred_log_qc.to_numpy(float)
        m=.5*(r+p);low=(r<0)|(p<0)
        y=np.where(low,np.minimum(m,-1e-9),np.maximum(m,0.))
        return PredictionBundle(point_prediction_frame(query,y),'tabpfn_graph')

class SharedRisk:
    """Native empirical risk head v6_empirical_graph_union_local (independent of the entry's point path)."""
    def __init__(self,fold_id,device=None):
        self.graph_union=TabCombination(fold_id,device)
        self.distribution=EmpiricalDistribution()

    def initialize(self):
        self.graph_union.initialize()

    def condition(self,observed):
        self.graph_union.condition(observed);self.distribution.condition(observed)

    def predict(self,query,residual=None):
        y=self.graph_union.predict(query,residual).frame.pred_log_qc.to_numpy(float)
        return self.distribution.probability(query,y)

class PointFamily:
    def __init__(self,fold_id,kind):
        self.kind=kind
        self.base=StructuralBase(fold_id) if kind in ('balanced','recognition','count_shrink') else None
        self.joint=FastJointAdapter() if kind in ('balanced','count_shrink') else None
        self.recognition=FastRecognition(variant='recall60') if kind=='recognition' else None

    def condition(self,observed):
        self.observed=validate_observed(observed)
        if self.base:self.base.condition(observed)
        if self.joint:self.joint.condition(observed)
        if self.recognition:self.recognition.condition(observed)
        if self.kind in ('simplex',*HAAR_DETAIL):
            profiles=self.observed.pivot(index='location_id',columns='depth_index',values='log_qc').sort_index()
            if list(profiles.columns)!=list(range(24)) or profiles.isna().any().any():
                raise ValueError('Complete 24-depth profiles required')
            profiles_array=profiles.to_numpy(float)
            xy=self.observed.drop_duplicates('location_id').set_index('location_id').loc[profiles.index,['x_m','y_m']].to_numpy(float)
            # Triangulation/hull are built once per revealed context (same v7 weight rules).
            self.core=(_FastCore(xy,profiles_array,detail=HAAR_DETAIL[self.kind]) if self.kind in HAAR_DETAIL
                       else _SimplexGeometry(xy,profiles_array))

    def predict(self,query):
        query=query[[c for c in QUERY_COLUMNS if c in query]].copy().reset_index(drop=True)
        validate_query_site(query,self.observed)
        if self.kind in ('balanced','count_shrink'):
            base=self.base.values(query)
            y=self.joint.batch_predict(query,base,policy='reference' if self.kind=='balanced' else 'count_shrink')['point']
        elif self.kind=='recognition':
            y=self.recognition.predict(query,self.base.values(query,stack_only=True))['point']
        else:
            locations=query.location_id.drop_duplicates()
            xy=query.drop_duplicates('location_id').set_index('location_id').loc[locations,['x_m','y_m']].to_numpy(float)
            if self.kind in HAAR_DETAIL:
                base,haar=self.core.predict(xy);values=project_counts(base,(haar<0).sum(axis=1))
            else:
                values=self.core.predict(xy)
            y=values[pd.Index(locations).get_indexer(query.location_id),query.depth_index.to_numpy(int)]
        return PredictionBundle(point_prediction_frame(query,y),self.kind)

class TabPFNJoint:
    """v7_pair_joint_simplex point: base = 0.5*simplex + 0.5*guarded TabPFN residual,
    then the complete-curve empirical-error decision (FastJointAdapter, reference policy).
    With ``residual=None`` at predict time the own TabPFN residual is used; the
    full path passes the residual already computed for the shared risk head."""
    def __init__(self,fold_id,device=None,*,own_residual=True):
        self.residual=_residual(fold_id,device) if own_residual else None
        self.simplex=PointFamily(fold_id,'simplex');self.joint=FastJointAdapter()

    def initialize(self):
        if self.residual is not None:self.residual.initialize()

    def condition(self,observed):
        self.observed=validate_observed(observed)
        if self.residual is not None:self.residual.condition(observed)
        self.simplex.condition(observed);self.joint.condition(observed)

    def predict(self,query,residual=None):
        query=query[[c for c in QUERY_COLUMNS if c in query]].copy().reset_index(drop=True)
        validate_query_site(query,self.observed)
        r=_residual_values(self.residual,query) if residual is None else np.asarray(residual,float)
        s=self.simplex.predict(query).frame.pred_log_qc.to_numpy(float)
        y=self.joint.batch_predict(query,.5*s+.5*r,'error',policy='reference')['point']
        return PredictionBundle(point_prediction_frame(query,y),'tabpfn_joint')

class WithRisk:
    def __init__(self,fold_id,kind,device=None):
        self.risk=SharedRisk(fold_id,device);self.model_id=kind
        # tabpfn_joint reuses the risk head's TabPFN residual: one TabPFN forward per query.
        self.point=TabPFNJoint(fold_id,own_residual=False) if kind=='tabpfn_joint' else PointFamily(fold_id,kind)

    def initialize(self):
        self.risk.initialize()

    def condition(self,observed):
        self.point.condition(observed);self.risk.condition(observed)

    def predict(self,query):
        if isinstance(self.point,TabPFNJoint):
            query=query[[c for c in QUERY_COLUMNS if c in query]].copy().reset_index(drop=True)
            residual=_residual_values(self.risk.graph_union.residual,query)
            frame=self.point.predict(query,residual).frame
            frame['p_low_1']=self.risk.predict(query,residual)
        else:
            frame=self.point.predict(query).frame
            frame['p_low_1']=self.risk.predict(query)
        return PredictionBundle(frame,self.model_id)

def entry(model_id):
    rows={r['id']:r for r in catalog()}
    if model_id not in rows:raise ValueError('Model is not in the retained catalogue')
    return rows[model_id]

def _build_model(model_id,fold_id=0,*,point_only=False,device=None):
    record=entry(model_id)
    if point_only and not record.get('point_only_supported',False):
        raise ValueError(f'{model_id} has no separate risk head; its output is already the point path, so --point-only does not apply.')
    config=diagnostic_config()
    if model_id=='constant':
        from .models.constant import TargetConstantMedianPredictor
        return TargetConstantMedianPredictor()
    if model_id=='idw':
        from .models.idw_fast import OptimizedIDWPredictor
        return OptimizedIDWPredictor(config)
    if model_id=='gp':
        from .models.gp import GPPredictor
        raw=json.loads(model_asset(fold_id,'gp_trend.json').read_text())
        return GPPredictor(config,trend={int(k):v for k,v in raw.items()})
    if model_id=='xgb':
        from .models.xgb_residual import ResidualPredictor
        return ResidualPredictor(config,model_asset(fold_id,'xgb.json'))
    if model_id=='tabpfn_graph':return TabCombination(fold_id,device)
    if model_id=='cnp':
        from .models.cnp import NeuralProcessPredictor
        return NeuralProcessPredictor(artifact=model_asset(fold_id,'cnp.pt'),device='cpu')
    if model_id=='simplex':return PointFamily(fold_id,'simplex')
    if point_only:
        return TabPFNJoint(fold_id,device) if model_id=='tabpfn_joint' else PointFamily(fold_id,model_id)
    if model_id in SHARED_RISK_ENTRIES:return WithRisk(fold_id,model_id,device)
    raise ValueError(f'No builder for catalogue entry {model_id}')

class RetainedModel:
    def __init__(self,model_id,predictor):
        self.model_id,self.predictor=model_id,predictor
    def initialize(self):
        """Load lazily-initialized weights (TabPFN) now; a no-op for other entries."""
        if hasattr(self.predictor,'initialize'):self.predictor.initialize()
    def condition(self,observed):
        return self.predictor.condition(observed)
    def predict(self,query):
        return PredictionBundle(self.predictor.predict(query).frame,self.model_id)

def make_model(model_id,fold_id=0,*,point_only=False,device=None):
    """``device`` applies to TabPFN components only (None: configs/main.yaml / auto)."""
    return RetainedModel(model_id,_build_model(model_id,fold_id,point_only=point_only,device=device))
