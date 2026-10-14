"""The twelve retained model entries. Components are shared, not extra models."""
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
from .components.haar import _FastCore,project_counts
from .components.recognition import FastRecognition
from .components.geometry import simplex_weights

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
    def __init__(self,fold_id,kind):
        from .models.tabpfn_residual import FastGuardedPredictor
        from .components.functional import FunctionalProfilePredictor
        from .components.graph import GraphProfilePredictor
        self.kind=kind
        self.residual=FastGuardedPredictor(diagnostic_config(),artifact=model_asset(fold_id,'tabpfn.ckpt'))
        self.partner=(FunctionalProfilePredictor(load_basis(fold_id),method='krr') if kind=='fpca' else
                      GraphProfilePredictor(planned_geometry(),variant='huber'))

    def condition(self,observed):
        self.residual.condition(observed);self.partner.condition(observed)

    def predict(self,query):
        r=self.residual.predict(query).frame.pred_log_qc.to_numpy(float)
        p=self.partner.predict(query).frame.pred_log_qc.to_numpy(float)
        m=.5*(r+p);low=(r<0) if self.kind=='fpca' else ((r<0)|(p<0))
        y=np.where(low,np.minimum(m,-1e-9),np.maximum(m,0.))
        return PredictionBundle(point_prediction_frame(query,y),'tabpfn_'+self.kind)

class SharedRisk:
    def __init__(self,fold_id):
        self.graph_union=TabCombination(fold_id,'graph')
        self.distribution=EmpiricalDistribution()

    def condition(self,observed):
        self.graph_union.condition(observed);self.distribution.condition(observed)

    def predict(self,query):
        y=self.graph_union.predict(query).frame.pred_log_qc.to_numpy(float)
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
        if self.kind in ('thickness','simplex'):
            profiles=self.observed.pivot(index='location_id',columns='depth_index',values='log_qc').sort_index()
            if list(profiles.columns)!=list(range(24)) or profiles.isna().any().any():
                raise ValueError('Complete 24-depth profiles required')
            self.profiles=profiles.to_numpy(float)
            self.xy=self.observed.drop_duplicates('location_id').set_index('location_id').loc[profiles.index,['x_m','y_m']].to_numpy(float)
            if self.kind=='thickness':self.core=_FastCore(self.xy,self.profiles)
            else:
                self.xy,inverse=np.unique(self.xy,axis=0,return_inverse=True)
                self.profiles=np.asarray([self.profiles[inverse==i].mean(axis=0) for i in range(len(self.xy))])

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
            if self.kind=='thickness':
                base,haar=self.core.predict(xy);values=project_counts(base,(haar<0).sum(axis=1))
            else:
                weights=np.asarray([simplex_weights(self.xy,q)[0] for q in xy])
                values=weights@self.profiles
            y=values[pd.Index(locations).get_indexer(query.location_id),query.depth_index.to_numpy(int)]
        return PredictionBundle(point_prediction_frame(query,y),self.kind)

class WithRisk:
    def __init__(self,fold_id,kind):
        self.point=PointFamily(fold_id,kind);self.risk=SharedRisk(fold_id);self.model_id=kind

    def condition(self,observed):
        self.point.condition(observed);self.risk.condition(observed)

    def predict(self,query):
        frame=self.point.predict(query).frame
        frame['p_low_1']=self.risk.predict(query)
        return PredictionBundle(frame,self.model_id)

def _build_model(model_id,fold_id=0,*,point_only=False):
    if model_id not in {r['id'] for r in catalog()}:raise ValueError('Model is not in the retained catalogue')
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
    if model_id.startswith('tabpfn_'):return TabCombination(fold_id,model_id.split('_')[1])
    if model_id=='cnp':
        from .models.cnp import NeuralProcessPredictor
        return NeuralProcessPredictor(artifact=model_asset(fold_id,'cnp.pt'),device='cpu')
    if model_id=='simplex' or point_only:return PointFamily(fold_id,model_id)
    return WithRisk(fold_id,model_id)

class RetainedModel:
    def __init__(self,model_id,predictor):
        self.model_id,self.predictor=model_id,predictor
    def condition(self,observed):
        return self.predictor.condition(observed)
    def predict(self,query):
        return PredictionBundle(self.predictor.predict(query).frame,self.model_id)

def make_model(model_id,fold_id=0,*,point_only=False):
    return RetainedModel(model_id,_build_model(model_id,fold_id,point_only=point_only))
