"""Live point and empirical-risk pipelines; no frozen prediction replay."""
from functools import lru_cache
from time import perf_counter
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import ndtr
from cpt_budget.diagnostics.common import diagnostic_config,source_folds
from cpt_budget.types import load_dataset,PredictionBundle,QUERY_COLUMNS
from cpt_budget.models.distribution import point_prediction_frame
from cpt_budget.v5.graph import GraphProfilePredictor
from cpt_budget.v6.empirical_risk import support_error_frame,empirical_low_probability
from . import joint

@lru_cache(maxsize=1)
def planned_geometry():
    dataset=load_dataset(diagnostic_config())
    sites={s for f in source_folds() for s in f['validation_sites']}
    ids=dataset.locations().loc[lambda x:x.site_id.isin(sites),'location_id']
    return dataset.query(ids)

class FastEmpiricalDistribution:
    def condition(self,observed):
        self.context=joint.FastJointAdapter().condition(observed)
        self.z=observed[['depth_index','z_center_m']].drop_duplicates().sort_values('depth_index').z_center_m.to_numpy(float)
        return self

    def predict(self,query,point):
        context=self.context
        log_weight=-.5*(cdist(query[['x_m','y_m']].to_numpy(float),context.xy)/context.length)**2
        log_weight-=log_weight.max(axis=1,keepdims=True)
        weight=np.exp(log_weight);weight/=weight.sum(axis=1,keepdims=True)
        depth=-.5*((query.z_center_m.to_numpy(float)[:,None]-self.z[None,:])/.5)**2
        depth-=depth.max(axis=1,keepdims=True)
        depth=np.exp(depth);depth/=depth.sum(axis=1,keepdims=True)
        cdf=ndtr((-np.asarray(point)[:,None,None]-context.error_curves[None,:,:])/.15)
        return np.clip(np.einsum('ns,nd,nsd->n',weight,depth,cdf),0.,1.)

class EmpiricalRiskPipeline:
    """Recompute TabPFN/graph union plus the original support-error event head."""
    def __init__(self,fold_id,optimized=True):
        from . import tabpfn
        self.residual=(tabpfn.new_factory if optimized else tabpfn.old_factory)(fold_id)
        self.graph=GraphProfilePredictor(planned_geometry(),variant='huber')
        self.optimized=optimized
        self.model_id='v8_live_empirical_risk'
        self.stage_timings={}

    def initialize(self):
        self.residual.initialize()

    def condition(self,observed):
        self.observed=observed
        start=perf_counter()
        self.residual.condition(observed)
        after_residual=perf_counter()
        self.graph.condition(observed)
        after_graph=perf_counter()
        self.distribution=FastEmpiricalDistribution().condition(observed) if self.optimized else support_error_frame(observed)
        self.stage_timings={'risk_residual_condition_seconds':after_residual-start,
                            'risk_graph_condition_seconds':after_graph-after_residual,
                            'risk_error_condition_seconds':perf_counter()-after_graph}

    def predict(self,query):
        query=query[[c for c in QUERY_COLUMNS if c in query]].copy()
        start=perf_counter()
        residual=self.residual.predict(query).frame.pred_log_qc.to_numpy(float)
        after_residual=perf_counter()
        graph=self.graph.predict(query).frame.pred_log_qc.to_numpy(float)
        after_graph=perf_counter()
        mean=.5*(residual+graph)
        union=np.where((residual<0)|(graph<0),np.minimum(mean,-1e-9),np.maximum(mean,0.))
        probability=(self.distribution.predict(query,union) if self.optimized else
                     empirical_low_probability(self.distribution,query,union,mode='local')[0])
        self.stage_timings.update(risk_residual_predict_seconds=after_residual-start,
            risk_graph_predict_seconds=after_graph-after_residual,risk_cdf_seconds=perf_counter()-after_graph)
        frame=point_prediction_frame(query,union);frame['p_low_1']=probability
        return PredictionBundle(frame,self.model_id)

class BalancedPipeline:
    """Live leading point predictor, optionally including its true risk pipeline.

    Includes condition and prediction for source-basis structural stacking,
    whole-profile decisions, and (if requested) TabPFN/graph empirical risk.
    External validation fitting/calibration and I/O are not inference.
    """
    def __init__(self,fold_id,optimized=True,include_risk=False):
        self.point=(joint.new_factory if optimized else joint.old_factory)(fold_id)
        self.risk=EmpiricalRiskPipeline(fold_id,optimized) if include_risk else None
        self.model_id='v8_balanced_full' if include_risk else 'v8_balanced_point'
        self.stage_timings={}

    def initialize(self):
        if self.risk is not None:
            self.risk.initialize()

    def condition(self,observed):
        start=perf_counter()
        self.point.condition(observed)
        self.stage_timings={'point_condition_seconds':perf_counter()-start}
        if self.risk is not None:
            self.risk.condition(observed)
            self.stage_timings.update(self.risk.stage_timings)

    def predict(self,query):
        start=perf_counter()
        point=self.point.predict(query)
        frame=point.frame.copy()
        self.stage_timings['point_predict_seconds']=perf_counter()-start
        if self.risk is not None:
            risk=self.risk.predict(query)
            frame['p_low_1']=risk.frame.p_low_1.to_numpy(float)
            self.stage_timings.update(self.risk.stage_timings)
        return PredictionBundle(frame,self.model_id)

def balanced_point_old_factory(fold_id):
    return BalancedPipeline(fold_id,False,False)

def balanced_point_new_factory(fold_id):
    return BalancedPipeline(fold_id,True,False)

def balanced_full_old_factory(fold_id):
    return BalancedPipeline(fold_id,False,True)

def balanced_full_new_factory(fold_id):
    return BalancedPipeline(fold_id,True,True)

def benchmark_specs():
    from .benchmark import BenchmarkSpec
    return [BenchmarkSpec('balanced_point_old',balanced_point_old_factory,'complete_point_path_with_internal_joint_probability','balanced_point','old'),
            BenchmarkSpec('balanced_point_new',balanced_point_new_factory,'complete_point_path_with_internal_joint_probability','balanced_point','new'),
            BenchmarkSpec('balanced_full_old',balanced_full_old_factory,'complete_point_plus_native_empirical_risk_path','balanced_full','old','cuda'),
            BenchmarkSpec('balanced_full_new',balanced_full_new_factory,'complete_point_plus_native_empirical_risk_path','balanced_full','new','cuda')]
