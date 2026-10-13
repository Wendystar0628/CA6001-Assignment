"""Four frozen validation states; CPU-only GP restart/local-mean diagnosis."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import json, warnings, copy
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from cpt_budget.config import load_config
from cpt_budget.types import load_dataset
from cpt_budget.models.gp import GPPredictor
from cpt_budget.evaluate import evaluate
from cpt_budget.models.distribution import gaussian_prediction_frame

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
config = load_config(ROOT/'configs/main.yaml')
dataset = load_dataset(config)
states = pd.read_parquet(ROOT/'runs/main-20261013/states.parquet')
states = states[(states.role=='validation')&(states.fold_id==0)&(states.seed==17)&states.budget_n.isin([5,13])]
trend = {int(k):v for k,v in json.loads((ROOT/'runs/main-20261013/checkpoints/fold_0/gp_trend.json').read_text()).items()}
results = []
with threadpool_limits(limits=1):
    for state in states.itertuples():
        observed, query, truth = dataset.observed(state.observed_ids), dataset.query(state.evaluation_ids), dataset.truth(state.evaluation_ids)
        for variant, nrestart, localmean in [('default',0,False),('restart2',2,False),('restart2_localmean',2,True)]:
            cfg = copy.deepcopy(config)
            cfg['gp'] = {'n_restarts_optimizer': nrestart}
            # A local residual offset is learned only from revealed observations.
            offset = float((observed.log_qc - observed.depth_index.map(trend)).mean()) if localmean else 0.0
            model = GPPredictor(cfg, trend={k:v+offset for k,v in trend.items()})
            started = perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                model.condition(observed)
                pred = model.predict(query)
            metrics = evaluate(pred,truth,config)
            record = {'site':state.site_id,'budget':int(state.budget_n),'variant':variant,'source_role':'validation','seed':17,
                      'kernel':str(model.regressor.kernel_),'log_marginal_likelihood':float(model.regressor.log_marginal_likelihood_value_),
                      'offset':offset,'seconds':perf_counter()-started,
                      **{k:metrics[k] for k in ['log_mae','log_rmse','h_point_mae_m','recall','coverage_90']},
                      'warnings':[str(w.message) for w in caught]}
            results.append(record)
            print(json.dumps(record),flush=True)
(OUT/'gp_probe.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
