"""Validation-only estimator-count probe; frozen production configuration is untouched."""
from pathlib import Path
import gc
import json
import time
import pandas as pd
import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from cpt_budget.config import load_config
from cpt_budget.types import load_dataset
from cpt_budget.models.tabpfn_adapter import TabPFNPredictor, estimator_options
from cpt_budget.models.idw import IDWPredictor
from cpt_budget.evaluate import evaluate

root=Path(__file__).resolve().parents[2]
out=Path(__file__).resolve().parent/'ensemble'
out.mkdir(exist_ok=True)
config=load_config()
dataset=load_dataset(config)
run=root/'runs/main-20261013'
states=pd.read_parquet(run/'states.parquet')
states=states[(states.role=='validation')&(states.fold_id==0)&(states.seed==17)&
              (states.model_id=='gp')&(states.policy_id=='maximin')&states.budget_n.isin([5,13])]
assert len(states)==4 and states.site_id.nunique()==2
rows=[]
for state in states.sort_values(['site_id','budget_n']).itertuples(index=False):
    observed=dataset.observed(list(state.observed_ids))
    query=dataset.query(list(state.evaluation_ids))
    truth=dataset.truth(list(state.evaluation_ids))
    for model in ['idw','tabpfn','tabpfn_finetuned']:
        counts=[None] if model=='idw' else [1,4]
        for count in counts:
            start=time.perf_counter()
            if model=='idw':
                predictor=IDWPredictor(config)
            else:
                artifact=run/'checkpoints/fold_0/tabpfn_finetuned.ckpt' if model=='tabpfn_finetuned' else None
                predictor=TabPFNPredictor(config,artifact=artifact)
                options=estimator_options(config)
                options['n_estimators']=count
                if artifact is not None:
                    options['model_path']=artifact
                predictor._estimator=TabPFNRegressor.create_default_for_version(
                    ModelVersion.V2,fit_mode='fit_preprocessors',**options)
            torch.cuda.reset_peak_memory_stats()
            predictor.condition(observed)
            predictions=predictor.predict(query)
            metric=evaluate(predictions,truth,config)
            row={'site_id':state.site_id,'fold_id':0,'role':'validation','seed':17,
                 'budget_n':state.budget_n,'model_id':model,'n_estimators':count,
                 'seconds':time.perf_counter()-start,
                 'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),**metric}
            rows.append(row)
            pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
            print(state.site_id,state.budget_n,model,count,round(metric['log_mae'],4),flush=True)
            del predictor,predictions
            gc.collect()
            torch.cuda.empty_cache()
df=pd.DataFrame(rows)
metrics=['log_mae','h_point_mae_m','recall','coverage_90','width_90_log']
df.groupby(['model_id','n_estimators','budget_n'],dropna=False)[metrics].mean().to_csv(out/'macro.csv')
(out/'scope.json').write_text(json.dumps({'scope':'exploratory validation only','fold':0,'seed':17,
    'sites':states.site_id.unique().tolist(),'budgets':[5,13],
    'changes':'estimator count only, frozen original and fine-tuned v2 weights',
    'not_claimed':'new independent test evidence or 10 percent improvement'},indent=2),encoding='utf-8')
print(df.groupby(['model_id','n_estimators','budget_n'],dropna=False)[metrics].mean().round(4).to_string())
