"""The single retained evaluation protocol and explicit label boundary."""
from dataclasses import dataclass
import json
import pandas as pd
from .config import PROJECT_ROOT,load_config
from .types import load_dataset

STATE_KEYS=['fold_id','site_id','seed','budget_n']
BUDGETS=(5,7,9,11,13)
KEYS=STATE_KEYS+['location_id','depth_index']
ASSETS=PROJECT_ROOT/'artifacts'

def diagnostic_config():
    return load_config()

def catalog():
    return json.loads((PROJECT_ROOT/'configs/catalog.json').read_text(encoding='utf-8'))

def model_asset(fold_id,name):
    path=ASSETS/'models'/f'fold_{int(fold_id)}'/name
    if not path.is_file():
        raise FileNotFoundError(f'Required retained asset is missing: {path}. See docs/RUN_GUIDE.md.')
    return path

@dataclass
class EvaluationState:
    metadata:dict
    observed:pd.DataFrame
    query:pd.DataFrame
    truth:pd.DataFrame

    @property
    def key(self):
        m=self.metadata
        return f"f{m['fold_id']}_{m['site_id']}_s{m['seed']}_n{m['budget_n']}"

def iter_states(folds=None,seeds=None,budgets=None):
    dataset=load_dataset(diagnostic_config())
    states=pd.read_parquet(ASSETS/'evaluation/states.parquet')
    if len(states)!=60 or states.duplicated(STATE_KEYS).any() or not states.role.eq('validation').all():
        raise ValueError('Invalid retained validation manifest')
    for column,values in [('fold_id',folds),('seed',seeds),('budget_n',budgets)]:
        if values is not None:states=states[states[column].isin(values)]
    for row in states.sort_values(STATE_KEYS).itertuples(index=False):
        observed=dataset.observed(list(row.observed_ids));query=dataset.query(list(row.evaluation_ids))
        if set(observed.near_group_id)&set(query.near_group_id):
            raise ValueError('Support/query near groups overlap')
        meta={k:getattr(row,k) for k in STATE_KEYS}
        meta.update(observed_set_id=row.observed_set_id,investigation_group=row.investigation_group,
                    policy_id='maximin',role='evaluation',split_role='validation')
        yield EvaluationState(meta,observed,query,dataset.truth(list(row.evaluation_ids)))

def planned_geometry():
    dataset=load_dataset(diagnostic_config())
    sites=pd.read_parquet(ASSETS/'evaluation/states.parquet').site_id.unique()
    return dataset.query(dataset.locations().loc[lambda f:f.site_id.isin(sites),'location_id'])
