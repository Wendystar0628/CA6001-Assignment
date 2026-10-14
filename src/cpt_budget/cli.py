"""One small command surface for the retained project."""
import argparse,gc,json,sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .config import PROJECT_ROOT,load_config,write_json
from .experiment import catalog,iter_states,STATE_KEYS,ASSETS
from .runtime import make_model

def verify(models=None,folds=None,seeds=None,budgets=None):
    wanted=models or [m['id'] for m in catalog()]
    reference=pd.read_parquet(ASSETS/'evaluation/reference_predictions.parquet')
    states=list(iter_states(folds,seeds,budgets));rows=[]
    output=PROJECT_ROOT/'runs/current/verification';output.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        for name in wanted:
            predictor=None;last=None
            for state in states:
                if state.metadata['fold_id']!=last:
                    predictor=None;gc.collect()
                    predictor=make_model(name,state.metadata['fold_id']);last=state.metadata['fold_id']
                predictor.condition(state.observed);frame=predictor.predict(state.query).frame
                mask=reference.model_id.eq(name)
                for key in STATE_KEYS:mask &= reference[key].eq(state.metadata[key])
                old=reference.loc[mask].set_index(['location_id','depth_index']).sort_index()
                new=frame.set_index(['location_id','depth_index']).sort_index()
                if not new.index.equals(old.index):raise AssertionError(f'Grid mismatch: {name}/{state.key}')
                delta=np.abs(new.pred_log_qc.to_numpy()-old.pred_log_qc.to_numpy())
                changed=int(((new.pred_log_qc.to_numpy()<0)!=(old.pred_log_qc.to_numpy()<0)).sum())
                probability_error=None
                if 'p_low_1' in new and old.p_low_1.notna().all():
                    probability_error=float(np.abs(new.p_low_1.to_numpy()-old.p_low_1.to_numpy()).max())
                tolerance=1e-5 if name=='gp' else 1e-6
                passed=bool(delta.max()<tolerance and changed==0 and (probability_error is None or probability_error<tolerance))
                rows.append({'model_id':name,**state.metadata,'rows':len(new),'max_point_difference':float(delta.max()),
                             'label_changes':changed,'max_native_probability_difference':probability_error,'passed':passed})
                pd.DataFrame(rows).to_csv(output/'checks.csv',index=False)
                print(f'{name} {state.key}: {delta.max():.3g} labels={changed} pass={passed}',flush=True)
                if not passed:raise AssertionError(f'Retained implementation changed: {name}/{state.key}')
    write_json(output/'summary.json',{'models':wanted,'states_per_model':len(states),'checks':len(rows),
                                     'passed':all(r['passed'] for r in rows),'comparison':'Archived v0.8 predictions; same complete observed/query states.'})
    return rows

def main():
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('models');sub.add_parser('status')
    for command in ['restore-inputs','prepare','split']:
        p=sub.add_parser(command);p.add_argument('--config',default='configs/main.yaml')
    p=sub.add_parser('verify');p.add_argument('--models',nargs='+');p.add_argument('--folds',nargs='+',type=int)
    p.add_argument('--seeds',nargs='+',type=int);p.add_argument('--budgets',nargs='+',type=int)
    p=sub.add_parser('predict');p.add_argument('--model',required=True,choices=[r['id'] for r in catalog()])
    p.add_argument('--fold',type=int,default=0);p.add_argument('--site');p.add_argument('--seed',type=int,default=17)
    p.add_argument('--budget',type=int,default=5);p.add_argument('--point-only',action='store_true')
    p.add_argument('--output',default='runs/current/prediction.csv')
    args=parser.parse_args()
    if args.command=='models':
        for row in catalog():print(f"{row['id']:16} {row['group']:10} {row['name']}")
    elif args.command=='status':
        manifest=json.loads((ASSETS/'manifest.json').read_text(encoding='utf-8'))
        missing=[r['path'] for r in manifest['files'] if not (PROJECT_ROOT/r['path']).is_file()]
        print(json.dumps({'retained_models':len(catalog()),'missing_assets':missing,'state_document':'docs/IMPLEMENTATION_STATUS.md'},ensure_ascii=False,indent=2))
        if missing:raise SystemExit(1)
    elif args.command in ('restore-inputs','prepare','split'):
        config=load_config(args.config)
        if args.command=='restore-inputs':
            from .data.import_existing import restore_arrays
            result=restore_arrays(config)
        elif args.command=='prepare':
            from .data.normalize import prepare_data
            result=prepare_data(config)
        else:
            from .data.splits import make_splits
            from .types import load_dataset
            result=make_splits(load_dataset(config),config)
        print(result)
    elif args.command=='verify':verify(args.models,args.folds,args.seeds,args.budgets)
    else:
        state=next((s for s in iter_states([args.fold],[args.seed],[args.budget]) if args.site is None or s.metadata['site_id']==args.site),None)
        if state is None:raise ValueError('No matching registered state')
        model=make_model(args.model,args.fold,point_only=args.point_only)
        with threadpool_limits(limits=1):
            model.condition(state.observed);frame=model.predict(state.query).frame
        output=PROJECT_ROOT/args.output;output.parent.mkdir(parents=True,exist_ok=True)
        frame.to_csv(output,index=False);print(output)

if __name__=='__main__':main()
