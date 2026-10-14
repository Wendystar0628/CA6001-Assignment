"""One small command surface for the retained project."""
import argparse,gc,json,sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .config import PROJECT_ROOT,load_config,write_json
from .experiment import catalog,iter_states,STATE_KEYS,ASSETS
from .runtime import make_model

# Native output columns that verify compares whenever the reference carries them.
NATIVE_COLUMNS=['p_low_1','q05_log','q10_log','q25_log','q75_log','q90_log','q95_log']

def _native_columns(name,old,new):
    """Strict schema check: a native column must neither disappear nor appear."""
    errors={}
    for column in NATIVE_COLUMNS:
        expected=column in old and old[column].notna().any()
        if expected and not old[column].notna().all():
            raise AssertionError(f'Reference {column} is partially missing for {name}')
        present=column in new and new[column].notna().any()
        if expected!=present:
            raise AssertionError(f'{name}: native column {column} '+('disappeared' if expected else 'appeared unexpectedly'))
        if expected:
            values=new[column].to_numpy(float)
            if not np.isfinite(values).all():raise AssertionError(f'{name}: non-finite {column}')
            errors[column]=float(np.abs(values-old[column].to_numpy(float)).max())
    return errors

def verify(models=None,folds=None,seeds=None,budgets=None,output='runs/current/verification'):
    rows_by_id={m['id']:m for m in catalog()}
    wanted=models or list(rows_by_id)
    reference=pd.read_parquet(ASSETS/'evaluation/reference_predictions.parquet')
    missing=sorted(set(wanted)-set(reference.model_id))
    if missing:raise AssertionError(f'Reference predictions lack catalogue entries: {missing}')
    states=list(iter_states(folds,seeds,budgets));rows=[]
    output=PROJECT_ROOT/output;output.mkdir(parents=True,exist_ok=True)
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
                if old.empty or not new.index.equals(old.index):raise AssertionError(f'Grid mismatch: {name}/{state.key}')
                if old.source_model_id.nunique()!=1 or old.source_model_id.iloc[0]!=rows_by_id[name]['source_model_id']:
                    raise AssertionError(f'Reference source_model_id differs from the catalogue: {name}')
                native=_native_columns(name,old,new)
                if ('p_low_1' in native)!=(rows_by_id[name]['risk_source']!='platt_on_point'):
                    raise AssertionError(f'{name}: p_low_1 presence contradicts risk_source {rows_by_id[name]["risk_source"]}')
                delta=np.abs(new.pred_log_qc.to_numpy()-old.pred_log_qc.to_numpy())
                changed=int(((new.pred_log_qc.to_numpy()<0)!=(old.pred_log_qc.to_numpy()<0)).sum())
                tolerance=1e-5 if name=='gp' else 1e-6
                passed=bool(delta.max()<tolerance and changed==0 and all(v<tolerance for v in native.values()))
                rows.append({'model_id':name,**state.metadata,'rows':len(new),'max_point_difference':float(delta.max()),
                             'label_changes':changed,'max_native_probability_difference':native.get('p_low_1'),
                             'max_interval_difference':max([v for k,v in native.items() if k!='p_low_1'],default=None),
                             'native_columns_checked':','.join(native),'passed':passed})
                print(f'{name} {state.key}: {delta.max():.3g} labels={changed} native={len(native)} pass={passed}',flush=True)
                if not passed:
                    pd.DataFrame(rows).to_csv(output/'checks.csv',index=False)
                    raise AssertionError(f'Retained implementation changed: {name}/{state.key}')
            pd.DataFrame(rows).to_csv(output/'checks.csv',index=False)
    import torch
    write_json(output/'summary.json',{'models':wanted,'states_per_model':len(states),'checks':len(rows),
        'passed':all(r['passed'] for r in rows),'label_changes':int(sum(r['label_changes'] for r in rows)),
        'max_point_difference':{m:float(max(r['max_point_difference'] for r in rows if r['model_id']==m)) for m in wanted},
        'native_columns_checked':{m:next(r['native_columns_checked'] for r in rows if r['model_id']==m) for m in wanted},
        'tabpfn_device':'cuda' if torch.cuda.is_available() else 'cpu',
        'comparison':'artifacts/evaluation/reference_predictions.parquet (archived v0.8 predictions; v0.9.1 ports frozen after the per-state equivalence check in results/current/port_equivalence.csv); same complete observed/query states. Native columns (p_low_1, GP quantiles) fail if they disappear or appear.'})
    return rows

def status():
    manifest=json.loads((ASSETS/'manifest.json').read_text(encoding='utf-8'))
    missing,size_mismatch=[],[]
    for record in manifest['files']:
        path=PROJECT_ROOT/record['path']
        if not path.is_file():missing.append(record['path'])
        elif 'bytes' in record and path.stat().st_size!=int(record['bytes']):
            size_mismatch.append({'path':record['path'],'manifest_bytes':int(record['bytes']),'actual_bytes':path.stat().st_size})
    config=load_config()
    prepared=PROJECT_ROOT/config['project']['prepared_dir']
    prepared_missing=[f'{prepared.name}/{n}.parquet' for n in ('profiles','grid','sites') if not (prepared/f'{n}.parquet').is_file()]
    result={'retained_models':len(catalog()),'missing_assets':missing,'size_mismatches':size_mismatch,
            'missing_prepared_data':prepared_missing,'state_document':'docs/IMPLEMENTATION_STATUS.md'}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if missing or size_mismatch or prepared_missing:raise SystemExit(1)
    return result

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
    p.add_argument('--output',default='runs/current/verification')
    p=sub.add_parser('evaluate',help='Recompute metrics/curves/site metrics/ranking incl. cross-fitted risk calibration.')
    p.add_argument('--source',choices=['reference','live'],default='reference');p.add_argument('--models',nargs='+')
    p.add_argument('--output',default='results/current')
    p=sub.add_parser('port-check',help='Compare per-state metrics with the archived v0.8-engineering per-state metrics.')
    p.add_argument('--models',nargs='+');p.add_argument('--output',default='results/current/port_equivalence.csv')
    p=sub.add_parser('benchmark',help='Uniform catalogue runtime benchmark (12 states, warm-up + repeats).')
    p.add_argument('--models',nargs='+');p.add_argument('--cpu-core',type=int,default=None)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--warmups',type=int,default=1)
    p.add_argument('--no-point-only',action='store_true');p.add_argument('--output',default='runs/current/benchmark')
    p.add_argument('--publish',metavar='DIR',help='Also copy the outputs to DIR/runtime_*.csv|json (e.g. results/current).')
    p=sub.add_parser('final-test',help='Pre-registered final outer test (docs/FINAL_TEST_PROTOCOL.md).')
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true',help='Test states, leakage audit, validation-only risk calibrators.')
    mode.add_argument('--blind',action='store_true',help='Predict all test states without reading test truth.')
    mode.add_argument('--freeze',action='store_true',help='Record SHA-256 hashes in results/final_test/frozen.json.')
    mode.add_argument('--unblind',action='store_true',help='Score the frozen blind predictions (once; hashes must match).')
    p.add_argument('--models',nargs='+');p.add_argument('--sites',nargs='+');p.add_argument('--seeds',nargs='+',type=int)
    p.add_argument('--budgets',nargs='+',type=int);p.add_argument('--output',help='--blind only: output directory.')
    p.add_argument('--force-rerun',action='store_true',help='--unblind only: allow a logged rerun.')
    p.add_argument('--force-refreeze',action='store_true',help='--freeze only: replace frozen.json before any unblinding.')
    p=sub.add_parser('predict');p.add_argument('--model',required=True,choices=[r['id'] for r in catalog()])
    p.add_argument('--fold',type=int,default=0);p.add_argument('--site');p.add_argument('--seed',type=int,default=17)
    p.add_argument('--budget',type=int,default=5)
    p.add_argument('--point-only',action='store_true',help='Skip the separate shared risk head (only entries with point_only_supported in the catalogue; others are already point/native paths and are rejected).')
    p.add_argument('--output',default='runs/current/prediction.csv')
    args=parser.parse_args()
    if args.command=='models':
        for row in catalog():print(f"{row['id']:18} {row['group']:10} {row['risk_source']:22} {row['name']}")
    elif args.command=='status':status()
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
    elif args.command=='verify':verify(args.models,args.folds,args.seeds,args.budgets,args.output)
    elif args.command=='evaluate':
        from .report import run
        run(args.source,args.models,args.output)
    elif args.command=='port-check':
        from .report import port_check
        port_check(args.models,args.output)
    elif args.command=='final-test':
        from . import final_test as ft
        subset=any(v is not None for v in (args.models,args.sites,args.seeds,args.budgets))
        if (args.output or subset) and not args.blind:raise SystemExit('--models/--sites/--seeds/--budgets/--output apply to --blind only')
        if args.force_rerun and not args.unblind:raise SystemExit('--force-rerun applies to --unblind only')
        if args.force_refreeze and not args.freeze:raise SystemExit('--force-refreeze applies to --freeze only')
        if args.prepare:ft.prepare()
        elif args.blind:
            if subset and not args.output:raise SystemExit('A subset blind run must use --output (the default directory holds the complete run)')
            ft.run_blind(args.models,args.sites,args.seeds,args.budgets,args.output)
        elif args.freeze:
            doc=ft.freeze(force=args.force_refreeze)
            print(json.dumps({k:v for k,v in doc.items() if k!='listings'},ensure_ascii=False,indent=2))
        else:
            result=ft.unblind(force_rerun=args.force_rerun)
            print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    elif args.command=='benchmark':
        from .benchmark import run_catalog
        run_catalog(args.models,args.output,cpu_core=args.cpu_core,repeats=args.repeats,warmups=args.warmups,
                    point_only=not args.no_point_only,publish=args.publish)
    else:
        state=next((s for s in iter_states([args.fold],[args.seed],[args.budget]) if args.site is None or s.metadata['site_id']==args.site),None)
        if state is None:raise ValueError('No matching registered state')
        model=make_model(args.model,args.fold,point_only=args.point_only)
        with threadpool_limits(limits=1):
            model.condition(state.observed);frame=model.predict(state.query).frame
        output=PROJECT_ROOT/args.output;output.parent.mkdir(parents=True,exist_ok=True)
        frame.to_csv(output,index=False);print(output)

if __name__=='__main__':main()
