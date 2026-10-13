"""Isolated v0.2 CLI; the original cpt_budget entry point is unchanged."""
import argparse
import json
from pathlib import Path
from cpt_budget.config import load_config
from . import pipeline


def main(argv=None):
    parser=argparse.ArgumentParser(description='CPT v0.2 task-aware exploratory experiments')
    parser.add_argument('command',choices=['prepare','fit','validate-models','calibrate','select-models',
        'validate-policies','select-policies','test-selected','summarize','export','status','run'])
    parser.add_argument('--config')
    parser.add_argument('--run',dest='run_id')
    parser.add_argument('--fold',type=int)
    parser.add_argument('--models')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    config=load_config(args.config or 'configs/v2.yaml')
    if args.command in ['status','summarize','export'] and not args.config:
        output=Path(config['_root'])/config['project']['output_dir']
        args.run_id=args.run_id or json.loads((output/'latest.json').read_text())['run_id']
        stored=json.loads((output/args.run_id/'run.json').read_text(encoding='utf-8'))
        config={**stored['config'],'_root':config['_root']}
    models=args.models.split(',') if args.models else None
    if models:
        if set(models)-set(pipeline.MODEL_IDS):
            parser.error('Unknown model ID')
        config['v2']['prediction_models']=models
    if args.command=='prepare':
        result=pipeline.prepare(config)
    elif args.command=='status':
        result=pipeline._read(Path(config['_root'])/config['project']['output_dir']/args.run_id/'run.json')
    else:
        run=pipeline.open_run(config,args.run_id,resume=args.resume or args.command!='run')
        print(f'V2 run: {run}',flush=True)
        if args.command=='fit': result=pipeline.fit(config,run,args.fold,models)
        elif args.command=='validate-models': result=pipeline.validate_models(config,run,args.fold,models)
        elif args.command=='calibrate': result=pipeline.calibrate(config,run)
        elif args.command=='select-models': result=pipeline.select_models(config,run)
        elif args.command=='validate-policies': result=pipeline.validate_policies(config,run,args.fold)
        elif args.command=='select-policies': result=pipeline.select_policies(config,run)
        elif args.command=='test-selected': result=pipeline.validate_policies(config,run,args.fold,role='test')
        elif args.command=='summarize': result=pipeline.summarize(config,run)
        elif args.command=='export': result=pipeline.export(config,run)
        else: result=pipeline.run_all(config,run,args.fold,models)
    print(json.dumps(pipeline._plain(result),ensure_ascii=False,indent=2,default=str))


if __name__=='__main__':
    main()
