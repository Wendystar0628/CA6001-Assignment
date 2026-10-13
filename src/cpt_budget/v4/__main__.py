"""python -m cpt_budget.v4: isolated residual study on the main branch."""
import argparse
import json
from cpt_budget.config import load_config
from . import pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'fit', 'evaluate', 'run', 'status'])
    parser.add_argument('--config', default='configs/v4.yaml')
    parser.add_argument('--run', default='v4-residual-20261014')
    parser.add_argument('--fold', type=int, choices=[0, 1, 2])
    parser.add_argument('--variant', choices=['crps', 'task'])
    args = parser.parse_args()
    config = load_config(args.config)
    run = pipeline.open_run(config, args.run)
    variants = [args.variant] if args.variant else None
    if args.command == 'status':
        meta = pipeline._read(run/'run.json')
        result = {key: meta.get(key) for key in ['run_id', 'status', 'active_training', 'failures']}
        result['completed_stages'] = list(meta['stages'])
    elif args.command == 'prepare':
        result = pipeline.prepare(config, run)
    elif args.command == 'fit':
        pipeline.prepare(config, run)
        summaries = pipeline.fit(config, run, args.fold, variants)
        result = [{k: s.get(k) for k in ['fold_id', 'variant', 'actual_steps', 'best_step', 'checkpoint_path']}
                  for s in summaries]
    elif args.command == 'evaluate':
        result = pipeline.evaluate(config, run, args.fold, variants)
    else:
        pipeline.prepare(config, run)
        pipeline.fit(config, run, args.fold, variants)
        result = pipeline.evaluate(config, run, args.fold, variants)
    print(json.dumps(pipeline._plain(result), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
