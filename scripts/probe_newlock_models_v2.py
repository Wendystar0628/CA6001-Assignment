"""Finite target-only adaptation probe on the previously inspected v1 splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from cpt_budget.adaptive_data import AdaptiveDataView
from cpt_budget.adaptive_policy import select_batch
from cpt_budget.adaptive_v2_models import V2_MODEL_IDS, make_v2_predictor
from cpt_budget.config import load_config
from cpt_budget.evaluate import evaluate
from cpt_budget.types import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[17, 29, 43])
    parser.add_argument('--budgets', type=int, nargs='+', default=[5, 13, 25, 41, 64])
    parser.add_argument('--models', nargs='+', default=list(V2_MODEL_IDS))
    parser.add_argument('--sites', nargs='+', default=['newlock'])
    parser.add_argument('--output', default='results/adaptive_v2_diagnostics')
    parser.add_argument('--tag', default='')
    args = parser.parse_args()
    config = load_config()
    dataset = load_dataset(config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    prefix = 'adaptation' + ('_' + args.tag if args.tag else '')
    splits = json.loads(Path('results/adaptive_v1/splits.json').read_text(encoding='utf-8'))
    rows, manifest = [], []
    models = {m: make_v2_predictor(m) for m in args.models}
    for split in splits:
        if split['site_id'] not in args.sites or split['seed'] not in args.seeds:
            continue
        view = AdaptiveDataView(dataset, split)
        checkpoints = sorted(set(min(n, split['n_full']) for n in args.budgets))
        for budget in checkpoints:
            while len(view.revealed_ids) < budget:
                remaining = sorted(set(split['full_ids']) - set(view.revealed_ids))
                selected, _ = select_batch(view.geometry(), view.revealed_ids, remaining,
                                          split['near_groups'], policy='maximin',
                                          batch_size=min(2, budget-len(view.revealed_ids)))
                view.reveal(selected)
            observed, query = view.observed(), view.query(split['audit_ids'])
            for model_id, model in models.items():
                model.fold_id = split['fold_id']
                model.condition(observed)
                pred = model.predict(query)
                # Audit curves are read only after adaptation and prediction.
                metrics = evaluate(pred, dataset.truth(split['audit_ids']), config)
                row = dict(split_id=split['split_id'], site_id=split['site_id'], seed=split['seed'],
                           budget_n=len(view.revealed_ids), n_full=split['n_full'], model_id=model_id,
                           log_mae=metrics['log_mae'], thickness_mae_m=metrics['h_point_mae_m'],
                           recall=metrics['recall'], fpr=metrics['false_positive_rate'],
                           absolute_quality_pass=metrics['log_mae'] <= .626458169250689 and metrics['h_point_mae_m'] <= .88125,
                           low_quality_pass=metrics['recall'] is not None and metrics['recall'] >= .5 and metrics['false_positive_rate'] <= .1,
                           selected_candidate=json.dumps(model.context_manifest['candidate'], sort_keys=True),
                           cv_log_mae=model.cv_metrics['log_mae'], cv_h_mae=model.cv_metrics['thickness_mae_m'])
                rows.append(row)
                manifest.append(dict(split_id=split['split_id'], budget_n=row['budget_n'], **model.context_manifest))
                print(json.dumps(row), flush=True)
            pd.DataFrame(rows).to_csv(output / f'{prefix}_probe.csv', index=False)
            (output / f'{prefix}_contexts.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    table = pd.DataFrame(rows)
    if len(table):
        full = table.loc[table.budget_n.eq(table.n_full), ['split_id', 'model_id', 'log_mae', 'thickness_mae_m']]
        full = full.rename(columns={'log_mae': 'full_log_mae', 'thickness_mae_m': 'full_thickness_mae_m'})
        table = table.merge(full, on=['split_id', 'model_id'], validate='many_to_one')
        table['relative_full_quality_pass'] = ((table.log_mae <= table.full_log_mae + .05) &
                                               (table.thickness_mae_m <= table.full_thickness_mae_m + .25))
        table['joint_quality_pass'] = table.absolute_quality_pass & table.relative_full_quality_pass
        table['all_quality_pass'] = table.joint_quality_pass & table.low_quality_pass
        table.to_csv(output / f'{prefix}_probe.csv', index=False)
    (output / f'{prefix}_protocol.json').write_text(json.dumps(dict(
        kind='post_unblinding_exploratory_diagnostic', states='original v1 geometry-only maximin',
        sites=args.sites, seeds=args.seeds, budgets=args.budgets, models=args.models,
        online_label_access='revealed near groups only; no audit or unrevealed candidate curves',
        source_weights_used=False, timing_compared=False,
        parameter_selection='finite candidates; revealed-group cross-validation only',
        quality=dict(log_mae_max=.626458169250689, thickness_mae_max_m=.88125,
                     relative_log_tolerance=.05, relative_thickness_tolerance_m=.25,
                     low_recall_min=.5, low_fpr_max=.1)), indent=2), encoding='utf-8')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
