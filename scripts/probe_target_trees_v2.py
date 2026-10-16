"""Post-unblind finite target-tree probe; audit labels only score final predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from cpt_budget.adaptive_data import AdaptiveDataView
from cpt_budget.adaptive_policy import select_batch
from cpt_budget.adaptive_v2_target_trees import TARGET_TREE_MODEL_IDS, make_target_tree_predictor
from cpt_budget.config import load_config
from cpt_budget.evaluate import evaluate
from cpt_budget.types import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[17, 29, 43])
    parser.add_argument('--budgets', type=int, nargs='+', default=[5, 13, 25, 41, 64])
    parser.add_argument('--models', nargs='+', default=[name for name in TARGET_TREE_MODEL_IDS
        if not name.startswith('target_et_median_') or name == 'target_et_median_count_cv'])
    parser.add_argument('--output', default='results/adaptive_v2_diagnostics')
    parser.add_argument('--output-prefix', default='target_trees')
    args = parser.parse_args()
    config = load_config()
    dataset = load_dataset(config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    splits = json.loads(Path('results/adaptive_v1/splits.json').read_text(encoding='utf-8'))
    rows, contexts = [], []
    for split in splits:
        if split['site_id'] != 'newlock' or split['seed'] not in args.seeds:
            continue
        view = AdaptiveDataView(dataset, split)
        checkpoints = sorted(set(min(n, split['n_full']) for n in args.budgets))
        for budget in checkpoints:
            while len(view.revealed_ids) < budget:
                remaining = sorted(set(split['full_ids']) - set(view.revealed_ids))
                selected, _ = select_batch(view.geometry(), view.revealed_ids, remaining,
                    split['near_groups'], policy='maximin', batch_size=min(2, budget-len(view.revealed_ids)))
                view.reveal(selected)
            observed, query = view.observed(), view.query(split['audit_ids'])
            joint = (make_target_tree_predictor('target_tree_cv', split['fold_id']).initialize().condition(observed)
                if any(name in ('target_tree_cv','target_tree_log_cv') for name in args.models) else None)
            for model_id in args.models:
                if model_id == 'target_tree_cv':
                    model = joint
                elif model_id == 'target_tree_log_cv':
                    # Reuse crossfits while making the different selection explicit.
                    selected = min(joint.cv_candidates, key=lambda c: c['log_mae'])['name']
                    model = make_target_tree_predictor(selected, split['fold_id']).condition(observed)
                else:
                    model = make_target_tree_predictor(model_id, split['fold_id']).condition(observed)
                prediction = model.predict(query)
                # No target audit truth is accessed until the fitted model has predicted.
                metrics = evaluate(prediction, dataset.truth(split['audit_ids']), config)
                row = dict(split_id=split['split_id'], site_id=split['site_id'], seed=split['seed'],
                    budget_n=len(view.revealed_ids), n_full=split['n_full'], model_id=model_id,
                    selected_candidate=model.selected_candidate.name,
                    log_mae=metrics['log_mae'], thickness_mae_m=metrics['h_point_mae_m'],
                    recall=metrics['recall'], fpr=metrics['false_positive_rate'],
                    absolute_quality_pass=metrics['log_mae'] <= .626458169250689 and metrics['h_point_mae_m'] <= .88125,
                    low_quality_pass=metrics['recall'] is not None and metrics['recall'] >= .5 and metrics['false_positive_rate'] <= .1,
                    cv_log_mae=model.cv_metrics['log_mae'], cv_h_mae=model.cv_metrics['thickness_mae_m'])
                rows.append(row)
                print(json.dumps(row), flush=True)
                if model_id in ('target_tree_cv', 'target_et_median_count_cv', 'target_tree_augmented_cv'):
                    contexts.append(dict(split_id=split['split_id'], budget_n=len(view.revealed_ids), **model.context_manifest))
            pd.DataFrame(rows).to_csv(output/(args.output_prefix+'_probe.csv'), index=False)
            (output/(args.output_prefix+'_contexts.json')).write_text(json.dumps(contexts, indent=2), encoding='utf-8')
    frame = pd.DataFrame(rows)
    full = frame[frame.budget_n == frame.n_full][['split_id', 'model_id', 'log_mae', 'thickness_mae_m']]
    full = full.rename(columns={'log_mae':'full_log_mae', 'thickness_mae_m':'full_thickness_mae_m'})
    frame = frame.merge(full, on=['split_id', 'model_id'], how='left', validate='many_to_one')
    frame['relative_quality_pass'] = ((frame.log_mae <= frame.full_log_mae+.05) &
        (frame.thickness_mae_m <= frame.full_thickness_mae_m+.25))
    frame['reconstruction_quality_pass'] = frame.absolute_quality_pass & frame.relative_quality_pass
    frame['combined_quality_pass'] = frame.reconstruction_quality_pass & frame.low_quality_pass
    frame['saved_fraction'] = 1-frame.budget_n/frame.n_full
    frame.to_csv(output/(args.output_prefix+'_probe.csv'), index=False)
    (output/(args.output_prefix+'_protocol.json')).write_text(json.dumps(dict(
        kind='post_unblinding_exploratory_diagnostic', sites=['newlock'], seeds=args.seeds,
        budgets=args.budgets, models=args.models,
        online_label_access='currently revealed near groups only; no audit or future candidate labels',
        source_weights_used=False, parameter_selection='finite six target trees or one median-tree mechanism with native/half/full count modes; revealed-near-group CV',
        quality=dict(log_mae_max=.626458169250689, thickness_mae_max_m=.88125,
            relative_log_tolerance=.05, relative_thickness_tolerance_m=.25,
            low_recall_min=.5, low_fpr_max=.1)), indent=2), encoding='utf-8')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
