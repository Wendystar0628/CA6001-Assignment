"""Acquisition evidence; oracle measurements are confined to an explicit branch.

No runtime comparisons and no report are produced. Candidate oracle tests use
the GP point predictor only, not the TabPFN/graph primary model. Their purpose
is to test whether marginal uncertainty predicts error reduction for that GP.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from cpt_budget.adaptive_data import AdaptiveDataView, audit_truth
from cpt_budget.adaptive_policy import _groups, select_batch as select_v1
from cpt_budget.adaptive_v2_acquisition import latent_posterior, select_batch
from cpt_budget.config import PROJECT_ROOT, load_config
from cpt_budget.runtime import make_model
from cpt_budget.types import load_dataset


def score(prediction, truth):
    joined = prediction[['location_id', 'depth_index', 'pred_log_qc']].merge(
        truth[['location_id', 'depth_index', 'log_qc']], on=['location_id', 'depth_index'], validate='one_to_one')
    mae = float(joined.assign(error=lambda frame: abs(frame.log_qc - frame.pred_log_qc)).groupby('location_id').error.mean().mean())
    thickness = float(joined.assign(true_low=lambda frame: frame.log_qc < 0,
                                  pred_low=lambda frame: frame.pred_log_qc < 0)
                      .groupby('location_id').apply(lambda group: abs(group.true_low.sum() - group.pred_low.sum()) * .25,
                                                    include_groups=False).mean())
    return mae, thickness


def trajectory_evidence(output, sites):
    trajectories = pd.read_csv(PROJECT_ROOT / 'results/adaptive_v1/trajectories.csv')
    quality = pd.read_csv(PROJECT_ROOT / 'results/adaptive_v1/quality_audit.csv')
    keys = ['site_id', 'seed', 'policy_id', 'round', 'variant']
    selected = trajectories.loc[trajectories.site_id.isin(sites) & trajectories.role.eq('evaluation')]
    joined = selected.merge(quality[keys + ['log_mae', 'thickness_mae_m', 'audit_pass']], on=keys, validate='one_to_one')
    rows = []
    for _, group in joined.groupby(['site_id', 'seed', 'policy_id', 'variant'], sort=True):
        group = group.sort_values('round').copy()
        group['next_log_error_reduction'] = group.log_mae - group.log_mae.shift(-1)
        group['next_thickness_error_reduction_m'] = group.thickness_mae_m - group.thickness_mae_m.shift(-1)
        group['next_added_cpt'] = group.n_observed.shift(-1) - group.n_observed
        group['selected_group_cpt_count'] = group.selected_ids.map(lambda value: len(json.loads(value)))
        rows.append(group[['site_id', 'seed', 'policy_id', 'variant', 'round', 'n_observed', 'n_full',
                           'selected_group_cpt_count', 'uncertainty_q90', 'delta_q90', 'delta_h90',
                           'support_log_mae', 'coverage_fraction', 'pass_streak', 'stopped', 'stop_reason',
                           'log_mae', 'thickness_mae_m', 'audit_pass', 'next_log_error_reduction',
                           'next_thickness_error_reduction_m', 'next_added_cpt']])
    frame = pd.concat(rows, ignore_index=True)
    frame.to_csv(output / 'acquisition_trajectory.csv', index=False)
    summary = []
    for keys, group in frame.groupby(['site_id', 'seed', 'policy_id', 'variant']):
        valid = group.dropna(subset=['next_log_error_reduction'])
        summary.append(dict(zip(['site_id', 'seed', 'policy_id', 'variant'], keys),
                            n_final=int(group.n_observed.max()), steps=int(len(valid)),
                            error_worsening_steps=int((valid.next_log_error_reduction < 0).sum()),
                            mean_log_reduction_per_cpt=float((valid.next_log_error_reduction / valid.next_added_cpt).mean()),
                            uncertainty_vs_next_gain_spearman=(float(spearmanr(valid.uncertainty_q90, valid.next_log_error_reduction).statistic)
                                                                if len(valid) > 2 and valid.uncertainty_q90.nunique() > 1 else None)))
    pd.DataFrame(summary).to_csv(output / 'acquisition_trajectory_summary.csv', index=False)
    return frame


def diagnose_initial(dataset, split, output, *, offline_oracle=False, oracle_max_groups=8):
    view = AdaptiveDataView(dataset, split)
    gp = make_model('gp', split['fold_id'])
    gp.condition(view.observed())
    query = view.query(split['full_ids'])
    geometry = view.geometry()
    known, candidates = view.revealed_ids, split['candidate_ids']
    covariance, noise = latent_posterior(gp, query)
    native = gp.predict(query).frame
    latent = np.diag(covariance)
    row = {'site_id': split['site_id'], 'seed': split['seed'], 'n_observed': len(known),
           'noise_variance': noise, 'fitted_kernel': str(gp.predictor.regressor.kernel_),
           'mean_latent_variance': float(latent.mean()),
           'mean_native_variance': float((latent + noise).mean()),
           'noise_fraction_of_native_variance': float(noise / (latent.mean() + noise)),
           'latent_std_mean': float(np.sqrt(latent).mean()),
           'native_std_mean': float(np.sqrt(latent + noise).mean()),
           'native_std_span': float(np.ptp(np.sqrt(latent + noise))),
           'noise_std_floor': float(np.sqrt(noise)),
           'audit_or_unrevealed_labels_used_by_policy': False}
    selection, ivr_scores = select_batch(geometry, known, candidates, split['near_groups'], gp_predictor=gp, query_grid=query)
    signals = native.assign(native_std=(native.q95_log - native.q05_log) / (2 * 1.6448536269514722))
    uncertainty = signals.groupby('location_id').native_std.mean().to_dict()
    us_selection, us_scores = select_v1(geometry, known, candidates, split['near_groups'],
                                      uncertainty=uncertainty, weights=(.7, 0, .3))
    maximin, maximin_scores = select_v1(geometry, known, candidates, split['near_groups'], policy='maximin')
    row.update(latent_ivr_selected=json.dumps(selection), us_selected=json.dumps(us_selection), maximin_selected=json.dumps(maximin))
    ivr_scores.insert(0, 'seed', split['seed'])
    ivr_scores.insert(0, 'site_id', split['site_id'])
    ivr_scores.to_csv(output / f"acquisition_initial_scores_{split['site_id']}_s{split['seed']}.csv", index=False)

    # This branch is an OFFLINE GP ORACLE diagnostic. Audit truth and hidden
    # candidate curves exist only here, after every deployable score is fixed.
    oracle_rows = []
    if offline_oracle:
        mapping = _groups(split['near_groups'], geometry)
        top = set(selection + us_selection + maximin)
        ranked = ivr_scores.sort_values('score', ascending=False)
        requested_groups = []
        for loc in sorted(top) + ranked.location_id.tolist():
            gid = mapping[loc]
            if gid not in requested_groups:
                requested_groups.append(gid)
            if len(requested_groups) >= oracle_max_groups:
                break
        truth = audit_truth(dataset, split)
        baseline = score(gp.predict(view.query(split['audit_ids'])).frame, truth)
        for gid in requested_groups:
            members = [loc for loc in candidates if mapping[loc] == gid]
            hypothetical = make_model('gp', split['fold_id'])
            # Deliberately labelled offline oracle; never fed into select_batch.
            hypothetical.condition(dataset.observed(known + members))
            new = score(hypothetical.predict(view.query(split['audit_ids'])).frame, truth)
            u = us_scores.loc[us_scores.near_group_id.eq(gid)].iloc[0]
            ivr = ivr_scores.loc[ivr_scores.near_group_id.eq(gid)].iloc[0]
            oracle_rows.append({'site_id': split['site_id'], 'seed': split['seed'],
                                'diagnostic_only': True, 'offline_oracle': True, 'point_model': 'gp',
                                'near_group_id': gid, 'group_size': len(members),
                                'gp_native_uncertainty': u.uncertainty,
                                'gp_native_uncertainty_rank': u.rank_uncertainty,
                                'us_score': u.score, 'latent_ivr_score': ivr.score,
                                'integrated_reduction_per_cpt': ivr.reduction_per_cpt,
                                'baseline_gp_audit_log_mae': baseline[0], 'new_gp_audit_log_mae': new[0],
                                'oracle_log_reduction_per_cpt': (baseline[0] - new[0]) / len(members),
                                'oracle_thickness_reduction_per_cpt_m': (baseline[1] - new[1]) / len(members),
                                'candidate_group_sample': 'label_free_top_ivr_us_maximin_not_all_groups'})
        # Diagnostic rankings of realised gains are strictly kept out of policy.
        frame = pd.DataFrame(oracle_rows)
        frame.to_csv(output / f"acquisition_oracle_gp_{split['site_id']}_s{split['seed']}.csv", index=False)
    return row, oracle_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sites', nargs='+', default=['hollywood', 'newlock'])
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--offline-oracle', action='store_true')
    parser.add_argument('--oracle-max-groups', type=int, default=8)
    args = parser.parse_args()
    output = PROJECT_ROOT / 'results/adaptive_v2_diagnostics'
    output.mkdir(parents=True, exist_ok=True)
    trajectory_evidence(output, args.sites)
    dataset = load_dataset(load_config())
    splits = json.loads((PROJECT_ROOT / 'results/adaptive_v1/splits.json').read_text(encoding='utf-8'))
    rows, oracles = [], []
    with threadpool_limits(limits=1):
        for split in splits:
            if split['site_id'] in args.sites and split['seed'] == args.seed:
                row, oracle = diagnose_initial(dataset, split, output, offline_oracle=args.offline_oracle,
                                               oracle_max_groups=args.oracle_max_groups)
                rows.append(row)
                oracles.extend(oracle)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    pd.DataFrame(rows).to_csv(output / 'acquisition_gp_noise.csv', index=False)
    metadata = {'policies': ['latent_ivr', 'target_ivr'], 'coverage_weight': .15,
                'primary_policy': 'latent_ivr', 'retrospective_diagnostic_only': True,
                'oracle_enabled': args.offline_oracle, 'oracle_point_model': 'gp_only',
                'oracle_scope': 'initial_state_label_free_selected_group_sample',
                'oracle_does_not_estimate_tabpfn_graph_candidate_gain': True,
                'information_boundary': 'policy_full_plan_geometry_and_revealed_gp_context_only',
                'v1_rank_issue': 'group_mean_signals_are_ranked_equally_despite_group_size_and_arbitrarily_small_differences',
                'v1_batch_issue': 'U_and_D_ranks_remain_stale_within_batch_only_distance_is_updated',
                'v2_objective': 'average_full_plan_latent_variance_reduction_per_added_CPT_then_hypothetical_covariance_update',
                'target_ivr': 'continuous_primary_helper_profile_discrepancy_weights_normalised_by_observed_MAD',
                'quality_targets_changed': False}
    (output / 'acquisition_diagnostic_config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
