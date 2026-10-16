"""Isolated v2 acquisition experiments with revealed-only model contexts.

Audit predictions are saved for scoring after the whole online simulation.
With no development-calibrated controller, a sweep continues to the full plan;
the earliest audit-passing round is only an offline opportunity diagnostic.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .adaptive_data import AdaptiveDataView, validate_split
from .adaptive_models import MAIN_MODEL_IDS, ABLATION_MODEL_IDS, make_predictor
from .adaptive_policy import coverage_fraction, select_batch as select_v1
from .adaptive_v2_acquisition import select_batch as select_ivr
from .adaptive_v2_stopping import remaining_grid_signals, support_quality_signals


POLICIES = ('maximin', 'UDS', 'latent_ivr', 'target_ivr')


def make_experimental_predictor(model_id, split, geometry):
    """Keep immutable source models and revealed-only target models separate."""
    if model_id in MAIN_MODEL_IDS + ABLATION_MODEL_IDS:
        return make_predictor(model_id, split['fold_id'], role=split['role'], geometry=geometry,
                              device='cuda').initialize()
    if model_id.startswith(('target_tree_', 'target_et_', 'target_rf_', 'target_xgb_')):
        from .adaptive_v2_target_trees import make_target_tree_predictor
        return make_target_tree_predictor(model_id, split['fold_id']).initialize()
    from .adaptive_v2_models import make_v2_predictor
    return make_v2_predictor(model_id, split['fold_id']).initialize()


@dataclass(frozen=True)
class QualityStopRule:
    """Explicitly calibrated error bounds; no audit score is accepted online."""
    log_target: float
    thickness_target_m: float
    log_margin: float
    thickness_margin_m: float
    delta_q90_max: float
    delta_h90_max: float
    min_observed: int = 9
    consecutive_rounds: int = 2
    support_max_groups: int = 5
    calibration_sites: tuple[str, ...] = ()

    def __post_init__(self):
        numbers = [self.log_target, self.thickness_target_m, self.log_margin,
                   self.thickness_margin_m, self.delta_q90_max, self.delta_h90_max]
        if not np.isfinite(numbers).all() or min(numbers) < 0:
            raise ValueError('Stop parameters must be finite and nonnegative')
        if self.min_observed < 5 or self.consecutive_rounds < 1 or self.support_max_groups < 1:
            raise ValueError('Invalid minimum support or streak length')
        if not self.calibration_sites:
            raise ValueError('Online early stopping requires named development calibration sites')

    def checks(self, signals, n_observed):
        q_upper = signals.get('support_log_mae', np.nan) + self.log_margin
        h_upper = signals.get('support_thickness_mae_m', np.nan) + self.thickness_margin_m
        return dict(
            minimum=n_observed >= self.min_observed,
            log_quality=bool(np.isfinite(q_upper) and q_upper <= self.log_target),
            thickness_quality=bool(np.isfinite(h_upper) and h_upper <= self.thickness_target_m),
            delta_q=bool(np.isfinite(signals['delta_q90']) and signals['delta_q90'] <= self.delta_q90_max),
            delta_h=bool(np.isfinite(signals['delta_h90']) and signals['delta_h90'] <= self.delta_h90_max),
            coverage=bool(signals['coverage_ok']),
            geometry_supported=bool(signals.get('candidate_geometry_supported', False)),
        )


def _frame(value):
    return value.frame if hasattr(value, 'frame') else value


def run_v2_trajectory(dataset, split, model_id, policy, *, predictor=None,
                      gp_predictor=None, support_predictor=None, stop_rule=None,
                      diagnostics=True, support_max_groups=5, log=print):
    """Whole-group replay; no full-plan/audit measurements enter any decision.

    Batches are selected before any member is revealed. The GP auxiliary is
    conditioned solely on the current observations. Diagnostic sweeps use no
    stop rule and must not be described as achieved online savings.
    """
    if policy not in POLICIES:
        raise ValueError(f'Unknown acquisition policy: {policy}')
    validate_split(split)
    view = AdaptiveDataView(dataset, split)
    plan_ids, audit_ids = list(split['full_ids']), list(split['audit_ids'])
    geometry, query = view.geometry(), view.query(plan_ids + audit_ids)
    plan_query = view.query(plan_ids)
    predictor = predictor or make_experimental_predictor(model_id, split, query)
    gp_predictor = gp_predictor or (predictor if model_id == 'gp' else make_experimental_predictor('gp', split, plan_query))
    # A GP compared with itself has identically zero disagreement. Its helper
    # is instead a revealed-only depth median; other models use the source GP.
    helper_predictor = (make_experimental_predictor('target_depth_median', split, plan_query)
                        if model_id == 'gp' and policy in {'UDS', 'target_ivr'} else None)
    if diagnostics and support_predictor is None:
        support_predictor = make_experimental_predictor(model_id, split, query)
    if stop_rule and set(stop_rule.calibration_sites) & {split['site_id']} and split['role'] == 'evaluation':
        raise ValueError('Evaluation site cannot be included in stop calibration')
    meta = dict(site_id=split['site_id'], seed=int(split['seed']), fold_id=int(split['fold_id']),
                role=split['role'], model_id=model_id, policy=policy,
                run_id=f"{split['site_id']}__s{split['seed']}__{model_id}__{policy}")
    rows, predictions, selections, support_details = [], [], [], []
    previous, selected, selected_scores = None, list(split['initial_ids']), None
    streak, round_index = 0, 0
    while True:
        known = view.revealed_ids
        remaining = sorted(set(plan_ids) - set(known))
        # Source TabPFN preprocessing has small row-order sensitivity. A v2
        # context is a stable set representation for every acquisition policy.
        observed = view.observed().sort_values(['location_id', 'depth_index'], kind='stable').reset_index(drop=True)
        predictor.condition(observed)
        main = _frame(predictor.predict(query)).reset_index(drop=True)
        keys = ['location_id', 'depth_index']
        if not main[keys].equals(query[keys].reset_index(drop=True)):
            raise ValueError('Predictions must preserve the requested keys and order')
        if gp_predictor is not predictor:
            gp_predictor.condition(observed)
        gp = _frame(gp_predictor.predict(plan_query)).reset_index(drop=True)
        if not gp[keys].equals(plan_query[keys].reset_index(drop=True)):
            raise ValueError('Auxiliary GP must preserve planned query keys and order')
        grid = main.loc[main.location_id.isin(plan_ids)].drop(columns='std_log_qc', errors='ignore').reset_index(drop=True)
        grid = grid.merge(gp[keys + ['std_log_qc']], on=keys, validate='one_to_one')
        helper = gp
        if helper_predictor is not None:
            helper_predictor.condition(observed)
            helper = _frame(helper_predictor.predict(plan_query))
        signals = remaining_grid_signals(previous, grid, remaining)
        coverage = coverage_fraction(geometry, known, plan_ids, split.get('mandatory_ids', ()))
        signals.update(coverage_fraction=coverage, coverage_ok=bool(coverage >= 1 - 1e-12))
        if diagnostics and len(known) >= 9 and remaining:
            support, detail = support_quality_signals(
                observed, geometry, split['near_groups'], support_predictor, view.query,
                remaining_ids=remaining, max_groups=stop_rule.support_max_groups if stop_rule else support_max_groups)
            signals.update(support)
            support_details.append(detail.assign(**meta, round=round_index, n_observed=len(known)))
        checks = stop_rule.checks(signals, len(known)) if stop_rule and remaining else {}
        streak = streak + 1 if checks and all(checks.values()) else 0
        stopped = not remaining or bool(stop_rule and streak >= stop_rule.consecutive_rounds)
        reason = 'full_plan_complete' if not remaining else ('estimated_quality' if stopped else None)
        record = {**meta, 'round': round_index, 'n_observed': len(known), 'n_full': len(plan_ids),
                  'known_ids': json.dumps(known), 'revealed_ids': json.dumps(selected),
                  'selected_ids': '[]', **signals, 'pass_streak': streak, 'stopped': stopped,
                  'stop_reason': reason, 'stop_checks': json.dumps(checks, sort_keys=True),
                  'controller': 'development_calibrated' if stop_rule else 'full_sweep_no_online_stop'}
        audit = main.loc[main.location_id.isin(audit_ids)].copy()
        if {'log_qc', 'qc_mpa', 'truth', 'true_log_qc'} & set(audit):
            raise ValueError('Predictions may not expose audit labels')
        predictions.append(audit.assign(**meta, round=round_index, n_observed=len(known)))
        for loc in selected:
            score = np.nan if selected_scores is None else selected_scores.loc[loc, 'selection_score']
            selections.append({**meta, 'round': round_index, 'location_id': loc,
                               'selection_order': len(selections) + 1, 'selection_score': score})
        if not stopped:
            if policy in {'maximin', 'UDS'}:
                # Retained as a controlled comparison with v1's rank heuristic.
                paired = grid[keys + ['pred_log_qc']].merge(
                    helper[keys + ['pred_log_qc']], on=keys, suffixes=('_primary', '_gp'), validate='one_to_one')
                signals_u = gp.groupby('location_id').std_log_qc.mean().to_dict()
                signals_d = ((paired.pred_log_qc_primary < 0) != (paired.pred_log_qc_gp < 0)).groupby(paired.location_id).mean().to_dict()
                next_ids, scores = select_v1(geometry, known, remaining, split['near_groups'],
                    uncertainty=signals_u, disagreement=signals_d, policy='maximin' if policy == 'maximin' else 'adaptive')
            else:
                native_gp = gp_predictor.suite.models['gp']
                next_ids, scores = select_ivr(geometry, known, remaining, split['near_groups'],
                    gp_predictor=native_gp, query_grid=plan_query, policy=policy,
                    primary_predictions=grid, helper_predictions=helper)
            if not next_ids or set(next_ids) - set(remaining):
                raise ValueError('Selection must reveal nonempty remaining whole groups')
            selected_scores = scores.set_index('location_id')
            record['selected_ids'] = json.dumps(next_ids)
        rows.append(record)
        if log:
            log(f"{meta['run_id']} n={len(known)}/{len(plan_ids)} stop={reason or '-'}")
        if stopped:
            break
        # There are no label-dependent calls between selection and batch reveal.
        view.reveal(next_ids)
        previous, selected, round_index = grid, next_ids, round_index + 1
    return dict(trajectories=pd.DataFrame(rows), predictions=pd.concat(predictions, ignore_index=True),
                selections=pd.DataFrame(selections),
                support=pd.concat(support_details, ignore_index=True) if support_details else pd.DataFrame(),
                metadata={**meta, 'audit_labels_used_online': False, 'n_full': len(plan_ids),
                          'scope': 'post_unblind_exploration', 'batch_size': 2,
                          'disagreement_helper': 'target_depth_median' if helper_predictor else 'gp',
                          'stop_rule': asdict(stop_rule) if stop_rule else None})
