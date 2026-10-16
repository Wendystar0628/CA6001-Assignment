import pandas as pd

from cpt_budget.adaptive_run import _decision, quality_flags


def test_frozen_online_stop_is_not_moved_to_a_later_passing_audit():
    meta = dict(run_id='r', site_id='s', seed=17, fold_id=0, role='evaluation',
                model_id='gp', policy_id='P_adapt', variant='US')
    trace = pd.DataFrame([{**meta, 'round': i, 'n_observed': n, 'n_full': 17,
                           'delta_q90': .1, 'delta_h90': .1, 'support_log_mae': .1,
                           'uncertainty_q90': .1, 'coverage_ok': True}
                          for i, n in enumerate([5, 7, 9, 11, 13, 15, 17])])
    audit = pd.DataFrame([dict(run_id='r', round=i, audit_pass=(i >= 4), low_qc_audit_pass=False,
                              log_mae=.4, thickness_mae_m=.3, precision=.5, recall=.4, f1=.44,
                              fpr=.1, log_error_p90=.7) for i in range(7)])
    rule = dict(delta_q90_max=.2, delta_h90_max=.2, support_log_mae_max=.2,
                uncertainty_q90_max=.2, min_observed=9, consecutive_rounds=2)
    result = _decision(trace, audit, rule)
    assert result['n_stop'] == 11
    assert result['false_early_stop']
    assert not result['audit_pass']
    assert result['n_batches'] == 4


def test_absolute_quality_is_required_even_when_close_to_a_poor_full_plan():
    quality = dict(absolute_log_mae_max=.6, absolute_thickness_mae_max_m=.5,
                   relative_log_tolerance=.05, relative_thickness_tolerance_m=.25,
                   low_qc_recall_target=.5, low_qc_fpr_target=.1)
    row = dict(log_mae=.9, thickness_mae_m=.7, recall=.7, fpr=.05)
    full = dict(log_mae=.91, thickness_mae_m=.71)
    flags = quality_flags(row, full, quality)
    assert flags['relative_quality_pass']
    assert not flags['absolute_quality_pass']
    assert not flags['audit_pass']
    assert flags['low_qc_audit_pass']


def test_saved_live_stop_is_preserved_when_scoring_serialised_evaluation():
    meta = dict(run_id='live', site_id='s', seed=17, fold_id=0, role='evaluation',
                model_id='gp', policy_id='P_adapt', variant='US')
    trace = pd.DataFrame([{**meta, 'round': i, 'n_observed': n, 'n_full': 17,
                           'stopped': i == 2, 'coverage_ok': True,
                           'delta_q90': 0., 'delta_h90': 0., 'support_log_mae': 0., 'uncertainty_q90': 0.}
                          for i, n in enumerate([5, 7, 9])])
    audit = pd.DataFrame([dict(run_id='live', round=i, audit_pass=False, low_qc_audit_pass=None,
                              log_mae=.8, thickness_mae_m=.6, precision=None, recall=None,
                              f1=0., fpr=0., log_error_p90=1.) for i in range(3)])
    # Re-scoring never recomputes or moves the controller's already saved stop.
    strict_rule = dict(delta_q90_max=0., delta_h90_max=0., support_log_mae_max=0.,
                       uncertainty_q90_max=0., min_observed=15, consecutive_rounds=2)
    result = _decision(trace, audit, strict_rule)
    assert result['n_stop'] == 9
    assert result['false_early_stop']
    assert result['low_qc_audit_pass'] is None
