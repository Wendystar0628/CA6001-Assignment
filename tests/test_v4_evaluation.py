import numpy as np
import pandas as pd
import pytest

from cpt_budget.v4.evaluation import (
    METRICS, candidate_ids, select_trained_candidates, training_gain_tables,
    validate_prediction_matrix, matched_xgb_gain_tables, _reference_predictions,
    calibration_effects,
)


def _prediction_rows():
    return pd.DataFrame([
        dict(fold_id=0, site_id='site', seed=17, budget_n=5,
             location_id='query', depth_index=i, model_id='idw',
             observed_set_id='frozen_support', investigation_group='group',
             role='evaluation', split_role='validation', policy_id='maximin',
             pred_log_qc=.1, truth_log_qc=float(i), truth_qc_mpa=float(np.exp(i)))
        for i in range(2)
    ])


@pytest.mark.parametrize('damage', ['missing_depth', 'wrong_truth', 'wrong_support', 'test_role'])
def test_comparison_refuses_incomplete_or_nonpaired_predictions(damage):
    expected = _prediction_rows()
    actual = expected.assign(model_id='residual_ft_crps_a025')
    if damage == 'missing_depth':
        actual = actual.iloc[:1].copy()
    elif damage == 'wrong_truth':
        actual.loc[0, 'truth_log_qc'] = 99.
    elif damage == 'wrong_support':
        actual.loc[0, 'observed_set_id'] = 'different_support'
    else:
        actual.loc[0, 'split_role'] = 'test'
    with pytest.raises(ValueError):
        validate_prediction_matrix(actual, expected, ['residual_ft_crps_a025'])


def _curves():
    rows = []
    for fold, sites in [(0, ['a', 'b']), (1, ['a', 'b']), (2, ['c', 'd'])]:
        for site in sites:
            for seed in [17, 29]:
                for model in ['idw', 'gp', 'residual_ft_crps_a025', 'tabpfn_residual_a025', 'xgb_residual_matched']:
                    error = .8 if site == 'a' and model == 'residual_ft_crps_a025' else 1.
                    if model == 'gp':
                        error = 1.1
                    if model == 'tabpfn_residual_a025':
                        error = .9
                    if model == 'xgb_residual_matched':
                        error = .85
                    values = dict.fromkeys(METRICS, .1)
                    values.update(log_mae=error, risk_recall=.8, recall=.8)
                    rows.append(dict(fold_id=fold, site_id=site, seed=seed, model_id=model, **values))
    return pd.DataFrame(rows)


def test_global_gate_counts_repeated_sites_once_and_requires_majority():
    settings = dict(h_tolerance_m=.05, recall_tolerance=.02, fpr_tolerance=.02,
                    min_relative_improvement=0.)
    result, gates, _, sites = select_trained_candidates(
        _curves(), ['residual_ft_crps_a025'], settings)
    assert gates.passed.tolist() == [True, True, False]
    assert result['unique_sites'] == 4
    assert result['improved_sites'] == 1
    assert result['method']['log_mae'] == pytest.approx(.95)
    assert not result['passed']
    assert not result['outer_test_launched']
    assert result['folds']['2']['fallback']


def test_training_effect_uses_same_alpha_untrained_control_not_weak_direct_model():
    curves = _curves()
    paired, folds, global_gain = training_gain_tables(curves, ['residual_ft_crps_a025'])
    # It beats IDW on one site, but worsens its matched untrained residual on
    # three sites. Equal weighting of four unique sites yields .95 versus .90.
    row = global_gain.iloc[0]
    assert row.untrained_model_id == 'tabpfn_residual_a025'
    assert row.log_mae_trained == pytest.approx(.95)
    assert row.log_mae_untrained == pytest.approx(.9)
    assert row.log_mae_delta == pytest.approx(.05)
    assert not row.training_improves_log_mae
    assert row.improved_unique_sites == 1


def test_partial_fold_never_authorizes_promotion():
    with pytest.raises(ValueError, match='All three folds'):
        select_trained_candidates(_curves().query('fold_id == 0'),
                                  ['residual_ft_crps_a025'], {})


def test_missing_matched_training_control_is_an_error():
    curves = _curves()
    curves = curves[~((curves.model_id == 'tabpfn_residual_a025') & (curves.site_id == 'd'))]
    with pytest.raises(ValueError, match='Missing matched'):
        training_gain_tables(curves, ['residual_ft_crps_a025'])


def test_matched_xgb_control_separates_training_data_from_model_effect():
    paired, folds, gain = matched_xgb_gain_tables(_curves(), ['residual_ft_crps_a025'])
    row = gain.iloc[0]
    assert row.matched_control_model_id == 'xgb_residual_matched'
    assert row.log_mae_tabpfn == pytest.approx(.95)
    assert row.log_mae_matched_xgb == pytest.approx(.85)
    assert row.log_mae_delta == pytest.approx(.10)
    assert not row.beats_matched_xgb_log_mae
    assert row.improved_unique_sites == 1
    assert row.unique_sites == 4


def test_missing_matched_xgb_states_or_artifact_cannot_be_ignored(tmp_path):
    curves = _curves()
    curves = curves[~((curves.model_id == 'xgb_residual_matched') & (curves.site_id == 'd'))]
    with pytest.raises(ValueError, match='Missing same-trajectory'):
        matched_xgb_gain_tables(curves, ['residual_ft_crps_a025'])
    with pytest.raises(FileNotFoundError, match='same-trajectory XGBoost'):
        _reference_predictions({}, tmp_path)


def _calibration_predictions():
    rows = []
    for fold, sites in [(0, ['a', 'b']), (1, ['a', 'b']), (2, ['c', 'd'])]:
        for site in sites:
            for seed in [17, 29]:
                for budget in [5, 7, 9, 11, 13]:
                    for model in ['idw', 'gp']:
                        for depth, truth in [(0, .5), (1, 2.)]:
                            native = .6 if depth == 0 else .1
                            oof = .8 if depth == 0 else (.3 if site == 'a' and budget > 5 else .1)
                            rows.append(dict(fold_id=fold, site_id=site, seed=seed, budget_n=budget,
                                location_id=f'query_{site}', depth_index=depth, model_id=model,
                                observed_set_id=f'{fold}_{site}_{seed}_{budget}', investigation_group=f'group_{site}',
                                role='evaluation', split_role='validation', policy_id='maximin',
                                pred_log_qc=.1, truth_log_qc=np.log(truth), truth_qc_mpa=truth,
                                p_low_1=native if model == 'gp' else np.nan, risk_probability=oof))
    frame = pd.DataFrame(rows)
    return frame, frame[frame.model_id == 'idw'].copy()


def test_calibration_effects_use_same_threshold_budget_integral_and_equal_sites():
    predictions, expected = _calibration_predictions()
    table, detail = calibration_effects(predictions, expected, ['idw', 'gp'])
    assert table.model_id.tolist() == ['gp']
    row = table.iloc[0]
    assert row.native_brier == pytest.approx(.085)
    assert row.oof_brier == pytest.approx(.03375)
    assert row.native_recall == row.oof_recall == 1.
    assert row.native_fpr == 0.
    # Site a occurs twice but receives 1/4 weight. Its FPR curve is [0,1,1,1,1].
    assert row.oof_fpr == pytest.approx(.875 / 4)
    assert row.fpr_delta_oof_minus_native == pytest.approx(.21875)
    assert next(r for r in detail['availability'] if r['model_id'] == 'idw')['status'] == 'not_available'


def test_partial_native_probability_never_becomes_a_partial_site_average():
    predictions, expected = _calibration_predictions()
    predictions.loc[predictions.model_id.eq('gp').idxmax(), 'p_low_1'] = np.nan
    table, detail = calibration_effects(predictions, expected, ['idw', 'gp'])
    assert table.empty
    assert next(r for r in detail['availability'] if r['model_id'] == 'gp')['reason'] == 'native_probability_missing_or_invalid_in_some_cells'


def test_mixed_point_and_distribution_system_has_no_partial_interval_claim():
    curves = _curves()
    curves.loc[curves.model_id.eq('idw'), ['coverage_90', 'width_90_log']] = np.nan
    settings = dict(h_tolerance_m=.05, recall_tolerance=.02, fpr_tolerance=.02, min_relative_improvement=0.)
    result, _, _, _ = select_trained_candidates(curves, ['residual_ft_crps_a025'], settings)
    assert np.isnan(result['method']['coverage_90'])
    assert np.isnan(result['baseline']['coverage_90'])
    assert result['interval_availability']['method']['coverage_90']['defined_curves'] == 8
    assert result['interval_availability']['method']['coverage_90']['total_curves'] == 12
    assert result['method']['log_mae'] == pytest.approx(.95)
    assert not result['passed']
