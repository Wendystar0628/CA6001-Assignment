import numpy as np
import pandas as pd
import pytest

from cpt_budget.v4.evaluation import (
    METRICS, candidate_ids, select_trained_candidates, training_gain_tables,
    validate_prediction_matrix, matched_xgb_gain_tables, _reference_predictions,
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
