"""Retained inference component, consolidated from cpt_budget.v2.risk (v0.8-engineering)."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from cpt_budget.evaluate import join_evaluation
CALIBRATOR_VERSION = 'v2_monotone_platt_1'
PROBABILITY_EPSILON = 1e-06
DEFAULT_RISK_THRESHOLD = 0.2

def _score(frame: pd.DataFrame, kind: str | None=None) -> tuple[np.ndarray, str]:
    if kind is None:
        native = 'p_low_1' in frame and frame['p_low_1'].notna().any()
        kind = 'native_probability_logit' if native else 'negative_point_log_prediction'
    if kind == 'native_probability_logit':
        if 'p_low_1' not in frame:
            raise ValueError('Native-probability calibration requires p_low_1')
        probability = frame['p_low_1'].to_numpy(float)
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError('Native probabilities must all be finite and in [0, 1]')
        score = logit(np.clip(probability, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON))
    elif kind == 'negative_point_log_prediction':
        if 'pred_log_qc' not in frame:
            raise ValueError('Point-model calibration requires pred_log_qc')
        score = -frame['pred_log_qc'].to_numpy(float)
    else:
        raise ValueError(f'Unknown risk score kind: {kind}')
    if not np.isfinite(score).all():
        raise ValueError('Calibration scores must be finite')
    return (score, kind)

def _validation_frame(predictions: pd.DataFrame, group_column: str) -> pd.DataFrame:
    required = {'split_role', 'role', 'site_id', group_column, 'truth_qc_mpa'}
    if not required.issubset(predictions):
        raise ValueError(f'Calibration requires explicit validation evaluation rows: {sorted(required - set(predictions))}')
    if predictions.empty:
        raise ValueError('Cannot fit calibration without validation rows')
    if not predictions.split_role.eq('validation').all() or not predictions.role.eq('evaluation').all():
        raise ValueError("Calibration accepts only split_role='validation', role='evaluation'")
    if predictions[['site_id', group_column]].isna().any().any():
        raise ValueError('Calibration requires a known site and investigation group for every row')
    for column in ('model_id', 'fold_id'):
        if column in predictions and predictions[column].nunique(dropna=False) != 1:
            raise ValueError('Fit each model and outer fold separately')
    if (predictions.groupby('site_id')[group_column].nunique() != 1).any():
        raise ValueError('Each calibration site must belong to one investigation group')
    truth = predictions.truth_qc_mpa.to_numpy(float)
    if not np.isfinite(truth).all() or (truth <= 0).any():
        raise ValueError('Calibration truth must be finite positive qc')
    return predictions.copy()

def fit_risk_calibrator(predictions: pd.DataFrame, *, group_column: str='investigation_group') -> dict:
    """Fit one-dimensional Platt scaling with a nonnegative slope.

    Each site contributes the same total weight, irrespective of its number of
    evaluation cells. A single-class fit explicitly falls back to that weighted
    empirical prior (zero or one), rather than pretending to learn a slope.
    """
    frame = _validation_frame(predictions, group_column)
    score, kind = _score(frame)
    labels = (frame.truth_qc_mpa.to_numpy(float) < 1.0).astype(float)
    counts = frame.groupby('site_id').site_id.transform('size').to_numpy(float)
    weights = 1.0 / counts
    weights /= weights.sum()
    prior = float(np.dot(weights, labels))
    center = float(np.dot(weights, score))
    scale = float(np.sqrt(np.dot(weights, (score - center) ** 2)))
    result = {'calibrator_version': CALIBRATOR_VERSION, 'source_role': 'validation', 'fit_role': 'evaluation', 'score_kind': kind, 'tau_mpa': 1.0, 'probability_epsilon': PROBABILITY_EPSILON, 'weighting': 'equal_sites_then_equal_rows_within_site', 'group_column': group_column, 'training_groups': sorted(frame[group_column].astype(str).unique().tolist()), 'training_sites': sorted(frame.site_id.astype(str).unique().tolist()), 'n_training_rows': len(frame), 'n_training_sites': int(frame.site_id.nunique()), 'n_training_groups': int(frame[group_column].nunique()), 'n_positive_rows': int(labels.sum()), 'n_negative_rows': int(len(labels) - labels.sum()), 'weighted_positive_prior': prior, 'score_center': center, 'score_scale': scale if scale > 1e-12 else 1.0, 'slope': 0.0, 'intercept': float(logit(np.clip(prior, 1e-09, 1 - 1e-09))), 'constant_probability': None, 'fallback_reason': None, 'risk_threshold': DEFAULT_RISK_THRESHOLD, 'threshold_rationale': 'course_scenario_false_negative_cost_4_false_positive_cost_1'}
    if not labels.any() or labels.all() or scale <= 1e-12:
        result.update(status='constant_prior', constant_probability=prior, fallback_reason='single_class_validation' if not labels.any() or labels.all() else 'constant_validation_score')
        return result
    standardized = (score - center) / scale
    regularization = 1e-06

    def objective(parameters):
        slope, intercept = parameters
        logits = slope * standardized + intercept
        loss = np.dot(weights, np.logaddexp(0, logits) - labels * logits)
        residual = weights * (expit(logits) - labels)
        gradient = np.array([np.dot(residual, standardized) + regularization * slope, residual.sum()])
        return (float(loss + regularization * slope * slope / 2), gradient)
    optimized = minimize(objective, [1.0, result['intercept']], jac=True, method='L-BFGS-B', bounds=[(0.0, None), (None, None)], options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-08})
    result['optimizer_success'] = bool(optimized.success)
    result['optimizer_message'] = str(optimized.message)
    result['slope_l2'] = regularization
    if not optimized.success or not np.isfinite(optimized.x).all():
        result.update(status='constant_prior', constant_probability=prior, fallback_reason='calibration_optimizer_failed')
    else:
        result.update(status='fitted', slope=float(optimized.x[0]), intercept=float(optimized.x[1]), objective=float(optimized.fun))
    return result

def _transform_score(score: np.ndarray, calibrator: dict) -> np.ndarray:
    if calibrator.get('calibrator_version') != CALIBRATOR_VERSION:
        raise ValueError('Unsupported or missing risk calibrator version')
    if calibrator.get('source_role') != 'validation':
        raise ValueError('Risk calibrator must have validation provenance')
    if calibrator.get('constant_probability') is not None:
        probability = float(calibrator['constant_probability'])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Invalid constant risk prior')
        return np.full(np.shape(score), probability, dtype=float)
    slope = float(calibrator['slope'])
    intercept = float(calibrator['intercept'])
    center = float(calibrator['score_center'])
    scale = float(calibrator['score_scale'])
    if not np.isfinite([slope, intercept, center, scale]).all() or slope < 0 or scale <= 0:
        raise ValueError('Risk calibrator must have a nonnegative finite slope and positive scale')
    return expit(slope * (score - center) / scale + intercept)

def apply_risk(frame: pd.DataFrame, calibrator: dict, *, threshold: float=DEFAULT_RISK_THRESHOLD) -> pd.DataFrame:
    """Apply a frozen mapping using only native p or a point prediction."""
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Risk threshold must lie in [0, 1]')
    score, kind = _score(frame, calibrator['score_kind'])
    output = frame.copy()
    output['risk_probability'] = _transform_score(score, calibrator)
    output['risk_label'] = output.risk_probability >= threshold
    output['risk_score_kind'] = kind
    output['risk_threshold'] = float(threshold)
    output['risk_calibration_status'] = calibrator['status']
    return output

def crossfit_risk(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Leave one validation investigation group out, then fit final deployment map."""
    group_column = 'investigation_group'
    frame = _validation_frame(predictions, group_column)
    groups = sorted(frame[group_column].astype(str).unique())
    if len(groups) < 2:
        raise ValueError('Crossfit calibration requires at least two validation investigation groups')
    final = fit_risk_calibrator(frame, group_column=group_column)
    original_index = frame.index
    frame = frame.reset_index(drop=True)
    pieces, fits = ([], {})
    for held_out in groups:
        mask = frame[group_column].astype(str).eq(held_out)
        fitted = fit_risk_calibrator(frame.loc[~mask], group_column=group_column)
        fitted['excluded_groups'] = [held_out]
        held = apply_risk(frame.loc[mask], fitted)
        held['risk_calibration_mode'] = 'leave_investigation_group_out'
        held['risk_calibration_held_out_group'] = held_out
        held['risk_calibration_training_groups'] = json.dumps(fitted['training_groups'])
        pieces.append(held)
        fits[held_out] = fitted
    output = pd.concat(pieces).sort_index()
    output.index = original_index
    final['crossfit'] = {'mode': 'leave_investigation_group_out', 'complete': True, 'n_groups': len(groups), 'held_out_fits': fits}
    final['oof_calibrators'] = fits
    final['excluded_groups'] = []
    return (output, final)

def evaluate_risk(prediction_frame: pd.DataFrame, truth: pd.DataFrame, config: dict) -> dict:
    """Independent risk metrics on the same fixed holdout used by v1 evaluate."""
    frame = join_evaluation(prediction_frame, truth)
    if not {'risk_probability', 'risk_label'}.issubset(frame):
        raise ValueError('Risk evaluation requires calibrated risk_probability and risk_label')
    tau = float(config.get('data', {}).get('tau_mpa', 1.0))
    step = float(config.get('data', {}).get('depth_step_m', 0.25))
    if tau != 1.0 or not np.isfinite(step) or step <= 0:
        raise ValueError('This risk calibrator defines the 1 MPa event; depth step must be positive')
    probability = frame.risk_probability.to_numpy(float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError('Calibrated risk probabilities must be finite and in [0, 1]')
    if frame.risk_label.isna().any() or not frame.risk_label.isin([True, False, 0, 1]).all():
        raise ValueError('Risk labels must be known Boolean decisions')
    labels = frame.truth_qc_mpa.to_numpy(float) < tau
    decisions = frame.risk_label.to_numpy(bool)
    tp, fp = (int((labels & decisions).sum()), int((~labels & decisions).sum()))
    fn, tn = (int((labels & ~decisions).sum()), int((~labels & ~decisions).sum()))
    frame = frame.assign(_risk_brier=(probability - labels) ** 2, _true_low=labels)
    group_keys = ['site_id', 'location_id'] if 'site_id' in frame else ['location_id']
    locations = frame.groupby(group_keys, sort=False)
    expected = locations.risk_probability.sum() * step
    actual = locations._true_low.sum() * step
    result = {'risk_recall': float(tp / (tp + fn)) if tp + fn else None, 'risk_precision': float(tp / (tp + fp)) if tp + fp else None, 'risk_fpr': float(fp / (fp + tn)) if fp + tn else None, 'risk_brier': float(locations._risk_brier.mean().mean()), 'h_risk_expected_mae_m': float(abs(expected - actual).mean()), 'risk_tp': tp, 'risk_fp': fp, 'risk_fn': fn, 'risk_tn': tn, 'risk_n_true_positive': tp + fn, 'risk_n_true_negative': tn + fp, 'risk_n_predicted_positive': tp + fp, 'risk_n_eval_locations': int(len(expected)), 'risk_n_eval_cells': len(frame)}
    reasons = {}
    if not tp + fn:
        reasons['risk_recall'] = 'no_positive_labels'
    if not tp + fp:
        reasons['risk_precision'] = 'no_predicted_positive_labels'
    if not fp + tn:
        reasons['risk_fpr'] = 'no_negative_labels'
    for key in ('risk_recall', 'risk_precision', 'risk_fpr'):
        result[f'{key}_reason'] = reasons.get(key)
    result['risk_missing_reasons'] = json.dumps(reasons, sort_keys=True)
    return result
