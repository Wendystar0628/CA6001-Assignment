"""Regenerate results/current from reference (or live) predictions.

Pipeline, unchanged from the v0.8 analysis: per model and outer fold a
leave-one-investigation-group-out Platt calibration (risk.crossfit_risk) gives
risk_probability; risk_label = risk_probability >= 0.2 (4:1 FN:FP cost). The
Platt input is logit(native p_low_1) for entries with a native or shared risk
head and -pred_log_qc otherwise. Per-state metrics -> normalized trapezoid over
budgets 5..13 (analysis.curve_metrics) -> mean over (fold, seed) per site ->
equal-weight mean over the 4 validation sites.
"""
from __future__ import annotations
import gc
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .analysis import METRICS, curve_metrics
from .config import PROJECT_ROOT, write_json
from .evaluate import evaluate
from .experiment import ASSETS, STATE_KEYS, catalog, diagnostic_config, iter_states
from .risk import crossfit_risk, evaluate_risk

REFERENCE = ASSETS / 'evaluation/reference_predictions.parquet'
RISK_SOURCES = {'shared_empirical_head': 'native_probability_logit', 'native_gp': 'native_probability_logit',
                'platt_on_point': 'negative_point_log_prediction'}
RISK_SOURCE_NOTE = {
    'shared_empirical_head': 'p_low_1 of the shared empirical risk head v6_empirical_graph_union_local '
                             '(computed from the tabpfn_graph point path, independent of the entry point path); '
                             'identical risk metrics for every entry with this source',
    'native_gp': 'GP Gaussian predictive CDF p_low_1',
    'platt_on_point': 'no native probability; evaluation-time Platt calibration of -pred_log_qc'}
GEOMETRY = ['site_id', 'location_id', 'profile_id', 'near_group_id', 'depth_index', 'z_center_m', 'x_m', 'y_m',
            'geometry_scale_m']
NATIVE = ['q05_log', 'q10_log', 'q25_log', 'q75_log', 'q90_log', 'q95_log', 'p_low_1']
META = ['fold_id', 'seed', 'model_id', 'policy_id', 'budget_n', 'investigation_group', 'role', 'split_role',
        'observed_set_id']
STATE_META = ['fold_id', 'site_id', 'seed', 'budget_n', 'model_id', 'policy_id', 'investigation_group',
              'observed_set_id', 'role', 'split_role', 'study_version']


def entries(models=None):
    rows = catalog()
    if models is None:
        return rows
    by_id = {r['id']: r for r in rows}
    unknown = sorted(set(models) - set(by_id))
    if unknown:
        raise ValueError(f'Not in the catalogue: {unknown}')
    return [by_id[m] for m in models]


def state_frame(frame, state, record):
    """Attach state metadata and (outside the predictor) the query truth."""
    out = frame.copy()
    for key, value in state.metadata.items():
        if key in META or key in STATE_KEYS:
            out[key] = value
    out['model_id'] = record['id']
    out['source_model_id'] = record['source_model_id']
    truth = state.truth[['site_id', 'location_id', 'depth_index', 'qc_mpa', 'log_qc']].rename(
        columns={'qc_mpa': 'truth_qc_mpa', 'log_qc': 'truth_log_qc'})
    out = out.merge(truth, on=['site_id', 'location_id', 'depth_index'], how='left', validate='one_to_one')
    if out.truth_qc_mpa.isna().any() or len(out) != len(state.query):
        raise ValueError(f'Prediction grid does not match the frozen query grid: {record["id"]}/{state.key}')
    keep = [c for c in GEOMETRY + ['pred_log_qc', 'pred_qc_mpa', 'truth_qc_mpa', 'truth_log_qc'] + META + NATIVE
            + ['source_model_id'] if c in out]
    return out[keep]


def live_predictions(models=None, *, folds=None, seeds=None, budgets=None, device=None, log=print):
    """Real condition/predict on the frozen states (threads limited to 1)."""
    states = list(iter_states(folds, seeds, budgets))
    from .runtime import make_model
    frames = []
    with threadpool_limits(limits=1):
        for record in entries(models):
            predictor, last = None, None
            for state in states:
                if state.metadata['fold_id'] != last:
                    predictor = None
                    gc.collect()
                    predictor = make_model(record['id'], state.metadata['fold_id'], device=device)
                    last = state.metadata['fold_id']
                predictor.condition(state.observed)
                frames.append(state_frame(predictor.predict(state.query).frame, state, record))
            if log:
                log(f'live predictions: {record["id"]} ({len(states)} states)')
    return pd.concat(frames, ignore_index=True)


def _score_frame(group, record):
    """Keep p_low_1 only where the catalogue declares a native/shared risk head."""
    frame = group.drop(columns=[c for c in ('risk_probability', 'risk_label', 'risk_threshold') if c in group])
    source = record['risk_source']
    if source not in RISK_SOURCES:
        raise ValueError(f'Unknown risk_source {source} for {record["id"]}')
    if source == 'platt_on_point':
        if 'p_low_1' in frame and frame.p_low_1.notna().any():
            raise ValueError(f'{record["id"]} declares platt_on_point but carries a native p_low_1')
        frame = frame.drop(columns=['p_low_1'], errors='ignore')
    elif 'p_low_1' not in frame or not frame.p_low_1.notna().all():
        raise ValueError(f'{record["id"]} declares {source} but p_low_1 is missing')
    return frame.drop(columns=[c for c in NATIVE if c in frame and frame[c].isna().all()])


def calibrate(predictions, models=None):
    """Cross-fitted risk columns, per catalogue entry and outer fold."""
    records = {r['id']: r for r in entries(models)}
    pieces = []
    for (model_id, _), group in predictions.groupby(['model_id', 'fold_id'], sort=False):
        record = records[model_id]
        frame = _score_frame(group, record)
        calibrated, _ = crossfit_risk(frame)
        kind = set(calibrated.risk_score_kind)
        if kind != {RISK_SOURCES[record['risk_source']]}:
            raise AssertionError(f'{model_id}: risk score kind {kind} contradicts risk_source {record["risk_source"]}')
        out = group.copy()
        out['risk_probability'] = calibrated.risk_probability.to_numpy(float)
        out['risk_label'] = calibrated.risk_label.to_numpy(bool)
        out['risk_threshold'] = calibrated.risk_threshold.to_numpy(float)
        pieces.append(out)
    return pd.concat(pieces).sort_index()


def state_metrics(calibrated, models=None):
    config = diagnostic_config()
    records = {r['id']: r for r in entries(models)}
    rows = []
    for key, group in calibrated.groupby(['model_id'] + STATE_KEYS, sort=False):
        record = records[key[0]]
        truth = group[['site_id', 'location_id', 'depth_index', 'truth_qc_mpa', 'truth_log_qc']]
        prediction = _score_frame(group, record).drop(columns=['truth_qc_mpa', 'truth_log_qc'])
        prediction[['risk_probability', 'risk_label', 'risk_threshold']] = group[
            ['risk_probability', 'risk_label', 'risk_threshold']]
        result = {'source_model_id': record['source_model_id'], **evaluate(prediction, truth, config)}
        first = group.iloc[0]
        result.update({c: first[c] for c in STATE_META if c in group})
        result['role'] = first['split_role']  # archived convention: the state role, not the row role 'evaluation'
        result['study_version'] = record.get('study_version')
        result.update(evaluate_risk(prediction, truth, config))
        result['risk_source'] = record['risk_source']
        rows.append(result)
    frame = pd.DataFrame(rows)
    order = {r['id']: i for i, r in enumerate(entries(models))}
    return frame.sort_values(['model_id'] + STATE_KEYS, key=lambda s: s.map(order) if s.name == 'model_id' else s,
                             kind='stable').reset_index(drop=True)


def aggregate(metrics):
    """Budget AUBC -> per site (mean over fold x seed) -> equal-site mean."""
    sources = metrics.drop_duplicates('model_id').set_index('model_id')[['source_model_id', 'risk_source']]
    curves = curve_metrics(metrics)
    curves = curves.join(sources, on='model_id')
    curves = curves[['source_model_id', 'fold_id', 'site_id', 'seed', 'model_id', 'risk_source', *METRICS]]
    curves = curves.sort_values(['fold_id', 'site_id', 'seed', 'source_model_id'], kind='stable').reset_index(drop=True)
    sites = curves.groupby(['source_model_id', 'site_id', 'model_id', 'risk_source'], sort=False)[METRICS].mean().reset_index()
    sites = sites.sort_values(['site_id', 'source_model_id'], kind='stable').reset_index(drop=True)
    ranking = sites.groupby(['source_model_id', 'model_id', 'risk_source'], sort=False)[METRICS].mean().reset_index()
    if sites.groupby('model_id').site_id.nunique().ne(4).any():
        raise ValueError('Every entry must have all 4 validation sites')
    ranking = ranking.sort_values('log_mae', kind='stable').reset_index(drop=True)
    return curves, sites, ranking


def run(source='reference', models=None, output=None, *, device=None, log=print):
    """Recompute metrics/curves/site metrics/ranking and write them to ``output``."""
    output = PROJECT_ROOT / (output or 'results/current')
    wanted = [r['id'] for r in entries(models)]
    if source == 'reference':
        predictions = pd.read_parquet(REFERENCE)
        missing = sorted(set(wanted) - set(predictions.model_id))
        if missing:
            raise ValueError(f'Reference predictions lack catalogue entries: {missing}')
        predictions = predictions[predictions.model_id.isin(wanted)].reset_index(drop=True)
    elif source == 'live':
        predictions = live_predictions(wanted, device=device, log=log)
    else:
        raise ValueError('source must be reference or live')
    calibrated = calibrate(predictions, wanted)
    metrics = state_metrics(calibrated, wanted)
    if metrics.groupby('model_id').size().ne(60).any():
        raise ValueError('Every entry must be evaluated on the 60 frozen states')
    curves, sites, ranking = aggregate(metrics)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / 'metrics.csv', index=False)
    curves.to_csv(output / 'curves.csv', index=False)
    sites.to_csv(output / 'unique_site_metrics.csv', index=False)
    ranking.to_csv(output / 'all_model_ranking.csv', index=False)
    write_json(output / 'evaluation_provenance.json', {
        'command': f'python -m cpt_budget evaluate --source {source}',
        'prediction_source': str(REFERENCE.relative_to(PROJECT_ROOT)) if source == 'reference' else 'live inference',
        'models': wanted, 'states_per_model': 60,
        'risk': {'calibration': 'risk.crossfit_risk per model and outer fold: leave one validation investigation group out, '
                                'monotone Platt, equal site weights', 'threshold': 0.2,
                 'threshold_rationale': 'course scenario false-negative cost 4 : false-positive cost 1',
                 'risk_source': RISK_SOURCE_NOTE},
        'aggregation': 'normalized trapezoid over budgets 5/7/9/11/13 -> mean over (fold, seed) per site -> '
                       'equal-weight mean over 4 validation sites',
        'caveat': 'Exploratory: the same 4 validation sites were reused for model selection; no outer test.'})
    if log:
        log(f'wrote {output}: {len(metrics)} state rows, {len(ranking)} entries')
    return metrics, curves, sites, ranking


ARCHIVE_TAG = 'v0.8-engineering'
ARCHIVE_METRICS = 'results/v8-engineering-20261014/comparison/metrics.csv'
# Risk/probability metrics of a new point x head pairing are checked against the archived head.
ARCHIVED_RISK_REFERENCE = {'thickness': 'v6_empirical_graph_union_local'}
POINT_COLUMNS = ['log_mae', 'log_rmse', 'qc_mae_mpa', 'h_point_mae_m', 'tp', 'fp', 'fn', 'tn']
PROBABILITY_COLUMNS = ['h_expected_mae_m', 'average_precision', 'risk_brier', 'h_risk_expected_mae_m', 'risk_tp',
                       'risk_fp', 'risk_fn', 'risk_tn']


def archived_metrics():
    import io
    import subprocess
    text = subprocess.check_output(['git', 'show', f'{ARCHIVE_TAG}:{ARCHIVE_METRICS}'], cwd=PROJECT_ROOT, text=True,
                                   encoding='utf-8')
    return pd.read_csv(io.StringIO(text), low_memory=False)


def port_check(models=None, output='results/current/port_equivalence.csv', *, metrics=None, log=print):
    """Per-state equality of v0.9 metrics with the archived v0.8 per-state metrics.

    Point columns are compared with the archived source_model_id; probability and
    risk columns with the archived risk head (the source id unless the entry is a
    new point x head pairing, see ARCHIVED_RISK_REFERENCE). Counts must be equal;
    real-valued metrics are reported as max absolute differences.
    """
    records = entries(models)
    if metrics is None:
        predictions = pd.read_parquet(REFERENCE)
        predictions = predictions[predictions.model_id.isin([r['id'] for r in records])].reset_index(drop=True)
        metrics = state_metrics(calibrate(predictions, [r['id'] for r in records]), [r['id'] for r in records])
    archive = archived_metrics()
    rows = []
    for record in records:
        new = metrics[metrics.model_id.eq(record['id'])].set_index(STATE_KEYS).sort_index()
        groups = [('point', record['source_model_id'], POINT_COLUMNS + (
            ['average_precision'] if record['risk_source'] == 'platt_on_point' else [])),
                  ('probability_and_risk', ARCHIVED_RISK_REFERENCE.get(record['id'], record['source_model_id']),
                   [c for c in PROBABILITY_COLUMNS if not (record['risk_source'] == 'platt_on_point' and c in (
                       'average_precision', 'h_expected_mae_m'))])]
        for part, archived_id, columns in groups:
            old = archive[archive.model_id.eq(archived_id)].set_index(STATE_KEYS).sort_index()
            if len(old) != 60 or len(new) != 60 or not old.index.equals(new.index):
                raise AssertionError(f'{record["id"]}: archived {archived_id} does not cover the same 60 states')
            row = {'model_id': record['id'], 'part': part, 'archived_model_id': archived_id, 'states': len(new)}
            ok = True
            for column in columns:
                a, b = pd.to_numeric(old[column], errors='coerce'), pd.to_numeric(new[column], errors='coerce')
                nan_mismatch = int((a.isna() != b.isna()).sum())
                both = a.notna() & b.notna()
                difference = float((a[both] - b[both]).abs().max()) if both.any() else 0.0
                exact = column in ('tp', 'fp', 'fn', 'tn') or column.startswith('risk_t') or column.startswith('risk_f')
                ok &= nan_mismatch == 0 and (difference == 0 if exact else difference <= 1e-9)
                row[f'max_abs_diff_{column}'] = difference
                if nan_mismatch:
                    row[f'nan_mismatch_{column}'] = nan_mismatch
            row['equivalent'] = bool(ok)
            rows.append(row)
            if log:
                log(f'{record["id"]:18} {part:21} vs {archived_id}: equivalent={ok}')
    frame = pd.DataFrame(rows)
    if output:
        path = PROJECT_ROOT / output
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    if not frame.equivalent.all():
        raise AssertionError('Per-state metrics differ from the archived v0.8 metrics')
    return frame
