"""Frozen-state, validation-only evaluation of source-trained residual TabPFN.

One native residual prediction is shared by all alpha and point-guard variants.
Training gain is reported against the corresponding *untrained residual* at the
same alpha, separately from the established IDW/GP promotion gate. Nothing in
this module launches an outer test or modifies the frozen v0.2/v0.3 results.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from cpt_budget.config import PROJECT_ROOT, write_json
from cpt_budget.diagnostics.analyze import METRICS, curve_metrics, gate_values
from cpt_budget.diagnostics.common import (
    OUTPUT_ROOT, SOURCE_RUN, STATE_KEYS, BUDGETS, evaluated_frame,
    iter_validation_states, metrics_for,
)
from cpt_budget.diagnostics.residual_adapter import (
    BaselineResidualTabPFNPredictor, guarded_residual_frame,
)
from cpt_budget.types import PredictionBundle
from cpt_budget.v2.pipeline import _gate, _plain
from cpt_budget.v2.risk import crossfit_risk, evaluate_risk


ALPHAS = (.25, .5, 1.)
GUARDED_ALPHAS = (.25, .5)
VARIANTS = ('crps', 'task')
IDENTITY = [*STATE_KEYS, 'location_id', 'depth_index']
REFERENCE_MODELS = ('idw', 'gp', 'idw_residual', 'tabpfn_spatial',
                    'tabpfn_crps', 'tabpfn_task', 'xgb24',
                    'tabpfn_residual_a025', 'tabpfn_residual_a050',
                    'tabpfn_residual_a100', 'tabpfn_guarded_a025',
                    'tabpfn_guarded_a050')


def candidate_ids(variants=VARIANTS):
    return [f'residual_ft_{v}{guard}_a{int(a * 100):03d}'
            for v in variants for guard, alphas in (('', ALPHAS), ('_guarded', GUARDED_ALPHAS))
            for a in alphas]


def _matched_untrained(model):
    for variant in VARIANTS:
        prefix = f'residual_ft_{variant}'
        if model.startswith(prefix + '_guarded_a'):
            return 'tabpfn_guarded' + model[len(prefix + '_guarded'):]
        if model.startswith(prefix + '_a'):
            return 'tabpfn_residual' + model[len(prefix):]
    raise ValueError(f'No registered untrained comparison for {model}')


def _without_calibration(frame):
    return frame.drop(columns=[c for c in frame
                               if c.startswith('risk_') or c == 'calibration_scope'])


def _atomic_parquet(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.parquet')
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _checkpoint(run, fid, variant):
    summary_path = run / 'checkpoints' / f'fold_{fid}' / f'residual_{variant}' / 'summary.json'
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    if int(summary.get('fold_id', fid)) != fid or summary.get('variant', variant) != variant:
        raise ValueError(f'Checkpoint provenance disagrees with fold/variant: {summary_path}')
    raw = Path(summary['checkpoint_path'])
    choices = [raw] if raw.is_absolute() else [run / raw, summary_path.parent / raw, PROJECT_ROOT / raw]
    path = next((p.resolve() for p in choices if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f'Training checkpoint not found: {raw}')
    stat = path.stat()
    identity = {'checkpoint_path': str(path), 'checkpoint_size': stat.st_size,
                'checkpoint_mtime_ns': stat.st_mtime_ns,
                'summary_sha256': hashlib.sha256(summary_bytes).hexdigest()}
    return path, identity


def _expected_frame(config):
    """The source IDW grid is checked against the frozen sixty state manifest."""
    reference = pd.read_parquet(SOURCE_RUN / 'predictions_oof.parquet')
    reference = _without_calibration(reference)
    idw = reference.loc[reference.model_id.eq('idw')].copy()
    state_rows = []
    for state in iter_validation_states(config):
        meta = state.metadata
        state_rows.append({k: meta[k] for k in [*STATE_KEYS, 'observed_set_id', 'investigation_group']})
        mask = np.ones(len(idw), dtype=bool)
        for k in STATE_KEYS:
            mask &= idw[k].eq(meta[k]).to_numpy()
        rows = idw.loc[mask]
        expected_grid = set(state.query[['location_id', 'depth_index']].itertuples(index=False, name=None))
        if set(rows[['location_id', 'depth_index']].itertuples(index=False, name=None)) != expected_grid:
            raise ValueError(f'IDW reference query grid differs from frozen state {state.key}')
    states = pd.DataFrame(state_rows)
    if len(states) != 60 or states.duplicated(STATE_KEYS).any():
        raise ValueError('The registered validation manifest must contain exactly 60 unique states')
    if idw.duplicated(IDENTITY).any():
        raise ValueError('Duplicate reference evaluation cells')
    # Legacy frames lack the support fingerprint; bind it from their exact frozen state.
    reference = reference.drop(columns='observed_set_id', errors='ignore').merge(
        states[STATE_KEYS + ['observed_set_id']], on=STATE_KEYS, how='left', validate='many_to_one')
    if reference.observed_set_id.isna().any():
        raise ValueError('Reference predictions include states outside the frozen manifest')
    return reference, reference.loc[reference.model_id.eq('idw')].copy()


def validate_prediction_matrix(frame, expected, required_models):
    """Reject missing cells, reused truth, wrong support, and mixed evaluation roles."""
    required = set(IDENTITY + ['model_id', 'observed_set_id', 'investigation_group',
                              'role', 'split_role', 'policy_id', 'pred_log_qc',
                              'truth_log_qc', 'truth_qc_mpa'])
    if not required.issubset(frame):
        raise ValueError(f'Missing prediction columns: {sorted(required - set(frame))}')
    if (not frame.role.eq('evaluation').all() or not frame.split_role.eq('validation').all()
            or not frame.policy_id.eq('maximin').all()):
        raise ValueError('Only frozen maximin validation evaluation rows are allowed')
    if set(frame.model_id) != set(required_models):
        raise ValueError('The registered candidate matrix is incomplete or contains extra models')
    if frame[IDENTITY + ['observed_set_id', 'investigation_group']].isna().any().any():
        raise ValueError('Evaluation keys/support/group must not be missing')
    truth_cols = ['truth_log_qc', 'truth_qc_mpa']
    expected_keys = set(expected[IDENTITY].itertuples(index=False, name=None))
    expected_values = expected.set_index(IDENTITY).sort_index()
    for model, block in frame.groupby('model_id'):
        if (block.duplicated(IDENTITY).any()
                or set(block[IDENTITY].itertuples(index=False, name=None)) != expected_keys):
            raise ValueError(f'Missing, extra or duplicate evaluation cells for {model}')
        ordered = block.set_index(IDENTITY).sort_index()
        for column in ['observed_set_id', 'investigation_group']:
            if not ordered[column].equals(expected_values[column]):
                raise ValueError(f'Frozen {column} differs for {model}')
        if not np.allclose(ordered[truth_cols].to_numpy(float),
                           expected_values[truth_cols].to_numpy(float), rtol=0, atol=1e-12):
            raise ValueError(f'Evaluation truth differs for {model}')
        if not np.isfinite(ordered.pred_log_qc.to_numpy(float)).all():
            raise ValueError(f'Nonfinite point prediction for {model}')


def _reference_predictions(config):
    reference, expected = _expected_frame(config)
    paths = [OUTPUT_ROOT / 'xgb24/validation/predictions.parquet']
    paths.extend(OUTPUT_ROOT / 'residual_adapter/validation' / f'tabpfn_residual_a{int(a*100):03d}' / 'predictions.parquet'
                 for a in ALPHAS)
    paths.extend(OUTPUT_ROOT / 'point_guarded' / f'tabpfn_guarded_a{int(a*100):03d}' / 'predictions.parquet'
                 for a in GUARDED_ALPHAS)
    parts = [reference]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Frozen comparison is required: {path}')
        parts.append(_without_calibration(pd.read_parquet(path)))
    combined = pd.concat(parts, ignore_index=True)
    validate_prediction_matrix(combined, expected, REFERENCE_MODELS)
    return combined, expected


def _calibrate_and_measure(combined, config, output):
    frames, records, calibrators = [], [], {}
    threshold = float(config['v2']['risk']['decision_threshold'])
    if threshold != .2:
        raise ValueError('v4 retains the frozen 0.2 risk threshold')
    for (fid, model), block in combined.groupby(['fold_id', 'model_id'], sort=True):
        calibrated, final = crossfit_risk(block.dropna(axis=1, how='all').copy())
        calibrated['risk_label'] = calibrated.risk_probability >= threshold
        calibrated['calibration_scope'] = 'leave_investigation_group_out'
        frames.append(calibrated)
        calibrators.setdefault(str(fid), {})[model] = final
        for _, rows in calibrated.groupby(STATE_KEYS, sort=True):
            record = metrics_for(rows, config)
            record['study_version'] = '0.4-residual'
            truth = rows[['site_id', 'location_id', 'depth_index', 'truth_log_qc', 'truth_qc_mpa']].rename(
                columns={'truth_log_qc': 'log_qc', 'truth_qc_mpa': 'qc_mpa'})
            risk = evaluate_risk(rows.drop(columns=['truth_log_qc', 'truth_qc_mpa']), truth, config)
            records.append({**record, **risk})
    _atomic_parquet(pd.concat(frames, ignore_index=True), output / 'predictions_oof.parquet')
    write_json(output / 'calibrators.json', calibrators)
    metrics = pd.DataFrame(records)
    metrics.to_csv(output / 'metrics.csv', index=False)
    return metrics


def training_gain_tables(curves, candidates):
    """Equal-alpha training effects: seed, then fold, then distinct site averaging."""
    paired_parts, fold_rows, global_rows = [], [], []
    for model in candidates:
        control = _matched_untrained(model)
        pair = curves[curves.model_id.eq(model)].merge(
            curves[curves.model_id.eq(control)], on=['fold_id', 'site_id', 'seed'],
            suffixes=('_trained', '_untrained'), validate='one_to_one', how='outer', indicator=True)
        if pair.empty or not pair['_merge'].eq('both').all():
            raise ValueError(f'Missing matched untrained residual states for {model}')
        pair = pair.drop(columns='_merge')
        for name in METRICS:
            pair[f'{name}_delta'] = pair[f'{name}_trained'] - pair[f'{name}_untrained']
        paired_parts.append(pair)
        columns = [f'{name}_{suffix}' for name in METRICS for suffix in ['trained', 'untrained', 'delta']]
        sites_by_fold = pair.groupby(['fold_id', 'site_id'])[columns].mean()
        for fid, group in sites_by_fold.groupby(level='fold_id'):
            values = group.mean().to_dict()
            fold_rows.append({'fold_id': fid, 'model_id': model, 'untrained_model_id': control,
                              **values, 'training_improves_log_mae': values['log_mae_delta'] < 0})
        sites = sites_by_fold.groupby(level='site_id').mean()
        values = sites.mean().to_dict()
        global_rows.append({'model_id': model, 'untrained_model_id': control, **values,
                            'relative_log_mae_reduction': 1 - values['log_mae_trained']/values['log_mae_untrained'],
                            'improved_unique_sites': int((sites.log_mae_delta < 0).sum()),
                            'unique_sites': len(sites),
                            'training_improves_log_mae': values['log_mae_delta'] < 0})
    return pd.concat(paired_parts, ignore_index=True), pd.DataFrame(fold_rows), pd.DataFrame(global_rows)


def select_trained_candidates(curves, candidates, settings):
    """Use unchanged v2 gates, while preserving all failed trained candidates."""
    folds_present = set(curves.fold_id)
    if folds_present != {0, 1, 2}:
        raise ValueError('All three folds are required for a selection decision')
    fold = curves.groupby(['fold_id', 'site_id', 'model_id'])[METRICS].mean().groupby(['fold_id', 'model_id']).mean()
    selections, gate_rows, paired_parts = {}, [], []
    for fid in sorted(folds_present):
        baseline_id = min(['idw', 'gp'], key=lambda name: fold.loc[(fid, name), 'log_mae'])
        baseline = gate_values(fold.loc[(fid, baseline_id)])
        eligible = []
        for model in candidates:
            values = gate_values(fold.loc[(fid, model)])
            checks = _gate(values, baseline, settings)
            row = {'fold_id': fid, 'model_id': model, 'baseline_id': baseline_id,
                   **values, **checks, 'passed': all(checks.values()),
                   'matched_untrained_model': _matched_untrained(model),
                   'training_log_mae_delta': float(fold.loc[(fid, model), 'log_mae']
                       - fold.loc[(fid, _matched_untrained(model)), 'log_mae'])}
            gate_rows.append(row)
            if row['passed']:
                eligible.append(model)
        selected = min(eligible, key=lambda name: fold.loc[(fid, name), 'log_mae']) if eligible else baseline_id
        selections[str(fid)] = {'baseline_model': baseline_id, 'selected_model': selected,
                                'fallback': not eligible, 'eligible_candidates': eligible}
        sub = curves[curves.fold_id.eq(fid)]
        paired_parts.append(sub[sub.model_id.eq(selected)].merge(sub[sub.model_id.eq(baseline_id)],
            on=['fold_id', 'site_id', 'seed'], suffixes=('_method', '_baseline'), validate='one_to_one'))
    paired = pd.concat(paired_parts, ignore_index=True)
    columns = [f'{name}_{suffix}' for name in METRICS for suffix in ['method', 'baseline']]
    sites = paired.groupby(['site_id', 'fold_id'])[columns].mean().groupby('site_id').mean()
    method = {name: float(sites[f'{name}_method'].mean()) for name in METRICS}
    baseline = {name: float(sites[f'{name}_baseline'].mean()) for name in METRICS}
    checks = _gate(gate_values(method), gate_values(baseline), settings)
    improved = int((sites.log_mae_method < sites.log_mae_baseline).sum())
    checks.update(majority_unique_sites_improve=improved > len(sites)/2, all_folds_complete=True)
    passed = all(checks.values())
    result = {'scope': 'exploratory_validation_only', 'folds': selections,
              'global_checks': checks, 'passed': passed, 'improved_sites': improved,
              'unique_sites': len(sites), 'method': method, 'baseline': baseline,
              'relative_log_mae_reduction': 1 - method['log_mae']/baseline['log_mae'],
              'candidates': list(candidates), 'no_test_used': True, 'outer_test_launched': False,
              'outer_test_recommendation': 'consider_frozen_extension' if passed else 'do_not_expand',
              'training_effect_claim': 'See same-alpha matched controls; passing IDW/GP gates alone does not prove training gain.'}
    return result, pd.DataFrame(gate_rows), paired, sites


def analyze_trained_residuals(config, run_dir, *, variants=VARIANTS):
    """CPU-only analysis, deliberately refusing partial candidate matrices."""
    run = Path(run_dir)
    if len(tuple(variants)) != len(VARIANTS) or set(variants) != set(VARIANTS):
        raise ValueError('Both registered training variants are required for a selection decision')
    candidates = candidate_ids(variants)
    reference, expected = _reference_predictions(config)
    parts = [reference]
    for model in candidates:
        path = run / 'validation' / model / 'predictions.parquet'
        if not path.is_file():
            raise FileNotFoundError(f'All registered trained candidates are required: {path}')
        parts.append(_without_calibration(pd.read_parquet(path)))
    combined = pd.concat(parts, ignore_index=True)
    validate_prediction_matrix(combined, expected, [*REFERENCE_MODELS, *candidates])
    output = run / 'comparison'
    output.mkdir(parents=True, exist_ok=True)
    metrics = _calibrate_and_measure(combined, config, output)
    curves = curve_metrics(metrics)
    curves.to_csv(output / 'curves.csv', index=False)
    curves.groupby(['fold_id', 'site_id', 'model_id'])[METRICS].mean().groupby(['fold_id', 'model_id']).mean().to_csv(output / 'fold_macro.csv')
    sites = curves.groupby(['fold_id', 'site_id', 'model_id'])[METRICS].mean().groupby(['site_id', 'model_id']).mean()
    sites.to_csv(output / 'unique_site_metrics.csv')
    sites.groupby('model_id').mean().sort_values('log_mae').to_csv(output / 'unique_site_macro.csv')
    paired_gain, fold_gain, gain = training_gain_tables(curves, candidates)
    paired_gain.to_csv(output / 'training_gain_paired.csv', index=False)
    fold_gain.to_csv(output / 'training_gain_by_fold.csv', index=False)
    gain.to_csv(output / 'training_gain_unique_sites.csv', index=False)
    result, gates, paired, selected_sites = select_trained_candidates(curves, candidates, config['v2']['gate'])
    gates.to_csv(output / 'candidate_gates.csv', index=False)
    paired.to_csv(output / 'selected_system_paired.csv', index=False)
    selected_sites.to_csv(output / 'selected_system_sites.csv')
    result.update(status='complete', training_gain=gain.to_dict('records'),
                  comparison_path=str(output), registered_states=60,
                  number_of_trained_candidates=len(candidates))
    write_json(output / 'selection.json', _plain(result))
    return _plain(result)


def evaluate_trained_residuals(config, run_dir, *, variants=VARIANTS, fold_ids=None):
    """Run/resume registered forwards, then analyze only when every fold exists."""
    run = Path(run_dir)
    variants = tuple(variants)
    fids = tuple(fold_ids) if fold_ids is not None else (0, 1, 2)
    if not variants or len(set(variants)) != len(variants) or set(variants) - set(VARIANTS):
        raise ValueError('Variants must be unique registered crps/task names')
    if not fids or len(set(fids)) != len(fids) or set(fids) - {0, 1, 2}:
        raise ValueError('Fold IDs must be unique members of 0, 1, 2')
    _, expected = _expected_frame(config)
    expected_index = expected.set_index(IDENTITY)
    config_signature = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
    recovery = []
    for fid in fids:
        for variant in variants:
            artifact, provenance = _checkpoint(run, fid, variant)
            provenance['config_sha256'] = config_signature
            marker_path = run / 'validation' / 'provenance' / f'fold_{fid}_{variant}.json'
            if marker_path.is_file() and json.loads(marker_path.read_text(encoding='utf-8')) != provenance:
                raise ValueError(f'Cached inference belongs to a different checkpoint/config: {marker_path}')
            write_json(marker_path, provenance)
            predictor = None
            try:
                for state in iter_validation_states(config, fold_ids=(fid,)):
                    names = candidate_ids((variant,))
                    paths = {name: run / 'validation' / name / 'states' / f'{state.key}.parquet' for name in names}
                    zero_path = run / 'validation' / 'alpha_zero_checks' / f'{variant}_{state.key}.json'
                    if all(p.is_file() for p in paths.values()) and zero_path.is_file():
                        recovery.append(json.loads(zero_path.read_text(encoding='utf-8')))
                        continue
                    if predictor is None:
                        predictor = BaselineResidualTabPFNPredictor(config, artifact=artifact)
                    started = time.perf_counter()
                    predictor.condition(state.observed.copy())
                    bundles = predictor.predict_variants(state.query.copy(), alphas=(0., *ALPHAS))
                    native = bundles[1.].frame
                    zero = evaluated_frame(bundles[0.], state).set_index(IDENTITY).sort_index()
                    baseline = expected_index.loc[zero.index].sort_index()
                    difference = float(np.max(np.abs(zero.pred_log_qc.to_numpy() - baseline.pred_log_qc.to_numpy())))
                    if difference > 1e-10:
                        raise ValueError(f'Alpha zero failed exact frozen IDW recovery: {state.key}')
                    for guarded, alphas in ((False, ALPHAS), (True, GUARDED_ALPHAS)):
                        for alpha in alphas:
                            name = f"residual_ft_{variant}{'_guarded' if guarded else ''}_a{int(alpha*100):03d}"
                            if guarded:
                                raw = guarded_residual_frame(state.query, native.baseline_log_qc.to_numpy(float),
                                                             native.residual_pred_log_qc.to_numpy(float), alpha)
                                bundle = PredictionBundle(raw, name)
                            else:
                                bundle = bundles[alpha]
                                bundle.model_id = name
                            frame = evaluated_frame(bundle, state)
                            frame['study_version'] = '0.4-residual'
                            frame['run_id'] = run.name
                            frame['shared_prediction_seconds'] = time.perf_counter() - started
                            _atomic_parquet(frame, paths[name])
                    check = {**state.metadata, 'variant': variant, 'max_abs_difference': difference}
                    write_json(zero_path, check)
                    recovery.append(check)
                    print(f'residual_ft_{variant}: {state.key}', flush=True)
            finally:
                del predictor
                gc.collect()
    available, missing = {}, []
    # A caller may infer one variant/fold at a time, but selection still needs
    # all ten pre-registered candidates, including the ones not run this call.
    for model in candidate_ids(VARIANTS):
        paths = sorted((run / 'validation' / model / 'states').glob('*.parquet'))
        if paths:
            frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
            _atomic_parquet(frame, run / 'validation' / model / 'predictions.parquet')
            available[model] = len(frame[STATE_KEYS].drop_duplicates())
        else:
            available[model] = 0
        if available[model] != 60:
            missing.append(model)
    write_json(run / 'validation' / 'alpha_zero_recovery.json', {
        'scope': 'current_invocation', 'checks': recovery,
        'max_abs_difference': max((r['max_abs_difference'] for r in recovery), default=None)})
    if missing:
        result = {'status': 'partial', 'scope': 'exploratory_validation_only',
                  'available_states': available, 'incomplete_models': missing,
                  'selection_available': False, 'outer_test_launched': False,
                  'outer_test_recommendation': 'wait_for_complete_matrix'}
        write_json(run / 'validation' / 'status.json', result)
        return result
    result = analyze_trained_residuals(config, run, variants=VARIANTS)
    write_json(run / 'validation' / 'status.json', result)
    return result
