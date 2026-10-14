"""Serial, paired runtime measurements on frozen geometry/support states.

The benchmark executes real condition/predict calls, never final predictions.
Optional prepare_state(observed, query, metadata) is outside the timer and must
be declared in each spec's scope. Initialization is reported separately.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import gc
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Callable

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from cpt_budget.config import PROJECT_ROOT, write_json
from cpt_budget.diagnostics.common import SOURCE_RUN, STATE_KEYS, diagnostic_config, iter_validation_states


RUN = PROJECT_ROOT / 'runs/v8-engineering-20261014/benchmark'


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    factory: Callable[[int], object]
    scope: str
    pair_id: str = ''
    version: str = 'baseline'
    device: str = 'cpu'
    setup_note: str = 'No per-state preparation outside timing.'
    equivalence_atol: float = 1e-8


def resident_memory_mib():
    """Absolute process working set/RSS; this is not model-exclusive memory."""
    if os.name == 'nt':
        class Counters(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                       *[(name, ctypes.c_size_t) for name in (
                           'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                           'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                           'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]]
        value = Counters()
        value.cb = ctypes.sizeof(value)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb):
            return value.WorkingSetSize / 1024**2
    elif Path('/proc/self/status').exists():
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return float(line.split()[1]) / 1024
    return float('nan')


def _torch_for(spec):
    if not spec.device.startswith('cuda'):
        return None
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f'{spec.name} requires actual CUDA; CPU fallback is forbidden.')
    return torch


def _sync(spec, torch):
    if torch is not None:
        torch.cuda.synchronize(spec.device)


def numerical_output(output):
    """Keep numeric prediction payloads for paired equivalence diagnostics."""
    if hasattr(output, 'frame'):
        frame = output.frame.sort_values(['location_id', 'depth_index'])
        columns = [column for column in ('pred_log_qc', 'p_low_1') if column in frame]
        return {column: frame[column].to_numpy(float) for column in columns}
    if isinstance(output, dict):
        return {str(key): np.asarray(value, float).reshape(-1)
                for key, value in output.items() if isinstance(value, (np.ndarray, list, tuple))}
    if isinstance(output, tuple):
        return {f'output_{i}': np.asarray(value, float).reshape(-1)
                for i, value in enumerate(output)}
    return {'output': np.asarray(output, float).reshape(-1)}


def _check_output(output):
    values = numerical_output(output)
    if not values or not all(len(value) and np.isfinite(value).all() for value in values.values()):
        raise ValueError('The timed predictor must return finite nonempty predictions.')
    return values


def _hardware():
    processor_name = platform.processor()
    if os.name == 'nt':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                    r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                processor_name = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    versions = {}
    for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'torch', 'tabpfn', 'threadpoolctl'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                              '--format=csv,noheader'], capture_output=True,
                             text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        gpu = None
    try:
        source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
            cwd=PROJECT_ROOT, text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        source_commit = None
    return {'platform': platform.platform(), 'processor': processor_name,
            'logical_cpu_count': os.cpu_count(), 'python': sys.version,
            'package_versions': versions, 'nvidia_smi': gpu, 'threadpools': threadpool_info(),
            'source_commit_at_run_start': source_commit}


def baseline_specs():
    from cpt_budget.models.idw import IDWPredictor
    from cpt_budget.models.gp import GPPredictor
    from cpt_budget.v5.weak_baselines import TargetConstantMedianPredictor
    config = diagnostic_config()

    def gp_factory(fold_id):
        path = SOURCE_RUN / 'checkpoints' / f'fold_{fold_id}' / 'gp_trend.json'
        trend = {int(key): float(value) for key, value in json.loads(path.read_text()).items()}
        return GPPredictor(config, trend=trend)

    return [BenchmarkSpec('target_constant_median', lambda fid: TargetConstantMedianPredictor(),
                          'full point inference: support median + query prediction'),
            BenchmarkSpec('idw', lambda fid: IDWPredictor(config),
                          'full point inference: condition + IDW geometry/features + prediction'),
            BenchmarkSpec('gp', gp_factory,
                          'full native-distribution inference: GP fit + mean/std/event prediction')]


class JointOperatorPredictor:
    """Only the joint/recognition operator is timed; a real base is prepared.

    An IDW base allows the old and new operator to receive identical realistic
    shapes without reading any final prediction cache. The complete optimized
    quality pipelines are separate specs supplied by runtime.py.
    """
    def __init__(self, adapter, *, recognition=False):
        self.adapter, self.recognition = adapter, recognition

    def prepare_state(self, observed, query, metadata):
        from cpt_budget.models.idw import IDWPredictor
        base_model = IDWPredictor(diagnostic_config())
        base_model.condition(observed)
        self.base = base_model.predict(query).frame.pred_log_qc.to_numpy(float)

    def condition(self, observed):
        self.adapter.condition(observed)

    def predict(self, query):
        result = (self.adapter.predict(query, self.base) if self.recognition else
                  self.adapter.batch_predict(query, self.base, kind='error', policy='reference'))
        return {'point': np.asarray(result['point']), 'probability': np.asarray(result['probability'])}


def engineering_specs(groups=('structural', 'joint', 'haar', 'recognition', 'tabpfn')):
    """Lazy imports keep --help and lightweight tests independent of GPU setup."""
    result = []
    if 'structural' in groups:
        from cpt_budget.v8 import structural
        for version, factory in [('old', structural.old_factory), ('new', structural.new_factory)]:
            result.append(BenchmarkSpec('structural_'+version, factory,
                'full point inference: source FPCA initialization separate; support LOO stack + simplex blend',
                pair_id='structural_point', version=version))
    if 'joint' in groups:
        from cpt_budget.v8.joint import ReferenceJointAdapter, FastJointAdapter
        for version, adapter in [('old', ReferenceJointAdapter), ('new', FastJointAdapter)]:
            result.append(BenchmarkSpec('joint_operator_'+version,
                lambda fid, adapter=adapter: JointOperatorPredictor(adapter()),
                'component only: complete-curve error-mixture operator; IDW base computation excluded',
                pair_id='joint_operator', version=version,
                setup_note='Real IDW base recomputed from current support/query outside timing; preparation_ms reported. This is not complete balanced-system latency.'))
    if 'haar' in groups:
        from cpt_budget.v8.haar import make_predictor
        for version, optimized in [('old', False), ('new', True)]:
            result.append(BenchmarkSpec('haar_'+version,
                lambda fid, optimized=optimized: make_predictor(variant='equivalent', optimized=optimized),
                'full point inference: Haar detail transfer + simplex + thickness-count decision; separate empirical risk head excluded',
                pair_id='haar_point', version=version))
    if 'recognition' in groups:
        from cpt_budget.v8.recognition import ReferenceRecognition, FastRecognition
        for version, adapter in [('old', ReferenceRecognition), ('new', FastRecognition)]:
            result.append(BenchmarkSpec('recognition_operator_'+version,
                lambda fid, adapter=adapter: JointOperatorPredictor(adapter(), recognition=True),
                'component only: complete-curve analog-mixture recognition operator; IDW base and independent risk head excluded',
                pair_id='recognition_operator', version=version,
                setup_note='Real IDW base recomputed outside timing; preparation_ms reported. This is not complete recognition-system latency.'))
    if 'tabpfn' in groups:
        from cpt_budget.v8 import tabpfn
        for version, factory in [('old', tabpfn.old_factory), ('new', tabpfn.new_factory)]:
            result.append(BenchmarkSpec('tabpfn_residual_'+version, factory,
                'full guarded residual point inference: feature preparation + trained TabPFN CUDA forward + IDW guard; source checkpoint load separate',
                pair_id='tabpfn_residual', version=version, device='cuda', equivalence_atol=1e-5))
    return result


def _write_outputs(output, samples, initializations, correctness, specs):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sample_frame, init_frame = pd.DataFrame(samples), pd.DataFrame(initializations)
    sample_frame.to_csv(output / 'samples.csv', index=False)
    init_frame.to_csv(output / 'initializations.csv', index=False)
    fields = ['condition_ms', 'predict_ms', 'total_ms', 'rss_after_mib', 'cuda_peak_allocated_mib']
    state_records, summary_records = [], []
    for name, group in sample_frame.groupby('model_id', sort=False):
        spec = next(spec for spec in specs if spec.name == name)
        record = dict(model_id=name, pair_id=spec.pair_id, version=spec.version, scope=spec.scope,
                      device=spec.device, states=group.state_key.nunique(), measured_samples=len(group))
        for field in fields:
            record[field + '_median'] = group[field].median()
            record[field + '_p95'] = group[field].quantile(.95)
        summary_records.append(record)
        for state_key, cells in group.groupby('state_key', sort=False):
            state_records.append(dict(model_id=name, pair_id=spec.pair_id, version=spec.version,
                scope=spec.scope, state_key=state_key,
                **{key: cells[key].iloc[0] for key in STATE_KEYS},
                **{field: cells[field].median() for field in fields},
                samples=len(cells), preparation_ms=cells.preparation_ms.iloc[0]))
    states = pd.DataFrame(state_records)
    states.to_csv(output / 'state_summary.csv', index=False)
    pd.DataFrame(summary_records).to_csv(output / 'summary.csv', index=False)
    paired, paired_summary = [], []
    for pair_id, group in states.loc[states.pair_id.ne('')].groupby('pair_id', sort=False):
        old, new = group.loc[group.version.eq('old')], group.loc[group.version.eq('new')]
        if old.empty or new.empty:
            continue
        join = old.merge(new, on='state_key', suffixes=('_old', '_new'), validate='one_to_one')
        if len(join) != len(old) or len(join) != len(new):
            raise ValueError(f'{pair_id} old/new runtime states do not match.')
        for row in join.to_dict('records'):
            record = {'pair_id': pair_id, 'state_key': row['state_key'],
                      'scope': row['scope_new'],
                      **{key: row[key+'_old'] for key in STATE_KEYS}}
            for metric in ('condition_ms', 'predict_ms', 'total_ms'):
                record[metric+'_old'] = row[metric+'_old']
                record[metric+'_new'] = row[metric+'_new']
                record[metric+'_speedup_old_over_new'] = row[metric+'_old']/row[metric+'_new']
            paired.append(record)
        one = pd.DataFrame(paired).loc[lambda frame: frame.pair_id.eq(pair_id)]
        paired_summary.append({'pair_id': pair_id, 'states': len(one), 'scope': group.scope.iloc[0],
            'total_speedup_paired_state_median': one.total_ms_speedup_old_over_new.median(),
            'total_speedup_paired_state_min': one.total_ms_speedup_old_over_new.min(),
            'total_speedup_paired_state_max': one.total_ms_speedup_old_over_new.max(),
            'total_speedup_ratio_of_state_medians': one.total_ms_old.median()/one.total_ms_new.median()})
    pd.DataFrame(paired).to_csv(output / 'paired_state_ratios.csv', index=False)
    pd.DataFrame(paired_summary).to_csv(output / 'paired_summary.csv', index=False)
    pd.DataFrame(correctness).to_csv(output / 'paired_output_checks.csv', index=False)
    return pd.DataFrame(summary_records)


def run_specs(specs, states, output=RUN, *, repeats=3, warmups=1, native_reference=None):
    """Run sequentially; the caller must stop unrelated jobs before invoking."""
    specs, states = list(specs), list(states)
    if not specs or not states or repeats < 1 or warmups < 1:
        raise ValueError('Nonempty specs/states, positive warmups and repeats required.')
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError('Benchmark model names must be unique.')
    samples, initializations, correctness, remembered, native_checks = [], [], [], {}, []
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    # A loaded torch runtime can launch independent CPU pools even for CPU specs.
    if 'torch' in sys.modules or any(spec.device.startswith('cuda') for spec in specs):
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    with threadpool_limits(limits=1):
        hardware = _hardware()
        for spec in specs:
            torch = _torch_for(spec)
            predictor, previous_fold = None, None
            try:
                for state in states:
                    fid = int(state.metadata['fold_id'])
                    if fid != previous_fold:
                        predictor = None
                        gc.collect()
                        if torch is not None:
                            torch.cuda.empty_cache()
                        _sync(spec, torch)
                        started = time.perf_counter()
                        predictor = spec.factory(fid)
                        _sync(spec, torch)
                        after_factory = time.perf_counter()
                        if hasattr(predictor, 'initialize'):
                            predictor.initialize()
                            _sync(spec, torch)
                        after_initialize = time.perf_counter()
                        initializations.append({'model_id': spec.name, 'fold_id': fid,
                            'scope': spec.scope, 'device': spec.device,
                            'factory_ms': (after_factory-started)*1000,
                            'explicit_weight_initialization_ms': (after_initialize-after_factory)*1000,
                            'initialization_ms': (after_initialize-started)*1000,
                            'rss_after_initialization_mib': resident_memory_mib()})
                        previous_fold = fid
                    started = time.perf_counter()
                    if hasattr(predictor, 'prepare_state'):
                        predictor.prepare_state(state.observed.copy(), state.query.copy(),
                                                dict(state.metadata))
                        _sync(spec, torch)
                    prepare_ms = (time.perf_counter()-started)*1000
                    payload = None
                    for repeat in range(-warmups, repeats):
                        observed, query = state.observed.copy(), state.query.copy()
                        if torch is not None:
                            torch.cuda.reset_peak_memory_stats(spec.device)
                        _sync(spec, torch)
                        start = time.perf_counter()
                        predictor.condition(observed)
                        _sync(spec, torch)
                        after_condition = time.perf_counter()
                        prediction = predictor.predict(query)
                        _sync(spec, torch)
                        after_prediction = time.perf_counter()
                        payload = _check_output(prediction)
                        if repeat >= 0:
                            samples.append({'model_id': spec.name, 'state_key': state.key,
                                **{key: state.metadata[key] for key in STATE_KEYS}, 'repeat': repeat,
                                'preparation_ms': prepare_ms,
                                'condition_ms': (after_condition-start)*1000,
                                'predict_ms': (after_prediction-after_condition)*1000,
                                'total_ms': (after_prediction-start)*1000,
                                'rss_after_mib': resident_memory_mib(),
                                'cuda_peak_allocated_mib': (torch.cuda.max_memory_allocated(spec.device)/1024**2
                                    if torch is not None else float('nan')),
                                'stage_timings_json': json.dumps(getattr(predictor, 'stage_timings', {}))})
                    if spec.pair_id:
                        key = spec.pair_id, state.key
                        if key in remembered:
                            other_version, other = remembered.pop(key)
                            if set(payload) != set(other):
                                raise ValueError(f'{key}: paired output fields differ.')
                            for column in payload:
                                if payload[column].shape != other[column].shape:
                                    raise ValueError(f'{key}: paired output dimensions differ.')
                                correctness.append({'pair_id': spec.pair_id, 'state_key': state.key,
                                    'field': column, 'values': len(payload[column]),
                                    'max_abs_difference': np.max(np.abs(payload[column]-other[column])),
                                    'absolute_tolerance': spec.equivalence_atol,
                                    'equivalent': np.allclose(payload[column], other[column],
                                        atol=spec.equivalence_atol, rtol=0),
                                    'versions': '/'.join([other_version, spec.version])})
                        else:
                            remembered[key] = spec.version, payload
                    if native_reference is not None and spec.pair_id == 'balanced_full':
                        # Evaluation-only comparison after all timed forwards.
                        expected = native_reference[state.key]
                        actual = payload['p_low_1']
                        if expected.shape != actual.shape:
                            raise ValueError('Native reference shape differs from real full-pipeline inference.')
                        native_checks.append({'model_id': spec.name, 'state_key': state.key,
                            'values': len(actual), 'reference_model': 'v6_empirical_graph_union_local',
                            'max_abs_difference': np.max(np.abs(actual-expected)),
                            'mean_abs_difference': np.mean(np.abs(actual-expected)),
                            'risk_label_changes_at_0p2': int(((actual >= .2) != (expected >= .2)).sum())})
                    print(f'benchmark {spec.name}: {state.key}', flush=True)
                _write_outputs(output, samples, initializations, correctness, specs)
            finally:
                del predictor
                gc.collect()
                if torch is not None:
                    torch.cuda.empty_cache()
        summary = _write_outputs(output, samples, initializations, correctness, specs)
        if native_checks:
            pd.DataFrame(native_checks).to_csv(output / 'native_reference_checks.csv', index=False)
        write_json(output / 'protocol.json', {'scope': 'serial hot-state inference benchmark',
            'states_per_model': len(states), 'warmups_per_state': warmups,
            'measured_repeats_per_state': repeats, 'cpu_threads': 1,
            'state_selection': 'folds 0/1/2, two validation sites each, seed 17, budgets 5/13',
            'timing': 'perf_counter, device synchronize at timing boundaries for CUDA; fresh condition every repeat',
            'excluded': 'data loading, input DataFrame copies, optional declared prepare_state, evaluation and calibration',
            'initialization': 'One measured construction + optional initialize() weight load per factory/fold; separate from warm inference; no cold-start distribution claimed. Framework import and CUDA context startup are outside factory timing.',
            'quantiles': 'Summary median/p95 over all 3 repetitions x 12 states; descriptive heterogeneous warm-state samples, not a stable tail-latency estimate',
            'pairing': 'Per-state medians old/new; ratios old divided by new. No cold reference versus warm optimized comparison.',
            'memory': 'RSS is absolute process working set sampled after inference, not model-exclusive or a per-model peak; CUDA peak allocated includes live weights and tensors and excludes reserved allocator memory',
            'prediction_cache': 'Final prediction files are not accepted by this runner.',
            'models': [{key: value for key, value in spec.__dict__.items() if key != 'factory'} for spec in specs],
            'hardware': hardware})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RUN)
    parser.add_argument('--registry', action='append', default=[],
                        help='Optional module:callable returning a list of BenchmarkSpec objects.')
    parser.add_argument('--groups', nargs='*', choices=['structural', 'joint', 'haar', 'recognition', 'tabpfn'],
                        default=['structural', 'joint', 'haar', 'recognition', 'tabpfn'])
    parser.add_argument('--without-baselines', action='store_true')
    args = parser.parse_args()
    specs = [] if args.without_baselines else baseline_specs()
    specs.extend(engineering_specs(args.groups))
    for entry in args.registry:
        module, callable_name = entry.split(':', 1)
        specs.extend(getattr(importlib.import_module(module), callable_name)())
    states = list(iter_validation_states(seeds=(17,), budgets=(5, 13)))
    if len(states) != 12:
        raise ValueError('The runtime comparison requires exactly 12 frozen states.')
    native_reference = None
    if any(spec.pair_id == 'balanced_full' for spec in specs):
        # This frozen reference is never returned to any predictor or adapter.
        reference = pd.read_parquet(PROJECT_ROOT / 'runs/v7-discovery-20261014/final_extended/predictions_oof.parquet',
            columns=[*STATE_KEYS, 'location_id', 'depth_index', 'p_low_1'],
            filters=[('model_id', '==', 'v6_empirical_graph_union_local')])
        native_reference = {}
        for state in states:
            mask = np.ones(len(reference), bool)
            for key in STATE_KEYS:
                mask &= reference[key].eq(state.metadata[key]).to_numpy()
            selected = reference.loc[mask].sort_values(['location_id', 'depth_index'])
            if len(selected) != len(state.query) or selected.p_low_1.isna().any():
                raise ValueError('The evaluation-only native risk reference is incomplete.')
            native_reference[state.key] = selected.p_low_1.to_numpy(float)
    run_specs(specs, states, args.output, native_reference=native_reference)


if __name__ == '__main__':
    main()
