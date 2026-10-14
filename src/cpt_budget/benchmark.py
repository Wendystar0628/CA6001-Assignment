"""Uniform catalogue runtime benchmark (protocol ported from v0.8-engineering src/cpt_budget/v8/benchmark.py).

Every catalogue entry is timed with real condition/predict calls on the same 12
frozen states (folds 0/1/2 x two validation sites x seed 17 x budgets 5/13),
1 warm-up + 3 measured repeats per state, fresh condition every repeat,
perf_counter with CUDA synchronisation at timing boundaries, one process pinned
to one logical core with 1 BLAS/OpenMP/torch thread. Weight loading
(construction + explicit initialize()) is timed separately. Paths:

* ``full``: the catalogue pipeline (point + native/shared risk head where the
  entry has one) on the entry's policy device (``device`` in configs/catalog.json:
  ``cuda`` = TabPFN components on CUDA, hard failure if CUDA is unavailable;
  all non-TabPFN numerical work stays on the pinned CPU core).
* ``point_only``: entries with ``point_only_supported`` only, all on CPU
  (tabpfn_joint therefore runs its TabPFN residual on CPU).

Outputs are compared with the frozen reference predictions after timing.
Evaluation, calibration and data loading are excluded.
"""
from __future__ import annotations
import ctypes
import gc
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits
from .config import PROJECT_ROOT, write_json
from .experiment import ASSETS, STATE_KEYS, catalog, iter_states

FIELDS = ['condition_ms', 'predict_ms', 'total_ms', 'rss_after_mib', 'cuda_peak_allocated_mib']


def resident_memory_mib():
    """Absolute process working set/RSS; this is not model-exclusive memory."""
    if os.name == 'nt':
        class Counters(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                        *[(name, ctypes.c_size_t) for name in (
                            'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
                            'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage',
                            'PeakPagefileUsage')]]
        value = Counters()
        value.cb = ctypes.sizeof(value)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb):
            return value.WorkingSetSize / 1024 ** 2
    elif Path('/proc/self/status').exists():
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return float(line.split()[1]) / 1024
    return float('nan')


def apply_cpu_affinity(core):
    """Restrict this benchmark process only; identify a logical CPU, not its type."""
    if core is None:
        return {'requested_logical_core': None, 'applied': False,
                'note': 'Unpinned process; OS may migrate work between logical CPUs.'}
    if int(core) != core or core < 0:
        raise ValueError('CPU affinity must name a nonnegative logical core.')
    core = int(core)
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                                  ctypes.POINTER(ctypes.c_size_t)]
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        handle = kernel.GetCurrentProcess()
        process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(handle, ctypes.byref(process_mask), ctypes.byref(system_mask)):
            raise OSError(ctypes.get_last_error(), 'Cannot read process CPU affinity.')
        previous = process_mask.value
        requested = 1 << core
        if not previous & requested:
            raise ValueError(f'Logical core {core} is not in allowed mask {previous:#x}.')
        if not kernel.SetProcessAffinityMask(handle, requested):
            raise OSError(ctypes.get_last_error(), 'Cannot set process CPU affinity.')
        if (not kernel.GetProcessAffinityMask(handle, ctypes.byref(process_mask), ctypes.byref(system_mask))
                or process_mask.value != requested):
            raise RuntimeError('CPU affinity verification failed; aborting benchmark.')
        return {'requested_logical_core': core, 'applied': True, 'previous_allowed_mask': hex(previous),
                'verified_process_mask': hex(process_mask.value), 'core_type': 'not inferred',
                'frequency_locked': False}
    if hasattr(os, 'sched_getaffinity') and hasattr(os, 'sched_setaffinity'):
        allowed = os.sched_getaffinity(0)
        if core not in allowed:
            raise ValueError(f'Logical core {core} is not in allowed set {sorted(allowed)}.')
        os.sched_setaffinity(0, {core})
        if os.sched_getaffinity(0) != {core}:
            raise RuntimeError('CPU affinity verification failed.')
        return {'requested_logical_core': core, 'applied': True, 'previous_allowed_cores': sorted(allowed),
                'verified_cores': [core], 'core_type': 'not inferred', 'frequency_locked': False}
    raise RuntimeError('This platform cannot enforce the requested CPU affinity.')


def _nvidia(query):
    try:
        return subprocess.run(['nvidia-smi', f'--query-{query}', '--format=csv,noheader'], capture_output=True,
                              text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def gpu_state():
    """GPU load snapshot; compute apps other than this process mean the GPU is not idle."""
    return {'gpu': _nvidia('gpu=name,driver_version,memory.total,memory.used,utilization.gpu,clocks.sm,clocks.max.sm'),
            'compute_apps': _nvidia('compute-apps=pid,process_name,used_memory'),
            'note': 'GPU/CPU clocks are not locked.'}


def _hardware():
    processor_name = platform.processor()
    if os.name == 'nt':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                processor_name = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    versions = {}
    for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'torch', 'tabpfn', 'xgboost', 'threadpoolctl'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, text=True, timeout=10).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain', 'src', 'configs'], cwd=PROJECT_ROOT,
                                             text=True, timeout=10).strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    return {'platform': platform.platform(), 'processor': processor_name, 'logical_cpu_count': os.cpu_count(),
            'python': sys.version, 'package_versions': versions, 'threadpools': threadpool_info(),
            'source_commit_at_run_start': commit, 'uncommitted_src_or_configs_changes': dirty}


def specs(models=None, *, point_only=True):
    """Catalogue-driven specs: all full paths first (catalogue order), then point-only CPU paths."""
    rows = catalog() if models is None else [r for r in catalog() if r['id'] in set(models)]
    if models is not None and len(rows) != len(set(models)):
        raise ValueError('Unknown catalogue ids in --models')
    full = []
    for r in rows:
        tabpfn = r['device'] == 'cuda'
        full.append({'model_id': r['id'], 'path': 'full', 'source_model_id': r['source_model_id'],
                     'device': r['device'], 'tabpfn_device': 'cuda' if tabpfn else None,
                     'risk_source': r['risk_source'],
                     'scope': ('point + shared empirical risk head (TabPFN graph-union point + empirical error CDF)'
                               if r['risk_source'] == 'shared_empirical_head' else
                               'point + native GP predictive distribution' if r['risk_source'] == 'native_gp' else
                               'point prediction (no native risk output; risk is evaluation-time Platt)'),
                     'device_policy': ('TabPFN on CUDA (required, no CPU fallback); other numerical work on the pinned '
                                       'CPU core with 1 thread' if tabpfn else
                                       'CPU only, pinned core, 1 thread' + (' (XGBoost n_jobs=1)' if r['id'] == 'xgb' else '')
                                       + (' (torch CPU, 1 thread)' if r['id'] == 'cnp' else ''))})
    extra = []
    if point_only:
        for r in rows:
            if r.get('point_only_supported'):
                extra.append({'model_id': r['id'], 'path': 'point_only', 'source_model_id': r['source_model_id'],
                              'device': 'cpu', 'tabpfn_device': 'cpu' if r['id'] == 'tabpfn_joint' else None,
                              'risk_source': None,
                              'scope': 'point path only (--point-only; shared risk head skipped)',
                              'device_policy': 'CPU only, pinned core, 1 thread' + (
                                  ' (TabPFN residual on CPU)' if r['id'] == 'tabpfn_joint' else '')})
    return full + extra


def _factory(spec, fold_id):
    from .runtime import make_model
    model = make_model(spec['model_id'], fold_id, point_only=spec['path'] == 'point_only',
                       device=spec['tabpfn_device'])
    regressor = getattr(model.predictor, 'regressor', None)
    if spec['model_id'] == 'xgb' and regressor is not None:
        regressor.set_params(n_jobs=1)
    return model


def _reference_lookup(models):
    reference = pd.read_parquet(ASSETS / 'evaluation/reference_predictions.parquet',
                                columns=[*STATE_KEYS, 'model_id', 'location_id', 'depth_index', 'pred_log_qc', 'p_low_1'])
    return reference[reference.model_id.isin(models)]


def _output_check(spec, state, frame, reference):
    mask = reference.model_id.eq(spec['model_id']).to_numpy().copy()
    for key in STATE_KEYS:
        mask &= reference[key].eq(state.metadata[key]).to_numpy()
    old = reference.loc[mask].set_index(['location_id', 'depth_index']).sort_index()
    new = frame.set_index(['location_id', 'depth_index']).sort_index()
    if not new.index.equals(old.index):
        raise AssertionError(f'Benchmark grid mismatch: {spec["model_id"]}/{state.key}')
    point = np.abs(new.pred_log_qc.to_numpy() - old.pred_log_qc.to_numpy())
    labels = int(((new.pred_log_qc.to_numpy() < 0) != (old.pred_log_qc.to_numpy() < 0)).sum())
    has_p = 'p_low_1' in new
    if has_p != (spec['path'] == 'full' and old.p_low_1.notna().all()):
        raise AssertionError(f'{spec["model_id"]}/{spec["path"]}: unexpected p_low_1 presence')
    probability = float(np.abs(new.p_low_1.to_numpy() - old.p_low_1.to_numpy()).max()) if has_p else None
    return {'model_id': spec['model_id'], 'path': spec['path'], 'device': spec['device'], 'state_key': state.key,
            'max_point_difference': float(point.max()), 'label_changes': labels,
            'max_native_probability_difference': probability,
            'matches_reference': bool(point.max() < 1e-5 and labels == 0 and (probability is None or probability < 1e-5))}


def _summaries(samples, initializations):
    sample_frame, init_frame = pd.DataFrame(samples), pd.DataFrame(initializations)
    keys = ['model_id', 'path']
    summary, states = [], []
    for (model_id, path), group in sample_frame.groupby(keys, sort=False):
        first = group.iloc[0]
        record = {'model_id': model_id, 'path': path, 'source_model_id': first.source_model_id,
                  'device': first.device, 'tabpfn_device': first.tabpfn_device, 'device_policy': first.device_policy,
                  'scope': first.scope, 'states': group.state_key.nunique(), 'measured_samples': len(group)}
        for field in FIELDS:
            record[field + '_median'] = group[field].median()
            record[field + '_p95'] = group[field].quantile(.95)
        init = init_frame[(init_frame.model_id == model_id) & (init_frame.path == path)]
        record['initialization_ms_median'] = init.initialization_ms.median()
        summary.append(record)
        for state_key, cells in group.groupby('state_key', sort=False):
            states.append({'model_id': model_id, 'path': path, 'device': first.device, 'state_key': state_key,
                           **{key: cells[key].iloc[0] for key in STATE_KEYS},
                           **{field: cells[field].median() for field in FIELDS}, 'samples': len(cells)})
    return sample_frame, init_frame, pd.DataFrame(states), pd.DataFrame(summary)


def run_catalog(models=None, output='runs/current/benchmark', *, cpu_core=None, repeats=3, warmups=1, point_only=True,
                publish=None, log=print):
    """Run sequentially; stop unrelated GPU/CPU jobs before invoking."""
    plan = specs(models, point_only=point_only)
    states = list(iter_states(seeds=[17], budgets=[5, 13]))
    if len(states) != 12:
        raise ValueError('The runtime protocol requires exactly 12 frozen states.')
    if repeats < 1 or warmups < 1:
        raise ValueError('Positive warm-ups and repeats are required.')
    output = PROJECT_ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    gpu_before = gpu_state()
    affinity = apply_cpu_affinity(cpu_core)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if any(s['tabpfn_device'] == 'cuda' for s in plan) and not torch.cuda.is_available():
        raise RuntimeError('CUDA policy entries require actual CUDA; CPU fallback is forbidden.')
    reference = _reference_lookup({s['model_id'] for s in plan})
    samples, initializations, checks = [], [], []
    started_all = time.perf_counter()
    with threadpool_limits(limits=1):
        hardware = _hardware()
        for spec in plan:
            cuda = spec['tabpfn_device'] == 'cuda'

            def sync():
                if cuda:
                    torch.cuda.synchronize()
            predictor, previous_fold = None, None
            try:
                for state in states:
                    fold = int(state.metadata['fold_id'])
                    if fold != previous_fold:
                        predictor = None
                        gc.collect()
                        if cuda:
                            torch.cuda.empty_cache()
                        sync()
                        start = time.perf_counter()
                        predictor = _factory(spec, fold)
                        sync()
                        after_factory = time.perf_counter()
                        predictor.initialize()
                        sync()
                        after_initialize = time.perf_counter()
                        initializations.append({'model_id': spec['model_id'], 'path': spec['path'], 'fold_id': fold,
                                                'device': spec['device'],
                                                'factory_ms': (after_factory - start) * 1000,
                                                'explicit_weight_initialization_ms': (after_initialize - after_factory) * 1000,
                                                'initialization_ms': (after_initialize - start) * 1000,
                                                'rss_after_initialization_mib': resident_memory_mib()})
                        previous_fold = fold
                    frame = None
                    for repeat in range(-warmups, repeats):
                        observed, query = state.observed.copy(), state.query.copy()
                        if cuda:
                            torch.cuda.reset_peak_memory_stats()
                        sync()
                        start = time.perf_counter()
                        predictor.condition(observed)
                        sync()
                        after_condition = time.perf_counter()
                        frame = predictor.predict(query).frame
                        sync()
                        after_prediction = time.perf_counter()
                        if repeat >= 0:
                            samples.append({**{k: spec[k] for k in ('model_id', 'path', 'source_model_id', 'device',
                                                                    'tabpfn_device', 'device_policy', 'scope')},
                                            'state_key': state.key, **{k: state.metadata[k] for k in STATE_KEYS},
                                            'repeat': repeat, 'condition_ms': (after_condition - start) * 1000,
                                            'predict_ms': (after_prediction - after_condition) * 1000,
                                            'total_ms': (after_prediction - start) * 1000,
                                            'rss_after_mib': resident_memory_mib(),
                                            'cuda_peak_allocated_mib': (torch.cuda.max_memory_allocated() / 1024 ** 2
                                                                        if cuda else float('nan'))})
                    checks.append(_output_check(spec, state, frame, reference))  # outside the timer
                if log:
                    log(f'benchmark {spec["model_id"]}/{spec["path"]} ({spec["device"]}) done')
                    sys.stdout.flush()
            finally:
                del predictor
                gc.collect()
                if cuda:
                    torch.cuda.empty_cache()
    sample_frame, init_frame, state_frame, summary = _summaries(samples, initializations)
    check_frame = pd.DataFrame(checks)
    agreement = check_frame.groupby(['model_id', 'path'], sort=False).agg(
        max_point_difference=('max_point_difference', 'max'), label_changes=('label_changes', 'sum'),
        max_native_probability_difference=('max_native_probability_difference', 'max'),
        matches_reference=('matches_reference', 'all')).reset_index()
    summary = summary.merge(agreement, on=['model_id', 'path'], how='left')
    sample_frame.to_csv(output / 'samples.csv', index=False)
    init_frame.to_csv(output / 'initializations.csv', index=False)
    state_frame.to_csv(output / 'state_summary.csv', index=False)
    summary.to_csv(output / 'summary.csv', index=False)
    check_frame.to_csv(output / 'output_checks.csv', index=False)
    protocol = {
        'scope': 'v0.9.1 uniform serial hot-state inference benchmark over every catalogue entry',
        'command': 'python -m cpt_budget benchmark' + (f' --cpu-core {cpu_core}' if cpu_core is not None else ''),
        'ported_from': 'v0.8-engineering src/cpt_budget/v8/benchmark.py (run_specs, apply_cpu_affinity, _hardware)',
        'states_per_path': len(states), 'warmups_per_state': warmups, 'measured_repeats_per_state': repeats,
        'cpu_threads': 1, 'cpu_affinity': affinity,
        'execution_order': [f"{s['model_id']}/{s['path']}" for s in plan],
        'device_policy': {
            'full': 'catalogue device: cuda = TabPFN residual components on CUDA (hard failure without CUDA); '
                    'cpu = CPU only. All other numerical work runs on the pinned CPU core with 1 thread.',
            'point_only': 'CPU only for every entry with point_only_supported; tabpfn_joint runs TabPFN on CPU.',
            'xgb': 'XGBoost n_jobs forced to 1 for timing (configs/main.yaml keeps 4 for normal runs; outputs identical).',
            'cnp': 'torch CPU (hard-coded in runtime).'},
        'state_selection': 'folds 0/1/2, two validation sites each, seed 17, budgets 5/13',
        'timing': 'perf_counter, torch.cuda.synchronize at timing boundaries for CUDA paths; fresh condition every repeat',
        'excluded': 'data loading, input DataFrame copies, output checks against the reference, evaluation and calibration',
        'initialization': 'One construction + explicit initialize() (TabPFN weight load) per path and fold, '
                          'separate from warm inference. Framework import and CUDA context start-up are outside timing.',
        'quantiles': 'median/p95 over 3 repeats x 12 states; descriptive warm-state samples, not a stable tail estimate',
        'memory': 'RSS is the absolute process working set after inference; CUDA peak allocated includes live weights',
        'remaining_timing_limits': 'Entries run sequentially in the recorded order; CPU/GPU frequencies are not locked; '
                                   'order effects remain possible.',
        'output_check': 'Each state output compared with artifacts/evaluation/reference_predictions.parquet after timing '
                        '(output_checks.csv; summary columns matches_reference etc.).',
        'gpu_before': gpu_before, 'gpu_after': gpu_state(), 'wall_seconds': time.perf_counter() - started_all,
        'hardware': hardware}
    write_json(output / 'protocol.json', protocol)
    if publish:
        target = PROJECT_ROOT / publish
        for source, name in [('summary.csv', 'runtime_summary.csv'), ('state_summary.csv', 'runtime_state_summary.csv'),
                             ('samples.csv', 'runtime_samples.csv'), ('initializations.csv', 'runtime_initializations.csv'),
                             ('output_checks.csv', 'runtime_output_checks.csv'), ('protocol.json', 'runtime_protocol.json')]:
            shutil.copy2(output / source, target / name)
    return summary
