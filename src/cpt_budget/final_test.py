"""Final independent outer test, pre-registered in docs/FINAL_TEST_PROTOCOL.md.

Stages (CLI ``final-test``):
  --prepare  test states (coordinates only), artifact leakage audit, per-fold risk
             calibrators fitted on that fold's validation labels only.
  --blind    all catalogue entries on all test states. Structurally truth-free: the
             evaluation cells of every state are removed from the in-memory grid
             before any model call, and the truth loader is never called.
  --freeze   SHA-256 of protocol, test states, catalogue, calibrators, code tree,
             model artifacts and blind predictions -> results/final_test/frozen.json.
  --unblind  refuses unless every frozen hash matches; logs the run; refuses a
             second run unless --force-rerun (then marked as a rerun).
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import gc
import getpass
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, load_config, write_json
from .experiment import ASSETS, BUDGETS, STATE_KEYS, catalog
from .types import PreparedDataset, QUERY_COLUMNS, load_dataset

# ---------------------------------------------------------------- fixed protocol
# Test site -> the only outer fold in which it is a TEST group (folds.json).
TEST_SITES = {'christchurch_stadium': 0, 'wellington_airport': 0, 'newlock': 1, 'hollywood': 2}
DISPLAY = {'christchurch_stadium': 'Chch Stadium', 'wellington_airport': 'Wellington Airport',
           'newlock': 'New Lock', 'hollywood': 'Hollywood'}
# Replay-eligible sites that are excluded, with the reason (they are TEST groups in some fold too).
EXCLUDED_SITES = {
    'lower_hutt_town_hall': 'validation site of the 60 frozen states (v0.9.1 model selection/validation)',
    'tauranga_farmers_building': 'validation site of the 60 frozen states (v0.9.1 model selection/validation)',
    'christchurch_metro_sports_facility': 'validation site of the 60 frozen states (v0.9.1 model selection/validation)',
    'marshland': 'validation site of the 60 frozen states (v0.9.1 model selection/validation)'}
VERIFIED_SUBSET = ('christchurch_stadium', 'wellington_airport')  # grouping_verified sensitivity analysis
SEEDS = (17, 29, 43, 71, 101)
POLICY = 'maximin'
RISK_THRESHOLD = 0.2
MEASUREMENTS = ['qc_mpa', 'log_qc', 'low_0_5', 'thickness_0_5_m', 'low_1', 'thickness_1_m', 'low_2', 'thickness_2_m']

FINAL_DIR = PROJECT_ROOT / 'results/final_test'
TEST_STATES = ASSETS / 'evaluation/test_states.parquet'
PROTOCOL = PROJECT_ROOT / 'docs/FINAL_TEST_PROTOCOL.md'
CALIBRATORS = FINAL_DIR / 'calibrators.json'
FROZEN = FINAL_DIR / 'frozen.json'
BLIND_DIR = FINAL_DIR / 'blind'
UNBLIND_DIR = FINAL_DIR / 'unblind'
RUN_LOG = FINAL_DIR / 'unblind_log.jsonl'
VALIDATION_RANKING = PROJECT_ROOT / 'results/current/all_model_ranking.csv'
REFERENCE = ASSETS / 'evaluation/reference_predictions.parquet'

# Metrics and direction (True = lower is better), as in analysis.METRICS.
DIRECTION = {'log_mae': True, 'h_point_mae_m': True, 'recall': False, 'false_positive_rate': True,
             'risk_recall': False, 'risk_fpr': True, 'risk_brier': True, 'coverage_90': None, 'width_90_log': True}


class TruthAccessError(RuntimeError):
    """Raised when the blind path would touch test-location truth."""


def _now():
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec='seconds')


def _folds():
    return json.loads((ASSETS / 'evaluation/folds.json').read_text(encoding='utf-8'))


# ---------------------------------------------------------------- site roles
def site_roles(dataset=None):
    """Role of every replay-eligible site in every fold, and the test decision."""
    dataset = dataset or coordinate_dataset()
    folds = _folds()
    frozen = pd.read_parquet(ASSETS / 'evaluation/states.parquet')
    eligible = dataset.sites.loc[dataset.sites.replay_eligible.astype(bool)].set_index('site_id')
    rows = []
    for site, info in eligible.iterrows():
        row = {'site_id': site, 'display_name': info.display_name, 'investigation_group': info.investigation_group,
               'n_locations': int(info.n_locations), 'n_near_groups': int(info.n_near_groups),
               'grouping_verified': bool(info.grouping_verified)}
        for fold in folds['folds']:
            role = [r for r in ('train', 'validation', 'test') if site in fold[f'{r}_sites']]
            if len(role) != 1:
                raise ValueError(f'{site} has no unique role in fold {fold["fold_id"]}')
            row[f'fold_{fold["fold_id"]}'] = role[0]
        used = frozen[frozen.site_id.eq(site)]
        row['in_frozen_validation_states'] = bool(len(used))
        row['frozen_validation_folds'] = ','.join(map(str, sorted(used.fold_id.unique())))
        row['test_fold'] = [f['fold_id'] for f in folds['folds'] if site in f['test_sites']][0]
        row['included'] = site in TEST_SITES
        row['reason'] = ('outer-test site: not used for v0.9.1 validation/selection' if site in TEST_SITES
                         else EXCLUDED_SITES.get(site, 'UNCLASSIFIED'))
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values('site_id').reset_index(drop=True)
    derived = set(frame.loc[~frame.in_frozen_validation_states, 'site_id'])
    if derived != set(TEST_SITES) or set(frame.loc[frame.in_frozen_validation_states, 'site_id']) != set(EXCLUDED_SITES):
        raise AssertionError(f'Derived test sites {sorted(derived)} differ from the protocol {sorted(TEST_SITES)}')
    for site, fold in TEST_SITES.items():
        if frame.set_index('site_id').loc[site, 'test_fold'] != fold:
            raise AssertionError(f'{site}: protocol fold {fold} is not its test fold')
    return frame


# ---------------------------------------------------------------- states
def coordinate_dataset(config=None):
    """Dataset view with NO measurement columns: generator and roles see coordinates only."""
    config = config or load_config()
    full = load_dataset(config)
    grid = full.grid[['site_id', 'location_id']].drop_duplicates().reset_index(drop=True)
    return PreparedDataset(full.profiles, grid, full.sites, full.root)


def _rollout(split, locations, config):
    from .acquisition import select_batch
    observed = list(split['initial_ids'])
    sequence = {split['budgets'][0]: list(observed)}
    for budget in split['budgets'][1:]:
        remaining = sorted(set(split['pool_ids']) - set(observed))
        selected, _ = select_batch(locations[locations.location_id.isin(observed)],
                                   locations[locations.location_id.isin(remaining)], None, config,
                                   policy=POLICY, min_spacing_m=split['min_spacing_m'], batch_size=2)
        observed = observed + selected
        sequence[budget] = list(observed)
    return sequence


def _state_id(ids):
    return hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()[:16]


def _splits(dataset, config, seeds):
    from .data.splits import make_replay_splits
    config = json.loads(json.dumps({k: v for k, v in config.items()}))
    config['replay']['seeds'] = list(seeds)
    splits, skipped = make_replay_splits(dataset, config)
    return splits, skipped, config


def reproduce_frozen_states(dataset=None, config=None):
    """The generator must reproduce all 60 frozen validation states exactly."""
    config = config or load_config()
    dataset = dataset or coordinate_dataset(config)
    frozen = pd.read_parquet(ASSETS / 'evaluation/states.parquet')
    splits, _, config = _splits(dataset, config, sorted(frozen.seed.unique()))
    checked = mismatches = 0
    for split in splits:
        rows = frozen[frozen.site_id.eq(split['site_id']) & frozen.seed.eq(split['seed'])]
        if rows.empty:
            continue
        locations = dataset.locations(split['site_id'])[['location_id', 'x_m', 'y_m']]
        sequence = _rollout(split, locations, config)
        for row in rows.itertuples():
            checked += 1
            mismatches += not (list(row.observed_ids) == sequence[row.budget_n]
                               and sorted(row.evaluation_ids) == sorted(split['evaluation_ids']))
    return {'checked': checked, 'mismatches': mismatches}


def generate_test_states(write=True):
    config = load_config()
    dataset = coordinate_dataset(config)
    if tuple(config['replay']['budgets']) != BUDGETS or config['replay']['initial_locations'] != 5:
        raise ValueError('Configured budgets/initial count differ from the frozen protocol')
    check = reproduce_frozen_states(dataset, config)
    if check['checked'] != 60 or check['mismatches']:
        raise AssertionError(f'Generator does not reproduce the frozen validation states: {check}')
    splits, skipped, config = _splits(dataset, config, SEEDS)
    bad = [s for s in skipped if s['site_id'] in TEST_SITES]
    if bad:
        raise ValueError(f'Test-site splits could not be generated: {bad}')
    rows = []
    for split in splits:
        site = split['site_id']
        if site not in TEST_SITES:
            continue
        locations = dataset.locations(site)
        sequence = _rollout(split, locations[['location_id', 'x_m', 'y_m']], config)
        groups = locations.set_index('location_id').near_group_id
        for round_id, budget in enumerate(BUDGETS):
            observed = sequence[budget]
            if set(groups.loc[observed]) & set(groups.loc[split['evaluation_ids']]):
                raise AssertionError('Observed and evaluation near groups overlap')
            rows.append({'fold_id': TEST_SITES[site], 'site_id': site, 'seed': int(split['seed']), 'budget_n': budget,
                         'policy_id': POLICY, 'round_id': round_id, 'investigation_group': split['investigation_group'],
                         'grouping_verified': bool(split['grouping_verified']),
                         'observed_set_id': _state_id(observed), 'observed_ids': list(observed),
                         'remaining_ids': sorted(set(split['pool_ids']) - set(observed)),
                         'evaluation_ids': list(split['evaluation_ids']),
                         'evaluation_near_groups': list(split['evaluation_near_groups']),
                         'n_evaluation': int(split['n_evaluation']),
                         'evaluation_fraction_actual': float(split['evaluation_fraction_actual']),
                         'min_spacing_m': float(split['min_spacing_m']), 'role': 'test',
                         'generator': 'data.splits.make_replay_splits + acquisition.select_batch(maximin, predictions=None)'})
    states = pd.DataFrame(rows).sort_values(STATE_KEYS).reset_index(drop=True)
    expected = len(TEST_SITES) * len(SEEDS) * len(BUDGETS)
    if len(states) != expected or states.duplicated(STATE_KEYS).any():
        raise AssertionError(f'Expected {expected} unique test states, got {len(states)}')
    if write:
        TEST_STATES.parent.mkdir(parents=True, exist_ok=True)
        states.to_parquet(TEST_STATES, index=False)
        _register_manifest(TEST_STATES, 'final outer test states (final-test --prepare; seeds 17/29/43/71/101, maximin, budgets 5-13)')
    return states, check


def _register_manifest(path, origin):
    manifest_path = ASSETS / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    record = {'path': rel, 'bytes': path.stat().st_size, 'origin': origin, 'sha256': sha256_file(path)}
    files = [r for r in manifest['files'] if r['path'] != rel] + [record]
    manifest['files'] = files
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def load_test_states():
    states = pd.read_parquet(TEST_STATES)
    expected = len(TEST_SITES) * len(SEEDS) * len(BUDGETS)
    if len(states) != expected or states.duplicated(STATE_KEYS).any() or not states.role.eq('test').all():
        raise ValueError('Invalid final test-state manifest')
    if dict(states.groupby('site_id').fold_id.first()) != TEST_SITES or states.groupby('site_id').fold_id.nunique().ne(1).any():
        raise ValueError('Test states must use exactly the protocol fold of each test site')
    return states


# ---------------------------------------------------------------- blind inputs
class BlindDataset:
    """Per-state view: evaluation cells carry no measurements; truth() always raises."""

    def __init__(self, dataset):
        self._dataset = dataset
        # Geometry-only copy of the grid: the query side never sees a measurement column.
        geometry = dataset.grid[[c for c in dataset.grid if c not in MEASUREMENTS]].copy()
        self._geometry = PreparedDataset(dataset.profiles, geometry, dataset.sites, dataset.root)

    def truth(self, *_args, **_kwargs):
        raise TruthAccessError('Blind final-test path must not read test truth')

    def state_inputs(self, row):
        observed_ids, evaluation_ids = list(row.observed_ids), list(row.evaluation_ids)
        if set(observed_ids) & set(evaluation_ids):
            raise TruthAccessError('An evaluation location is in the revealed set')
        base = self._dataset
        # The revealed side is drawn from a grid from which every evaluation row was removed.
        masked = PreparedDataset(base.profiles, base.grid.loc[~base.grid.location_id.isin(evaluation_ids)],
                                 base.sites, base.root)
        observed = masked.observed(observed_ids)
        if set(observed.location_id) & set(evaluation_ids) or observed[['qc_mpa', 'log_qc']].isna().any().any():
            raise TruthAccessError('Revealed profiles must not include evaluation cells')
        query = self._geometry.query(evaluation_ids)
        if set(query) - set(QUERY_COLUMNS) or set(query) & set(MEASUREMENTS):
            raise TruthAccessError('Query frame carries non-geometry columns')
        if set(observed.near_group_id) & set(query.near_group_id):
            raise ValueError('Support/query near groups overlap')
        return observed, query


def iter_blind_states(states, dataset):
    view = BlindDataset(dataset)
    for row in states.sort_values(STATE_KEYS).itertuples(index=False):
        observed, query = view.state_inputs(row)
        meta = {k: getattr(row, k) for k in STATE_KEYS}
        meta.update(observed_set_id=row.observed_set_id, investigation_group=row.investigation_group,
                    policy_id=POLICY, role='evaluation', split_role='test')
        yield meta, observed, query


@contextlib.contextmanager
def forbid_truth():
    """Belt and braces: any PreparedDataset.truth call during the blind run raises."""
    original = PreparedDataset.truth

    def _raise(self, *args, **kwargs):
        raise TruthAccessError('PreparedDataset.truth called during the blind final test')
    PreparedDataset.truth = _raise
    try:
        yield
    finally:
        PreparedDataset.truth = original


# ---------------------------------------------------------------- risk calibration
def fit_test_calibrators(models=None, write=True):
    """Per entry and fold: monotone Platt fitted on ALL validation states of that fold
    (reference predictions; validation labels only), threshold 0.2."""
    from .report import RISK_SOURCES, _score_frame, entries
    from .risk import fit_risk_calibrator
    reference = pd.read_parquet(REFERENCE)
    folds = _folds()['folds']
    result = {}
    for record in entries(models):
        for fold in sorted(set(TEST_SITES.values())):
            frame = reference[reference.model_id.eq(record['id']) & reference.fold_id.eq(fold)]
            if len(frame.drop_duplicates(STATE_KEYS)) != 20:
                raise ValueError(f'{record["id"]} fold {fold}: expected 20 validation states')
            calibrator = fit_risk_calibrator(_score_frame(frame, record))
            if calibrator['score_kind'] != RISK_SOURCES[record['risk_source']]:
                raise AssertionError(f'{record["id"]}: wrong calibration score kind')
            if set(calibrator['training_sites']) & set(TEST_SITES):
                raise AssertionError(f'{record["id"]} fold {fold}: calibrator saw a test site')
            if not set(calibrator['training_sites']) <= set(folds[fold]['validation_sites']):
                raise AssertionError(f'{record["id"]} fold {fold}: calibrator sites are not validation sites')
            calibrator.update(model_id=record['id'], fold_id=fold, risk_source=record['risk_source'],
                              fit_source='artifacts/evaluation/reference_predictions.parquet (validation states of this fold)')
            result[f'{record["id"]}|{fold}'] = calibrator
    if write:
        write_json(CALIBRATORS, {'threshold': RISK_THRESHOLD, 'created': _now(), 'calibrators': result})
    return result


def apply_test_risk(frame, record, calibrator):
    from .report import _score_frame
    from .risk import apply_risk
    scored = apply_risk(_score_frame(frame, record), calibrator, threshold=RISK_THRESHOLD)
    out = frame.copy()
    out['risk_probability'] = scored.risk_probability.to_numpy(float)
    out['risk_label'] = scored.risk_label.to_numpy(bool)
    out['risk_threshold'] = RISK_THRESHOLD
    return out


# ---------------------------------------------------------------- leakage audit
def audit_artifacts(write=True):
    """Every inference artifact of a test site's fold must be fitted without that site."""
    from .experiment import model_asset
    from .models.gp import fit_gp_trend
    folds = _folds()['folds']
    calibrators = json.loads(CALIBRATORS.read_text(encoding='utf-8'))['calibrators'] if CALIBRATORS.is_file() else {}
    rows, gp_checks = [], {}
    full = None
    for site, fold_id in TEST_SITES.items():
        fold = folds[fold_id]
        checks = {'folds.json role': [site] if site in fold['test_sites'] else []}
        if site in fold['train_sites'] or site in fold['validation_sites']:
            checks['folds.json train/validation'] = [site]
        basis = json.loads(model_asset(fold_id, 'basis_manifest.json').read_text(encoding='utf-8'))
        sources = {
            'basis_manifest.source_site_ids (FPCA)': basis['source_site_ids'],
            'tabpfn_training.train_sites': None, 'tabpfn_training.validation_sites': None,
            'tabpfn_training.exposure.per_site': None,
            'xgb.train_sites': None, 'xgb.validation_sites': None,
            'cnp.train_sites': None, 'cnp.validation_sites': None, 'cnp.source_normalizer_sites': None}
        tab = json.loads(model_asset(fold_id, 'tabpfn_training.json').read_text(encoding='utf-8'))
        xgb = json.loads(model_asset(fold_id, 'xgb.json').read_text(encoding='utf-8'))
        cnp = json.loads(model_asset(fold_id, 'cnp_training.json').read_text(encoding='utf-8'))
        sources.update({'tabpfn_training.train_sites': tab['train_sites'], 'tabpfn_training.validation_sites': tab['validation_sites'],
                        'tabpfn_training.exposure.per_site': list(tab['exposure']['per_site']),
                        'xgb.train_sites': xgb['train_sites'], 'xgb.validation_sites': xgb['validation_sites'],
                        'cnp.train_sites': cnp['train_sites'], 'cnp.validation_sites': cnp['validation_sites'],
                        'cnp.source_normalizer_sites': cnp['source_normalizer_sites']})
        for name, values in sources.items():
            if site in values:
                checks[name] = [site]
        # Fold integrity: every training list is inside this fold's train/validation sites.
        allowed = set(fold['train_sites']) | set(fold['validation_sites'])
        outside = {name: sorted(set(v) - allowed) for name, v in sources.items() if set(v) - allowed}
        if fold_id not in gp_checks:
            if full is None:
                full = load_dataset(load_config())
            stored = pd.Series({int(k): v for k, v in json.loads(model_asset(fold_id, 'gp_trend.json').read_text()).items()}).sort_index()
            refit = fit_gp_trend(full, fold['train_sites']).sort_index()
            gp_checks[fold_id] = float(np.abs(stored.to_numpy() - refit.to_numpy()).max())
        cal_sites = sorted({s for key, c in calibrators.items() if key.endswith(f'|{fold_id}') for s in c['training_sites']})
        if site in cal_sites:
            checks['risk calibrators.training_sites'] = [site]
        leaks = {k: v for k, v in checks.items() if k != 'folds.json role' and v}
        rows.append({'site_id': site, 'test_fold': fold_id, 'is_test_group_in_fold': bool(checks['folds.json role']),
                     'leaks': leaks, 'lists_outside_fold_sources': outside,
                     'gp_trend_refit_from_fold_train_sites_max_abs_diff': gp_checks[fold_id],
                     'gp_trend_fit_sites': 'folds.json train_sites (fit_gp_trend, equal site weights)',
                     'risk_calibrator_training_sites': cal_sites,
                     'runtime_target_only_components': 'simplex/Haar/structural stacking weights/joint error atoms/'
                                                       'graph/IDW/GP hyperparameters are fitted per state on revealed CPTs only',
                     'graph_planned_geometry': 'coordinates of all locations of the site (no measurements), as for validation',
                     'passed': bool(checks['folds.json role']) and not leaks and not outside
                     and gp_checks[fold_id] < 1e-12})
    document = {'created': _now(), 'all_passed': all(r['passed'] for r in rows), 'sites': rows,
                'note': 'The GP trend refit reads the training-site labels of the fold (already embedded in the stored '
                        'artifact); only the max absolute difference is recorded.'}
    if write:
        write_json(FINAL_DIR / 'leakage_audit.json', document)
    if not document['all_passed']:
        raise AssertionError(f'Leakage audit failed: {rows}')
    return document


def prepare():
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    if FROZEN.is_file():
        raise RuntimeError('The protocol is frozen; --prepare would change frozen inputs')
    roles = site_roles()
    roles.to_csv(FINAL_DIR / 'site_roles.csv', index=False)
    states, check = generate_test_states(write=True)
    fit_test_calibrators(write=True)
    audit = audit_artifacts(write=True)
    summary = {'test_states': len(states), 'distinct_observed_sets': int(states.observed_set_id.nunique()),
               'distinct_evaluation_sets': int(states.drop_duplicates(['site_id', 'seed']).evaluation_ids.map(lambda v: tuple(sorted(v))).nunique()),
               'per_site': states.drop_duplicates(['site_id', 'seed']).groupby('site_id').n_evaluation.agg(['min', 'max', 'sum']).to_dict('index'),
               'frozen_state_reproduction': check, 'leakage_audit_passed': audit['all_passed']}
    write_json(FINAL_DIR / 'prepare_summary.json', summary)
    print(json.dumps(summary, indent=2, default=str))
    return summary


# ---------------------------------------------------------------- blind run
NATIVE = ['p_low_1', 'q05_log', 'q10_log', 'q25_log', 'q75_log', 'q90_log', 'q95_log']
GEOMETRY = ['site_id', 'location_id', 'profile_id', 'near_group_id', 'depth_index', 'z_center_m', 'x_m', 'y_m', 'geometry_scale_m']
META = ['fold_id', 'seed', 'budget_n', 'policy_id', 'investigation_group', 'observed_set_id', 'role', 'split_role']


def _sanity(frame, query, record):
    keys = ['location_id', 'depth_index']
    expected = query[keys].sort_values(keys).reset_index(drop=True)
    got = frame[keys].sort_values(keys).reset_index(drop=True)
    point = frame[['pred_log_qc', 'pred_qc_mpa']].to_numpy(float)
    native = [c for c in NATIVE if c in frame and frame[c].notna().any()]
    row = {'rows': len(frame), 'expected_rows': len(query), 'keys_match': bool(expected.equals(got)),
           'n_nan_point': int((~np.isfinite(point)).sum()), 'pred_log_qc_min': float(np.nanmin(point[:, 0])),
           'pred_log_qc_max': float(np.nanmax(point[:, 0])), 'pred_qc_positive': bool((point[:, 1] > 0).all()),
           'native_columns': ','.join(native), 'n_pred_low': int((point[:, 0] < 0).sum())}
    if 'p_low_1' in native:
        p = frame.p_low_1.to_numpy(float)
        row.update(p_low_min=float(p.min()), p_low_max=float(p.max()), p_low_valid=bool(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()))
    expect_p = record['risk_source'] != 'platt_on_point'
    row['native_schema_ok'] = (('p_low_1' in native) == expect_p) and (
        (record['id'] == 'gp') == all(c in native for c in NATIVE[1:]))
    if record['id'] == 'gp':
        q = frame[NATIVE[1:]].to_numpy(float)
        row['quantiles_ordered'] = bool(np.isfinite(q).all() and (np.diff(q, axis=1) >= -1e-12).all())
    r = frame.risk_probability.to_numpy(float)
    row.update(risk_probability_min=float(r.min()), risk_probability_max=float(r.max()),
               n_risk_label=int(frame.risk_label.sum()))
    row['passed'] = bool(row['keys_match'] and row['rows'] == row['expected_rows'] and not row['n_nan_point']
                         and row['pred_qc_positive'] and row['native_schema_ok'] and row.get('p_low_valid', True)
                         and row.get('quantiles_ordered', True) and np.isfinite(r).all() and (r >= 0).all() and (r <= 1).all())
    return row


def run_blind(models=None, sites=None, seeds=None, budgets=None, output=None, log=print):
    from threadpoolctl import threadpool_limits
    from .report import entries
    from .runtime import make_model
    import torch
    output = Path(output) if output else BLIND_DIR
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if output.resolve() == BLIND_DIR.resolve() and FROZEN.is_file():
        raise RuntimeError('Frozen blind predictions exist; write a re-run to another --output')
    states = load_test_states()
    for column, values in (('site_id', sites), ('seed', seeds), ('budget_n', budgets)):
        if values is not None:
            states = states[states[column].isin(values)]
    if states.empty:
        raise ValueError('No test states selected')
    calibrators = json.loads(CALIBRATORS.read_text(encoding='utf-8'))
    if calibrators['threshold'] != RISK_THRESHOLD:
        raise ValueError('Calibrator threshold differs from the protocol')
    dataset = load_dataset(load_config())
    blind_states = list(iter_blind_states(states, dataset))
    del dataset
    gc.collect()
    frames, sanity, runtime, cuda_peak = [], [], [], {}
    started = time.perf_counter()
    with forbid_truth(), threadpool_limits(limits=1):
        for record in entries(models):
            device = 'cuda' if record['device'] == 'cuda' else None
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                baseline_bytes = torch.cuda.memory_allocated()
            predictor, last = None, None
            for meta, observed, query in blind_states:
                if meta['fold_id'] != last:
                    predictor = None
                    gc.collect()
                    t0 = time.perf_counter()
                    predictor = make_model(record['id'], meta['fold_id'], device=device)
                    predictor.initialize()
                    runtime.append({'model_id': record['id'], 'fold_id': meta['fold_id'], 'stage': 'build_and_initialize',
                                    'seconds': time.perf_counter() - t0})
                    last = meta['fold_id']
                t0 = time.perf_counter()
                predictor.condition(observed)
                frame = predictor.predict(query).frame
                if record['device'] == 'cuda':
                    torch.cuda.synchronize()
                seconds = time.perf_counter() - t0
                frame = frame.merge(query[[c for c in GEOMETRY if c not in ('location_id', 'depth_index')] + ['location_id', 'depth_index']],
                                    on=['location_id', 'depth_index'], how='left', suffixes=('_pred', ''), validate='one_to_one')
                frame = frame.drop(columns=[c for c in frame if c.endswith('_pred')])
                for key in META + ['site_id']:
                    frame[key] = meta[key]
                frame['model_id'] = record['id']
                frame['source_model_id'] = record['source_model_id']
                frame = apply_test_risk(frame, record, calibrators['calibrators'][f'{record["id"]}|{meta["fold_id"]}'])
                keep = [c for c in GEOMETRY + ['pred_log_qc', 'pred_qc_mpa'] + NATIVE + ['risk_probability', 'risk_label', 'risk_threshold']
                        + META + ['model_id', 'source_model_id'] if c in frame]
                if set(frame) & set(MEASUREMENTS + ['truth_qc_mpa', 'truth_log_qc']):
                    raise TruthAccessError('A measurement column reached the prediction frame')
                frame = frame[keep]
                frames.append(frame)
                sanity.append({'model_id': record['id'], **{k: meta[k] for k in STATE_KEYS}, **_sanity(frame, query, record)})
                runtime.append({'model_id': record['id'], **{k: meta[k] for k in STATE_KEYS}, 'stage': 'condition_predict',
                                'seconds': seconds})
            if torch.cuda.is_available():
                cuda_peak[record['id']] = {'catalog_device': record['device'],
                                           'cuda_peak_extra_mb': (torch.cuda.max_memory_allocated() - baseline_bytes) / 2 ** 20}
            if log:
                log(f'blind: {record["id"]:18} {len(blind_states)} states')
    total = time.perf_counter() - started
    predictions = pd.concat(frames, ignore_index=True)
    order = {r['id']: i for i, r in enumerate(catalog())}
    predictions = predictions.sort_values(['model_id'] + STATE_KEYS + ['location_id', 'depth_index'],
                                          key=lambda s: s.map(order) if s.name == 'model_id' else s, kind='stable').reset_index(drop=True)
    sanity = pd.DataFrame(sanity)
    runtime = pd.DataFrame(runtime)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output / 'predictions.parquet', index=False)
    sanity.to_csv(output / 'sanity.csv', index=False)
    runtime.to_csv(output / 'runtime.csv', index=False)
    per_model = runtime[runtime.stage.eq('condition_predict')].groupby('model_id').seconds.agg(['count', 'mean', 'median', 'max', 'sum'])
    summary = {'created': _now(), 'models': [r['id'] for r in entries(models)], 'states': len(blind_states),
               'prediction_rows': len(predictions), 'all_sanity_passed': bool(sanity.passed.all()),
               'n_sanity_failed': int((~sanity.passed).sum()), 'total_seconds': total,
               'tabpfn_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
               'cuda_available': bool(torch.cuda.is_available()),
               'per_model_state_seconds': per_model.round(4).to_dict('index'),
               'cuda_memory_evidence': cuda_peak,
               'cuda_used_where_catalog_says': all((v['cuda_peak_extra_mb'] > 1) == (v['catalog_device'] == 'cuda')
                                                   for v in cuda_peak.values()) if cuda_peak else False,
               'predictions_sha256': sha256_file(output / 'predictions.parquet'),
               'prediction_content_sha256': frame_hash(predictions),
               'truth_access': 'none: evaluation cells masked before model calls; PreparedDataset.truth patched to raise'}
    write_json(output / 'blind_summary.json', summary)
    if log:
        log(json.dumps({k: summary[k] for k in ('states', 'prediction_rows', 'all_sanity_passed', 'total_seconds', 'tabpfn_device')}))
    if not summary['all_sanity_passed']:
        raise AssertionError('Blind sanity checks failed; see sanity.csv')
    return predictions, sanity, summary


# ---------------------------------------------------------------- hashing / freezing
def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def frame_hash(frame):
    """Content hash independent of Parquet encoding (values rounded to 1e-9)."""
    values = frame.copy()
    for column in values:
        if pd.api.types.is_float_dtype(values[column]):
            values[column] = values[column].round(9)
    return hashlib.sha256(pd.util.hash_pandas_object(values, index=False).to_numpy().tobytes()).hexdigest()


def tree_hash(paths):
    digest = hashlib.sha256()
    listing = {}
    for path in sorted(paths, key=lambda p: p.relative_to(PROJECT_ROOT).as_posix()):
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        listing[rel] = sha256_file(path)
        digest.update(f'{rel}\0{listing[rel]}\n'.encode())
    return digest.hexdigest(), listing


def code_files():
    files = [p for p in (PROJECT_ROOT / 'src/cpt_budget').rglob('*.py') if '__pycache__' not in p.parts]
    return files + [PROJECT_ROOT / 'configs/main.yaml', PROJECT_ROOT / 'pyproject.toml']


def artifact_files():
    files = sorted((ASSETS / 'models').rglob('*'))
    files = [p for p in files if p.is_file()]
    return files + [ASSETS / 'evaluation/folds.json', ASSETS / 'evaluation/states.parquet', REFERENCE] + [
        PROJECT_ROOT / 'data_processed_v2' / f'{n}.parquet' for n in ('profiles', 'grid', 'sites')]


def current_hashes(blind_dir=None):
    blind_dir = Path(blind_dir) if blind_dir else BLIND_DIR
    code, code_listing = tree_hash(code_files())
    artifacts, artifact_listing = tree_hash(artifact_files())
    return {'protocol_sha256': sha256_file(PROTOCOL), 'test_states_sha256': sha256_file(TEST_STATES),
            'catalog_sha256': sha256_file(PROJECT_ROOT / 'configs/catalog.json'),
            'calibrators_sha256': sha256_file(CALIBRATORS),
            'code_tree_sha256': code, 'artifact_tree_sha256': artifacts,
            'validation_ranking_sha256': sha256_file(VALIDATION_RANKING),
            'blind_predictions_sha256': sha256_file(blind_dir / 'predictions.parquet'),
            'listings': {'code': code_listing, 'artifacts': artifact_listing}}


def freeze(force=False):
    if RUN_LOG.is_file():
        raise RuntimeError('Unblinding has already happened; the protocol cannot be re-frozen')
    if FROZEN.is_file() and not force:
        raise RuntimeError('frozen.json exists; pass --force-refreeze only before unblinding')
    for path in (PROTOCOL, TEST_STATES, CALIBRATORS, BLIND_DIR / 'predictions.parquet', BLIND_DIR / 'blind_summary.json'):
        if not path.is_file():
            raise FileNotFoundError(path)
    blind = json.loads((BLIND_DIR / 'blind_summary.json').read_text(encoding='utf-8'))
    if not blind['all_sanity_passed'] or len(blind['models']) != len(catalog()) or blind['states'] != len(load_test_states()):
        raise RuntimeError('Freeze requires a complete, passing blind run of all entries on all test states')
    import subprocess
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(['git', 'status', '--porcelain'], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip())
    document = {'frozen_at': _now(), 'project_version': '0.9.1', 'git_head': head, 'git_worktree_dirty': dirty,
                'test_sites': TEST_SITES, 'seeds': list(SEEDS), 'budgets': list(BUDGETS), 'risk_threshold': RISK_THRESHOLD,
                'models': [r['id'] for r in catalog()], **current_hashes(),
                'blind_prediction_content_sha256': blind['prediction_content_sha256']}
    write_json(FROZEN, document)
    return document


def check_frozen():
    if not FROZEN.is_file():
        raise RuntimeError('No frozen protocol: run final-test --freeze first')
    frozen = json.loads(FROZEN.read_text(encoding='utf-8'))
    now = current_hashes()
    keys = ['protocol_sha256', 'test_states_sha256', 'catalog_sha256', 'calibrators_sha256', 'code_tree_sha256',
            'artifact_tree_sha256', 'validation_ranking_sha256', 'blind_predictions_sha256']
    mismatched = [k for k in keys if frozen.get(k) != now[k]]
    if mismatched:
        detail = {}
        for kind in ('code', 'artifacts'):
            old, new = frozen['listings'][kind], now['listings'][kind]
            changed = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
            if changed:
                detail[kind] = changed
        raise RuntimeError(f'Refusing to unblind: frozen hashes differ for {mismatched} {detail}')
    return frozen


# ---------------------------------------------------------------- unblinding
def load_test_truth(states, dataset=None):
    """The ONLY function that reads test-location truth. Called by unblind() alone."""
    dataset = dataset or load_dataset(load_config())
    frames = []
    for (site, seed), group in states.groupby(['site_id', 'seed'], sort=True):
        ids = list(group.evaluation_ids.iloc[0])
        if any(sorted(v) != sorted(ids) for v in group.evaluation_ids):
            raise ValueError('Evaluation set must be fixed per (site, seed)')
        truth = dataset.truth(ids)[['site_id', 'location_id', 'depth_index', 'qc_mpa', 'log_qc']]
        frames.append(truth.assign(seed=seed))
    return pd.concat(frames, ignore_index=True).rename(columns={'qc_mpa': 'truth_qc_mpa', 'log_qc': 'truth_log_qc'})


def aggregate_test(metrics, sites=None):
    """Exact v0.9.1 aggregation: per-state pooled metrics -> trapezoid AUBC over budgets 5-13
    per (fold, site, seed) -> mean per site (pandas skipna) -> equal-weight mean over sites."""
    from .analysis import METRICS, curve_metrics
    sources = metrics.drop_duplicates('model_id').set_index('model_id')[['source_model_id', 'risk_source']]
    curves = curve_metrics(metrics).join(sources, on='model_id')
    if sites is not None:
        curves = curves[curves.site_id.isin(sites)]
    by_site = curves.groupby(['source_model_id', 'site_id', 'model_id', 'risk_source'], sort=False)[METRICS].mean().reset_index()
    overall = by_site.groupby(['source_model_id', 'model_id', 'risk_source'], sort=False)[METRICS].mean().reset_index()
    overall['n_sites'] = by_site.groupby('model_id', sort=False).site_id.nunique().reindex(overall.model_id).to_numpy()
    for metric in METRICS:
        overall[f'n_sites_{metric}'] = by_site.groupby('model_id', sort=False)[metric].count().reindex(overall.model_id).to_numpy()
    return curves, by_site, overall


def per_budget(metrics, sites):
    from .analysis import METRICS
    frame = metrics[metrics.site_id.isin(sites)]
    site_budget = frame.groupby(['model_id', 'site_id', 'budget_n'], sort=False)[METRICS].mean().reset_index()
    overall = site_budget.groupby(['model_id', 'budget_n'], sort=False)[METRICS].mean().reset_index()
    return site_budget, overall


def _sign_p(wins, losses):
    n = wins + losses
    if not n:
        return None
    k = min(wins, losses)
    return float(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n))


def paired_vs(by_site, curves, reference='idw'):
    """Per-site and per-(site, seed) differences entry - reference with W/T/L by site."""
    from .analysis import METRICS
    rows = []
    ref_site = by_site[by_site.model_id.eq(reference)].set_index('site_id')
    ref_curve = curves[curves.model_id.eq(reference)].set_index(['site_id', 'seed'])
    for model_id, group in by_site.groupby('model_id', sort=False):
        site = group.set_index('site_id')
        curve = curves[curves.model_id.eq(model_id)].set_index(['site_id', 'seed'])
        for metric in METRICS:
            lower = DIRECTION[metric]
            if lower is None:
                continue
            diff = (site[metric] - ref_site[metric].reindex(site.index)).dropna()
            better = diff < -1e-12 if lower else diff > 1e-12
            worse = diff > 1e-12 if lower else diff < -1e-12
            seed_diff = (curve[metric] - ref_curve[metric].reindex(curve.index)).dropna()
            seed_better = seed_diff < -1e-12 if lower else seed_diff > 1e-12
            seed_worse = seed_diff > 1e-12 if lower else seed_diff < -1e-12
            rows.append({'model_id': model_id, 'reference': reference, 'metric': metric, 'lower_is_better': lower,
                         'mean_entry': float(site[metric].mean()) if site[metric].notna().any() else None,
                         'mean_reference': float(ref_site[metric].mean()) if ref_site[metric].notna().any() else None,
                         'mean_site_difference': float(diff.mean()) if len(diff) else None,
                         'n_sites': int(len(diff)), 'site_wins': int(better.sum()), 'site_ties': int(len(diff) - better.sum() - worse.sum()),
                         'site_losses': int(worse.sum()), 'site_sign_test_p_two_sided': _sign_p(int(better.sum()), int(worse.sum())),
                         'n_site_seed_pairs': int(len(seed_diff)), 'site_seed_wins': int(seed_better.sum()),
                         'site_seed_losses': int(seed_worse.sum()),
                         'site_seed_mean_difference': float(seed_diff.mean()) if len(seed_diff) else None,
                         **{f'diff_{s}': float(v) for s, v in diff.items()}})
    return pd.DataFrame(rows)


def _classify(row):
    """Pre-registered outcome of one entry-vs-reference comparison on one metric."""
    if row is None or row['n_sites'] == 0 or row['mean_entry'] is None or row['mean_reference'] is None:
        return 'undefined'
    delta = row['mean_entry'] - row['mean_reference']
    improved = delta < -1e-12 if row['lower_is_better'] else delta > 1e-12
    worse = delta > 1e-12 if row['lower_is_better'] else delta < -1e-12
    need = math.ceil(0.75 * row['n_sites'])
    if improved and row['site_wins'] >= need:
        return 'consistent_improvement'
    if improved:
        return 'mean_improvement_only'
    if worse and row['site_losses'] >= need:
        return 'consistent_deterioration'
    return 'no_improvement'


def decisions(paired, overall, validation, guard_pp=0.02):
    lookup = {(r.model_id, r.reference, r.metric): r._asdict() for r in paired.itertuples(index=False)}

    def get(model, metric, ref='idw'):
        return lookup.get((model, ref, metric))

    def summary(model, metric, ref='idw'):
        row = get(model, metric, ref)
        return {'outcome': _classify(row), **({k: row[k] for k in ('mean_entry', 'mean_reference', 'mean_site_difference',
                                                                    'n_sites', 'site_wins', 'site_ties', 'site_losses')} if row else {})}
    ov = overall.set_index('model_id')
    idw_fpr = float(ov.loc['idw', 'false_positive_rate'])
    result = {}
    q1 = {m: summary('balanced', m) for m in ('log_mae', 'h_point_mae_m')}
    both = [q1[m]['outcome'] for m in q1]
    if all(o == 'consistent_improvement' for o in both):
        q1['decision'] = 'Balanced一致优于IDW（log-MAE与厚度MAE均是）'
    elif any(o in ('consistent_improvement', 'mean_improvement_only') for o in both):
        q1['decision'] = 'Balanced对IDW只有部分改进：' + '、'.join(m for m, o in zip(q1, both) if o in ('consistent_improvement', 'mean_improvement_only'))
    elif all(o == 'consistent_deterioration' for o in both):
        q1['decision'] = 'Balanced在两项指标上一致劣于IDW'
    else:
        q1['decision'] = 'Balanced相对IDW无改进'
    result['Q1_balanced_vs_idw'] = q1
    thick_fpr = float(ov.loc['thickness', 'false_positive_rate'])
    result['Q2_thickness'] = {
        'vs_idw_h_point_mae_m': summary('thickness', 'h_point_mae_m'),
        'vs_thickness_guarded_h_point_mae_m': summary('thickness', 'h_point_mae_m', 'thickness_guarded'),
        'thickness_point_fpr': thick_fpr, 'thickness_guarded_point_fpr': float(ov.loc['thickness_guarded', 'false_positive_rate']),
        'idw_point_fpr': idw_fpr, 'fpr_guard_idw_plus_2pp': idw_fpr + guard_pp,
        'thickness_within_fpr_guard': bool(thick_fpr <= idw_fpr + guard_pp)}
    rec_fpr = float(ov.loc['recognition', 'false_positive_rate'])
    result['Q3_recognition'] = {'recall_vs_idw': summary('recognition', 'recall'),
                                'fpr_vs_idw': summary('recognition', 'false_positive_rate'),
                                'recognition_point_fpr': rec_fpr, 'idw_point_fpr': idw_fpr,
                                'fpr_increase_pp': 100 * (rec_fpr - idw_fpr),
                                'within_fpr_guard_idw_plus_2pp': bool(rec_fpr <= idw_fpr + guard_pp)}
    result['Q4_tabpfn_vs_idw'] = {m: {k: summary(m, k) for k in ('log_mae', 'h_point_mae_m')} for m in ('tabpfn_joint', 'tabpfn_graph')}
    from scipy.stats import spearmanr
    q5 = {}
    val = validation.set_index('model_id')
    for metric in ('log_mae', 'h_point_mae_m'):
        common = [m for m in ov.index if m in val.index]
        rho = float(spearmanr(val.loc[common, metric], ov.loc[common, metric]).statistic)
        label = 'stable' if rho >= 0.7 else 'moderate' if rho >= 0.4 else 'unstable'
        top_val = list(val.loc[common, metric].sort_values(kind='stable').index[:3])
        top_test = list(ov.loc[common, metric].sort_values(kind='stable').index[:3])
        q5[metric] = {'spearman_rho': rho, 'stability': label, 'validation_top3': top_val, 'test_top3': top_test,
                      'top3_overlap': len(set(top_val) & set(top_test)),
                      'balanced_test_rank': int(ov.loc[common, metric].rank(method='min').loc['balanced'])}
    result['Q5_rank_stability'] = q5
    return result


def score(predictions, truth, n_states, models=None):
    """All pre-registered tables and answers from frozen predictions + truth."""
    from .report import state_metrics
    joined = predictions.merge(truth, on=['site_id', 'seed', 'location_id', 'depth_index'], how='left', validate='many_to_one')
    if joined.truth_qc_mpa.isna().any() or len(joined) != len(predictions):
        raise ValueError('Every prediction cell needs exactly one truth cell')
    metrics = state_metrics(joined, models)
    wanted = models or [r['id'] for r in catalog()]
    if metrics.groupby('model_id').size().ne(n_states).any() or set(metrics.model_id) != set(wanted):
        raise ValueError(f'Every entry must be scored on all {n_states} test states')
    sites = sorted(metrics.site_id.unique())
    curves, by_site, overall = aggregate_test(metrics)
    _, _, verified = aggregate_test(metrics, sites=[s for s in VERIFIED_SUBSET if s in sites])
    site_budget, budget_overall = per_budget(metrics, sites)
    paired = pd.concat([paired_vs(by_site, curves, 'idw'), paired_vs(by_site, curves, 'thickness_guarded')], ignore_index=True)
    result = decisions(paired, overall, pd.read_csv(VALIDATION_RANKING))
    tables = {'metrics': metrics, 'curves': curves, 'site_metrics': by_site, 'test_ranking': overall,
              'test_ranking_verified_sites': verified, 'site_budget_metrics': site_budget,
              'budget_metrics': budget_overall, 'paired_comparisons': paired}
    return tables, result


def unblind(force_rerun=False, log=print):
    frozen = check_frozen()
    previous = []
    if RUN_LOG.is_file():
        previous = [json.loads(line) for line in RUN_LOG.read_text(encoding='utf-8').splitlines() if line.strip()]
    if previous and not force_rerun:
        raise RuntimeError(f'Unblinding already ran {len(previous)} time(s); pass --force-rerun to rerun (it is logged as a rerun)')
    run_index = len(previous) + 1
    is_rerun = bool(previous)
    entry = {'event': 'unblind_started', 'run_index': run_index, 'rerun': is_rerun, 'time': _now(), 'user': getpass.getuser(),
             'frozen_at': frozen['frozen_at'], 'protocol_sha256': frozen['protocol_sha256'],
             'blind_predictions_sha256': frozen['blind_predictions_sha256']}
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open('a', encoding='utf-8') as handle:  # written BEFORE any truth is read
        handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
    output = UNBLIND_DIR if not is_rerun else FINAL_DIR / f'unblind_rerun_{run_index}'
    output.mkdir(parents=True, exist_ok=True)
    states = load_test_states()
    predictions = pd.read_parquet(BLIND_DIR / 'predictions.parquet')
    truth = load_test_truth(states)
    tables, result = score(predictions, truth, len(states))
    for name, table in tables.items():
        table.to_csv(output / f'{name}.csv', index=False)
    write_json(output / 'decisions.json', {'run_index': run_index, 'rerun': is_rerun, 'created': _now(),
                                           'protocol_sha256': frozen['protocol_sha256'], 'answers': result})
    done = {'event': 'unblind_finished', 'run_index': run_index, 'rerun': is_rerun, 'time': _now(), 'output': output.relative_to(PROJECT_ROOT).as_posix(),
            'outputs_sha256': {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()}}
    with RUN_LOG.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(done, ensure_ascii=False) + '\n')
    if log:
        log(f'unblind run {run_index} ({"RERUN" if is_rerun else "first run"}) -> {output}')
    return result
