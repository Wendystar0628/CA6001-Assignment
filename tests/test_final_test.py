"""Final outer test: blindness, state generator, calibration and aggregation guards.

None of these tests reads test-site truth: the blind run is executed with the truth
loaders patched to raise, and the aggregation test uses validation metrics only.
"""
import json
import numpy as np
import pandas as pd
import pytest
from cpt_budget import final_test as ft
from cpt_budget.config import PROJECT_ROOT
from cpt_budget.types import PreparedDataset

HAVE_STATES = ft.TEST_STATES.is_file() and ft.CALIBRATORS.is_file()


def test_coordinate_dataset_carries_no_measurements():
    dataset = ft.coordinate_dataset()
    assert set(dataset.grid) == {'site_id', 'location_id'}
    assert not set(dataset.profiles) & set(ft.MEASUREMENTS)
    with pytest.raises(Exception):
        dataset.observed(dataset.locations().location_id[:1].tolist()).log_qc


def test_generator_reproduces_all_frozen_validation_states():
    assert ft.reproduce_frozen_states() == {'checked': 60, 'mismatches': 0}


def test_test_sites_are_exactly_the_unused_replay_sites():
    roles = ft.site_roles().set_index('site_id')
    assert set(roles.index[roles.included]) == set(ft.TEST_SITES)
    assert not roles.loc[list(ft.TEST_SITES), 'in_frozen_validation_states'].any()
    for site, fold in ft.TEST_SITES.items():
        assert roles.loc[site, f'fold_{fold}'] == 'test'


@pytest.mark.skipif(not HAVE_STATES, reason='run final-test --prepare first')
def test_test_states_are_well_formed():
    states = ft.load_test_states()
    assert len(states) == 100 and set(states.seed) == set(ft.SEEDS)
    for row in states.itertuples():
        assert not set(row.observed_ids) & set(row.evaluation_ids)
        assert len(row.observed_ids) == row.budget_n
    first = states.sort_values('budget_n').groupby(['site_id', 'seed'])
    assert first.evaluation_ids.apply(lambda s: len({tuple(sorted(v)) for v in s})).eq(1).all()


@pytest.mark.skipif(not HAVE_STATES, reason='run final-test --prepare first')
def test_blind_inputs_mask_evaluation_cells():
    dataset = ft.load_dataset(ft.load_config())
    row = ft.load_test_states().iloc[0]
    observed, query = ft.BlindDataset(dataset).state_inputs(row)
    assert not set(observed.location_id) & set(row.evaluation_ids)
    assert not set(query) & set(ft.MEASUREMENTS) and set(query.location_id) == set(row.evaluation_ids)
    # Same model inputs as the validation path (iter_states), only built from masked grids.
    pd.testing.assert_frame_equal(query, dataset.query(list(row.evaluation_ids)))
    pd.testing.assert_frame_equal(observed, dataset.observed(list(row.observed_ids)))
    leaked = row.copy()
    leaked['observed_ids'] = list(row.observed_ids) + [row.evaluation_ids[0]]
    with pytest.raises(ft.TruthAccessError):
        ft.BlindDataset(dataset).state_inputs(leaked)
    with pytest.raises(ft.TruthAccessError):
        ft.BlindDataset(dataset).truth(row.evaluation_ids)


@pytest.mark.skipif(not HAVE_STATES, reason='run final-test --prepare first')
def test_blind_run_never_touches_truth(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise AssertionError('truth loader called on the blind path')
    monkeypatch.setattr(PreparedDataset, 'truth', boom)
    monkeypatch.setattr(ft, 'load_test_truth', boom)
    predictions, sanity, summary = ft.run_blind(['constant', 'idw', 'gp'], ['christchurch_stadium'], [17], [5, 13],
                                                output=tmp_path, log=None)
    assert summary['all_sanity_passed'] and len(sanity) == 6
    assert not set(predictions) & set(ft.MEASUREMENTS + ['truth_qc_mpa', 'truth_log_qc'])
    assert predictions.risk_probability.between(0, 1).all()
    assert PreparedDataset.truth is boom  # restored after the blind context


def test_blind_run_refuses_to_overwrite_frozen_predictions(monkeypatch, tmp_path):
    frozen = tmp_path / 'frozen.json'
    frozen.write_text('{}')
    monkeypatch.setattr(ft, 'FROZEN', frozen)
    monkeypatch.setattr(ft, 'BLIND_DIR', tmp_path / 'blind')
    with pytest.raises(RuntimeError, match='Frozen blind predictions'):
        ft.run_blind(output=tmp_path / 'blind', log=None)


def test_calibrators_use_only_the_folds_validation_labels():
    from cpt_budget.report import _score_frame, entries
    from cpt_budget.risk import crossfit_risk
    folds = ft._folds()['folds']
    fitted = ft.fit_test_calibrators(['idw', 'balanced'], write=False)
    reference = pd.read_parquet(ft.REFERENCE)
    records = {r['id']: r for r in entries(['idw', 'balanced'])}
    for key, calibrator in fitted.items():
        model, fold = key.split('|')
        assert not set(calibrator['training_sites']) & set(ft.TEST_SITES)
        assert set(calibrator['training_sites']) <= set(folds[int(fold)]['validation_sites'])
        frame = reference[reference.model_id.eq(model) & reference.fold_id.eq(int(fold))]
        _, final = crossfit_risk(_score_frame(frame, records[model]))
        assert np.isclose(final['slope'], calibrator['slope']) and np.isclose(final['intercept'], calibrator['intercept'])


def test_aggregation_reproduces_the_v091_validation_ranking():
    metrics = pd.read_csv(PROJECT_ROOT / 'results/current/metrics.csv', low_memory=False)
    _, _, overall = ft.aggregate_test(metrics)
    published = pd.read_csv(PROJECT_ROOT / 'results/current/all_model_ranking.csv').set_index('model_id')
    new = overall.set_index('model_id').loc[published.index]
    columns = ['log_mae', 'h_point_mae_m', 'recall', 'false_positive_rate', 'risk_recall', 'risk_fpr', 'risk_brier', 'coverage_90']
    np.testing.assert_allclose(new[columns].to_numpy(float), published[columns].to_numpy(float), atol=1e-12, rtol=0)


def _guarded(monkeypatch, tmp_path, frozen_hashes, current):
    monkeypatch.setattr(ft, 'FROZEN', tmp_path / 'frozen.json')
    monkeypatch.setattr(ft, 'RUN_LOG', tmp_path / 'unblind_log.jsonl')
    (tmp_path / 'frozen.json').write_text(json.dumps({'frozen_at': 'x', 'listings': {'code': {}, 'artifacts': {}}, **frozen_hashes}))
    monkeypatch.setattr(ft, 'current_hashes', lambda *a, **k: {'listings': {'code': {}, 'artifacts': {}}, **current})

    def boom(*args, **kwargs):
        raise AssertionError('truth read before the guards passed')
    monkeypatch.setattr(ft, 'load_test_truth', boom)
    monkeypatch.setattr(PreparedDataset, 'truth', boom)


KEYS = ['protocol_sha256', 'test_states_sha256', 'catalog_sha256', 'calibrators_sha256', 'code_tree_sha256',
        'artifact_tree_sha256', 'validation_ranking_sha256', 'blind_predictions_sha256']


def test_unblind_refuses_when_the_protocol_changed(monkeypatch, tmp_path):
    hashes = {k: 'a' for k in KEYS}
    _guarded(monkeypatch, tmp_path, hashes, {**hashes, 'protocol_sha256': 'b'})
    with pytest.raises(RuntimeError, match='protocol_sha256'):
        ft.unblind(log=None)
    assert not (tmp_path / 'unblind_log.jsonl').exists()


def test_unblind_refuses_a_second_run_without_force(monkeypatch, tmp_path):
    hashes = {k: 'a' for k in KEYS}
    _guarded(monkeypatch, tmp_path, hashes, hashes)
    (tmp_path / 'unblind_log.jsonl').write_text(json.dumps({'event': 'unblind_started', 'run_index': 1}) + '\n')
    with pytest.raises(RuntimeError, match='already ran'):
        ft.unblind(log=None)
    with pytest.raises(RuntimeError, match='re-frozen'):
        ft.freeze()


def test_outcome_classification_rules():
    row = dict(n_sites=4, mean_entry=0.5, mean_reference=0.6, lower_is_better=True, site_wins=3, site_losses=1)
    assert ft._classify(row) == 'consistent_improvement'
    assert ft._classify({**row, 'site_wins': 2, 'site_losses': 2}) == 'mean_improvement_only'
    assert ft._classify({**row, 'mean_entry': 0.7, 'site_wins': 1, 'site_losses': 3}) == 'consistent_deterioration'
    assert ft._classify({**row, 'mean_entry': 0.7, 'site_wins': 2, 'site_losses': 2}) == 'no_improvement'
    assert ft._classify({**row, 'lower_is_better': False, 'site_wins': 0, 'site_losses': 4}) == 'consistent_deterioration'


@pytest.mark.skipif(not (ft.BLIND_DIR / 'predictions.parquet').is_file(), reason='run final-test --blind first')
def test_scoring_pipeline_runs_on_synthetic_truth(monkeypatch):
    """Phase-2 code path on frozen predictions with SYNTHETIC truth (no test labels read)."""
    def boom(*args, **kwargs):
        raise AssertionError('real truth must not be read in tests')
    monkeypatch.setattr(PreparedDataset, 'truth', boom)
    monkeypatch.setattr(ft, 'load_test_truth', boom)
    predictions = pd.read_parquet(ft.BLIND_DIR / 'predictions.parquet')
    predictions = predictions[predictions.site_id.isin(['christchurch_stadium', 'newlock']) & predictions.seed.isin([17, 29])]
    cells = predictions[['site_id', 'seed', 'location_id', 'depth_index']].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(0)
    log_qc = rng.normal(0.8, 1.0, len(cells))
    truth = cells.assign(truth_log_qc=log_qc, truth_qc_mpa=np.exp(log_qc))
    tables, answers = ft.score(predictions, truth, n_states=2 * 2 * 5)
    assert set(answers) == {'Q1_balanced_vs_idw', 'Q2_thickness', 'Q3_recognition', 'Q4_tabpfn_vs_idw', 'Q5_rank_stability'}
    ranking = tables['test_ranking']
    assert len(ranking) == 13 and ranking.n_sites.eq(2).all()
    assert tables['paired_comparisons'].query("model_id=='idw' and reference=='idw'").site_wins.eq(0).all()
