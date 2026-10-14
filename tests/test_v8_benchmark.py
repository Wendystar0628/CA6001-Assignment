from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cpt_budget.v8 import benchmark


def fake_state():
    return SimpleNamespace(key='f0_s_s17_n5',
        metadata={'fold_id': 0, 'site_id': 's', 'seed': 17, 'budget_n': 5},
        observed=pd.DataFrame({'log_qc': [1., 2.]}),
        query=pd.DataFrame({'depth_index': [0, 1]}),
        truth=pd.DataFrame({'secret': [42., 43.]}))


def test_serial_runner_reconditions_and_prepares_without_truth(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, '_hardware', lambda: {'test': True})
    calls = []

    class Predictor:
        def __init__(self, version):
            self.version = version

        def initialize(self):
            calls.append((self.version, 'initialize'))

        def prepare_state(self, observed, query, metadata):
            assert 'secret' not in observed and 'secret' not in query
            assert set(metadata) == {'fold_id', 'site_id', 'seed', 'budget_n'}
            calls.append((self.version, 'prepare'))

        def condition(self, observed):
            calls.append((self.version, 'condition'))
            self.value = observed.log_qc.mean()

        def predict(self, query):
            calls.append((self.version, 'predict'))
            return np.full(len(query), self.value)

    specs = [benchmark.BenchmarkSpec(version, lambda fid, version=version: Predictor(version),
                                    'dummy complete point path', pair_id='test', version=version)
             for version in ('old', 'new')]
    summary = benchmark.run_specs(specs, [fake_state()], tmp_path)
    assert summary.measured_samples.tolist() == [3, 3]
    assert summary.states.tolist() == [1, 1]
    for version in ('old', 'new'):
        assert calls.count((version, 'initialize')) == 1
        assert calls.count((version, 'prepare')) == 1
        assert calls.count((version, 'condition')) == 4
        assert calls.count((version, 'predict')) == 4
    samples = pd.read_csv(tmp_path/'samples.csv')
    assert len(samples) == 6
    assert samples.repeat.tolist() == [0, 1, 2, 0, 1, 2]
    pairs = pd.read_csv(tmp_path/'paired_state_ratios.csv')
    assert len(pairs) == 1
    assert pairs.total_ms_speedup_old_over_new.iloc[0] > 0
    checks = pd.read_csv(tmp_path/'paired_output_checks.csv')
    assert checks.equivalent.all() and checks.max_abs_difference.max() == 0


def test_invalid_or_nonfinite_timed_output_rejected():
    with pytest.raises(ValueError, match='finite nonempty'):
        benchmark._check_output(np.array([np.nan]))
    with pytest.raises(ValueError, match='finite nonempty'):
        benchmark._check_output(np.array([]))


def test_pair_checks_detect_algorithm_difference(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, '_hardware', lambda: {})

    class Predictor:
        def __init__(self, value):
            self.value = value
        def condition(self, observed):
            pass
        def predict(self, query):
            return np.full(len(query), self.value)

    specs = [benchmark.BenchmarkSpec(version, lambda fid, value=value: Predictor(value),
                                    'test', pair_id='test', version=version)
             for version, value in [('old', 1.), ('new', 2.)]]
    benchmark.run_specs(specs, [fake_state()], tmp_path, repeats=1)
    checks = pd.read_csv(tmp_path/'paired_output_checks.csv')
    assert not checks.equivalent.any()
    assert checks.max_abs_difference.max() == 1
