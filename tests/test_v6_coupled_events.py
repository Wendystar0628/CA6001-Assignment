"""Cached coupling must align points and never ingest query labels or risk."""
import numpy as np
import pandas as pd
import pytest

from cpt_budget.v6.common import KEYS
from cpt_budget.v6.coupled_events import load_base_points


def keys():
    return pd.DataFrame({'fold_id': [0, 0], 'site_id': ['s', 's'], 'seed': [17, 17],
                         'budget_n': [5, 5], 'location_id': ['p0', 'p0'], 'depth_index': [0, 1]})


def test_cache_reads_only_keys_and_points_and_aligns_rows(tmp_path, monkeypatch):
    wide = keys()
    path = tmp_path / 'points.parquet'
    wide.assign(pred_log_qc=[-0.3, 0.8], truth_log_qc=[999., 999.], risk_probability=[0., 0.]).iloc[::-1].to_parquet(path)
    actual_reader = pd.read_parquet
    captured = []
    def reader(path, **kwargs):
        captured.append(kwargs.get('columns'))
        return actual_reader(path, **kwargs)
    monkeypatch.setattr(pd, 'read_parquet', reader)
    np.testing.assert_array_equal(load_base_points(path, wide), [-0.3, 0.8])
    assert captured == [[*KEYS, 'pred_log_qc']]


def test_cache_rejects_wrong_or_duplicate_identity(tmp_path):
    wide = keys()
    path = tmp_path / 'points.parquet'
    wide.assign(depth_index=[0, 2], pred_log_qc=[0., 1.]).to_parquet(path)
    with pytest.raises(ValueError, match='keys differ'):
        load_base_points(path, wide)
    wide.assign(depth_index=[0, 0], pred_log_qc=[0., 1.]).to_parquet(path)
    with pytest.raises(ValueError, match='unique'):
        load_base_points(path, wide)
