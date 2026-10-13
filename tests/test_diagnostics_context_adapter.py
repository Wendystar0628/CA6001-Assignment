"""Information-boundary checks for the two exploratory context constructions."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cpt_budget.diagnostics.context_adapter import (
    ContextAdapterPredictor, SourceContextBank, build_source_context_bank,
    context_feature_diagnostics, honest_anchor_split,
)
from cpt_budget.types import PreparedDataset
from cpt_budget.v2.features import FEATURE_COLUMNS_V2, build_features_v2


def _dataset(tmp_path):
    rows = []
    for site_index, site in enumerate(("source_a", "source_b", "target")):
        for i in range(8):
            for depth in range(24):
                value = site_index + i / 4 + np.sin(depth / 3)
                rows.append(dict(site_id=site, location_id=f"{site}_{i}", near_group_id=f"{site}_{i}",
                                 x_m=i * 12., y_m=(i % 3) * 7., geometry_scale_m=20.,
                                 depth_index=depth, z_center_m=2.125 + depth * .25,
                                 log_qc=value, qc_mpa=np.exp(value)))
    grid = pd.DataFrame(rows)
    profiles = grid.drop_duplicates("location_id").drop(columns=["depth_index", "z_center_m", "log_qc", "qc_mpa"])
    sites = pd.DataFrame([dict(site_id=s, investigation_group=s) for s in ("source_a", "source_b", "target")])
    dataset = PreparedDataset(profiles, grid, sites, Path(tmp_path))
    episodes = pd.DataFrame([dict(fold_id=0, role="train", site_id=s, episode_id=f"{s}_episode",
                                  support_location_ids=[f"{s}_{i}" for i in range(5)],
                                  query_location_ids=[f"{s}_{i}" for i in range(5, 8)])
                             for s in ("source_a", "source_b")])
    return dataset, episodes


def test_source_bank_is_balanced_deterministic_and_roundtrips(tmp_path):
    dataset, episodes = _dataset(tmp_path)
    kwargs = dict(fold_id=0, source_sites=["source_a", "source_b"], budgets=[5, 7])
    bank = build_source_context_bank(dataset, episodes, **kwargs)
    reordered = build_source_context_bank(dataset, episodes.iloc[::-1], **kwargs)
    pd.testing.assert_frame_equal(bank.rows, reordered.rows)
    assert bank.rows.groupby(["target_budget", "source_site_id"]).size().eq(24).all()
    assert len(bank.for_budget(5)[1]) == 48
    bank.save(tmp_path / "bank")
    loaded = SourceContextBank.load(tmp_path / "bank")
    pd.testing.assert_frame_equal(bank.rows, loaded.rows)
    assert bank.provenance == loaded.provenance


def test_source_bank_never_reads_nontraining_truth_and_rejects_wrong_role(tmp_path):
    dataset, episodes = _dataset(tmp_path)
    kwargs = dict(fold_id=0, source_sites=["source_a", "source_b"], budgets=[5])
    before = build_source_context_bank(dataset, episodes, **kwargs)
    dataset.grid.loc[dataset.grid.site_id.eq("target"), ["log_qc", "qc_mpa"]] = 1e6
    after = build_source_context_bank(dataset, episodes, **kwargs)
    pd.testing.assert_frame_equal(before.rows, after.rows)
    with pytest.raises(ValueError, match="validation, test"):
        build_source_context_bank(dataset, episodes.assign(role="validation"), **kwargs)
    with pytest.raises(ValueError, match="outside"):
        build_source_context_bank(dataset, episodes, **{**kwargs, "source_sites": ["source_a"]})


def test_honest_features_share_anchors_and_ignore_labelled_context_values(tmp_path):
    dataset, _ = _dataset(tmp_path)
    observed = dataset.observed([f"target_{i}" for i in range(5)])
    anchors, labelled = honest_anchor_split(observed)
    assert anchors.location_id.nunique() == 3 and labelled.location_id.nunique() == 2
    assert not set(anchors.location_id) & set(labelled.location_id)
    predictor = ContextAdapterPredictor(mode="honest")
    x, y = predictor.context_arrays(observed)
    query = dataset.query(["target_5", "target_6"])
    q = predictor.query_features(query)
    support_count_index = FEATURE_COLUMNS_V2.index("support_count")
    assert np.all(x[:, support_count_index] == 3) and np.all(q[:, support_count_index] == 3)
    poisoned = observed.copy()
    poisoned.loc[poisoned.location_id.isin(labelled.location_id), "log_qc"] += 1000
    x2, y2 = predictor.context_arrays(poisoned)
    np.testing.assert_allclose(x, x2, equal_nan=True)
    np.testing.assert_allclose(y2, y + 1000)
    np.testing.assert_allclose(q, predictor.query_features(query.assign(log_qc=-999, qc_mpa=1e6)), equal_nan=True)


def test_source_context_preserves_target_loo_and_rejects_source_target_overlap(tmp_path):
    dataset, episodes = _dataset(tmp_path)
    bank = build_source_context_bank(dataset, episodes, fold_id=0, source_sites=["source_a", "source_b"], budgets=[5])
    predictor = ContextAdapterPredictor(source_bank=bank)
    assert predictor.config["tabpfn"]["target_context_only"] is False
    observed = dataset.observed([f"target_{i}" for i in range(5)])
    x, y = predictor.context_arrays(observed)
    assert predictor.context_manifest["target_context_only"] is False
    assert x.shape == (168, 24) and len(y) == 168
    np.testing.assert_allclose(x[-120:], build_features_v2(observed, observed, leave_location_out=True), equal_nan=True)
    with pytest.raises(ValueError, match="overlaps"):
        predictor.context_arrays(dataset.observed([f"source_a_{i}" for i in range(5)]))


def test_raw_feature_diagnostic_identifies_support_count_shift_without_truth(tmp_path):
    dataset, _ = _dataset(tmp_path)
    observed = dataset.observed([f"target_{i}" for i in range(5)])
    query = dataset.query(["target_5", "target_6"])
    result = context_feature_diagnostics(observed, query.assign(log_qc=1e9))
    assert result["standard_support_counts"] == [4.]
    assert result["standard_query_counts"] == [5.]
    assert result["honest_context_counts"] == result["honest_query_counts"] == [3.]
