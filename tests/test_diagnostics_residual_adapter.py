"""Residual target isolation and exact physical CDF reconstruction, CPU only."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cpt_budget.diagnostics import residual_adapter as adapter
from cpt_budget.models.distribution import QUANTILES, QUANTILE_COLUMNS, PROBABILITY_COLUMNS, THRESHOLDS
from cpt_budget.models.idw import IDWPredictor


def context():
    rows = []
    for i, x in enumerate([0., 0., 10., 25., 50.]):
        for depth in range(3):
            target = .25 * i + .15 * depth
            rows.append(dict(site_id="S", location_id=f"L{i}", profile_id=f"L{i}",
                             near_group_id=f"L{i}", x_m=x, y_m=0., geometry_scale_m=10.,
                             depth_index=depth, z_center_m=2.125 + .25 * depth,
                             log_qc=target, qc_mpa=np.exp(target)))
    return pd.DataFrame(rows)


def query():
    frame = context().loc[lambda x: x.location_id.isin(["L2", "L4"])].copy()
    frame["location_id"] = frame.location_id.map({"L2": "Q0", "L4": "Q1"})
    frame["x_m"] += 5
    return frame.iloc[[0, 3, 1, 4, 2, 5]].reset_index(drop=True)


def full_distribution(n):
    torch = pytest.importorskip("torch")
    pytest.importorskip("tabpfn")
    from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
    criterion = FullSupportBarDistribution(torch.tensor([-3., -1., -.2, .5, 2.]))
    logits = torch.tensor([.1, -.2, .5, .2]).repeat(n, 1)
    quantiles = [criterion.icdf(logits, q).numpy() for q in QUANTILES]
    return dict(criterion=criterion, logits=logits, quantiles=quantiles)


def test_whole_location_and_coincident_cpts_are_excluded_from_context_reference():
    observed = context()
    original = adapter.residual_context(observed)
    changed = observed.copy()
    own = changed.location_id == "L0"
    coincident = changed.location_id == "L1"
    changed.loc[own, "log_qc"] += 10
    changed.loc[coincident, "log_qc"] -= 7
    rebuilt = adapter.residual_context(changed)
    # Changing all depths of both coincident CPTs must not affect L0 features
    # or its reference, although changing its own label changes its residual.
    np.testing.assert_allclose(original["X_support"][own], rebuilt["X_support"][own], equal_nan=True)
    np.testing.assert_allclose(original["loo_baseline_log_qc"][own], rebuilt["loo_baseline_log_qc"][own])
    np.testing.assert_allclose(rebuilt["y_residual"][own] - original["y_residual"][own], 10.)
    # Self-inclusive IDW would return the coincident CPT mean, not this value.
    assert not np.allclose(original["loo_baseline_log_qc"][own], .125 + .15 * np.arange(3))


def test_context_rejects_no_independent_reference_instead_of_zero_residual():
    only_coincident = context().loc[lambda x: x.location_id.isin(["L0", "L1"])]
    with pytest.raises(ValueError, match="noncoincident CPT"):
        adapter.residual_context(only_coincident)


@pytest.mark.parametrize("alpha", [.25, .5, 1.])
def test_physical_cdf_uses_query_specific_shift_and_residual_scale(alpha):
    import torch
    query_frame = query()
    native = full_distribution(len(query_frame))
    baseline = np.array([-1.3, -.6, -.2, .1, .7, 1.4])
    result = adapter.reconstruct_residual_output(native, query_frame, baseline, alpha=alpha)
    expected_thresholds = torch.as_tensor((np.log(THRESHOLDS)[None, :] - baseline[:, None]) / alpha,
                                          dtype=native["logits"].dtype)
    expected_probabilities = native["criterion"].cdf(native["logits"], expected_thresholds).numpy()
    np.testing.assert_allclose(result[list(PROBABILITY_COLUMNS)], expected_probabilities)
    expected_quantiles = baseline[:, None] + alpha * np.column_stack(native["quantiles"])
    np.testing.assert_allclose(result[list(QUANTILE_COLUMNS)], expected_quantiles)
    np.testing.assert_allclose(result.pred_qc_mpa, np.exp(expected_quantiles[:, 3]))
    assert result.p_low_1.iloc[0] > result.p_low_1.iloc[-1]
    assert "log_qc" not in result and "qc_mpa" not in result


def test_zero_alpha_is_exact_idw_without_native_uncertainty_or_model_loading(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("alpha=0 must not load a neural model")
    monkeypatch.setattr(adapter, "make_estimator", forbidden)
    predictor = adapter.BaselineResidualTabPFNPredictor(alpha=0)
    predictor.condition(context())
    baseline = IDWPredictor()
    baseline.condition(context())
    frame = predictor.predict(query()).frame
    np.testing.assert_allclose(frame.pred_log_qc, baseline.predict(query()).frame.pred_log_qc)
    assert not set(QUANTILE_COLUMNS).intersection(frame.columns)
    assert not set(PROBABILITY_COLUMNS).intersection(frame.columns)


def test_variants_reuse_native_prediction_ignore_query_truth_and_preserve_order(monkeypatch):
    class FakeEstimator:
        def __init__(self):
            self.models_ = []
            self.fits = 0
            self.predicts = 0
            self.features = []

        def fit(self, X, y):
            self.fits += 1
            self.X = np.array(X, copy=True)
            self.y = np.array(y, copy=True)

        def predict(self, X, **kwargs):
            self.predicts += 1
            self.features.append(np.array(X, copy=True))
            return full_distribution(len(X))

    fake = FakeEstimator()
    monkeypatch.setattr(adapter, "make_estimator", lambda *args: fake)
    predictor = adapter.BaselineResidualTabPFNPredictor({"episodes": {"query_profiles_per_batch": 1}})
    predictor.condition(context())
    expected = adapter.residual_context(context())
    first = predictor.predict_variants(query())
    assert fake.fits == 1 and fake.predicts == 2  # two whole query profiles, not 3 alphas * 2
    np.testing.assert_allclose(fake.X, expected["X_support"], equal_nan=True)
    np.testing.assert_allclose(fake.y, expected["y_residual"])
    changed_query = query()
    changed_query["log_qc"] = 999
    changed_query["qc_mpa"] = 999
    second = predictor.predict_variants(changed_query)
    assert fake.fits == 1 and fake.predicts == 4
    for alpha in first:
        pd.testing.assert_frame_equal(first[alpha].frame, second[alpha].frame)
        assert first[alpha].frame.location_id.tolist() == query().location_id.tolist()
        assert first[alpha].frame.depth_index.tolist() == query().depth_index.tolist()
    for i in range(2):
        np.testing.assert_allclose(fake.features[i], fake.features[i + 2], equal_nan=True)


def test_unregistered_baselines_and_alphas_rejected():
    with pytest.raises(ValueError, match="only the IDW"):
        adapter.BaselineResidualTabPFNPredictor(baseline="gp")
    with pytest.raises(ValueError, match="pre-registered"):
        adapter.BaselineResidualTabPFNPredictor(alpha=.75)
