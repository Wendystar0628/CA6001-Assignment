import numpy as np
import pandas as pd
import pytest

from cpt_budget.v7.metric_kernel import (
    MetricKernelPredictor, fit_source_pairs, learn_support_metric, radial_features)


def profiles(site="a", n=6):
    rows = []
    for i in range(n):
        for d in range(6):
            rows.append(dict(site_id=site, location_id=f"{site}_{i}", depth_index=d,
                x_m=float(i * 10), y_m=float((i % 2) * 17),
                log_qc=float(np.sin(d) + 0.3 * i * (d < 3))))
    return pd.DataFrame(rows)


def test_support_metric_is_positive_definite_bounded_and_translation_invariant():
    xy = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [2., 1.]])
    values = np.column_stack([xy[:, 0] * 4, xy[:, 0] * 3])
    metric = learn_support_metric(xy, values)
    eigenvalues = np.linalg.eigvalsh(metric)
    assert np.all(eigenvalues > 0)
    assert eigenvalues.max() / eigenvalues.min() <= 4.0000001
    np.testing.assert_allclose(np.linalg.det(metric), 1.)
    np.testing.assert_allclose(learn_support_metric(xy + 913, values), metric)


def test_radial_features_nonnegative_and_monotonic():
    features = radial_features([0, 0.1, 1, 10])
    assert np.all(features >= 0)
    assert np.all(np.diff(features, axis=0) >= 0)


@pytest.mark.parametrize("method", ["source_pair_variance", "support_psd_metric",
                                   "source_pair_transport", "metric_pair_transport",
                                   "euclidean_all_control"])
def test_query_labels_are_ignored_and_observed_profiles_exact(method):
    source = fit_source_pairs(profiles("source"))
    model = MetricKernelPredictor(source, method=method)
    support = profiles("target")
    model.condition(support)
    exact = model.predict(support.drop(columns="log_qc")).frame
    np.testing.assert_allclose(exact.pred_log_qc, support.log_qc, atol=1e-10)
    query = support.copy()
    query["location_id"] += "_new"
    query["x_m"] += 3
    clean = model.predict(query.drop(columns="log_qc")).frame
    query["log_qc"] = -10000
    polluted = model.predict(query).frame
    np.testing.assert_allclose(clean.pred_log_qc, polluted.pred_log_qc)
    assert np.isfinite(clean.pred_log_qc).all()
    assert source.manifest["source_site_ids"] == ["source"]
