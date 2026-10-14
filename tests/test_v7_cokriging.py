import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtri

from cpt_budget.v7.cokriging import CoKrigingPredictor, fit_profile_covariance


def profiles(site, n=4):
    return pd.DataFrame([
        dict(site_id=site, location_id=f"{site}_{i}", depth_index=d,
             x_m=float(i * 10), y_m=float(i % 2) * 5,
             log_qc=0.3 * d + (i - n / 2) * np.sin(1 + d))
        for i in range(n) for d in range(5)
    ])


def test_covariance_equal_site_weight_and_psd():
    a, b = profiles("a"), profiles("b")
    b["log_qc"] += b.x_m * np.cos(b.depth_index) * 0.03
    repeat = a.copy()
    repeat["location_id"] += "_copy"
    first = fit_profile_covariance(pd.concat([a, b]))
    second = fit_profile_covariance(pd.concat([a, repeat, b]))
    np.testing.assert_allclose(first.covariance, second.covariance, atol=1e-12)
    assert np.linalg.eigvalsh(first.covariance).min() > -1e-12


@pytest.mark.parametrize("variant", ["full", "lmc", "rank6"])
def test_gaussian_conditioning_matches_direct_linear_algebra(variant):
    source = fit_profile_covariance(profiles("source"))
    model = CoKrigingPredictor(source, variant=variant)
    target = profiles("target")
    model.condition(target)
    query = target.iloc[:5].copy().drop(columns="log_qc")
    query["location_id"] = "new"
    query["x_m"] += 4
    frame = model.predict(query).frame
    xy = np.array([[4.0, 0.0]])
    covariance = model.covariance_matrix(model.xy, model.xy, model.length_scale_m)
    covariance += (model.nugget_variance + model.jitter) * np.eye(len(covariance))
    cross = model.covariance_matrix(xy, model.xy, model.length_scale_m)
    y = target.pivot(index="location_id", columns="depth_index", values="log_qc").to_numpy()
    exact_mean = y.mean(axis=0) + cross @ np.linalg.solve(covariance, (y - y.mean(axis=0)).reshape(-1))
    prior = model.covariance_matrix(xy, xy, model.length_scale_m)
    posterior = prior + model.nugget_variance * np.eye(5) - cross @ np.linalg.solve(covariance, cross.T)
    np.testing.assert_allclose(frame.pred_log_qc, exact_mean, atol=1e-10)
    np.testing.assert_allclose((frame.q95_log - frame.q50_log) / ndtri(.95), np.sqrt(np.diag(posterior)), atol=1e-10)
    assert np.linalg.eigvalsh(posterior).min() > 0
    assert frame[["p_low_0_5", "p_low_1", "p_low_2"]].ge(0).all().all()
    assert frame[["p_low_0_5", "p_low_1", "p_low_2"]].le(1).all().all()


def test_support_loo_and_query_order_do_not_use_query_truth():
    model = CoKrigingPredictor(fit_profile_covariance(profiles("source")), variant="loo")
    model.condition(profiles("target"))
    assert len(model.loo_scores) == 3
    query = profiles("target").drop(columns="log_qc")
    query["location_id"] += "_new"
    query["x_m"] += 3
    first = model.predict(query).frame
    query["log_qc"] = 999
    query = query.sample(frac=1, random_state=11)
    second = model.predict(query).frame
    keys = ["location_id", "depth_index"]
    np.testing.assert_allclose(first.set_index(keys).sort_index().pred_log_qc,
                               second.set_index(keys).sort_index().pred_log_qc)
    assert list(second.location_id) == list(query.location_id)
    with pytest.raises(ValueError, match="disjoint"):
        model.condition(profiles("source"))


def test_loo_excludes_entire_near_group():
    model = CoKrigingPredictor(fit_profile_covariance(profiles("source")), variant="loo")
    support = profiles("target")
    support["near_group_id"] = "one_independent_group"
    with pytest.raises(ValueError, match="independent coordinate/near groups"):
        model.condition(support)
