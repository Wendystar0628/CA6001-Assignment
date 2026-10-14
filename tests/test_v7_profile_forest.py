import numpy as np
import pandas as pd
import pytest

from cpt_budget.v7.profile_forest import build_source, site_weights, fit_forests, predict_profiles
from cpt_budget.v6.thickness_head import profile_features


def test_source_role_boundary_before_read():
    plan = pd.DataFrame([dict(fold_id=0, role="train", site_id="target")])
    with pytest.raises(ValueError, match="role violation"):
        build_source(None, plan, {"fold_id": 0, "train_sites": ["source"]}, "train")
    with pytest.raises(ValueError, match="Only source"):
        build_source(None, plan, {"fold_id": 0}, "test")


def test_whole_profile_features_ignore_query_truth_and_equal_site_weight():
    rows = lambda ids: pd.DataFrame([dict(site_id="s", location_id=f"l{i}", depth_index=d,
        x_m=float(i*10), y_m=float(i % 2), log_qc=np.sin(d/3)+i/10.) for i in ids for d in range(24)])
    support, query = rows(range(5)), rows([8, 9])
    first = profile_features(support, query.drop(columns="log_qc"))
    query["log_qc"] = 1e20
    second = profile_features(support, query)
    np.testing.assert_array_equal(first[1], second[1])
    sites = np.array(["a", "a", "a", "b"])
    weights = site_weights(sites)
    assert weights[sites == "a"].sum() == weights[sites == "b"].sum()


def test_shared_multioutput_trees_and_residual_addition():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(20, 5))
    idw = np.tile(np.arange(24)/24, (20, 1))
    truth = idw + .5
    models = fit_forests((x, truth, idw, np.repeat(["a", "b"], 10), np.arange(20)),
                        {"n_estimators": 4, "random_state": 7, "n_jobs": 1})
    assert models["direct"].n_outputs_ == 24
    assert models["residual"].estimators_[0].n_outputs_ == 24
    direct, residual = predict_profiles(models, x, idw)
    np.testing.assert_allclose(direct, truth)
    np.testing.assert_allclose(residual, truth)
