import numpy as np
import pandas as pd
import pytest
import json

from cpt_budget.v5.functional import FunctionalProfilePredictor, fit_profile_basis


def profiles(site, n=5):
    rows = []
    for i in range(n):
        for d in range(6):
            rows.append(dict(site_id=site, location_id=f"{site}_{i}", depth_index=d,
                             x_m=float(i * 10), y_m=float((i % 2) * 5),
                             log_qc=1 + 0.2 * d + (i - n / 2) * np.sin(d / 2)))
    return pd.DataFrame(rows)


def test_source_site_weight_invariant_to_site_profile_replication():
    a, b = profiles("a"), profiles("b")
    b["log_qc"] += np.cos(b.depth_index) * b.x_m / 10
    original = fit_profile_basis(pd.concat([a, b]), max_components=2)
    repeated = a.copy()
    repeated["location_id"] += "_copy"
    duplicate = fit_profile_basis(pd.concat([a, repeated, b]), max_components=2)
    np.testing.assert_allclose(original.eigenvalues, duplicate.eigenvalues, atol=1e-12)
    np.testing.assert_allclose(original.components.T @ original.components,
                               duplicate.components.T @ duplicate.components, atol=1e-12)


def test_basis_artifact_saves_actual_rank_and_variance(tmp_path):
    basis = fit_profile_basis(profiles("source"))
    basis.save(tmp_path)
    manifest = json.loads((tmp_path / "basis_manifest.json").read_text())
    arrays = np.load(tmp_path / "basis.npz")
    assert manifest["n_components"] == len(basis.components)
    assert manifest["source_site_ids"] == ["source"]
    np.testing.assert_array_equal(arrays["depth_indices"], basis.depth_indices)


@pytest.mark.parametrize("method", ["krr", "linear"])
def test_prediction_keeps_query_order_and_never_reads_query_labels(method):
    basis = fit_profile_basis(profiles("source"))
    model = FunctionalProfilePredictor(basis, method=method)
    model.condition(profiles("target"))
    query = profiles("target").copy()
    query["location_id"] += "_new"
    query["x_m"] += 2
    clean = model.predict(query.drop(columns="log_qc")).frame
    query["log_qc"] = 1000
    reordered = query.sample(frac=1, random_state=7)
    changed = model.predict(reordered).frame
    keys = ["location_id", "depth_index"]
    np.testing.assert_allclose(clean.set_index(keys).sort_index().pred_log_qc,
                               changed.set_index(keys).sort_index().pred_log_qc)
    assert list(changed.location_id) == list(reordered.location_id)
    assert np.isfinite(changed.pred_log_qc).all()


def test_low_rank_is_full_profile_constraint_and_missing_support_depth_rejected():
    basis = fit_profile_basis(profiles("source"))
    model = FunctionalProfilePredictor(basis)
    support = profiles("target")
    with pytest.raises(ValueError, match="complete finite"):
        model.condition(support.drop(index=0))
    model.condition(support)
    query = profiles("target").drop(columns="log_qc")
    query["location_id"] += "_new"
    prediction = model.predict(query).frame.pivot(index="location_id", columns="depth_index", values="pred_log_qc")
    differences = prediction.to_numpy() - model.mean_profile
    np.testing.assert_allclose(differences,
        differences @ basis.components.T @ basis.components, atol=1e-12)
