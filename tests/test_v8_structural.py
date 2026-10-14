import numpy as np
import pandas as pd
import pytest

from cpt_budget.v5.functional import fit_profile_basis
from cpt_budget.v5.layer_alignment import bounded_warp
from cpt_budget.v8.structural import (OriginalStructuralPredictor, StructuralPredictor,
                                       fast_bounded_warp)


def profiles(site="target", n=5):
    rows = []
    for i in range(n):
        for d in range(8):
            y = .4 + .1 * i - np.exp(-.5 * ((d - 3 - i / 3) / 1.1) ** 2)
            rows.append(dict(site_id=site, location_id=f"{site}_{i}", depth_index=d,
                             x_m=10. * i, y_m=2. * (i % 2), z_center_m=2.125 + .25 * d,
                             geometry_scale_m=30., log_qc=y, qc_mpa=np.exp(y)))
    return pd.DataFrame(rows)


def test_vectorized_warp_preserves_scalar_paths_and_ties():
    rng = np.random.default_rng(13)
    for n in (8, 24):
        for _ in range(5):
            a, b = rng.normal(size=(2, n))
            np.testing.assert_array_equal(fast_bounded_warp(a, b), bounded_warp(a, b))
        np.testing.assert_array_equal(fast_bounded_warp(np.zeros(n), np.zeros(n)), np.arange(n))


def test_exact_live_path_matches_original_and_preserves_query_order():
    observed = profiles()
    basis = fit_profile_basis(profiles("source"))
    query = profiles(n=2).assign(location_id=lambda x: x.location_id + "_q", x_m=lambda x: x.x_m + 3)
    old, new = OriginalStructuralPredictor(basis=basis), StructuralPredictor(basis=basis)
    old.condition(observed)
    new.condition(observed)
    expected = old.predict(query).frame.pred_log_qc.to_numpy()
    actual = new.predict(query).frame.pred_log_qc.to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    order = np.random.default_rng(4).permutation(len(query))
    dirty = query.iloc[order].assign(log_qc=-1000, qc_mpa=1e50)
    np.testing.assert_allclose(new.predict(dirty).frame.pred_log_qc, actual[order], rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="Query sites"):
        new.predict(query.assign(site_id="foreign"))


def test_fast_loo_excludes_near_and_colocated_profiles():
    observed = profiles()
    observed["near_group_id"] = observed.location_id
    observed.loc[observed.location_id.eq("target_1"), "near_group_id"] = "target_0"
    observed.loc[observed.location_id.eq("target_2"), ["x_m", "y_m"]] = [0., 0.]
    basis = fit_profile_basis(profiles("source"))
    a, b = StructuralPredictor(basis=basis, quality_context=True), StructuralPredictor(basis=basis, quality_context=True)
    a.condition(observed)
    changed = observed.copy()
    changed.loc[changed.location_id.isin(["target_0", "target_1", "target_2"]), "log_qc"] += 50
    b.condition(changed)
    held = observed.location_id.eq("target_0")
    np.testing.assert_allclose(a.loo_predictions[held], b.loo_predictions[held], rtol=0, atol=1e-12)
    assert set(a.engine.exclusions[0]["excluded_location_ids"]) == {"target_0", "target_1", "target_2"}
    for variant in ("loo_simplex", "reliability"):
        a.variant = variant
        prediction = a.predict(observed.drop(columns=["log_qc", "qc_mpa"]))
        assert np.isfinite(prediction.frame.pred_log_qc).all()
