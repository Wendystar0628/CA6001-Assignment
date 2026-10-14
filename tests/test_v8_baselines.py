import numpy as np
import pandas as pd
import pytest

from cpt_budget.models.idw import IDWPredictor
from cpt_budget.v8.baselines import OptimizedIDWPredictor


def support():
    coordinates = [(0., 0.), (0., 0.), (-2., 0.), (2., 0.), (0., 2.)]
    return pd.DataFrame([dict(site_id="s", location_id=f"p{i}", depth_index=d,
                             x_m=x, y_m=y, z_center_m=2.125 + .25*d,
                             geometry_scale_m=10., log_qc=.3*i-.8*d)
                         for i, (x, y) in enumerate(coordinates) for d in range(4)])


def paired(observed, query):
    old, new = IDWPredictor(), OptimizedIDWPredictor()
    old.condition(observed)
    new.condition(observed)
    expected = old.predict(query).frame.pred_log_qc.to_numpy()
    actual = new.predict(query).frame.pred_log_qc.to_numpy()
    np.testing.assert_array_equal(actual, expected)
    return new, actual


def test_complete_profiles_preserve_ties_coincidence_and_query_order():
    observed = support().sample(frac=1, random_state=19)
    query = pd.concat([support().iloc[:4].assign(location_id="zero", x_m=0., y_m=0.),
                       support().iloc[:4].assign(location_id="ties", x_m=0., y_m=1.),
                       support().iloc[:4].assign(location_id="far", x_m=4., y_m=3.)], ignore_index=True)
    model, expected = paired(observed, query)
    assert model.complete_profiles
    # The two coincident supports among the three nearest receive equal share.
    np.testing.assert_allclose(expected[:4], .15 - .8*np.arange(4), rtol=0, atol=1e-15)
    order = np.random.default_rng(4).permutation(len(query))
    dirty = query.iloc[order].assign(log_qc=-999., qc_mpa=1e99)
    np.testing.assert_array_equal(model.predict(dirty).frame.pred_log_qc.to_numpy(), expected[order])


def test_unequal_depth_and_depth_varying_coordinates_use_equivalent_fallback():
    observed = support()
    observed = observed.loc[~(observed.location_id.eq("p0") & observed.depth_index.eq(1))].copy()
    observed.loc[observed.location_id.eq("p4") & observed.depth_index.eq(2), "x_m"] = 3.
    query = support().iloc[:4].assign(location_id="new", x_m=.7, y_m=.3)
    model, _ = paired(observed, query)
    assert not model.complete_profiles
    with pytest.raises(ValueError, match="requested query depth"):
        model.predict(query.iloc[:1].assign(depth_index=99))
    with pytest.raises(ValueError, match="Query sites"):
        model.predict(query.assign(site_id="foreign"))


def test_reconditioning_does_not_reuse_old_values_or_neighbor_formula():
    observed = support()
    query = observed.iloc[:4].assign(location_id="new", x_m=.7, y_m=.3)
    model, before = paired(observed, query)
    changed = observed.assign(log_qc=observed.log_qc + 2.)
    model.condition(changed)
    after = model.predict(query).frame.pred_log_qc.to_numpy()
    np.testing.assert_allclose(after, before + 2., rtol=0, atol=1e-14)
    with pytest.raises(ValueError, match="fixed to original"):
        OptimizedIDWPredictor({"idw": {"nearest_neighbors": 5}})
