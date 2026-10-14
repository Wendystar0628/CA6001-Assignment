import numpy as np
import pandas as pd
import pytest
from scipy.fft import dct, idct

from cpt_budget.v7.multiscale import haar_forward, haar_inverse, MultiscaleProfilePredictor, VARIANTS


def profiles():
    return pd.DataFrame([dict(site_id="target", location_id=f"c{i}", depth_index=d,
                             x_m=10.0*i, y_m=3.0*(i%2), log_qc=np.sin(d/3) + i/5)
                         for i in range(5) for d in range(24)])


def test_complete_transforms_reconstruct_spike_and_preserve_energy():
    values = np.random.default_rng(12).normal(size=(5, 24))
    values[0] = 1
    values[0, 11] = -2  # One 0.25 m low-resistance cell must survive the representation.
    haar = haar_forward(values)
    np.testing.assert_allclose(haar_inverse(haar), values, atol=1e-14)
    np.testing.assert_allclose(np.sum(haar**2, axis=1), np.sum(values**2, axis=1), atol=1e-13)
    cosine = dct(values, type=2, norm="ortho", axis=1)
    np.testing.assert_allclose(idct(cosine, type=2, norm="ortho", axis=1), values, atol=1e-14)


@pytest.mark.parametrize("variant", VARIANTS)
def test_query_labels_ignored_order_preserved_and_observed_coordinates_reproduced(variant):
    support = profiles()
    model = MultiscaleProfilePredictor(variant)
    model.condition(support)
    query = support.drop(columns="log_qc").copy()
    query.location_id += "_query"
    exact = model.predict(query).frame
    np.testing.assert_allclose(exact.pred_log_qc, support.log_qc, atol=1e-12)
    query.x_m += 2.0
    clean = model.predict(query).frame
    query["log_qc"] = -1000.0
    dirty = model.predict(query.sample(frac=1, random_state=5)).frame
    keys = ["location_id", "depth_index"]
    np.testing.assert_allclose(clean.set_index(keys).sort_index().pred_log_qc,
                               dirty.set_index(keys).sort_index().pred_log_qc)
    assert np.isfinite(dirty.pred_log_qc).all()


def test_incomplete_revealed_profile_rejected():
    model = MultiscaleProfilePredictor(VARIANTS[0])
    with pytest.raises(ValueError, match="complete finite"):
        model.condition(profiles().iloc[1:])
