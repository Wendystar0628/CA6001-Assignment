import numpy as np
import pandas as pd

from cpt_budget.v6.complementary import complementary_points, decode_state


def test_frozen_mixtures_and_robust_even_median():
    result = complementary_points([0, 2], [1, 1000], [2, 3], [1000, 4],
                                  [10, 20], [20, 40], [30, 60])
    np.testing.assert_allclose(result["adaptive_graph"], [20, 40])
    np.testing.assert_allclose(result["fpca_graph"], [25, 50])
    np.testing.assert_allclose(result["median_four"], [1.5, 3.5])


def test_decoder_ignores_query_labels_preserves_order_and_reports_actual_event_sign():
    observed = pd.DataFrame([dict(site_id="s", location_id=f"s{i}", depth_index=d,
        z_center_m=2.125 + .25*d, x_m=i*10., y_m=0., log_qc=-.7 if d in (1, 2) else 1.)
        for i in range(3) for d in range(6)])
    query = observed.loc[observed.location_id.eq("s0")].assign(location_id="q", x_m=4.)
    base = {"a": np.full(6, .2), "b": np.array([1., -.1, -.2, 1., 1., 1.])}
    gp = np.array([.01, .95, .95, .01, .01, .01])
    values, probability, records = decode_state(observed, query, base, gp)
    poisoned, marginal, _ = decode_state(observed, query.assign(log_qc=-1000., qc_mpa=1e30,
                                            risk_probability=1.), base, gp)
    for name in base:
        np.testing.assert_allclose(values[name], poisoned[name])
        np.testing.assert_allclose(probability[name], marginal[name])
        assert np.all((probability[name] >= 0) & (probability[name] <= 1))
        record = next(r for r in records if r["base"] == name)
        assert record["predicted_count"] == int((values[name] < 0).sum())
        assert record["support_count_prior"] == 2
    permutation = [4, 0, 3, 2, 5, 1]
    shuffled, shuffled_p, _ = decode_state(observed, query.iloc[permutation],
                    {k: v[permutation] for k, v in base.items()}, gp[permutation])
    for name in base:
        np.testing.assert_allclose(shuffled[name], values[name][permutation])
        np.testing.assert_allclose(shuffled_p[name], probability[name][permutation])
