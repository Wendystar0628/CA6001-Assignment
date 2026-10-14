import numpy as np
import pandas as pd

from cpt_budget.v5.functional import fit_profile_basis
from cpt_budget.v6.adaptive import (weighted_median_predict, member_predictions,
                                    leave_one_location_out, fit_weights)


def profiles(site="target", n=4):
    rows = []
    for i in range(n):
        for d in range(8):
            value = 0.4 + 0.1 * i - 1.0 * np.exp(-0.5 * ((d - 3 - i / 3) / 1.1) ** 2)
            rows.append(dict(site_id=site, location_id=f"{site}_{i}", depth_index=d,
                             x_m=10.0 * i, y_m=2.0 * (i % 2), z_center_m=2.125 + .25 * d,
                             geometry_scale_m=30.0, log_qc=value, qc_mpa=np.exp(value)))
    return pd.DataFrame(rows)


def test_weighted_median_is_l1_optimum_and_colocated_exact():
    observed = profiles()
    query = observed.loc[observed.location_id.eq("target_0")].drop(columns=["log_qc", "qc_mpa"])
    np.testing.assert_allclose(weighted_median_predict(observed, query),
                               observed.loc[observed.location_id.eq("target_0"), "log_qc"])
    query = query.assign(location_id="new", x_m=14.0, y_m=0.0)
    actual = weighted_median_predict(observed, query)
    xy = observed.drop_duplicates("location_id")[["x_m", "y_m"]].to_numpy()
    dist = np.linalg.norm(xy - [14.0, 0.0], axis=1)
    nearest = np.argsort(dist)[:3]
    weights = dist[nearest] ** -2
    values = observed.pivot(index="location_id", columns="depth_index", values="log_qc").to_numpy()[nearest]
    actual_loss = (weights[:, None] * np.abs(values - actual)).sum(axis=0)
    for candidate in values:
        assert np.all(actual_loss <= (weights[:, None] * np.abs(values - candidate)).sum(axis=0) + 1e-12)


def test_loo_withholds_entire_profile_and_does_not_use_its_labels():
    observed = profiles()
    basis = fit_profile_basis(profiles("source"))
    ordered, before = leave_one_location_out(observed, basis)
    poisoned = observed.copy()
    withheld = poisoned.location_id.eq("target_0")
    poisoned.loc[withheld, "log_qc"] += 100
    _, after = leave_one_location_out(poisoned, basis)
    np.testing.assert_allclose(before[ordered.location_id.eq("target_0")],
                               after[ordered.location_id.eq("target_0")])
    assert not np.allclose(before[~ordered.location_id.eq("target_0")],
                           after[~ordered.location_id.eq("target_0")])


def test_members_ignore_poisoned_query_labels_and_keep_order():
    observed = profiles()
    basis = fit_profile_basis(profiles("source"))
    query = profiles("target", 2).assign(location_id=lambda x: x.location_id + "_new", x_m=lambda x: x.x_m + 3)
    clean = member_predictions(observed, query.drop(columns=["log_qc", "qc_mpa"]), basis)
    dirty = member_predictions(observed, query.assign(log_qc=-1000, qc_mpa=1e30), basis)
    np.testing.assert_allclose(clean, dirty)
    permutation = np.random.default_rng(3).permutation(len(query))
    reordered = member_predictions(observed, query.iloc[permutation], basis)
    np.testing.assert_allclose(reordered, clean[permutation])


def test_stack_selects_predictive_member_under_both_objectives():
    observed = profiles()
    truth = observed.log_qc.to_numpy()
    predictions = np.column_stack([truth, truth + 3, truth + 4, truth + 5])
    for objective in ("l1", "structural"):
        weights, record = fit_weights(observed, predictions, objective=objective)
        np.testing.assert_allclose(weights, [1, 0, 0, 0])
        assert record["loo_log_mae"] == 0


def test_loo_excludes_near_groups_and_colocated_labels():
    observed = profiles(n=5)
    observed["near_group_id"] = observed.location_id
    observed.loc[observed.location_id.eq("target_1"), "near_group_id"] = "target_0"
    observed.loc[observed.location_id.eq("target_2"), ["x_m", "y_m"]] = [0.0, 0.0]
    basis = fit_profile_basis(profiles("source"))
    ordered, before = leave_one_location_out(observed, basis)
    changed = observed.copy()
    changed.loc[changed.location_id.isin(["target_0", "target_1", "target_2"]), "log_qc"] += 50
    _, after = leave_one_location_out(changed, basis)
    np.testing.assert_allclose(before[ordered.location_id.eq("target_0")],
                               after[ordered.location_id.eq("target_0")])
    first = ordered.attrs["loo_exclusions"][0]
    assert set(first["excluded_location_ids"]) == {"target_0", "target_1", "target_2"}
    assert first["n_remaining_locations"] == 2
