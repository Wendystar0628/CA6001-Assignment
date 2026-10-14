import numpy as np
import pandas as pd
import pytest

from cpt_budget.v6.thickness_head import (
    profile_features, constrained_labels, minimally_relabel, build_source, PROFILE_NAMES, EVENT_NAMES,
)


def rows(ids):
    return pd.DataFrame([dict(site_id="s", location_id=f"l{i}", depth_index=d,
        x_m=float(i*10), y_m=float(i % 2), log_qc=np.sin(d/3)+i/10.) for i in ids for d in range(24)])


def test_features_ignore_query_labels_and_support_row_order():
    support, query = rows(range(5)), rows([8, 9])
    expected = profile_features(support, query.drop(columns="log_qc"))
    query["log_qc"] = 1e20
    changed = profile_features(support.sample(frac=1., random_state=2), query.sample(frac=1., random_state=4))
    assert expected[0] == changed[0]
    assert expected[1].shape[1] == len(PROFILE_NAMES)
    assert expected[2].shape[-1] == len(EVENT_NAMES)
    for a, b in zip(expected[1:], changed[1:]):
        np.testing.assert_allclose(a, b)


def test_count_constraint_and_point_thickness_agree_with_physical_threshold():
    probability = np.tile(np.linspace(0., 1., 24), (3, 1))
    base = np.tile(np.linspace(-1., 1., 24), (3, 1))
    desired, counts = constrained_labels(probability, [0., 3.5, 24.], base)
    output = minimally_relabel(base, desired)
    np.testing.assert_array_equal((output < 0.).sum(axis=1), [0, 4, 24])
    np.testing.assert_array_equal((output < 0.).sum(axis=1), counts)
    unchanged = (base < 0.) == desired
    np.testing.assert_array_equal(output[unchanged], base[unchanged])
    assert np.all(np.abs(output[~unchanged]) == .05)


def test_features_reject_incomplete_or_revealed_query_profiles():
    support, query = rows(range(5)), rows([8])
    with pytest.raises(ValueError, match="complete"):
        profile_features(support, query.iloc[1:])
    with pytest.raises(ValueError, match="revealed"):
        profile_features(support, rows([0]))


def test_source_role_violation_rejected_before_any_dataset_access():
    plan = pd.DataFrame([dict(fold_id=0, role="train", site_id="target")])
    with pytest.raises(ValueError, match="role violation"):
        build_source(None, plan, {"fold_id": 0, "train_sites": ["source"]}, "train")
