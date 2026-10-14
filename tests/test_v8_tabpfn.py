import numpy as np
import pandas as pd
import pytest

from cpt_budget.diagnostics.residual_adapter import leave_location_out_idw
from cpt_budget.models.tabpfn_adapter import geometry_only
from cpt_budget.v2.tabpfn import feature_array_v2
from cpt_budget.v8.tabpfn import ProfileFeatureContext


def profiles():
    rng = np.random.default_rng(7)
    rows = []
    for i, (x, y) in enumerate([(0, 0), (0, 0), (10, 0), (-10, 0), (3, 12)]):
        for d, z in enumerate([2.125, 2.375, 2.625, 2.875]):
            rows.append(dict(site_id='s', location_id=f'p{i}', depth_index=d,
                             z_center_m=z, x_m=x, y_m=y, geometry_scale_m=10., log_qc=rng.normal()))
    return pd.DataFrame(rows)


@pytest.mark.parametrize('loo', [False, True])
def test_complete_profile_features_match_reference_with_coincidence_and_ties(loo):
    observed = profiles().sample(frac=1, random_state=31).reset_index(drop=True)
    query = geometry_only(observed).sample(frac=1, random_state=13).reset_index(drop=True)
    fast = ProfileFeatureContext(observed).features(query, leave_location_out=loo)
    old = feature_array_v2(observed, query, leave_location_out=loo)
    # Exact reductions matter because official rank preprocessing separates
    # tied probabilities after otherwise harmless floating-point perturbations.
    np.testing.assert_array_equal(fast, old)


def test_support_reference_is_whole_location_loo_idw():
    observed = profiles()
    fast = ProfileFeatureContext(observed).features(geometry_only(observed), leave_location_out=True)
    np.testing.assert_allclose(fast[:, 13], leave_location_out_idw(observed), atol=2e-14, rtol=2e-14)


def test_query_labels_are_ignored_and_incomplete_support_falls_back():
    observed = profiles().iloc[1:].copy()
    query = profiles()
    context = ProfileFeatureContext(observed)
    assert not context.fast
    expected = feature_array_v2(observed, query)
    query['log_qc'] = 1e6
    np.testing.assert_allclose(context.features(query), expected, equal_nan=True)


def test_mismatched_query_depth_is_rejected():
    observed = profiles()
    query = geometry_only(observed)
    query.loc[0, 'z_center_m'] += .1
    with pytest.raises(ValueError, match='common depth grid'):
        ProfileFeatureContext(observed).features(query)
