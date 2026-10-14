import numpy as np
import pandas as pd
import pytest

from cpt_budget.v8.recognition import (
    FastRecognition, LocalEventEvidence, ReferenceRecognition, depth_stencil,
)


def support():
    return pd.DataFrame([
        dict(site_id='s', location_id=f'l{i}', near_group_id='g0' if i < 2 else f'g{i}',
             depth_index=k, x_m=float(i*13), y_m=float((i % 3)*9),
             z_center_m=2.125+.25*k, log_qc=np.sin(k/3.)+.2*i-.2)
        for i in range(6) for k in range(24)])


def query():
    template = support().loc[lambda f: f.location_id.eq('l0')].drop(columns='log_qc')
    return pd.concat([template.assign(location_id='q0', near_group_id='q0', x_m=20., y_m=14.),
                      template.assign(location_id='q1', near_group_id='q1', x_m=51., y_m=21.)],
                     ignore_index=True).sample(frac=1., random_state=7).reset_index(drop=True)


def test_fast_reference_analog_and_row_order_are_equivalent():
    q = query()
    base = .7*np.sin(q.depth_index.to_numpy()/3.)+.15
    ref = ReferenceRecognition().condition(support()).predict(q, base)
    fast = FastRecognition().condition(support()).predict(q, base)
    np.testing.assert_allclose(ref['point'], fast['point'], atol=1e-14)
    np.testing.assert_allclose(ref['probability'], fast['probability'], atol=1e-14)
    np.testing.assert_array_equal(ref['count_median'], fast['count_median'])


def test_whole_group_loo_excludes_alias_profile_event_values():
    observed = support()
    before = LocalEventEvidence(observed)
    revised = observed.copy()
    revised.loc[revised.location_id.eq('l1'), 'log_qc'] = -100.
    after = LocalEventEvidence(revised)
    np.testing.assert_array_equal(before.loo_local_probability[0], after.loo_local_probability[0])
    np.testing.assert_array_equal(before.loo_reference_probability[0], after.loo_reference_probability[0])
    assert np.all((before.local_weight >= 0.) & (before.local_weight <= .35))
    np.testing.assert_allclose(depth_stencil().sum(axis=1), 1.)


def test_amplitude_projection_preserves_count_and_unchanged_values():
    evidence = LocalEventEvidence(support())
    base = np.linspace(-.7, .8, 24)
    decision = base.copy()
    decision[2], decision[15] = .05, -.05
    output = evidence.project_crossings(base, decision, np.ones(6)/6)
    np.testing.assert_array_equal(output < 0., decision < 0.)
    unchanged = (base < 0.) == (decision < 0.)
    np.testing.assert_array_equal(output[unchanged], base[unchanged])
    assert np.any(output != decision)


def test_quality_candidates_preserve_selected_counts_and_reject_truth():
    q = query()
    base = .7*np.sin(q.depth_index.to_numpy()/3.)+.15
    for variant in ('localize', 'amplitude', 'recall60'):
        model = FastRecognition(variant).condition(support())
        result = model.predict(q, base)
        count = (result['point'][result['order']].reshape(-1, 24) < 0.).sum(axis=1)
        np.testing.assert_array_equal(count, result['chosen_count'])
        assert np.isfinite(result['point']).all()
        with pytest.raises(ValueError, match='geometry only'):
            model.predict(q.assign(truth_log_qc=-100.), base)
