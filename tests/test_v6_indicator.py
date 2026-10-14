"""Exact event inference and support/query isolation are the important risks."""
from itertools import product
import numpy as np
import pandas as pd

from cpt_budget.v6.indicator import IndicatorLayerPredictor, structured_event_profile


def test_dynamic_programming_matches_full_binary_enumeration():
    base = np.array([-0.1, 0.7, 0.2, -0.8])
    p = np.array([0.8, 0.4, 0.6, 0.5])
    target, tau, gamma, potts, margin = 2, 0.5, 0.25, 0.25, 0.05
    labels = np.array(list(product((0, 1), repeat=len(base))))
    amplitudes = np.where(labels, np.minimum(base, -margin), np.maximum(base, margin))
    energy = (-np.log(np.where(labels, p, 1 - p)) + (amplitudes - base)**2 / (2 * tau**2)).sum(axis=1)
    energy += potts * (labels[:, 1:] != labels[:, :-1]).sum(axis=1) + gamma * (labels.sum(axis=1) - target)**2
    weights = np.exp(-(energy - energy.min()))
    weights /= weights.sum()
    predicted, marginal, chosen = structured_event_profile(base, p, target)
    np.testing.assert_array_equal(chosen, labels[energy.argmin()])
    np.testing.assert_allclose(marginal, weights @ labels, atol=1e-12)
    np.testing.assert_allclose(predicted, amplitudes[energy.argmin()])
    np.testing.assert_array_equal(predicted < 0, chosen.astype(bool))


def sample_support():
    rows = []
    for i, profile in enumerate([[-.4, -.2, 1., 1.], [.5, -.1, -.3, 1.2], [.6, .1, 1., 1.]]):
        for d, value in enumerate(profile):
            rows.append({'site_id': 's', 'location_id': f'p{i}', 'x_m': i * 10., 'y_m': 0.,
                         'depth_index': d, 'z_center_m': 2.125 + .25 * d, 'log_qc': value})
    return pd.DataFrame(rows)


def test_predictions_ignore_query_labels_and_preserve_row_order():
    support = sample_support()
    query = support[support.location_id.eq('p0')].drop(columns='log_qc').assign(location_id='q', x_m=7.).iloc[::-1]
    model = IndicatorLayerPredictor(mode='adaptive')
    model.condition(support)
    first = model.predict(query).frame
    second = model.predict(query.assign(log_qc=-100., qc_mpa=1e-30)).frame
    np.testing.assert_allclose(first.pred_log_qc, second.pred_log_qc)
    np.testing.assert_allclose(first.p_low_1, second.p_low_1)
    np.testing.assert_array_equal(first.depth_index, query.depth_index)
    np.testing.assert_allclose(np.exp(first.pred_log_qc), first.pred_qc_mpa)
    assert len(model.loo_scores) == 9


def test_cardinality_is_soft_and_does_not_replace_point_labels():
    base = np.full(4, 8.)
    predicted, probability, labels = structured_event_profile(base, np.full(4, .6), 4)
    assert labels.sum() == 0  # overwhelming amplitude evidence can resist target count
    assert (predicted < 0).sum() == 0
    assert probability.sum() < 1e-10


def test_observed_leave_out_removes_entire_near_group_and_colocated_profiles():
    support = sample_support()
    support['near_group_id'] = support.location_id.map({'p0': 'g0', 'p1': 'g0', 'p2': 'g2'})
    model = IndicatorLayerPredictor()
    model.condition(support)
    base, _, count = model._spatial(model.xy[0], omit=0)
    np.testing.assert_array_equal(base, model.profiles[2])
    assert count == 0
    # Even distinct group labels cannot reintroduce colocated coordinates.
    support['near_group_id'] = support.location_id
    support.loc[support.location_id.eq('p1'), 'x_m'] = 0.
    model.condition(support)
    base, _, _ = model._spatial(model.xy[0], omit=0)
    np.testing.assert_array_equal(base, model.profiles[2])
