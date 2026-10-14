import numpy as np
import pandas as pd
import pytest

from cpt_budget.v6.graph_combinations import (
    mix_graph_residual, observed_bias_context, project_low_mask,
)
from cpt_budget.v5.graph import GraphProfilePredictor


def test_convex_mix_and_strict_sign_projection():
    graph = np.array([-2., 1., -.8, 0.])
    residual = np.array([1., -2., .3, 0.])
    mixed = mix_graph_residual(graph, residual, .5)
    assert np.all(mixed >= np.minimum(graph, residual))
    assert np.all(mixed <= np.maximum(graph, residual))
    guarded = project_low_mask(mixed, residual < 0)
    np.testing.assert_array_equal(guarded < 0, residual < 0)
    union = project_low_mask(mixed, (graph < 0) | (residual < 0))
    np.testing.assert_array_equal(union < 0, (graph < 0) | (residual < 0))
    with pytest.raises(ValueError, match='frozen'):
        mix_graph_residual(graph, residual, .1)


def test_loo_bias_is_whole_profile_and_does_not_mutate_support():
    observed = pd.DataFrame([
        dict(site_id='s', location_id=f'l{i}', depth_index=k, x_m=i*10., y_m=0.,
             z_center_m=2.125+.25*k, log_qc=float(i)+k*.3)
        for i in range(3) for k in range(4)
    ])
    original = observed.copy(deep=True)
    bias = observed_bias_context(observed)
    pd.testing.assert_frame_equal(observed, original)
    # The two edge CPTs cannot reproduce themselves from the other profiles.
    assert np.all(np.abs(bias.loc[bias.location_id.eq('l0'), 'log_qc']) > 1.)
    assert np.all(np.abs(bias.loc[bias.location_id.eq('l2'), 'log_qc']) > 1.)
    np.testing.assert_allclose(bias.loc[bias.location_id.eq('l1'), 'log_qc'], 0., atol=1e-12)
    # Equal vertical trends cancel from the LOO residual target.
    assert bias.groupby('location_id').log_qc.std().max() < 1e-12


def test_zero_observed_bias_produces_zero_graph_correction():
    plan = pd.DataFrame([
        dict(site_id='s', location_id=f'l{i}', depth_index=k, x_m=i*10., y_m=0.,
             z_center_m=2.125+.25*k)
        for i in range(5) for k in range(4)
    ])
    observed = plan.loc[plan.location_id.isin(['l0', 'l2', 'l4'])].assign(log_qc=1.25)
    graph = GraphProfilePredictor(plan, variant='huber')
    graph.condition(observed_bias_context(observed))
    np.testing.assert_allclose(graph.predict(plan).frame.pred_log_qc, 0., atol=1e-12)
    assert graph.solver_diagnostics['graph_boundary_max_error'] == 0.
