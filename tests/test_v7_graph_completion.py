import numpy as np
import pandas as pd
import pytest

from cpt_budget.v7.graph_completion import GraphCompletionPredictor, VARIANTS, nuclear_completion, _svt


def grid(n=9, d=8):
    return pd.DataFrame([dict(site_id='s', location_id=f'l{i}', depth_index=k,
        x_m=float(i)*10, y_m=0., z_center_m=2.125+.25*k) for i in range(n) for k in range(d)])


@pytest.mark.parametrize('variant', VARIANTS)
def test_boundary_constant_layer_profile_and_query_order(variant):
    plan = grid()
    observed = plan.loc[plan.location_id.isin(['l0', 'l4', 'l8'])].copy()
    observed['log_qc'] = np.where(observed.depth_index < 3, -1., 1.)
    predictor = GraphCompletionPredictor(plan, variant=variant)
    predictor.condition(observed)
    result = predictor.predict(plan).frame
    np.testing.assert_allclose(result.pred_log_qc, np.where(plan.depth_index < 3, -1., 1.), atol=1e-10)
    assert result.gmc_boundary_max_error.max() == 0
    assert result.gmc_converged.all()
    shuffled = predictor.predict(plan.sample(frac=1, random_state=1)).frame
    keys = ['location_id', 'depth_index']
    np.testing.assert_allclose(result.sort_values(keys).pred_log_qc, shuffled.sort_values(keys).pred_log_qc)


def test_nontrivial_nuclear_convergence_and_exact_support():
    plan = grid()
    observed = plan.loc[plan.location_id.isin(['l0', 'l4', 'l8'])].copy()
    observed['log_qc'] = np.sin(observed.depth_index)+observed.x_m/60
    for variant in ('nuclear', 'nuclear_gradient'):
        predictor = GraphCompletionPredictor(plan, variant=variant)
        predictor.condition(observed)
        pred = predictor.predict(observed.drop(columns='log_qc')).frame
        np.testing.assert_array_equal(pred.pred_log_qc, observed.log_qc)
        assert pred.gmc_converged.all()
        assert pred.gmc_primal_residual.iloc[0] <= pred.gmc_primal_tolerance.iloc[0]
        assert pred.gmc_dual_residual.iloc[0] <= pred.gmc_dual_tolerance.iloc[0]


def test_svt_is_singular_value_proximal_and_geometry_rejects_truth():
    np.testing.assert_allclose(_svt(np.diag([3., 1., .2]), .5), np.diag([2.5, .5, 0.]))
    with pytest.raises(ValueError, match='geometry-only'):
        GraphCompletionPredictor(grid().assign(truth_log_qc=1.))
