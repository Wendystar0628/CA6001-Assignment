import numpy as np
import pandas as pd
import pytest

from cpt_budget.v5.graph import GraphProfilePredictor, _horizontal_edges


def grid(n=7, d=8):
    return pd.DataFrame([
        dict(site_id='s', location_id=f'l{i}', depth_index=k,
             x_m=float(i)*10, y_m=0., z_center_m=2.125+0.25*k)
        for i in range(n) for k in range(d)])


@pytest.mark.parametrize('variant', ['harmonic', 'huber'])
def test_constant_field_exact_reconstruction_and_boundary(variant):
    plan = grid()
    observed = plan.loc[plan.location_id.isin(['l0', 'l6'])].assign(log_qc=1.7)
    predictor = GraphProfilePredictor(plan, variant=variant)
    predictor.condition(observed)
    result = predictor.predict(plan).frame
    np.testing.assert_allclose(result.pred_log_qc, 1.7, atol=1e-10)
    assert 'p_low_1' not in result
    assert result.graph_converged.all()
    assert result.graph_stationarity_max.max() < 1e-9


def test_geometry_boundary_rejects_label_columns_and_query_coordinate_drift():
    plan = grid()
    with pytest.raises(ValueError, match='geometry-only'):
        GraphProfilePredictor(plan.assign(log_qc=999))
    predictor = GraphProfilePredictor(plan)
    observed = plan.loc[plan.location_id.isin(['l0', 'l6'])].assign(log_qc=1.)
    predictor.condition(observed)
    with pytest.raises(ValueError, match='geometry-only'):
        predictor.predict(plan.assign(qc_mpa=999))
    with pytest.raises(ValueError, match='disagree'):
        predictor.predict(plan.assign(x_m=plan.x_m+1))


def test_huber_preserves_sharp_layer_more_than_quadratic_and_is_query_order_invariant():
    plan = grid()
    observed = plan.loc[plan.location_id.isin(['l0', 'l6'])].copy()
    observed['log_qc'] = np.where(observed.depth_index < 4, -2., 2.)
    predictions = {}
    for variant in ('harmonic', 'huber'):
        predictor = GraphProfilePredictor(plan, variant=variant)
        predictor.condition(observed)
        predictions[variant] = predictor.predict(plan).frame
        boundary = predictions[variant].merge(observed, on=['location_id', 'depth_index'])
        np.testing.assert_allclose(boundary.pred_log_qc, boundary.log_qc, atol=0)
        shuffled = predictor.predict(plan.sample(frac=1, random_state=3)).frame
        key = ['location_id', 'depth_index']
        np.testing.assert_allclose(predictions[variant].sort_values(key).pred_log_qc,
                                   shuffled.sort_values(key).pred_log_qc)
    target = np.where(plan.depth_index < 4, -2., 2.)
    errors = {v: float(np.mean(abs(f.pred_log_qc-target))) for v, f in predictions.items()}
    assert errors['huber'] < errors['harmonic']
    assert predictions['huber'].graph_objective_end.iloc[0] <= predictions['huber'].graph_objective_start.iloc[0]


def test_geometric_graph_connects_two_clusters_without_using_labels():
    xy = np.array([[0., 0.], [0., 0.], [1., 0.], [100., 0.], [101., 0.]])
    edges, weights, _ = _horizontal_edges(xy, neighbors=1)
    reached = {0}
    for _ in range(len(xy)):
        for i, j in edges:
            if i in reached or j in reached:
                reached.update([int(i), int(j)])
    assert len(reached) == len(xy)
    assert np.all(weights > 0)
    assert 2*weights.sum()/len(xy) == pytest.approx(1.)
