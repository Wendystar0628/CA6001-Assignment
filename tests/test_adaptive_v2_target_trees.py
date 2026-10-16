import numpy as np
import pandas as pd
import pytest

from cpt_budget.adaptive_v2_target_trees import make_target_tree_predictor


def _data():
    return pd.DataFrame([dict(site_id='target', location_id=f'l{i}', near_group_id=f'g{i//2}',
        depth_index=d, z_center_m=2.125+.25*d, x_m=i*10., y_m=i%2,
        log_qc=np.sin(d)+.15*i) for i in range(7) for d in range(6)])


def test_query_labels_and_order_cannot_change_target_tree_prediction():
    data = _data()
    observed, query = data[data.location_id != 'l6'], data[data.location_id == 'l6'].iloc[::-1]
    model = make_target_tree_predictor('target_et_residual').condition(observed)
    first = model.predict(query)
    second = model.predict(query.assign(log_qc=1e8, qc_mpa=1e8))
    np.testing.assert_allclose(first.pred_log_qc, second.pred_log_qc)
    assert first.depth_index.tolist() == query.depth_index.tolist()
    assert model.crossfit_predictions.shape == (6, 6)
    assert model.context_manifest['audit_labels_used'] is False
    assert set(model.context_manifest['observed_location_ids']) == set(observed.location_id)


def test_exact_revealed_values_and_site_boundary():
    data = _data()
    model = make_target_tree_predictor('target_et_raw').condition(data)
    np.testing.assert_allclose(model.predict(data).pred_log_qc, data.log_qc)
    with pytest.raises(ValueError, match='match'):
        model.predict(data.assign(site_id='other'))


def test_cv_refits_trend_and_geometry_without_sibling_labels():
    data = _data()
    first = make_target_tree_predictor('target_et_residual').condition(data)
    changed = data.copy()
    changed.loc[changed.near_group_id == 'g0', 'log_qc'] += 50
    second = make_target_tree_predictor('target_et_residual').condition(changed)
    # Every member of g0 is held out together; its labels cannot change its CV predictions.
    np.testing.assert_allclose(first.crossfit_predictions[:2], second.crossfit_predictions[:2])


def test_median_count_variant_selects_with_revealed_cv_and_one_curve():
    data = _data()
    observed, query = data[data.location_id != 'l6'], data[data.location_id == 'l6']
    model = make_target_tree_predictor('target_et_median_count_cv').condition(observed)
    first = model.predict(query)
    second = model.predict(query.assign(log_qc=-1e9))
    np.testing.assert_allclose(first.pred_log_qc, second.pred_log_qc)
    np.testing.assert_allclose(first.pred_qc_mpa, np.exp(first.pred_log_qc))
    assert len(model.cv_candidates) == 3
    assert {c['count_mode'] for c in model.cv_candidates} == {'native', 'half', 'full'}
