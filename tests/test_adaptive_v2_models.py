import numpy as np
import pandas as pd
import pytest

from cpt_budget.adaptive_v2_models import make_v2_predictor


def _data():
    rows = []
    for location in range(7):
        for depth in range(6):
            rows.append(dict(site_id='target', location_id=f'l{location}', near_group_id=f'g{location//2}',
                             depth_index=depth, z_center_m=2.125+.25*depth, x_m=location*10., y_m=location%2,
                             log_qc=np.sin(depth)+location*.1))
    return pd.DataFrame(rows)


def test_target_models_use_only_observed_labels_and_preserve_query_order():
    data = _data()
    observed, query = data[data.location_id!='l6'], data[data.location_id=='l6'].iloc[::-1].copy()
    for name in ('target_trend_gp', 'target_fpca_gp', 'target_profile_cv', 'target_joint_cv'):
        model = make_v2_predictor(name).condition(observed)
        first = model.predict(query)
        second = model.predict(query.assign(log_qc=1e8, qc_mpa=1e8))
        np.testing.assert_allclose(first.pred_log_qc, second.pred_log_qc)
        assert first.depth_index.tolist() == query.depth_index.tolist()
        assert not {'log_qc', 'qc_mpa'} & set(first)
        assert len(model.context_manifest['observed_location_ids']) == 6
        assert model.crossfit_predictions.shape == (6, 6)


def test_observed_query_values_are_measured_and_mixed_sites_rejected():
    data = _data()
    model = make_v2_predictor('target_depth_mean').condition(data)
    np.testing.assert_allclose(model.predict(data).pred_log_qc, data.log_qc)
    with pytest.raises(ValueError, match='match'):
        model.predict(data.assign(site_id='other'))
    with pytest.raises(ValueError, match='geometry'):
        model.predict(data.assign(x_m=999.))


def test_near_group_cv_does_not_borrow_sibling_measurements():
    from cpt_budget.adaptive_v2_models import _folds
    groups = np.array(['a', 'a', 'b', 'c', 'd', 'e'])
    masks = _folds(groups)
    assert all(mask[0] == mask[1] for mask in masks)
    assert np.stack(masks).sum(axis=0).tolist() == [1]*6
