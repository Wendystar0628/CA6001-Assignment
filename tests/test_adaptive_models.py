import numpy as np
import pandas as pd
import pytest

from cpt_budget.adaptive_models import AdaptiveModelSuite, audit_fold_sources
from cpt_budget.models.distribution import gaussian_prediction_frame
from cpt_budget.types import PredictionBundle


def test_evaluation_excludes_checkpoint_selection_and_source_training():
    assert audit_fold_sources(0, 'christchurch_stadium')['passed']
    assert not audit_fold_sources(0, 'marshland')['passed']
    assert not audit_fold_sources(0, 'lower_hutt_town_hall')['passed']
    assert audit_fold_sources(0, 'lower_hutt_town_hall', role='development')['passed']


class _FakeGP:
    def condition(self, observed):
        self.observed = observed

    def predict(self, query):
        assert not {'log_qc', 'qc_mpa'} & set(query)
        return PredictionBundle(gaussian_prediction_frame(query, np.full(len(query), .2), np.full(len(query), .7)), 'gp')


def test_fixed_grid_uses_only_revealed_labels_and_native_gp_std(monkeypatch):
    site = 'christchurch_stadium'
    observed = pd.DataFrame([dict(site_id=site, location_id='known', depth_index=0,
                                  x_m=0., y_m=0., z_center_m=2.125, log_qc=-.8)])
    query = pd.concat([observed, observed.assign(location_id='hidden', x_m=10., log_qc=-999.)], ignore_index=True)
    suite = AdaptiveModelSuite(0, ('gp',))
    suite.models['gp'] = _FakeGP()
    suite.condition(observed)
    frame = suite.predict(query)['gp'].frame
    np.testing.assert_allclose(frame.pred_log_qc, [-.8, .2])
    np.testing.assert_allclose(frame.std_log_qc, [0., .7])
    assert frame.is_revealed.tolist() == [True, False]
    assert 'log_qc' not in frame
    with pytest.raises(ValueError, match='geometry'):
        suite.predict(query.assign(x_m=99.))


def test_condition_rejects_source_seen_target_before_model_calls():
    observed = pd.DataFrame([dict(site_id='marshland', location_id='known', depth_index=0,
                                  x_m=0., y_m=0., z_center_m=2.125, log_qc=0.)])
    with pytest.raises(ValueError, match='contamination'):
        AdaptiveModelSuite(0, ('cnp',)).condition(observed)
