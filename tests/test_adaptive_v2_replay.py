"""The v2 controller and acquisition must not observe hidden measurements."""
import json
import numpy as np
import pandas as pd
import pytest

import cpt_budget.adaptive_v2_replay as replay
from cpt_budget.models.distribution import point_prediction_frame
from test_adaptive_replay import fixture


class FakePredictor:
    def __init__(self, gp=False):
        self.gp = gp

    def condition(self, observed):
        self.observed = observed.copy()
        return self

    def predict(self, query):
        assert not {'log_qc', 'qc_mpa'} & set(query)
        known = self.observed.set_index(['location_id', 'depth_index']).log_qc
        keys = pd.MultiIndex.from_frame(query[['location_id', 'depth_index']])
        values = np.full(len(query), .2)
        mask = keys.isin(known.index)
        values[mask] = known.reindex(keys[mask]).to_numpy()
        frame = point_prediction_frame(query, values)
        if self.gp:
            frame['std_log_qc'] = np.where(mask, 0., .5)
        return frame


def run(dataset, split, *, rule=None):
    return replay.run_v2_trajectory(dataset, split, 'fake', 'maximin',
        predictor=FakePredictor(), gp_predictor=FakePredictor(gp=True),
        support_predictor=FakePredictor(), stop_rule=rule, log=None)


def test_sweep_without_calibration_reaches_full_and_keeps_groups_atomic():
    dataset, split = fixture()
    result = run(dataset, split)
    assert result['trajectories'].iloc[-1].n_observed == split['n_full']
    assert result['trajectories'].iloc[-1].stop_reason == 'full_plan_complete'
    assert not result['metadata']['audit_labels_used_online']
    for selected in result['trajectories'].selected_ids.map(json.loads):
        assert not set(selected) & {'l05', 'l06'} or {'l05', 'l06'} <= set(selected)
    assert set(result['predictions'].location_id) == set(split['audit_ids'])
    assert not {'log_qc', 'qc_mpa'} & set(result['predictions'])


def test_hidden_labels_cannot_change_online_stop_or_selection():
    dataset, split = fixture(initial=9)
    rule = replay.QualityStopRule(100., 100., 0., 0., 100., 100.,
        consecutive_rounds=1, calibration_sites=('development_elsewhere',))
    first = run(dataset, split, rule=rule)
    known = set(json.loads(first['trajectories'].iloc[-1].known_ids))
    assert len(known) < split['n_full']
    poisoned, _ = fixture(initial=9)
    poisoned.grid.loc[~poisoned.grid.location_id.isin(known), ['log_qc', 'qc_mpa']] = -999.
    second = run(poisoned, split, rule=rule)
    pd.testing.assert_frame_equal(first['trajectories'], second['trajectories'])
    pd.testing.assert_frame_equal(first['selections'], second['selections'])
    pd.testing.assert_frame_equal(first['predictions'], second['predictions'])


def test_stop_requires_named_development_calibration_and_rejects_nan_error():
    with pytest.raises(ValueError, match='calibration'):
        replay.QualityStopRule(1., 1., 0., 0., .1, .25)
    rule = replay.QualityStopRule(1., 1., 0., 0., .1, .25, calibration_sites=('dev',))
    signals = dict(delta_q90=0., delta_h90=0., coverage_ok=True,
                   candidate_geometry_supported=True, support_log_mae=np.nan,
                   support_thickness_mae_m=0.)
    assert not rule.checks(signals, 11)['log_quality']


def test_gp_primary_retains_one_native_uncertainty_column():
    dataset, split = fixture()
    gp = FakePredictor(gp=True)
    result = replay.run_v2_trajectory(dataset, split, 'gp', 'maximin', predictor=gp,
        gp_predictor=gp, diagnostics=False, log=None)
    values = result['trajectories'].loc[lambda x: x.n_observed < x.n_full, 'uncertainty_q90']
    assert np.allclose(values, .5)
