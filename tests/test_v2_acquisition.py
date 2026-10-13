import numpy as np
import pandas as pd
import pytest
from scipy.linalg import cho_solve

from cpt_budget.acquisition import select_batch
from cpt_budget.models.gp import GPPredictor
from cpt_budget.types import PreparedDataset
from cpt_budget.v2.acquisition import _block_update, _latent_posterior, select_gp_batch


def _example(tmp_path):
    ids = [f"p{i:02d}" for i in range(12)]
    xy = np.array([[0, 0], [0, 20], [20, 0], [20, 20], [10, 10],
                   [5, 4], [5.1, 4.1], [14, 6], [30, 10], [40, 20], [12, 12], [18, 16]])
    profiles = pd.DataFrame({"site_id": "s", "profile_id": ids, "location_id": ids,
                             "near_group_id": ids, "x_m": xy[:, 0], "y_m": xy[:, 1],
                             "geometry_scale_m": 10., "usable": True, "is_representative": True})
    grid = pd.DataFrame([{"site_id": "s", "location_id": loc, "depth_index": d,
                          "z_center_m": 2.125 + .25 * d,
                          "log_qc": float(np.sin(d / 4) + .01 * xy[i, 0]),
                          "qc_mpa": float(np.exp(np.sin(d / 4) + .01 * xy[i, 0]))}
                         for i, loc in enumerate(ids) for d in range(24)])
    sites = pd.DataFrame({"site_id": ["s"], "investigation_group": ["g"]})
    dataset = PreparedDataset(profiles, grid, sites, tmp_path)
    config = {"run": {"mode": "pilot"}, "gp": {"optimize": False, "noise_level": .2},
              "data": {"depth_step_m": .25, "tau_mpa": 1.},
              "replay": {"budgets": [5, 7], "batch_size": 2},
              "v2": {"acquisition": {"shortlist_size": 3, "n_fantasies": 4, "seed": 9}}}
    observed = dataset.observed(ids[:5])
    gp = GPPredictor(config)
    gp.condition(observed)
    return dataset, config, gp, dataset.query(ids[5:10]), profiles.iloc[:5], ids


def test_latent_posterior_does_not_remove_observation_noise_as_information(tmp_path):
    _, _, gp, query, _, _ = _example(tmp_path)
    _, latent, noise = _latent_posterior(gp, query)
    x = gp._coordinates(query)
    cross = gp.regressor.kernel_.k1(x, gp.regressor.X_train_)
    expected = gp.regressor.kernel_.k1(x) - cross @ cho_solve((gp.regressor.L_, True), cross.T)
    assert np.allclose(latent, expected, atol=1e-9)
    assert noise == pytest.approx(.2)
    indices = np.arange(24)
    _, _, after_with_noise, _, _ = _block_update(latent, indices, noise)
    _, _, after_without_noise, _, _ = _block_update(latent, indices, 0.)
    assert (after_with_noise >= after_without_noise - 1e-8).all()
    assert after_with_noise[indices].mean() > .01


@pytest.mark.parametrize("policy", ["gp_alc", "gp_sur"])
def test_whole_block_determinism_and_conditional_second_scores(tmp_path, policy):
    _, config, gp, query, observed, _ = _example(tmp_path)
    selected, scores = select_gp_batch(gp, observed, query, config, policy=policy, min_spacing_m=2.)
    again, same = select_gp_batch(gp, observed, query.sample(frac=1, random_state=3), config,
                                 policy=policy, min_spacing_m=2.)
    assert selected == again and len(set(selected)) == 2
    pd.testing.assert_frame_equal(scores, same)
    assert scores.profile_cells.eq(24).all()
    remaining = scores.loc[scores.second_stage_profile_gain.notna()]
    assert (remaining.second_stage_profile_gain <= remaining.profile_gain + 1e-9).all()
    chosen = scores.loc[scores.selection_order.eq(2)].iloc[0]
    assert chosen.selection_score == chosen.second_stage_score
    assert chosen.second_stage_integrated_variance_gain < chosen.integrated_variance_gain
    assert chosen.distance_to_first_m >= 2.
    if policy == "gp_sur":
        assert scores.initial_shortlisted.sum() == 3
        assert scores.fantasies_per_candidate.eq(4).all()
        assert scores.loc[~scores.initial_shortlisted, "risk_gain"].isna().all()
    with pytest.raises(ValueError, match="24 depth"):
        select_gp_batch(gp, observed, query.iloc[1:], config, policy=policy, min_spacing_m=2.)
    with pytest.raises(ValueError, match="hidden truth"):
        select_gp_batch(gp, observed, query.assign(qc_mpa=99), config, policy=policy, min_spacing_m=2.)


def test_threshold_comparator_and_spacing_fallback_match_v1(tmp_path):
    _, config, gp, query, observed, _ = _example(tmp_path)
    selected, scores = select_gp_batch(gp, observed, query, config, policy="gp_threshold", min_spacing_m=500.)
    baseline, _ = select_batch(observed, query, gp.predict(query), config,
                               policy="threshold_uncertainty", min_spacing_m=500.)
    assert selected == baseline
    assert scores.loc[scores.selection_order.eq(2), "spacing_fallback"].all()
    for policy in ("gp_alc", "gp_sur"):
        _, scores = select_gp_batch(gp, observed, query, config, policy=policy, min_spacing_m=500.)
        second = scores.loc[scores.selection_order.eq(2)].iloc[0]
        assert second.spacing_fallback
        assert second.distance_to_first_m == scores.distance_to_first_m.max()


def test_replay_hidden_truth_cannot_affect_selection_or_reveal_one_mid_batch(tmp_path):
    from cpt_budget.v2.replay import run_replay_v2
    dataset, config, _, _, _, ids = _example(tmp_path)
    split = {"site_id": "s", "seed": 17, "evaluation_ids": ids[-2:], "pool_ids": ids[:-2],
             "initial_ids": ids[:5], "budgets": [5, 7], "min_spacing_m": 2., "role": "validation"}

    class SpyGP(GPPredictor):
        def __init__(self):
            super().__init__(config)
            self.observation_counts = []

        def condition(self, observed):
            self.observation_counts.append(observed.location_id.nunique())
            super().condition(observed)

        def predict(self, query):
            assert not {"qc_mpa", "log_qc", "truth_qc_mpa"}.intersection(query.columns)
            return super().predict(query)

    predictor = SpyGP()
    first = run_replay_v2(dataset, split, predictor, config, policy="gp_sur")
    changed = dataset.grid.copy()
    changed.loc[~changed.location_id.isin(ids[:5]), "qc_mpa"] = .01
    changed.loc[~changed.location_id.isin(ids[:5]), "log_qc"] = np.log(.01)
    second_dataset = PreparedDataset(dataset.profiles, changed, dataset.sites, tmp_path)
    second = run_replay_v2(second_dataset, split, SpyGP(), config, policy="gp_sur")
    assert first["recommendations"].location_id.tolist() == second["recommendations"].location_id.tolist()
    assert predictor.observation_counts == [5, 7]
    assert first["predictions"].query("role == 'candidate'").truth_qc_mpa.isna().all()
    assert first["metrics"].study_version.eq("exploratory_v2").all()
    assert first["metrics"].selection_model_id.eq("gp").all()
    assert first["predictions"].investigation_group.eq("g").all()
    assert "risk_probability" not in first["predictions"]
