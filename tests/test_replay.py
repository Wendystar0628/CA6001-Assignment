import numpy as np
import pandas as pd
import pytest

from cpt_budget.acquisition import select_batch
from cpt_budget.economics import freeze_quality_targets, selection_diagnostics, summarize_results
from cpt_budget.evaluate import evaluate
from cpt_budget.replay import run_replay
from cpt_budget.types import PredictionBundle, PreparedDataset


def _config():
    return {"data": {"tau_mpa": 1.0, "depth_step_m": 0.25},
            "replay": {"budgets": [5, 7, 9, 11, 13], "seeds": [17],
                       "initial_locations": 5, "batch_size": 2},
            "quality": {"reference_model": "gp", "reference_policy": "maximin", "reference_budget": 13},
            "economics": {}}


def _metric_curve(model, errors, **extra):
    return pd.DataFrame([{ "fold_id": 0, "site_id": "site", "seed": 17,
        "model_id": model, "policy_id": "maximin", "budget_n": n,
        "log_mae": error, "h_point_mae_m": error, "recall": 1.0,
        "n_true_positive": 1, "role": "test", **extra}
        for n, error in zip([5, 7, 9, 11, 13], errors)])


def _targets():
    return {"folds": {"0": {"E_star": 1.0, "H_star": 1.0, "R_star": .8, "complete": True}}}


def test_nonmonotonic_attainment_and_censored_costs_are_preserved():
    metrics = pd.concat([_metric_curve("gp", [2, .5, 2, .5, .5]),
                         _metric_curve("tabpfn", [.5, .5, .5, .5, 2]),
                         _metric_curve("idw", [2, 2, .5, .5, .5])])
    result = summarize_results(metrics, _targets(), _config())
    attainment = result["attainment"].set_index("model_id")
    assert attainment.loc["gp", "n_first_hit"] == 7
    assert attainment.loc["gp", "n_stable"] == 11
    assert attainment.loc["tabpfn", "n_first_hit"] == 5
    assert pd.isna(attainment.loc["tabpfn", "n_stable"])
    assert attainment.loc["tabpfn", "status"] == "not_reached_by_budget"
    costs = result["costs"].set_index("model_id")
    assert costs.loc["idw", "saved_cpt"] == 2
    assert pd.isna(costs.loc["tabpfn", "saved_cpt"])
    assert costs.loc["tabpfn", "savings_upper_bound_exclusive"] == -2
    assert result["costs"].net_savings.isna().all()
    # Integral of [2,.5,2,.5,.5], not a running best-case envelope.
    assert attainment.loc["gp", "aubc_log_mae"] == pytest.approx(1.0625)


def test_all_negative_labels_keep_continuous_errors_and_explicit_missing_classification():
    truth = pd.DataFrame({"location_id": ["a", "a"], "depth_index": [0, 1], "qc_mpa": [2., 3.]})
    frame = truth[["location_id", "depth_index"]].assign(pred_qc_mpa=[2., 2.5], pred_log_qc=np.log([2., 2.5]))
    result = evaluate(PredictionBundle(frame, "idw"), truth, _config())
    assert result["recall"] is None and result["average_precision"] is None
    assert result["precision"] is None and result["f1"] is None
    assert result["recall_reason"] == "no_positive_labels"
    assert result["false_positive_rate"] == 0
    assert result["log_mae"] > 0 and result["h_point_mae_m"] == 0
    frame["pred_qc_mpa"], frame["pred_log_qc"] = .5, np.log(.5)
    with_positive_truth = truth.copy()
    with_positive_truth["qc_mpa"] = .5
    missed = frame.assign(pred_qc_mpa=2., pred_log_qc=np.log(2.))
    missed_result = evaluate(PredictionBundle(missed, "idw"), with_positive_truth, _config())
    assert missed_result["precision"] is None and missed_result["f1"] == 0


def test_freeze_quality_uses_validation_site_then_seed_means_and_rejects_test_only_schema():
    val = pd.concat([_metric_curve("gp", [3, 3, 3, 3, 2], role="validation"),
                     _metric_curve("gp", [5, 5, 5, 5, 4], role="validation", site_id="other")])
    test = _metric_curve("gp", [.001] * 5)
    result = freeze_quality_targets(pd.concat([val, test]), _config())["folds"]["0"]
    assert result["E_star"] == 3 and result["H_star"] == 3
    assert result["R_star"] == pytest.approx(.95)
    assert set(result["validation_sites"]) == {"site", "other"}
    with pytest.raises(ValueError, match="explicit role"):
        freeze_quality_targets(test.drop(columns="role"), _config())
    incomplete = freeze_quality_targets(test, _config())["folds"]["0"]
    assert incomplete["status"] == "quality_target_incomplete"


def test_spacing_fallback_is_farthest_then_score_and_maximin_updates_geometry():
    observed = pd.DataFrame({"location_id": ["o"], "x_m": [0.], "y_m": [0.]})
    candidates = pd.DataFrame({"location_id": ["a", "b", "c"], "x_m": [10., 9., 2.], "y_m": [0., 0., 0.]})
    ids, scores = select_batch(observed, candidates, None, _config(), policy="maximin", min_spacing_m=20)
    assert ids == ["a", "c"]
    assert scores.set_index("location_id").loc["c", "spacing_fallback"]
    assert scores.set_index("location_id").loc["c", "selection_score"] == 2
    with pytest.raises(ValueError, match="distribution"):
        select_batch(observed, candidates, None, _config(), policy="uncertainty", min_spacing_m=0)


def test_hidden_values_cannot_change_selection_or_be_revealed_mid_batch(tmp_path):
    ids = [f"p{i}" for i in range(10)]
    profiles = pd.DataFrame({"site_id": "s", "profile_id": ids, "location_id": ids,
                             "near_group_id": ids, "x_m": np.arange(10.), "y_m": 0.,
                             "geometry_scale_m": 1., "usable": True, "is_representative": True})
    grid = pd.DataFrame([{ "site_id": "s", "location_id": loc, "depth_index": d,
                          "z_center_m": 2.125 + .25 * d, "qc_mpa": 2., "log_qc": np.log(2.)}
                         for loc in ids for d in range(2)])
    sites = pd.DataFrame({"site_id": ["s"]})
    split = {"site_id": "s", "seed": 17, "evaluation_ids": ids[-2:], "pool_ids": ids[:-2],
             "initial_ids": ids[:5], "budgets": [5, 7], "min_spacing_m": 1.}

    class SpyPredictor:
        def __init__(self):
            self.context_sizes = []

        def condition(self, observed):
            self.context_sizes.append(observed.location_id.nunique())
            self.mean = observed.log_qc.mean()

        def predict(self, query):
            assert not {"qc_mpa", "log_qc", "truth_qc_mpa"}.intersection(query.columns)
            return PredictionBundle(query.assign(pred_log_qc=self.mean, pred_qc_mpa=np.exp(self.mean),
                q05_log=self.mean - query.x_m / 10, q95_log=self.mean + query.x_m / 10), "gp")

    predictor = SpyPredictor()
    first = run_replay(PreparedDataset(profiles, grid, sites, tmp_path), split, predictor, _config(), policy="uncertainty")
    changed = grid.copy()
    changed.loc[~changed.location_id.isin(ids[:5]), "qc_mpa"] = .1
    changed.loc[~changed.location_id.isin(ids[:5]), "log_qc"] = np.log(.1)
    second = run_replay(PreparedDataset(profiles, changed, sites, tmp_path), split, SpyPredictor(), _config(), policy="uncertainty")
    assert first["recommendations"].location_id.tolist() == second["recommendations"].location_id.tolist()
    assert predictor.context_sizes == [5, 7]
    assert first["predictions"].query("role == 'candidate'").truth_qc_mpa.isna().all()
    assert first["states"].evaluation_ids.tolist() == [ids[-2:], ids[-2:]]


def test_macro_average_does_not_weight_sites_by_seed_count():
    one = _metric_curve("gp", [1] * 5)
    many = pd.concat([_metric_curve("gp", [3] * 5, site_id="large", seed=seed) for seed in [17, 29, 43]])
    summary = summarize_results(pd.concat([one, many]), _targets(), _config())
    macro = summary["aggregate"].query("level == 'macro' and metric == 'log_mae'")
    assert (macro.value == 2).all()
    assert (macro.n_sites == 2).all()


def test_pilot_curves_are_descriptive_and_cannot_claim_frozen_quality():
    summary = summarize_results(_metric_curve("gp", [.5] * 5, role="pilot"), _targets(), _config())
    assert summary["attainment"].iloc[0].status == "quality_target_incomplete"
    assert summary["attainment"].iloc[0].summary_role == "pilot"
    assert summary["attainment"].iloc[0].aubc_log_mae == .5


@pytest.mark.parametrize("observed_budgets", [[5, 7], [5, 7, 13]])
@pytest.mark.parametrize("incomplete_model", ["gp", "idw"])
def test_incomplete_budget_curves_never_claim_stable_attainment_or_costs(observed_budgets, incomplete_model):
    baseline = _metric_curve("gp", [.5] * 5)
    method = _metric_curve("idw", [.5] * 5)
    curves = pd.concat([baseline, method])
    curves = curves.loc[(curves.model_id != incomplete_model) | curves.budget_n.isin(observed_budgets)]
    config = _config()
    config["economics"] = {"currency": "GBP", "marginal_cost_per_cpt": 100,
                            "extra_analysis_cost": 0, "extra_mobilization_cost": 0}
    result = summarize_results(curves, _targets(), config)
    partial = result["attainment"].set_index("model_id").loc[incomplete_model]
    assert partial.status == "incomplete_budget_curve"
    assert pd.isna(partial.n_stable)
    assert partial.n_first_hit == 5
    assert partial.n_first_hit_scope == "observed_budget_points"
    cost = result["costs"].iloc[0]
    assert cost.pair_status == "incomplete_budget_curve"
    assert pd.isna(cost.saved_cpt) and pd.isna(cost.net_savings)
    assert pd.isna(cost.savings_lower_bound_exclusive) and pd.isna(cost.savings_upper_bound_exclusive)
    assert cost.n_both_reached == 0


def test_selection_diagnostics_only_compare_same_first_round_information():
    common = {"fold_id": 0, "site_id": "s", "seed": 17, "model_id": "gp",
              "budget_n": 5, "observed_set_id": "same", "round_id": 0, "role": "pilot"}
    scores = pd.DataFrame([{**common, "policy_id": policy, "location_id": loc, "score": score,
                            "selected": loc in selected}
        for policy, values, selected in [("maximin", [1., 2., 3.], {"a", "b"}),
                                         ("uncertainty", [3., 2., 1.], {"b", "c"}),
                                         ("threshold_uncertainty", [2., 2., 2.], {"b", "c"})]
        for loc, score in zip(["a", "b", "c"], values)])
    predictions = pd.DataFrame([{**common, "policy_id": policy, "location_id": loc,
        "depth_index": depth, "role": "candidate", "split_role": "pilot",
        "q05_log": -1. if depth == 0 else .5, "q95_log": 1.}
        for policy in scores.policy_id.unique() for loc in ["a", "b", "c"] for depth in [0, 1]])
    result = selection_diagnostics(scores, predictions)
    pair = result.query("policy_a == 'maximin' and policy_b == 'uncertainty'").iloc[0]
    assert pair.jaccard == pytest.approx(1 / 3)
    assert pair.score_spearman == pytest.approx(-1.)
    assert pair.crossing_fraction == .5
    constant = result.query("policy_a == 'threshold_uncertainty' or policy_b == 'threshold_uncertainty'")
    assert constant.score_spearman.isna().all()
    assert (constant.score_spearman_reason == "constant_candidate_scores").all()
    scores.loc[scores.policy_id == "threshold_uncertainty", "observed_set_id"] = "different"
    later = scores.assign(round_id=1, budget_n=7)
    filtered = selection_diagnostics(pd.concat([scores, later]), predictions)
    assert len(filtered) == 1 and filtered.iloc[0].budget_n == 5
    assert filtered.iloc[0].comparison_scope == "same_information_state"
