"""Consequential accounting checks, using synthetic decisions, not project results."""
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from cpt_budget.adaptive_economics import evaluate_economic_case, load_economics_config


def _config():
    path = Path(__file__).parents[1] / "configs" / "economics_case.yaml"
    return load_economics_config(path)


def _decisions(**changes):
    row = {"run_id": "synthetic_test", "fold_id": 0, "role": "test", "site_id": "synthetic",
        "seed": 17, "model_id": "gp", "policy_id": "P_adapt", "n_full": 24,
        "n_stop": 12, "n_initial": 4, "n_batches": 3, "audit_pass": True}
    row.update(changes)
    return pd.DataFrame([row])


def test_direct_ceil_calendar_wait_and_financial_social_separation():
    config = _config()
    result = evaluate_economic_case(_decisions(), config)
    row = result["economics"].iloc[0]
    # 24/5 -> 5 days; 12/5 -> 3 days, not 4.8 and 2.4.
    assert row.full_field_workdays == 5 and row.method_field_workdays == 3
    assert row.full_investigation_calendar_days == 7
    assert row.method_investigation_calendar_days == pytest.approx(4.7)
    assert row.direct_saving == 12 * 750  # Common 560 mobilization cancels.
    assert row.full_planned_metres == 24 * 20  # Not modelled 2–8 m interval.
    assert row.project_days_saved == pytest.approx(2.3)
    assert row.net_benefit == pytest.approx(9000 + 2.3 * (5000 + 3000) - 600)
    assert row.social_user_benefit == pytest.approx(2.3 * 20 * .5 * 200)
    assert not result["social_user"].included_in_owner_net_benefit.any()
    alpha = result["sensitivity"].query("sensitivity_kind == 'alpha'").set_index("alpha")
    assert alpha.loc[.5, "net_benefit"] == pytest.approx(9000 + 1.15 * 8000 - 600)


def test_false_and_unknown_early_stop_complete_plan_and_can_lose_money():
    for audit in (False, None, pd.NA, "false"):
        row = evaluate_economic_case(_decisions(audit_pass=audit), _config())["economics"].iloc[0]
        assert row.claimed_saved_cpt == 12 and row.saved_cpt == 0
        assert row.repaired_cpt == 12 and row.charged_cpt == 24
        assert row.repair_visits == 1 and row.direct_saving == -560
        # Separate repair requires another ceil, and a 2-day return delay.
        assert row.method_field_workdays == 6
        assert row.project_days_saved < 0 and row.net_benefit < 0
        assert not row.reliable_early_stop


def test_full_plan_has_zero_benefit_and_ai_not_double_counted():
    row = evaluate_economic_case(_decisions(policy_id="B_full", n_stop=24, n_batches=1),
        _config())["economics"].iloc[0]
    assert row.net_benefit == 0 and row.ai_incremental_cost == 0
    config = _config()
    config["parameters"]["review_workdays_per_batch"]["value"] = .1
    row = evaluate_economic_case(_decisions(), config)["economics"].iloc[0]
    assert row.additional_review_workdays == pytest.approx(.3)
    assert row.ai_incremental_cost == 600  # Not additionally four adviser hours.


def test_nonreplaceable_deep_task_cannot_claim_removed_whole_cpt():
    row = evaluate_economic_case(_decisions(scope_replaceable=False), _config())["economics"].iloc[0]
    assert row.saved_cpt == 0 and row.economic_status == "nonreplaceable_scope_completed"
    assert row.net_benefit < 0


def test_variable_full_depths_need_selected_depths_and_use_single_pricing_mode():
    config = _config()
    config["case"]["pricing_mode"] = "itemized_per_metre"
    config["parameters"].update(setup_cost_per_cpt={"value": 20},
        processing_cost_per_cpt={"value": 10}, penetration_cost_per_metre={"value": 5})
    decisions = _decisions(n_full=4, n_stop=2, n_initial=1, n_batches=1,
        full_depths_m="[10, 15, 20, 25]", measured_depths_m="[10, 20]")
    row = evaluate_economic_case(decisions, config)["economics"].iloc[0]
    assert row.direct_saving == 2 * (20 + 10) + 40 * 5
    assert row.full_planned_metres == 70 and row.charged_metres == 30
    with pytest.raises(ValueError, match="Variable full depths"):
        evaluate_economic_case(decisions.drop(columns="measured_depths_m"), config)
    with pytest.raises(ValueError, match="cannot exceed"):
        evaluate_economic_case(_decisions(n_stop=25), _config())


def test_batch_sizes_must_include_initial_and_provenance_keeps_assumptions():
    result = evaluate_economic_case(_decisions(batch_sizes="[4, 4, 4]"), _config())
    assert result["economics"].iloc[0].n_batches == 3
    assert result["parameters"].set_index("parameter").loc["cost_per_cpt", "source_type"] == "explicit_scenario_assumption"
    with pytest.raises(ValueError, match="sum to n_stop"):
        evaluate_economic_case(_decisions(batch_sizes="[4, 4]"), _config())
