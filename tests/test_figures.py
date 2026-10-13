"""Scientific scope is isolated before visual case selection or aggregation."""
import pandas as pd

from cpt_budget.figures import _select_result_scope


def test_test_results_exclude_validation_without_changing_prediction_roles():
    # Same site but different folds is a realistic source-validation/test overlap.
    metrics = pd.DataFrame({"site_id": ["newlock", "newlock", "newlock"],
                            "fold_id": [0, 1, 2], "role": ["validation", "test", "pilot"],
                            "log_mae": [99., .2, 88.]})
    predictions = pd.DataFrame({"site_id": ["newlock"] * 4,
                                "split_role": ["validation", "test", "test", "pilot"],
                                "role": ["evaluation", "evaluation", "candidate", "evaluation"],
                                "pred_qc_mpa": [99., .8, .9, 88.]})
    states = metrics[["site_id", "fold_id", "role"]].copy()
    scores = states.copy()
    scope, m, p, s, a = _select_result_scope(metrics, predictions, states, scores)
    assert scope == "test"
    assert m.log_mae.tolist() == [.2]
    assert p.pred_qc_mpa.tolist() == [.8, .9]
    assert p.role.tolist() == ["evaluation", "candidate"]
    assert s.fold_id.tolist() == a.fold_id.tolist() == [1]
    assert metrics.role.tolist() == ["validation", "test", "pilot"]


def test_scope_falls_back_to_pilot_then_explicit_validation():
    rows = pd.DataFrame({"role": ["validation", "pilot"]})
    predictions = pd.DataFrame({"split_role": ["validation", "pilot"], "role": ["evaluation", "candidate"]})
    scope, *_ = _select_result_scope(rows, predictions, rows, rows)
    assert scope == "pilot"
    scope, m, p, *_ = _select_result_scope(rows.iloc[:1], predictions.iloc[:1], rows.iloc[:1], rows.iloc[:1])
    assert scope == "validation"
    assert m.role.tolist() == ["validation"]
    assert p.role.tolist() == ["evaluation"]
