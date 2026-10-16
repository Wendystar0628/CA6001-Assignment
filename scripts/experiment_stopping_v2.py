"""Four-candidate development-selected stopping experiment on saved trajectories.

No model is rerun and no audit label is exposed to a sequential stopping
decision.  This is an exploratory re-evaluation after the historical evaluation
was inspected, not a newly blinded test or calibration of v2 remaining-grid
signals. Acquisition paths and all registered quality targets stay fixed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cpt_budget.adaptive_run import quality_flags


CANDIDATES = (
    {"candidate_id": "original", "h_max": None, "use_native_u": True, "changes": 0, "tie_order": 0},
    {"candidate_id": "h_one_depth_cell", "h_max": .25, "use_native_u": True, "changes": 1, "tie_order": 1},
    {"candidate_id": "without_native_u", "h_max": None, "use_native_u": False, "changes": 1, "tie_order": 2},
    {"candidate_id": "h_one_depth_cell_without_native_u", "h_max": .25, "use_native_u": False, "changes": 2, "tie_order": 3},
)
SIGNALS = ("delta_q90", "delta_h90", "support_log_mae", "uncertainty_q90")


def read_phase(root, role, variant):
    """Read only the requested role; development selection never opens eval."""
    pairs = []
    for folder in sorted((root / "runs/adaptive_v1" / role).glob("*")):
        if not (folder / "trajectories.csv").exists():
            continue
        trace = pd.read_csv(folder / "trajectories.csv")
        if not trace.role.eq(role).all():
            raise ValueError("A source trajectory crosses development/evaluation roles")
        if trace.policy_id.iloc[0] == "P_adapt" and trace.variant.iloc[0] != variant:
            continue
        pairs.append((trace, pd.read_csv(folder / "quality.csv")))
    if not pairs:
        raise ValueError(f"No saved {role} trajectories")
    return pairs


def decide_online(trace, rule, candidate):
    """Sequential early exit using online signal columns only, never outcomes."""
    trace = trace.sort_values("round").reset_index(drop=True)
    streak, records = 0, []
    for index, row in trace.iterrows():
        checks = {"minimum_observed": row.n_observed >= rule["min_observed"],
                  "coverage": bool(row.coverage_ok)}
        for signal in SIGNALS:
            if signal == "uncertainty_q90" and not candidate["use_native_u"]:
                continue
            threshold = (candidate["h_max"] if signal == "delta_h90" and candidate["h_max"] is not None
                         else rule[f"{signal}_max"])
            value = row[signal]
            checks[signal] = bool(np.isfinite(value) and value >= 0. and value <= threshold)
        streak = streak + 1 if all(checks.values()) else 0
        full = row.n_observed == row.n_full
        terminal = full or streak >= rule["consecutive_rounds"]
        records.append({"run_id": row.run_id, "candidate_id": candidate["candidate_id"],
                        "round": int(row["round"]), "n_observed": int(row.n_observed),
                        "pass_streak": streak, "would_stop": terminal,
                        "failed_conditions": json.dumps([name for name, value in checks.items() if not value]),
                        "audit_read_online": False, "future_round_read_online": False})
        if terminal:
            return {"index": index, "round": int(row["round"]), "n_stop": int(row.n_observed),
                    "stop_reason": "full_plan_complete" if full else "online_stability",
                    "trace_censored": False}, records
    # No fabricated continuation when an original evaluation already stopped.
    return {"index": None, "round": None, "n_stop": None,
            "stop_reason": "source_trace_censored", "trace_censored": True}, records


def score_decision(trace, audit, full, quality, decision, label, candidate_id):
    """Join the audit outcome only after a stopping decision is terminal."""
    first = trace.iloc[0]
    meta = {key: first[key] for key in ["run_id", "site_id", "seed", "fold_id", "role", "model_id", "policy_id", "variant", "n_full"]}
    common = {**meta, "method": label, "candidate_id": candidate_id,
              "n_stop": decision["n_stop"], "round": decision["round"],
              "stop_reason": decision["stop_reason"], "trace_censored": decision["trace_censored"],
              "audit_read_online": False, "future_round_read_online": False}
    if decision["trace_censored"]:
        return common | {"is_early_stop": None, "saved_cpt": None,
                         "verified_saved_cpt": None, "saved_fraction": None,
                         "verified_saved_fraction": None, "false_early_stop": None}
    scored = audit.loc[audit["round"].eq(decision["round"])]
    if len(scored) != 1:
        raise ValueError("A terminal decision must have exactly one saved audit outcome")
    row = scored.iloc[0]
    flags = quality_flags(row, full, quality)
    absolute_log = bool(np.isfinite(row.log_mae) and row.log_mae <= quality["absolute_log_mae_max"])
    absolute_h = bool(np.isfinite(row.thickness_mae_m) and row.thickness_mae_m <= quality["absolute_thickness_mae_max_m"])
    relative_log = bool(np.isfinite(row.log_mae) and row.log_mae <= full.log_mae + quality["relative_log_tolerance"])
    relative_h = bool(np.isfinite(row.thickness_mae_m) and row.thickness_mae_m <= full.thickness_mae_m + quality["relative_thickness_tolerance_m"])
    early = int(decision["n_stop"]) < int(first.n_full)
    saved = int(first.n_full - decision["n_stop"])
    return common | flags | {
        "is_early_stop": early, "saved_cpt": saved, "saved_fraction": saved / first.n_full,
        "verified_saved_cpt": saved if flags["audit_pass"] else 0,
        "verified_saved_fraction": saved / first.n_full if flags["audit_pass"] else 0.,
        "false_early_stop": bool(early and not flags["audit_pass"]),
        "absolute_log_pass": absolute_log, "absolute_thickness_pass": absolute_h,
        "relative_log_pass": relative_log, "relative_thickness_pass": relative_h,
        "low_qc_recall_pass": None if pd.isna(row.recall) else bool(row.recall >= quality["low_qc_recall_target"]),
        "low_qc_fpr_pass": None if pd.isna(row.fpr) else bool(row.fpr <= quality["low_qc_fpr_target"]),
        "log_mae": row.log_mae, "thickness_mae_m": row.thickness_mae_m,
        "recall": row.recall, "fpr": row.fpr,
        "full_log_mae": full.log_mae, "full_thickness_mae_m": full.thickness_mae_m,
    }


def phase_decisions(pairs, rule, quality, candidates, *, selected_id=None):
    full = {(trace.site_id.iloc[0], int(trace.seed.iloc[0])): audit.iloc[-1]
            for trace, audit in pairs if trace.policy_id.iloc[0] == "B_full"}
    scored, rounds = [], []
    for trace, audit in pairs:
        if trace.policy_id.iloc[0] == "B_full":
            continue
        actual_candidates = candidates if trace.policy_id.iloc[0] == "P_adapt" else [CANDIDATES[0]]
        for candidate in actual_candidates:
            decision, emitted = decide_online(trace, rule, candidate)
            if trace.policy_id.iloc[0] == "B_seq":
                label = "original_sequential"
            elif candidate["candidate_id"] == "original":
                label = "original_adaptive"
            elif candidate["candidate_id"] == selected_id:
                label = "development_selected_adaptive"
            else:
                label = candidate["candidate_id"]
            row = score_decision(trace, audit, full[(trace.site_id.iloc[0], int(trace.seed.iloc[0]))],
                                 quality, decision, label, candidate["candidate_id"])
            scored.append(row)
            for emitted_row in emitted:
                emitted_row.update({key: trace.iloc[0][key] for key in ["site_id", "seed", "role", "policy_id", "variant"]})
            rounds.extend(emitted)
    return pd.DataFrame(scored), pd.DataFrame(rounds)


def summarise(decisions, grouping):
    records = []
    for keys, group in decisions.groupby(grouping, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        observed = group.loc[~group.trace_censored]
        by_site = observed.groupby("site_id")[["verified_saved_fraction", "false_early_stop", "audit_pass"]].mean()
        records.append(dict(zip(grouping, keys)) | {
            "trajectories": len(group), "censored_trajectories": int(group.trace_censored.sum()),
            "quality_passing_decisions": int(observed.audit_pass.sum()),
            "early_stops": int(observed.is_early_stop.sum()),
            "false_early_stops": int(observed.false_early_stop.sum()),
            "saved_cpt": int(observed.saved_cpt.sum()),
            "verified_saved_cpt": int(observed.verified_saved_cpt.sum()),
            "site_equal_verified_saved_fraction": float(by_site.verified_saved_fraction.mean()),
            "site_equal_false_stop_rate": float(by_site.false_early_stop.mean()),
            "selection_score": float(by_site.verified_saved_fraction.mean() - by_site.false_early_stop.mean()),
        })
    return pd.DataFrame(records)


def run(root, output):
    frozen = json.loads((root / "results/adaptive_v1/frozen_config.json").read_text(encoding="utf-8"))
    rule, quality = frozen["stop_rule"], frozen["quality"]
    variant = frozen["acquisition_variant"]
    output.mkdir(parents=True, exist_ok=True)
    # Deliberate stage boundary: neither file discovery nor reading evaluation
    # occurs before all four candidates have been scored and a winner frozen.
    development = read_phase(root, "development", variant)
    dev_decisions, dev_rounds = phase_decisions(development, rule, quality, CANDIDATES)
    trial_decisions = dev_decisions.loc[dev_decisions.policy_id.eq("P_adapt")]
    trials = summarise(trial_decisions, ["candidate_id"])
    definitions = pd.DataFrame(CANDIDATES)[["candidate_id", "changes", "tie_order"]]
    trials = trials.merge(definitions, on="candidate_id", validate="one_to_one")
    # Scores equivalent at roundoff precision are ties. Prefer fewer changes;
    # the deterministic strictness order retains native U ahead of removing it.
    trials["ranking_score"] = trials.selection_score.round(12)
    trials = trials.sort_values(["ranking_score", "changes", "tie_order"],
                               ascending=[False, True, True], kind="stable").reset_index(drop=True)
    winner = str(trials.iloc[0].candidate_id)
    selected = next(candidate for candidate in CANDIDATES if candidate["candidate_id"] == winner)
    selection = {
        "selected_candidate_id": winner, "selected_candidate": selected,
        "candidate_count": len(CANDIDATES), "candidates": list(CANDIDATES),
        "selection_source": "development site-equal verified saving minus false stop rate only",
        "evaluation_accessed_before_selection": False,
        "tie_rule": "fewer changes, then original / retain native U with H=.25 / remove U only / both changes",
        "base_rule": rule, "quality_targets_unchanged": quality,
        "acquisition_variant_unchanged": variant, "primary_model_unchanged": frozen["primary_model"],
        "support_lgo_gate_retained": True,
        "audit_read_online": False, "future_round_read_online": False,
        "evaluation_scope": "finite exploratory re-evaluation after historical unblinding; no new blinded generalisation claim",
        "signal_scope": "uses original all-plan-grid signals only; does not calibrate v2 remaining-grid or dynamic-support signals",
        "censoring_policy": "monotone relaxations should stop no later than original; unavailable later rounds are never fabricated",
    }
    trials.to_csv(output / "stopping_v2_development_selection.csv", index=False)
    dev_decisions.to_csv(output / "stopping_v2_development_decisions.csv", index=False)
    dev_rounds.to_csv(output / "stopping_v2_development_online_trace.csv", index=False)
    selection_path = output / "stopping_v2_frozen_selection.json"
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")

    evaluation = read_phase(root, "evaluation", variant)
    eval_candidates = [CANDIDATES[0]] if winner == "original" else [CANDIDATES[0], selected]
    eval_decisions, eval_rounds = phase_decisions(evaluation, rule, quality, eval_candidates, selected_id=winner)
    # No-selected-change still gets an explicit selected-method comparison.
    if winner == "original":
        same = eval_decisions.loc[eval_decisions.method.eq("original_adaptive")].copy()
        same["method"] = "development_selected_adaptive"
        eval_decisions = pd.concat([eval_decisions, same], ignore_index=True)
    original_budget = eval_decisions.loc[eval_decisions.method.eq("original_adaptive")].set_index(["site_id", "seed"]).n_stop
    for _, row in eval_decisions.loc[eval_decisions.method.eq("development_selected_adaptive")].iterrows():
        if row.trace_censored or row.n_stop > original_budget.loc[(row.site_id, row.seed)]:
            raise ValueError("A monotone relaxation unexpectedly needs an unavailable later round")
    eval_summary = summarise(eval_decisions, ["method"])
    site_summary = summarise(eval_decisions, ["site_id", "method"])
    eval_decisions.to_csv(output / "stopping_v2_evaluation_decisions.csv", index=False)
    eval_rounds.to_csv(output / "stopping_v2_evaluation_online_trace.csv", index=False)
    eval_summary.to_csv(output / "stopping_v2_evaluation_summary.csv", index=False)
    site_summary.to_csv(output / "stopping_v2_site_summary.csv", index=False)
    print(trials[["candidate_id", "selection_score", "verified_saved_cpt", "false_early_stops"]].to_string(index=False))
    print(f"Development-selected candidate: {winner}")
    print(eval_summary.to_string(index=False))
    return trials, eval_decisions, eval_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("results/adaptive_v2_diagnostics"))
    args = parser.parse_args()
    run(args.root, args.output)
