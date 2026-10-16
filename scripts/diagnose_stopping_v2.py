"""Saved-trajectory stopping diagnosis; audit-oracle results are not deployable."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from cpt_budget.adaptive_run import quality_flags


SIGNALS = ("delta_q90", "delta_h90", "support_log_mae", "uncertainty_q90")


def diagnose(root, output):
    frozen = json.loads((root / "results/adaptive_v1/frozen_config.json").read_text(encoding="utf-8"))
    rule, quality = frozen["stop_rule"], frozen["quality"]
    pairs = []
    for phase in ("development", "evaluation"):
        for folder in sorted((root / "runs/adaptive_v1" / phase).glob("*")):
            if (folder / "trajectories.csv").exists() and (folder / "quality.csv").exists():
                pairs.append((pd.read_csv(folder / "trajectories.csv"), pd.read_csv(folder / "quality.csv")))
    full = {(trace.site_id.iloc[0], int(trace.seed.iloc[0])): audit.iloc[-1]
            for trace, audit in pairs if trace.policy_id.iloc[0] == "B_full"}
    details, decisions, oracles = [], [], []
    ablations = {"original": {}, "drop_delta_h90": {"omit": "delta_h90"},
                 "delta_h90_one_depth_cell": {"h_limit": .25},
                 **{f"drop_{s}": {"omit": s} for s in SIGNALS if s != "delta_h90"},
                 "drop_coverage": {"omit": "coverage"}}
    for trace, audit in pairs:
        if trace.policy_id.iloc[0] == "B_full":
            continue
        trace = trace.sort_values("round").reset_index(drop=True)
        scored = audit.drop(columns=[c for c in audit if c in trace and c not in ["run_id", "policy_id", "round"]])
        merged = trace.merge(scored, on=["run_id", "policy_id", "round"], validate="one_to_one")
        flags = pd.DataFrame([quality_flags(row, full[(row.site_id, int(row.seed))], quality)
                              for _, row in merged.iterrows()])
        merged = pd.concat([merged, flags], axis=1)
        meta = {key: merged.iloc[0][key] for key in ["run_id", "site_id", "seed", "role", "policy_id", "variant", "n_full"]}
        merged["minimum_pass"] = merged.n_observed >= rule["min_observed"]
        merged["coverage_pass"] = merged.coverage_ok.eq(True)
        for signal in SIGNALS:
            merged[f"{signal}_pass"] = np.isfinite(merged[signal]) & merged[signal].ge(0.) & merged[signal].le(rule[f"{signal}_max"])
        checks = ["minimum_pass", "coverage_pass", *[f"{s}_pass" for s in SIGNALS]]
        merged["blocked_conditions"] = merged.apply(lambda row: json.dumps([c.removesuffix("_pass") for c in checks if not row[c]]), axis=1)
        merged["sole_blocker"] = merged.apply(lambda row: next((c.removesuffix("_pass") for c in checks if not row[c]), "") if sum(not row[c] for c in checks) == 1 else "", axis=1)
        merged["known_fraction"] = merged.n_observed / merged.n_full
        # This flag identifies the geometric dilution regime, not a claim that
        # every model's observed-point uncertainty is exactly zero.
        merged["remaining_fraction_below_q90_tail"] = merged.known_fraction.ge(.9)
        merged["support_fraction_of_observed_locations"] = merged.n_support_fixed_groups / merged.n_observed
        for name, options in ablations.items():
            streak, chosen, reason = 0, len(merged) - 1, "available_trace_exhausted"
            for idx, row in merged.iterrows():
                active = {c.removesuffix("_pass"): bool(row[c]) for c in checks}
                if options.get("omit"):
                    active[options["omit"]] = True
                if "h_limit" in options:
                    active["delta_h90"] = bool(np.isfinite(row.delta_h90) and row.delta_h90 <= options["h_limit"])
                streak = streak + 1 if all(active.values()) else 0
                if row.n_observed == row.n_full:
                    chosen, reason = idx, "full_plan_complete"
                    break
                if streak >= rule["consecutive_rounds"]:
                    chosen, reason = idx, "online_stability"
                    break
            row = merged.iloc[chosen]
            early = reason == "online_stability" and row.n_observed < row.n_full
            saved = int(row.n_full - row.n_observed) if early else 0
            decisions.append({**meta, "ablation": name, "n_stop": int(row.n_observed),
                              "stop_reason": reason, "saved_cpt": saved,
                              "audit_pass": bool(row.audit_pass), "false_early_stop": bool(early and not row.audit_pass),
                              "verified_saved_fraction": float(saved / row.n_full) if row.audit_pass else 0.,
                              "low_qc_audit_pass": row.low_qc_audit_pass,
                              "log_mae": row.log_mae, "thickness_mae_m": row.thickness_mae_m})
        eligible = merged[merged.minimum_pass & merged.audit_pass & merged.n_observed.lt(merged.n_full)]
        oracle = eligible.iloc[0] if len(eligible) else None
        early_original = [d for d in decisions if d["run_id"] == meta["run_id"] and d["ablation"] == "original"][0]
        oracles.append({**meta, "oracle_not_deployable": True,
                        "observed_trace_complete": bool(merged.n_observed.iloc[-1] == merged.n_full.iloc[-1]),
                        "oracle_earliest_passing_budget": int(oracle.n_observed) if oracle is not None else None,
                        "oracle_available_saved_cpt": int(meta["n_full"] - oracle.n_observed) if oracle is not None else 0,
                        "any_saved_state_passes": oracle is not None,
                        "original_stop_budget": early_original["n_stop"],
                        "missed_quality_passing_early_state": bool(oracle is not None and early_original["n_stop"] > oracle.n_observed),
                        "best_log_mae": float(merged.log_mae.min()),
                        "best_thickness_mae_m": float(merged.thickness_mae_m.min()),
                        "full_log_mae": full[(meta["site_id"], int(meta["seed"]))]["log_mae"],
                        "full_thickness_mae_m": full[(meta["site_id"], int(meta["seed"]))]["thickness_mae_m"]})
        # Exclude machine timing columns from diagnostic results.
        details.append(merged.drop(columns=[c for c in merged if "runtime" in c]))
    output.mkdir(parents=True, exist_ok=True)
    detail, decision, oracle = pd.concat(details, ignore_index=True), pd.DataFrame(decisions), pd.DataFrame(oracles)
    detail.to_csv(output / "stopping_round_diagnostics.csv", index=False)
    decision.to_csv(output / "stopping_ablation_decisions.csv", index=False)
    oracle.to_csv(output / "stopping_oracle_diagnostic.csv", index=False)
    summary = []
    for keys, group in decision.groupby(["role", "policy_id", "variant", "ablation"], sort=True):
        site = group.groupby("site_id")[["verified_saved_fraction", "false_early_stop"]].mean()
        summary.append(dict(zip(["role", "policy_id", "variant", "ablation"], keys)) | {
            "trajectories": len(group), "false_early_stops": int(group.false_early_stop.sum()),
            "site_equal_verified_saved_fraction": float(site.verified_saved_fraction.mean()),
            "site_equal_false_stop_rate": float(site.false_early_stop.mean())})
    pd.DataFrame(summary).to_csv(output / "stopping_ablation_summary.csv", index=False)
    blocked = []
    for keys, group in detail[detail.minimum_pass & detail.n_observed.lt(detail.n_full)].groupby(["role", "site_id", "policy_id", "variant"]):
        for signal in ["coverage", *SIGNALS]:
            blocked.append(dict(zip(["role", "site_id", "policy_id", "variant"], keys)) | {
                "condition": signal, "eligible_rounds": len(group),
                "blocked_rounds": int((~group[f"{signal}_pass"]).sum()),
                "sole_blocker_rounds": int(group.sole_blocker.eq(signal).sum()),
                "quality_passing_sole_blocker_rounds": int((group.sole_blocker.eq(signal) & group.audit_pass).sum())})
    pd.DataFrame(blocked).to_csv(output / "stopping_blocker_counts.csv", index=False)
    info = {"quality_targets_unchanged": quality, "source_rule": rule,
            "oracle_not_deployable": True,
            "evaluation_scope": "post-unblinding exploratory diagnosis; audit used after decisions only",
            "ablation_scope": "All single-condition removals and H=.25 are post-hoc diagnostics, not development-frozen deployable policies",
            "h_one_depth_cell": ".25 m is one registered depth cell; this changes an online stability signal only and never changes audit thickness error targets",
            "corrected_signal_scope": "New remaining-grid and dynamic-support signals require new development calibration; original fixed-grid thresholds cannot be reused as calibrated v2 thresholds",
            "trace_count": len(pairs), "nonfull_trajectory_count": len(oracle),
            "development_variants": ["UDS", "US", "maximin"],
            "censoring": "Early-stopped evaluation traces do not contain later counterfactual rounds",
            "remaining_grid_diagnostic": "Known-plan dilution identified from geometry/counts; original full-grid predictions were not saved, so corrected q90 cannot be reconstructed from CSV alone"}
    (output / "stopping_diagnostic_meta.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return detail, decision, oracle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("results/adaptive_v2_diagnostics"))
    args = parser.parse_args()
    _, decisions, oracle = diagnose(args.root, args.output)
    print(oracle.query("role == 'evaluation'")[["site_id", "seed", "policy_id", "oracle_earliest_passing_budget", "original_stop_budget", "best_log_mae"]].to_string(index=False))
    print(decisions.query("role == 'evaluation' and site_id == 'newlock'").groupby("ablation").agg(false_stops=("false_early_stop", "sum"), saved=("saved_cpt", "sum")).to_string())
