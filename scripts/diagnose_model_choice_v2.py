"""Post-unblind model-choice diagnostics; never changes the frozen v1 run.

Only explicitly revealed profiles condition predictors. Audit values are read
after each prediction and are used solely for offline scoring. No timing table
or report is produced. Run from the repository root with the project Python.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cpt_budget.adaptive_models import AdaptiveModelSuite, MAIN_MODEL_IDS, ABLATION_MODEL_IDS
from cpt_budget.adaptive_run import _score, quality_flags, _e1_states, read_config
from cpt_budget.config import load_config
from cpt_budget.models.distribution import point_prediction_frame
from cpt_budget.runtime import make_model
from cpt_budget.types import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/adaptive_v2_diagnostics"
V1 = ROOT / "results/adaptive_v1"
SCOPE = "post-unblind exploratory diagnostics; original frozen splits and thresholds"


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def add_flags(frame, quality):
    frame = frame.copy()
    for name, values in {
        "absolute_quality_pass": (frame.log_mae <= quality["absolute_log_mae_max"]) & (frame.thickness_mae_m <= quality["absolute_thickness_mae_max_m"]),
        "low_qc_audit_pass": (frame.recall >= quality["low_qc_recall_target"]) & (frame.fpr <= quality["low_qc_fpr_target"]),
    }.items():
        frame[name] = values
    # No-positive sites cannot demonstrate recall; retain undefined rather than
    # silently count them as a failed (or successful) recognition measurement.
    frame["low_qc_audit_pass"] = frame.low_qc_audit_pass.astype(float).where(frame.recall.notna())
    frame["combined_absolute_pass"] = (frame.absolute_quality_pass & frame.low_qc_audit_pass.eq(1)).astype(float).where(frame.recall.notna())
    frame["profile_normalized"] = frame.log_mae / quality["absolute_log_mae_max"]
    frame["thickness_normalized"] = frame.thickness_mae_m / quality["absolute_thickness_mae_max_m"]
    frame["recognition_recall_shortfall"] = np.maximum(0, (quality["low_qc_recall_target"] - frame.recall) / quality["low_qc_recall_target"])
    frame["recognition_fpr_excess"] = np.maximum(0, (frame.fpr - quality["low_qc_fpr_target"]) / quality["low_qc_fpr_target"])
    # Diagnostic ranking only: fixed target-normalized joint violation, no new gate.
    frame["joint_violation"] = (np.maximum(0, frame.profile_normalized - 1) + np.maximum(0, frame.thickness_normalized - 1)
                                + frame.recognition_recall_shortfall + frame.recognition_fpr_excess)
    return frame


def frozen_summary():
    quality = json.loads((V1 / "frozen_config.json").read_text(encoding="utf-8"))["quality"]
    frames = [pd.read_csv(V1 / name) for name in ("model_comparison.csv", "ablation.csv")]
    all_rows = add_flags(pd.concat(frames, ignore_index=True), quality)
    # Match v1's main-model selection: each site is weighted equally, then each
    # seed/state equally within a site. Full endpoint was included in v1 E1.
    metrics = ["log_mae", "thickness_mae_m", "recall", "fpr", "absolute_quality_pass", "low_qc_audit_pass", "combined_absolute_pass", "joint_violation"]
    seed_states = all_rows.groupby(["role", "site_id", "seed", "model_id"])[metrics].mean().reset_index()
    site = seed_states.groupby(["role", "site_id", "model_id"])[metrics].mean().reset_index()
    summary = site.groupby(["role", "model_id"])[metrics].mean().reset_index()
    summary["rank_log_only"] = summary.groupby("role").log_mae.rank(method="min")
    summary["rank_joint_violation"] = summary.groupby("role").joint_violation.rank(method="min")
    summary.to_csv(OUTPUT / "model_choice_frozen_rankings.csv", index=False)
    all_rows.loc[all_rows.site_id.eq("newlock")].to_csv(OUTPUT / "model_choice_newlock_frozen.csv", index=False)
    full = all_rows.loc[all_rows.full_endpoint.astype(str).str.lower().eq("true")]
    full.loc[full.site_id.eq("newlock")].to_csv(OUTPUT / "model_choice_newlock_frozen_full.csv", index=False)
    print(summary.sort_values(["role", "rank_log_only"])[["role", "model_id", "log_mae", "thickness_mae_m", "recall", "fpr", "absolute_quality_pass", "combined_absolute_pass"]].to_string(index=False), flush=True)
    return quality


def point_prior(model_id, observed, query):
    """No source or target-future labels: only the current revealed context."""
    if model_id == "revealed_global_median":
        values = np.full(len(query), float(observed.log_qc.median()))
    else:
        operator = "mean" if model_id == "revealed_depth_mean" else "median"
        profile = observed.groupby("depth_index").log_qc.agg(operator)
        values = query.depth_index.map(profile).to_numpy(float)
    return point_prediction_frame(query, values)


def diagnostic_records(prediction, truth, query, observed, meta):
    matched = prediction[["location_id", "depth_index", "pred_log_qc"]].merge(
        truth[["location_id", "depth_index", "log_qc"]], on=["location_id", "depth_index"], validate="one_to_one")
    matched["absolute_log_error"] = abs(matched.pred_log_qc - matched.log_qc)
    matched["true_low"] = matched.log_qc < 0
    matched["pred_low"] = matched.pred_log_qc < 0
    matched["tp"] = matched.true_low & matched.pred_low
    matched["fp"] = ~matched.true_low & matched.pred_low
    matched["fn"] = matched.true_low & ~matched.pred_low
    points = query.drop_duplicates("location_id").set_index("location_id")
    obs_xy = observed.drop_duplicates("location_id")[["x_m", "y_m"]].to_numpy(float)
    holes = []
    for location, group in matched.groupby("location_id", sort=False):
        xy = points.loc[location, ["x_m", "y_m"]].to_numpy(float)
        holes.append({**meta, "location_id": location, "log_mae": group.absolute_log_error.mean(),
                      "true_thickness_m": group.true_low.sum() * .25, "pred_thickness_m": group.pred_low.sum() * .25,
                      "thickness_abs_error_m": abs(group.true_low.sum() - group.pred_low.sum()) * .25,
                      "nearest_revealed_distance_m": np.linalg.norm(obs_xy - xy, axis=1).min(),
                      "tp": int(group.tp.sum()), "fp": int(group.fp.sum()), "fn": int(group.fn.sum())})
    depths = []
    for depth, group in matched.groupby("depth_index", sort=True):
        depths.append({**meta, "depth_index": depth, "z_center_m": float(query.loc[query.depth_index.eq(depth), "z_center_m"].iloc[0]),
                       "log_mae": group.absolute_log_error.mean(), "true_low_fraction": group.true_low.mean(),
                       "pred_low_fraction": group.pred_low.mean(), "tp": int(group.tp.sum()),
                       "fp": int(group.fp.sum()), "fn": int(group.fn.sum())})
    return holes, depths


def endpoints(quality, device="cuda", families="all"):
    dataset = load_dataset(load_config())
    splits = json.loads((V1 / "splits.json").read_text(encoding="utf-8"))
    if isinstance(splits, dict):
        splits = splits["splits"]
    splits = [s for s in splits if s["site_id"] == "newlock"]
    assert len(splits) == 3 and all(s["fold_id"] == 1 for s in splits)
    # TabPFN preprocessing can depend on support row order. Reproduce the E1
    # revelation order, not just its final set, for the numerical cross-check.
    original_order = {}
    for split in splits:
        original_order[split["split_id"]] = list(_e1_states(split, dataset, read_config()))[-1][0]
        assert set(original_order[split["split_id"]]) == set(split["full_ids"])
    rows, holes, depths = [], [], []

    def score(model_id, prediction, split, state, known, observed, query):
        # Deliberately read audit truth only after the forward pass.
        truth = dataset.truth(split["audit_ids"])
        meta = {"split_id": split["split_id"], "site_id": "newlock", "seed": split["seed"],
                "fold_id": 1, "model_id": model_id, "state": state, "budget_n": len(known), "n_full": split["n_full"]}
        rows.append({**meta, **_score(prediction, truth)})
        hh, dd = diagnostic_records(prediction, truth, query, observed, meta)
        holes.extend(hh)
        depths.extend(dd)

    def flush():
        pd.DataFrame(rows).to_csv(OUTPUT / f"model_choice_endpoints_{families}.csv", index=False)
        pd.DataFrame(holes).to_csv(OUTPUT / f"model_choice_holes_{families}.csv", index=False)
        pd.DataFrame(depths).to_csv(OUTPUT / f"model_choice_depths_{families}.csv", index=False)

    if families in ("all", "cpu"):
        for split in splits:
            query = dataset.query(split["audit_ids"])
            for state, known in [("initial", split["initial_ids"]), ("full", original_order[split["split_id"]])]:
                assert not set(known) & set(split["audit_ids"])
                observed = dataset.observed(known)
                for model_id in ("balanced", "thickness", "recognition", "simplex", "idw"):
                    point_only = model_id in ("balanced", "thickness", "recognition")
                    model = make_model(model_id, 1, point_only=point_only)
                    model.initialize()
                    model.condition(observed)
                    score(model_id, model.predict(query).frame, split, state, known, observed, query)
                    del model
                for model_id in ("revealed_depth_mean", "revealed_depth_median", "revealed_global_median"):
                    score(model_id, point_prior(model_id, observed, query), split, state, known, observed, query)
                print(f"NewLock revealed priors/old point paths seed={split['seed']} state={state}", flush=True)
                flush()

    if families in ("all", "ai"):
        # The two weight generations run in separate suites; never keep both GPU
        # residual predictors initialized at the same time.
        groups = [tuple(m for m in MAIN_MODEL_IDS + ABLATION_MODEL_IDS if "unfinetuned" not in m),
                  tuple(m for m in ABLATION_MODEL_IDS if "unfinetuned" in m)]
        for model_ids in groups:
            suite = AdaptiveModelSuite(1, model_ids, device=device, role="evaluation").initialize()
            for split in splits:
                query = dataset.query(split["audit_ids"])
                for state, known in [("initial", split["initial_ids"]), ("full", original_order[split["split_id"]])]:
                    assert not set(known) & set(split["audit_ids"])
                    observed = dataset.observed(known)
                    suite.condition(observed)
                    predictions = suite.predict(query)
                    for model_id, bundle in predictions.items():
                        score(model_id, bundle.frame, split, state, known, observed, query)
                    print(f"NewLock same revealed state seed={split['seed']} state={state} models={','.join(model_ids)}", flush=True)
                    flush()
            del suite, predictions
            gc.collect()
            if device == "cuda":
                import torch
                torch.cuda.empty_cache()

    frame = add_flags(pd.DataFrame(rows), quality)
    flags = []
    for _, row in frame.iterrows():
        full = frame.loc[frame.seed.eq(row.seed) & frame.model_id.eq(row.model_id) & frame.state.eq("full")].iloc[0]
        flags.append(quality_flags(row, full, quality))
    frame = frame.drop(columns=[key for key in flags[0] if key in frame])
    frame = pd.concat([frame.reset_index(drop=True), pd.DataFrame(flags)], axis=1)
    frame["complete_task_pass"] = frame.audit_pass & frame.low_qc_audit_pass.fillna(False)
    frame["saved_fraction"] = 1 - frame.budget_n / frame.n_full
    frame.to_csv(OUTPUT / f"model_choice_endpoints_{families}.csv", index=False)
    summary = frame.groupby(["state", "model_id"])[["log_mae", "thickness_mae_m", "recall", "fpr", "audit_pass", "low_qc_audit_pass", "complete_task_pass"]].mean().reset_index()
    summary.to_csv(OUTPUT / f"model_choice_endpoint_summary_{families}.csv", index=False)
    print(summary.sort_values(["state", "log_mae"]).to_string(index=False), flush=True)
    json_write(OUTPUT / f"model_choice_metadata_{families}.json", {
        "scope": SCOPE, "site": "newlock", "fold_id": 1, "seeds": [s["seed"] for s in splits],
        "quality": quality, "states": ["original initial", "same-model original full"],
        "prediction_boundary": "Only observed(initial_ids/full_ids) conditions models; audit truth scored after prediction; point-only old paths",
        "model_ids": sorted(frame.model_id.unique()), "n_predictions": len(frame),
        "initial_complete_pass_models": frame.loc[frame.state.eq("initial") & frame.complete_task_pass, ["seed", "model_id", "budget_n"]].to_dict("records"),
        "full_complete_pass_models": frame.loc[frame.state.eq("full") & frame.complete_task_pass, ["seed", "model_id", "budget_n"]].to_dict("records"),
    })


def consolidate(quality):
    paths = [OUTPUT / f"model_choice_endpoints_{family}.csv" for family in ("cpu", "ai")]
    if not all(path.exists() for path in paths):
        return
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if frame.duplicated(["split_id", "state", "model_id"]).any():
        raise ValueError("Each model must have one prediction per original split/state")
    frame.to_csv(OUTPUT / "model_choice_newlock_endpoints.csv", index=False)
    summary = frame.groupby(["state", "model_id"])[["log_mae", "thickness_mae_m", "recall", "fpr", "audit_pass", "low_qc_audit_pass", "complete_task_pass"]].mean().reset_index()
    summary.to_csv(OUTPUT / "model_choice_newlock_summary.csv", index=False)
    dataset = load_dataset(load_config())
    names = dataset.locations("newlock")[["location_id", "source_file"]]
    hole_rows = pd.concat([pd.read_csv(OUTPUT / f"model_choice_holes_{family}.csv") for family in ("cpu", "ai")])
    hole_rows = hole_rows.merge(names, on="location_id", validate="many_to_one")
    hole_rows.to_csv(OUTPUT / "model_choice_newlock_holes.csv", index=False)
    depth_rows = pd.concat([pd.read_csv(OUTPUT / f"model_choice_depths_{family}.csv") for family in ("cpu", "ai")])
    depth_rows.to_csv(OUTPUT / "model_choice_newlock_depths.csv", index=False)
    ai = pd.read_csv(OUTPUT / "model_choice_endpoints_ai.csv")
    frozen = pd.read_csv(OUTPUT / "model_choice_newlock_frozen.csv")
    compare = ai.merge(frozen, on=["split_id", "model_id", "budget_n"], suffixes=("_actual", "_frozen"), validate="one_to_one")
    metric_error = {metric: float(abs(compare[f"{metric}_actual"] - compare[f"{metric}_frozen"]).max())
                    for metric in ("log_mae", "thickness_mae_m", "recall", "fpr")}
    json_write(OUTPUT / "model_choice_findings.json", {
        "scope": SCOPE,
        "quality": quality,
        "models": int(frame.model_id.nunique()),
        "split_state_model_predictions": len(frame),
        "absolute_profile_thickness_passes": int(frame.absolute_quality_pass.sum()),
        "complete_task_passes": int(frame.complete_task_pass.sum()),
        "ai_frozen_endpoint_matches": len(compare), "ai_frozen_max_abs_metric_difference": metric_error,
        "best_newlock_full_log_model": summary.loc[summary.state.eq("full")].sort_values("log_mae").iloc[0].to_dict(),
        "selection_diagnosis": "graph wins development log-MAE, joint improves thickness and absolute joint pass rate; neither passes NewLock even at the full endpoint",
        "next_model_direction": "Current endpoints require improved modeling of local heterogeneity; pure revealed depth/global priors do not solve the error. Full-plan failure alone does not prove every intermediate subset must fail; new trajectories still require actual quality scoring.",
        "simple_priors": "depthwise mean/median or global median of revealed log(qc) only; threshold labels always qc<1MPa",
    })


def gp_kernel_jump():
    """Exactly two recorded MLE fits; neither context receives hidden values."""
    from threadpoolctl import threadpool_limits
    dataset = load_dataset(load_config())
    split_data = json.loads((V1 / "splits.json").read_text(encoding="utf-8"))
    split_data = split_data["splits"] if isinstance(split_data, dict) else split_data
    split = next(s for s in split_data if s["site_id"] == "newlock" and s["seed"] == 29)
    trace = pd.read_csv(ROOT / "runs/adaptive_v2_acquisition/newlock__s29__gp__maximin/quality.csv")
    contexts = [trace.loc[trace.n_observed.eq(n)].iloc[0] for n in (63, 64)]
    assert set(json.loads(contexts[0].known_ids)) < set(json.loads(contexts[1].known_ids))
    query = dataset.query(split["audit_ids"])
    records, predictions = [], []
    with threadpool_limits(limits=1):
        for row in contexts:
            known = json.loads(row.known_ids)
            assert not set(known) & set(split["audit_ids"])
            observed = dataset.observed(known).sort_values(["location_id", "depth_index"], kind="stable").reset_index(drop=True)
            model = make_model("gp", 1)
            model.condition(observed)
            core, regressor = model.predictor, model.predictor.regressor
            kernel = regressor.kernel_
            train_prediction = model.predict(observed.drop(columns=["qc_mpa", "log_qc"], errors="ignore")).frame
            train_residual = observed.log_qc.to_numpy(float) - train_prediction.pred_log_qc.to_numpy(float)
            prediction = model.predict(query).frame
            predictions.append(prediction)
            # Offline audit scoring starts only after both fit and prediction.
            score = _score(prediction, dataset.truth(split["audit_ids"]))
            lengths = np.asarray(kernel.k1.k2.length_scale, float)
            added_ids = sorted(set(known) - set(json.loads(contexts[0].known_ids)))
            added_files = dataset.locations("newlock").set_index("location_id").loc[added_ids, "source_file"].tolist()
            records.append({"site_id": "newlock", "seed": 29, "fold_id": 1,
                            "n_observed": len(known), "source": "canonical revealed-only GP, unchanged v1 factory",
                            "signal_variance": float(kernel.k1.k1.constant_value),
                            "length_x_m": lengths[0], "length_y_m": lengths[1], "length_z_m": lengths[2],
                            "noise_variance": float(kernel.k2.noise_level),
                            "noise_sd": float(np.sqrt(kernel.k2.noise_level)),
                            "log_marginal_likelihood": float(regressor.log_marginal_likelihood_value_),
                            "kernel_repr": str(kernel), "kernel_theta": json.dumps(kernel.theta.tolist()),
                            "train_fit_residual_mae_log": float(abs(train_residual).mean()),
                            "train_fit_residual_rmse_log": float(np.sqrt(np.mean(train_residual ** 2))),
                            "train_fit_residual_sd_log": float(train_residual.std()),
                            "source_trend_residual_sd_log": float((observed.log_qc.to_numpy(float) - core._trend_at(observed)).std()),
                            "added_location_ids": json.dumps(added_ids), "added_source_files": json.dumps(added_files),
                            **score})
            print(f"kernel evidence n={len(known)} kernel={kernel} audit_Q={score['log_mae']:.6f}", flush=True)
    pd.DataFrame(records).to_csv(OUTPUT / "model_choice_gp_kernel_jump.csv", index=False)
    changed = query[["location_id", "depth_index", "z_center_m"]].copy()
    changed["pred_n63"] = predictions[0].pred_log_qc.to_numpy(float)
    changed["pred_n64"] = predictions[1].pred_log_qc.to_numpy(float)
    changed["signed_change_log"] = changed.pred_n64 - changed.pred_n63
    changed["abs_change_log"] = abs(changed.signed_change_log)
    changed["low_label_flipped"] = (changed.pred_n63 < 0) != (changed.pred_n64 < 0)
    by_hole = changed.groupby("location_id").agg(mean_abs_change_log=("abs_change_log", "mean"),
        max_abs_change_log=("abs_change_log", "max"), mean_signed_change_log=("signed_change_log", "mean"),
        low_label_flips=("low_label_flipped", "sum")).reset_index()
    by_hole = by_hole.merge(dataset.locations("newlock")[["location_id", "source_file"]], on="location_id")
    by_depth = changed.groupby("z_center_m").agg(mean_abs_change_log=("abs_change_log", "mean"),
        mean_signed_change_log=("signed_change_log", "mean"), low_label_flips=("low_label_flipped", "sum")).reset_index()
    json_write(OUTPUT / "model_choice_gp_kernel_jump.json", {
        "scope": SCOPE, "states": [63, 64], "model_id": "gp", "seed": 29, "fold_id": 1,
        "recorded_mle_fits": 2, "hyperparameter_search_expanded": False,
        "source_factory_changed": False, "full_kernel_backfilled_into_n63": False,
        "observation_boundary": "Each MLE fit uses its own revealed set only. Audit labels used only for post-prediction scoring.",
        "models": records,
        "audit_prediction_change": {"cells": len(changed), "mean_abs_log": float(changed.abs_change_log.mean()),
            "p90_abs_log": float(changed.abs_change_log.quantile(.9)), "max_abs_log": float(changed.abs_change_log.max()),
            "low_label_flips": int(changed.low_label_flipped.sum()),
            "by_hole": by_hole.to_dict("records"), "by_depth": by_depth.to_dict("records")},
        "interpretation_limit": "Two fitted states identify parameter and prediction changes; they do not isolate a causal fixed-kernel counterfactual.",
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--consolidate-only", action="store_true")
    parser.add_argument("--gp-jump-only", action="store_true")
    parser.add_argument("--families", choices=("all", "cpu", "ai"), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.gp_jump_only:
        gp_kernel_jump()
        return
    quality = frozen_summary()
    if not args.summary_only and not args.consolidate_only:
        import torch
        torch.set_num_threads(1)
        for family in (("cpu", "ai") if args.families == "all" else (args.families,)):
            endpoints(quality, args.device, family)
    consolidate(quality)


if __name__ == "__main__":
    main()
