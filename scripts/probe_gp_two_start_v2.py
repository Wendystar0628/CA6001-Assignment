"""Six-fit maximum probe of spatial GP MLE initialization at New Lock full64.

The two starts are fixed before fitting: native defaults and observed-geometry
median nearest-neighbour scale. Training LML selects the start before audit
labels are opened. No fitted full-plan kernel is reused for a smaller context.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from cpt_budget.adaptive_models import make_predictor, audit_fold_sources
from cpt_budget.adaptive_run import _score
from cpt_budget.config import load_config
from cpt_budget.types import load_dataset


def geometry_start(observed):
    xy = observed[["location_id", "x_m", "y_m"]].drop_duplicates("location_id")[["x_m", "y_m"]].to_numpy(float)
    distance = cdist(xy, xy)
    np.fill_diagonal(distance, np.inf)
    nearest = distance.min(axis=1)
    value = float(np.median(nearest))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Observed-position median nearest-neighbour distance must be positive")
    return [value, value, 1.], nearest


def train_one(template, observed, start):
    native = copy.deepcopy(template)
    native.settings = copy.deepcopy(template.settings)
    native.settings["length_scale_m"] = list(start)
    native.settings["n_restarts_optimizer"] = 0
    if not bool(native.settings.get("optimize", True)):
        raise ValueError("Native GP is not configured for MLE optimization")
    native.condition(observed)
    regressor = native.regressor
    coordinates = native._coordinates(native.observed)
    target_residual = native.observed.log_qc.to_numpy(float) - native._trend_at(native.observed)
    posterior_residual = regressor.predict(coordinates)
    errors = target_residual - posterior_residual
    kernel = regressor.kernel_
    return native, {
        "training_lml": float(regressor.log_marginal_likelihood_value_),
        "optimized_kernel": str(kernel),
        "optimized_length_x_m": float(kernel.k1.k2.length_scale[0]),
        "optimized_length_y_m": float(kernel.k1.k2.length_scale[1]),
        "optimized_length_z_m": float(kernel.k1.k2.length_scale[2]),
        "optimized_constant": float(kernel.k1.k1.constant_value),
        "optimized_noise": float(kernel.k2.noise_level),
        "training_residual_mean": float(errors.mean()),
        "training_residual_mae": float(abs(errors).mean()),
        "training_residual_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "training_residual_p90_abs": float(np.quantile(abs(errors), .9)),
        "training_target_residual_std": float(target_residual.std()),
        "training_error_scope": "in-sample posterior mean, not leave-one-out error",
        "optimizer_restarts": int(regressor.n_restarts_optimizer),
        "training_rows": len(observed),
    }


def run(root, output):
    frozen = json.loads((root / "results/adaptive_v1/frozen_config.json").read_text(encoding="utf-8"))
    split_object = json.loads((root / "results/adaptive_v1/splits.json").read_text(encoding="utf-8"))
    splits = split_object["splits"] if isinstance(split_object, dict) else split_object
    splits = sorted([s for s in splits if s["site_id"] == "newlock" and s["seed"] in [17, 29, 43]], key=lambda s: s["seed"])
    if len(splits) != 3 or any(s["n_full"] != 64 for s in splits):
        raise ValueError("Probe is limited to exactly the original three New Lock full64 splits")
    dataset = load_dataset(load_config())
    output.mkdir(parents=True, exist_ok=True)
    rows, selections, source_checks, fit_count = [], [], [], 0
    for split in splits:
        source = audit_fold_sources(split["fold_id"], "newlock", role="evaluation", model_ids=("gp",))
        if not source["passed"]:
            raise ValueError("Native GP source provenance failed")
        source_checks.append(source)
        adapter = make_predictor("gp", split["fold_id"], device="cpu", role="evaluation").initialize()
        retained = adapter.suite.models["gp"]
        template = retained.predictor
        observed = dataset.observed(split["full_ids"]).sort_values(["location_id", "depth_index"], kind="stable").reset_index(drop=True)
        if observed.location_id.nunique() != 64 or set(observed.location_id) & set(split["audit_ids"]):
            raise ValueError("Training must contain all64 planned positions and no audit positions")
        native_start = list(template.settings.get("length_scale_m", [20., 20., 1.]))
        if not np.array_equal(np.asarray(native_start), np.array([20., 20., 1.])):
            raise ValueError("Native default is no longer the registered [20,20,1] start")
        median_start, nearest = geometry_start(observed)
        starts = {"native_default": native_start, "observed_geometry_median_nn": median_start}
        fitted, statistics = {}, {}
        with threadpool_limits(limits=1):
            for name, start in starts.items():
                if name != "native_default" and np.array_equal(start, native_start):
                    fitted[name], statistics[name] = fitted["native_default"], dict(statistics["native_default"])
                    statistics[name]["same_fit_as"] = "native_default"
                else:
                    fitted[name], statistics[name] = train_one(template, observed, start)
                    statistics[name]["same_fit_as"] = None
                    fit_count += 1
                if fit_count > 6:
                    raise RuntimeError("Six-fit experimental cap exceeded")
            # Tie goes to the unchanged native default. No audit read/prediction
            # or evaluation quantity has happened before this selection.
            chosen = max(starts, key=lambda name: statistics[name]["training_lml"])
            lml_choice = {"site_id": "newlock", "seed": int(split["seed"]), "fold_id": int(split["fold_id"]),
                          "n_observed": 64, "n_full": 64, "selected_initialization": chosen,
                          "selection_source": "training_log_marginal_likelihood_only",
                          "audit_opened_before_selection": False,
                          "selected_training_lml": statistics[chosen]["training_lml"],
                          "lml_gain_vs_default": statistics[chosen]["training_lml"] - statistics["native_default"]["training_lml"],
                          "saved_cpt": 0, "savings_experiment": False}
            selections.append(lml_choice)
            from cpt_budget.adaptive_data import audit_truth
            truth = audit_truth(dataset, split)
            query = dataset.query(split["audit_ids"]).sort_values(["location_id", "depth_index"], kind="stable").reset_index(drop=True)
            for name, model in fitted.items():
                prediction = model.predict(query).frame
                score = _score(prediction, truth)
                qpass = score["log_mae"] <= frozen["quality"]["absolute_log_mae_max"]
                hpass = score["thickness_mae_m"] <= frozen["quality"]["absolute_thickness_mae_max_m"]
                lowpass = (bool(score["recall"] >= .5 and score["fpr"] <= .1)
                           if pd.notna(score["recall"]) and pd.notna(score["fpr"]) else None)
                rows.append({"site_id": "newlock", "seed": int(split["seed"]), "fold_id": int(split["fold_id"]),
                             "initialization": name, "selected_by_training_lml": name == chosen,
                             "initial_length_x_m": starts[name][0], "initial_length_y_m": starts[name][1],
                             "initial_length_z_m": starts[name][2], "observed_median_nn_m": float(np.median(nearest)),
                             "n_observed": 64, "n_full": 64, "n_audit": int(split["n_audit"]),
                             **statistics[name], **{key: score[key] for key in ["log_mae", "thickness_mae_m", "recall", "precision", "fpr"]},
                             "absolute_log_pass": bool(qpass), "absolute_thickness_pass": bool(hpass),
                             "absolute_quality_pass": bool(qpass and hpass), "low_qc_audit_pass": lowpass,
                             "saved_cpt": 0, "audit_read_for_fit_or_start_selection": False})
                print(f"newlock seed={split['seed']} {name} LML={statistics[name]['training_lml']:.6f} Q={score['log_mae']:.6f} H={score['thickness_mae_m']:.6f} chosen={name == chosen}", flush=True)
    frame = pd.DataFrame(rows)
    selection = pd.DataFrame(selections)
    chosen_scores = frame.loc[frame.selected_by_training_lml].set_index("seed")
    baseline = frame.loc[frame.initialization.eq("native_default")].set_index("seed")
    selection["selected_log_mae"] = selection.seed.map(chosen_scores.log_mae)
    selection["selected_thickness_mae_m"] = selection.seed.map(chosen_scores.thickness_mae_m)
    selection["selected_recall"] = selection.seed.map(chosen_scores.recall)
    selection["selected_fpr"] = selection.seed.map(chosen_scores.fpr)
    selection["selected_absolute_quality_pass"] = selection.seed.map(chosen_scores.absolute_quality_pass)
    selection["selected_low_qc_audit_pass"] = selection.seed.map(chosen_scores.low_qc_audit_pass)
    selection["log_mae_change_vs_default"] = selection.seed.map(chosen_scores.log_mae - baseline.log_mae)
    selection["thickness_mae_change_vs_default_m"] = selection.seed.map(chosen_scores.thickness_mae_m - baseline.thickness_mae_m)
    frame.to_csv(output / "gp_two_start_fits.csv", index=False)
    selection.to_csv(output / "gp_two_start_selection.csv", index=False)
    metadata = {
        "scope": "post-unblinding finite engineering probe, not a blinded evaluation",
        "actual_fits": fit_count, "fit_cap": 6, "seeds": [17, 29, 43], "budgets": [64],
        "initializations": ["native [20,20,1]", "current-observed-geometry [medianNN,medianNN,1]"],
        "model_template": "make_predictor('gp').suite.models['gp'].predictor deep copy (native GPPredictor)",
        "native_settings": template.settings, "source_checks": source_checks,
        "source_depth_trend_unchanged": True, "optimizer_restarts_each": 0,
        "row_order": "location_id then depth_index, stable sorted",
        "selection": "higher training LML; exact tie chooses native default",
        "audit_read_for_fit_or_start_selection": False,
        "audit_metrics_read_after_lml_selection": True,
        "kernels_reused_at_smaller_budget": False, "saved_cpt": 0,
        "quality_targets_unchanged": frozen["quality"],
        "quality_at_full_plan_scope": "absolute log and thickness targets plus separate low-value target; no early-stop or saving claim",
    }
    (output / "gp_two_start_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(selection.to_string(index=False))
    return frame, selection


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("results/adaptive_v2_diagnostics"))
    args = parser.parse_args()
    run(args.root, args.output)
