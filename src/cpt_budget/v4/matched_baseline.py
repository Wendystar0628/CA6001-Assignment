"""CPU XGBoost residual control trained on the same new v4 source episodes.

This separates the benefit of more varied training trajectories from the
benefit of training the Transformer.  It retains the registered v2 tree and
alpha-selection rules; it does not change TabPFN training or its checkpoints.
"""
from __future__ import annotations

import os

# Set before importing numerical libraries, and also limit loaded pools below.
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "2"

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from cpt_budget.config import PROJECT_ROOT, load_config, write_json
from cpt_budget.diagnostics.common import BUDGETS, SOURCE_RUN, STATE_KEYS, run_predictor
from cpt_budget.types import load_dataset
from cpt_budget.v2.features import FEATURE_COLUMNS_V2
from cpt_budget.v2.residual import ResidualPredictor, fit_residual


MODEL_ID = "xgb_residual_matched"
_POINT_METRICS = ["log_mae", "h_point_mae_m", "recall", "false_positive_rate"]


class MatchedResidualPredictor(ResidualPredictor):
    model_id = MODEL_ID


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _signature(train_path, validation_path, config):
    content = {
        "train_episode_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "validation_episode_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
        "feature_columns": FEATURE_COLUMNS_V2,
        "tree_settings": config["v2"]["residual"],
        "idw_settings": config.get("idw", {}),
        "split_seed": config.get("split", {}).get("seed", 20261013),
    }
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest(), content


def _point_summary(metrics):
    rows = []
    for keys, frame in metrics.groupby(["fold_id", "site_id", "seed"], sort=True):
        frame = frame.sort_values("budget_n")
        if frame.budget_n.tolist() != list(BUDGETS):
            raise ValueError("Matched baseline evaluation has incomplete or duplicate budgets")
        budget = frame.budget_n.to_numpy(float)
        row = dict(zip(["fold_id", "site_id", "seed"], keys))
        for name in _POINT_METRICS:
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
            row[name] = float(np.trapezoid(values, budget) / (budget[-1] - budget[0])) if np.isfinite(values).all() else np.nan
        rows.append(row)
    curves = pd.DataFrame(rows)
    fold_sites = curves.groupby(["fold_id", "site_id"])[_POINT_METRICS].mean()
    folds = fold_sites.groupby("fold_id").mean().reset_index()
    sites = fold_sites.groupby("site_id").mean().reset_index()
    macro = sites[_POINT_METRICS].mean().to_dict()
    return curves, folds, sites, macro


def run_matched_baseline(config, run_dir) -> dict:
    """Fit/reuse each source-only fold and evaluate all 60 frozen states on CPU."""
    config = copy.deepcopy(config)
    run_dir = Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = Path(config.get("_root", PROJECT_ROOT)) / run_dir
    output = run_dir / "matched_xgb"
    output.mkdir(parents=True, exist_ok=True)
    source = Path(config.get("_root", PROJECT_ROOT)) / config["v4"]["evaluation"]["source_run"]
    if source.resolve() != SOURCE_RUN.resolve():
        raise ValueError("The matched baseline requires the registered frozen v0.2 validation states")
    # Local execution copy only: never alter the stored v4 configuration.
    settings = config.setdefault("v2", {}).setdefault("residual", {})
    settings.update(max_depth=4, learning_rate=0.05, n_estimators=500,
                    early_stopping_rounds=30, n_jobs=2)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     cwd=config.get("_root", PROJECT_ROOT), text=True).strip()
    doc = _read(source / "inputs" / "folds.json")
    folds = doc["folds"] if isinstance(doc, dict) else doc
    dataset = load_dataset(config)
    summaries, artifacts, retrained = [], {}, False
    with threadpool_limits(limits=2):
        for fold in folds:
            fid = int(fold["fold_id"])
            train_path = run_dir / "episodes" / f"fold_{fid}_train.parquet"
            validation_path = run_dir / "episodes" / f"fold_{fid}_validation.parquet"
            signature, sources = _signature(train_path, validation_path, config)
            target = output / f"fold_{fid}"
            summary_path = target / "summary.json"
            artifact_path = target / "idw_residual.json"
            if summary_path.exists():
                summary = _read(summary_path)
                if summary.get("input_signature") != signature:
                    raise ValueError(f"Matched baseline inputs changed for fold {fid}; use a new run")
                if not artifact_path.exists() or not (target / summary["model_filename"]).exists():
                    raise ValueError(f"Incomplete saved matched baseline for fold {fid}")
            else:
                if artifact_path.exists():
                    raise ValueError(f"Matched baseline artifact lacks completion provenance: {artifact_path}")
                train, validation = pd.read_parquet(train_path), pd.read_parquet(validation_path)
                for role, plan in (("train", train), ("validation", validation)):
                    if not plan.fold_id.eq(fid).all() or not plan.role.eq(role).all():
                        raise ValueError("Saved source episode fold/role mismatch")
                    if not set(plan.site_id) <= set(fold[f"{role}_sites"]):
                        raise ValueError("Saved episodes include a site outside the frozen source role")
                print(f"Matched CPU XGBoost: fold {fid}, {len(train)} train / {len(validation)} validation episodes", flush=True)
                started = perf_counter()
                summary = fit_residual(dataset, train, validation, config, target)
                summary.update(
                    model_id=MODEL_ID, study_version="0.4-matched-control",
                    source_mode="v4_same_source_trajectory_episodes_as_tabpfn",
                    source_commit=commit, input_signature=signature, input_sources=sources,
                    training_episode_path=str(train_path), validation_episode_path=str(validation_path),
                    n_train_episodes=len(train), n_validation_episodes=len(validation),
                    train_scenario_seeds=sorted(map(int, train.scenario_seed.unique())),
                    validation_scenario_seeds=sorted(map(int, validation.scenario_seed.unique())),
                    training_seconds=perf_counter() - started,
                    same_information_note="Same 24 feature columns, source site roles and v4 whole-CPT trajectories. Tree optimization sees all episode query rows; TabPFN uses its fixed update/block budget.",
                    native_event_brier=None, native_event_brier_reason="Point predictor has no native CDF.",
                )
                write_json(artifact_path, summary)
                write_json(summary_path, summary)
                retrained = True
            artifacts[fid] = artifact_path
            summaries.append(summary)
        evaluation_dir = output / "validation"
        evaluated = run_predictor(
            lambda fid: MatchedResidualPredictor(config, artifact=artifacts[fid]),
            evaluation_dir, config=config, resume=not retrained,
        )
    predictions, metrics = evaluated["predictions"], evaluated["metrics"]
    if len(metrics) != 60 or set(predictions.model_id) != {MODEL_ID}:
        raise ValueError("Matched XGBoost must contain one complete 60-state evaluation")
    for frame in (predictions, metrics):
        frame["study_version"] = "0.4-matched-control"
        frame["run_id"] = run_dir.name
    predictions.to_parquet(evaluation_dir / "predictions.parquet", index=False)
    metrics.to_csv(evaluation_dir / "metrics.csv", index=False)
    curves, fold_points, site_points, macro = _point_summary(metrics)
    curves.to_csv(evaluation_dir / "point_curves.csv", index=False)
    fold_points.to_csv(evaluation_dir / "point_fold_macro.csv", index=False)
    site_points.to_csv(evaluation_dir / "point_unique_site_metrics.csv", index=False)
    protocol = {
        "model_id": MODEL_ID, "study_version": "0.4-matched-control",
        "source_run": str(source), "source_commit": commit,
        "scope": "exploratory_validation_only", "outer_test_executed": False,
        "purpose": "Separate new trajectory information from Transformer-specific training gain.",
        "fitting": "v2.residual.fit_residual with the new v4 source-only episode plans",
        "tree_settings": settings, "alpha_candidates": [0.0, 0.25, 0.5, 1.0],
        "seeds": [17, 29], "budgets": list(BUDGETS), "n_states": len(metrics),
        "n_prediction_rows": len(predictions), "threads_max": 2,
        "prediction_path": str(evaluation_dir / "predictions.parquet"),
        "metric_path": str(evaluation_dir / "metrics.csv"),
        "point_macro": macro, "fold_point_metrics": fold_points.to_dict("records"),
        "fold_training": summaries,
        "aggregation": "Budget trapezoid mean, seed mean, repeated-fold mean within site, equal unique-site mean.",
        "risk_calibration": "Not performed here; shared v4 evaluation calibrates all candidates together.",
        "engine_note": "Per-state caches originate from diagnostics.common.run_predictor; consolidated tables and this protocol identify v4 provenance.",
    }
    write_json(evaluation_dir / "evaluation_protocol.json", protocol)
    write_json(output / "summary.json", protocol)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v4.yaml")
    parser.add_argument("--run", default="v4-residual-20261014")
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", "..") or "/" in args.run or "\\" in args.run:
        parser.error("--run must be a simple run directory name")
    config = load_config(args.config)
    run = Path(config["_root"]) / config["project"]["output_dir"] / args.run
    summary = run_matched_baseline(config, run)
    print(json.dumps({"summary": str(run / "matched_xgb" / "summary.json"),
                      "point_macro": summary["point_macro"], "n_states": summary["n_states"]}, indent=2))


if __name__ == "__main__":
    main()
