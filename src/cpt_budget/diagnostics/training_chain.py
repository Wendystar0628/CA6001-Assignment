"""Source-only fixed-episode fitting probes; these are NOT generalization tests.

Run two independent official TabPFN v2 restarts on the same intact CPT context:
real query labels, and IDW teacher labels. No validation/test labels are read.
The trained checkpoint is a diagnostic artifact, never a deployable candidate.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from cpt_budget.config import load_config, write_json
from cpt_budget.models.idw import IDWPredictor
from cpt_budget.models.tabpfn_train import _ids
from cpt_budget.types import load_dataset
from cpt_budget.v2.tabpfn import (
    SpatialEpisodeAdapter, SpatialTabPFNPredictor, build_spatial_episode,
)


def select_source_episode(episodes, *, fold_id=0, query_profiles=2):
    """Predeclared deterministic selection, independent of labels or scores."""
    if query_profiles < 1:
        raise ValueError("query_profiles must be positive")
    source = episodes.loc[(episodes.fold_id == fold_id) & (episodes.role == "train")]
    for row in source.sort_values(["site_id", "episode_id"]).to_dict("records"):
        support = _ids(row["support_location_ids"])
        query = sorted(_ids(row["query_location_ids"]))
        if len(support) == 5 and len(query) >= query_profiles:
            return row, query[:query_profiles]
    raise ValueError("No eligible source/train episode with five support CPTs")


def official_training_loss(adapter, payload, *, seed):
    """Evaluate the exact v2 CRPS+MSE loss without an optimizer update."""
    import torch
    from tabpfn.architectures.interface import PerformanceOptions
    from tabpfn.finetuning._torch_compat import autocast, sdpa_kernel_context
    from tabpfn.finetuning.finetuned_regressor import (
        _compute_regression_loss, _targets_in_estimator_space,
    )
    batch = adapter.preprocess(**payload, seed=seed)
    adapter.estimator.raw_space_bardist_ = batch.raw_space_bardist
    adapter.estimator.znorm_space_bardist_ = batch.znorm_space_bardist
    previous_mode = adapter.model.training
    adapter.model.eval()
    adapter.estimator.fit_from_preprocessed(
        batch.X_context, batch.y_context, batch.cat_indices, batch.configs,
        performance_options=PerformanceOptions(
            force_recompute_layer=True, use_chunkwise_inference=False),
        no_refit=True,
    )
    with torch.no_grad(), autocast(enabled=adapter.use_amp), sdpa_kernel_context():
        _, per_member, _ = adapter.estimator.forward(
            batch.X_query, use_inference_mode=False)
        logits_qbel = torch.stack(per_member, dim=2)
        q, b, e, bars = logits_qbel.shape
        logits = logits_qbel.permute(1, 2, 0, 3).reshape(b * e, q, bars).float()
        targets = torch.stack([
            _targets_in_estimator_space(batch.y_query[0], c[0])
            for c in batch.configs
        ]).to(adapter.device)
        bardist = batch.znorm_space_bardist.to(device=logits.device, dtype=logits.dtype)
        loss = _compute_regression_loss(
            logits_BQL=logits, targets_BQ=targets, bardist_loss_fn=bardist,
            ce_loss_weight=0., crps_loss_weight=1., crls_loss_weight=0.,
            mse_loss_weight=1., mae_loss_weight=0.,
        )
    adapter.model.train(previous_mode)
    return float(loss.detach().cpu())


def prediction_metrics(frame, true_y, goal_y):
    prediction = frame.pred_log_qc.to_numpy(dtype=float)
    return {
        "official_inference_goal_log_mae": float(np.mean(np.abs(prediction - goal_y))),
        "official_inference_truth_log_mae": float(np.mean(np.abs(prediction - true_y))),
        "official_inference_goal_log_rmse": float(np.sqrt(np.mean((prediction - goal_y) ** 2))),
        "native_truth_brier": float(np.mean((frame.p_low_1.to_numpy() - (true_y < 0.)) ** 2)),
    }


def evaluate_probe(adapter, payload, observed, geometry, true_y, *, seed):
    loss = official_training_loss(adapter, payload, seed=seed)
    predictor = adapter.predictor()
    predictor.condition(observed)
    frame = predictor.predict(geometry).frame
    metrics = {"step": adapter.steps, "crps_plus_mse_eval": loss,
               **prediction_metrics(frame, true_y, payload["y_query"])}
    del predictor
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    return metrics, frame


def run(config, output_dir, *, steps=200, fold_id=0, query_profiles=2):
    if not 1 <= steps <= 200:
        raise ValueError("The predeclared diagnostic cap is 200 updates per probe")
    output_dir = Path(output_dir)
    if (output_dir / "summary.json").exists():
        raise FileExistsError("Completed evidence exists; choose a new output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(config)
    config.setdefault("tabpfn", {})["device"] = "cuda"
    config.setdefault("v2", {}).setdefault("training", {}).update({
        "learning_rate": 1e-5, "weight_decay": .01, "gradient_clip": 1.,
    })
    dataset = load_dataset(config)
    episodes = pd.read_parquet(Path(config["_root"]) / config["project"]["prepared_dir"] / "episodes.parquet")
    episode, query_ids = select_source_episode(
        episodes, fold_id=fold_id, query_profiles=query_profiles)
    spatial = build_spatial_episode(dataset, episode, query_ids)
    payload = {k: spatial[k] for k in ("X_support", "y_support", "X_query", "y_query")}
    observed = dataset.observed(spatial["support_location_ids"])
    geometry = dataset.query(query_ids)
    true_y = payload["y_query"].copy()
    idw = IDWPredictor(config)
    idw.condition(observed)
    teacher_frame = idw.predict(geometry).frame
    teacher_y = teacher_frame.pred_log_qc.to_numpy(dtype=float)
    seed = int(config["split"]["seed"])
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=config["_root"], text=True).strip()
    except subprocess.CalledProcessError:
        commit = None
    plan = {
        "study": "source_training_diagnostic_only", "generalization_claim": False,
        "source_commit": commit, "fold_id": fold_id, "source_role": "train",
        "site_id": episode["site_id"], "episode_id": episode["episode_id"],
        "support_location_ids": spatial["support_location_ids"], "query_location_ids": query_ids,
        "support_rows": len(observed), "query_rows": len(geometry),
        "selection": "first lexical site/episode with five support and >=query_profiles queries",
        "query_selection": "first lexical query IDs, all 24 depth cells retained",
        "query_labels_used_for_fitting": True, "validation_or_test_labels_read": False,
        "independent_pretrained_restarts": ["truth", "idw_teacher"],
        "support_labels": "real observed log qc for both probes",
        "loss": "official CRPS plus MSE", "updates_per_probe": steps,
        "learning_rate": 1e-5, "weight_decay": .01, "gradient_clip": 1.,
        "fixed_preprocessing_seed": seed, "no_early_stopping": True,
        "checkpoint_policy": "final fixed-budget checkpoint only, never validation selected",
        "evaluation_steps": sorted({0, steps} | {s for s in (10, 25, 50, 100, 150, 200) if s <= steps}),
        "idw_truth_log_mae": float(np.mean(np.abs(teacher_y - true_y))),
        "config": config,
    }
    write_json(output_dir / "plan.json", plan)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This registered diagnostic requires a real CUDA GPU")
    summaries = []
    for name, goal in (("truth", true_y), ("idw_teacher", teacher_y)):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % 2**32)
        started = perf_counter()
        probe_dir = output_dir / name
        probe_dir.mkdir(parents=True, exist_ok=True)
        adapter = SpatialEpisodeAdapter(config, variant="crps")
        initial = {n: p.detach().cpu().clone() for n, p in adapter.model.named_parameters()}
        task = {**payload, "y_query": goal.copy()}
        evaluations, updates = [], []
        metrics, initial_frame = evaluate_probe(adapter, task, observed, geometry, true_y, seed=seed)
        evaluations.append(metrics)
        initial_frame.to_parquet(probe_dir / "initial_predictions.parquet", index=False)
        print(json.dumps({"probe": name, "evaluation": metrics}), flush=True)
        for _ in range(steps):
            update = adapter.step(**task, seed=seed)
            updates.append(update)
            if adapter.steps in plan["evaluation_steps"]:
                metrics, final_frame = evaluate_probe(adapter, task, observed, geometry, true_y, seed=seed)
                evaluations.append(metrics)
                write_json(probe_dir / "history.json", {"evaluations": evaluations, "updates": updates})
                print(json.dumps({"probe": name, "evaluation": metrics}), flush=True)
        squared_delta = sum(float(torch.sum((p.detach().cpu() - initial[n]).double() ** 2))
                            for n, p in adapter.model.named_parameters())
        squared_original = sum(float(torch.sum(p.double() ** 2)) for p in initial.values())
        checkpoint = adapter.save(probe_dir / "diagnostic_only.ckpt", {
            "study": plan["study"], "generalization_claim": False,
            "probe": name, "source_site": episode["site_id"], "source_commit": commit,
        })
        final_frame.to_parquet(probe_dir / "final_predictions.parquet", index=False)
        reloaded = SpatialTabPFNPredictor(config, artifact=checkpoint, model_id=f"diagnostic_{name}")
        reloaded.condition(observed)
        reloaded_frame = reloaded.predict(geometry).frame
        columns = [c for c in final_frame if c.startswith(("pred_", "q0", "q1", "q2", "q5", "q7", "q9", "p_low_"))]
        reload_difference = float(np.max(np.abs(final_frame[columns].to_numpy() - reloaded_frame[columns].to_numpy())))
        summary = {
            "probe": name, "initial": evaluations[0], "final": evaluations[-1],
            "optimizer_updates": sum(u["optimizer_updates"] for u in updates),
            "parameters": sum(p.numel() for p in adapter.model.parameters()),
            "all_parameter_delta_l2": float(np.sqrt(squared_delta)),
            "relative_parameter_delta_l2": float(np.sqrt(squared_delta / squared_original)),
            "amp_overflow_count": sum(u["amp_overflow_count"] for u in updates),
            "amp_fp32_fallbacks": sum(u["amp_fp32_fallback"] for u in updates),
            "save_reload_max_absolute_difference": reload_difference,
            "save_reload_allclose_atol_1e_6": bool(np.allclose(final_frame[columns], reloaded_frame[columns], atol=1e-6, rtol=0)),
            "checkpoint": str(checkpoint.resolve()), "elapsed_seconds": perf_counter() - started,
            "generalization_claim": False,
        }
        write_json(probe_dir / "summary.json", summary)
        summaries.append(summary)
        del adapter, reloaded, initial
        gc.collect()
        torch.cuda.empty_cache()
    result = {
        "plan": plan, "device": torch.cuda.get_device_name(), "probes": summaries,
        "query_low_qc_count": int(np.sum(true_y < 0.)),
        "interpretation_limits": [
            "One fixed source-training episode; fitted query labels are reused for evaluation.",
            "Near-zero training error demonstrates fitting capacity, not held-out-site benefit.",
            "IDW teacher agreement tests imitation only; it does not add information beyond IDW.",
            "If no positive low-qc cells occur, this probe does not establish event discrimination.",
            "One fixed preprocessing seed and learning rate do not exhaust all training paths.",
        ],
    }
    write_json(output_dir / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v2.yaml")
    parser.add_argument("--output", default="runs/diagnostics-20261014/training_chain")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--query-profiles", type=int, default=2)
    args = parser.parse_args()
    result = run(load_config(args.config), args.output, steps=args.steps,
                 fold_id=args.fold, query_profiles=args.query_profiles)
    print(json.dumps({"completed": True, "probes": result["probes"]}, indent=2))


if __name__ == "__main__":
    main()
