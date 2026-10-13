"""Site-balanced CPT episode training, held-out site selection and checkpoints."""
from __future__ import annotations

import gc
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from cpt_budget.config import write_json
from cpt_budget.features import FEATURE_COLUMNS
from .tabpfn_adapter import EpisodeAdapter, feature_array, geometry_only


def _records(episodes):
    return episodes.to_dict("records") if isinstance(episodes, pd.DataFrame) else list(episodes)


def _ids(value):
    if isinstance(value, str):
        value = json.loads(value)
    return [str(item) for item in value]


def _check_episode(dataset, episode):
    support = _ids(episode["support_location_ids"])
    query = _ids(episode["query_location_ids"])
    if len(support) < 5 or not query or set(support) & set(query):
        raise ValueError("Episode needs >=5 support locations and disjoint, nonempty query locations")
    if len(set(support)) != len(support) or len(set(query)) != len(query):
        raise ValueError("Duplicate location IDs in episode")
    locations = dataset.locations()
    source = locations.set_index("location_id")
    selected = source.loc[support + query]
    if selected.site_id.nunique() != 1 or str(selected.site_id.iloc[0]) != str(episode["site_id"]):
        raise ValueError("An episode cannot mix sites")
    if "near_group_id" in source:
        if set(source.loc[support].near_group_id) & set(source.loc[query].near_group_id):
            raise ValueError("Near groups cannot cross the support/query boundary")
    return support, query


def build_episode(dataset, episode, *, query_cycle=0, query_limit=8):
    """Features see support only; labels of query are returned solely for loss."""
    support, query = _check_episode(dataset, episode)
    query_limit = max(1, min(8, int(query_limit)))
    if len(query) > query_limit:
        offset = int(query_cycle) * query_limit % len(query)
        query = (query[offset:] + query[:offset])[:query_limit]
    observed = dataset.observed(support)
    query_geometry = dataset.query(query)
    truth = dataset.truth(query)
    for name, frame in (("support", observed), ("query", query_geometry)):
        sizes = frame.groupby("location_id").size()
        if not sizes.eq(24).all() or frame.duplicated(["location_id", "depth_index"]).any():
            raise ValueError(f"Each {name} CPT must contain exactly 24 distinct depth cells")
    truth = query_geometry[["location_id", "depth_index"]].merge(
        truth[["location_id", "depth_index", "log_qc"]], on=["location_id", "depth_index"], validate="one_to_one", how="left")
    if not np.isfinite(truth.log_qc).all():
        raise ValueError("Query loss labels are missing or non-finite")
    return {
        "X_support": feature_array(observed, geometry_only(observed), leave_location_out=True),
        "y_support": observed.log_qc.to_numpy(dtype=np.float64),
        "X_query": feature_array(observed, query_geometry),
        "y_query": truth.log_qc.to_numpy(dtype=np.float64),
        "support_location_ids": support, "query_location_ids": query,
    }


def _validate(adapter, dataset, episodes):
    """Use exactly the target inference interface and equal weight per site."""
    predictor = adapter.predictor()
    site_episode_scores = {}
    for episode in episodes:
        support, query = _check_episode(dataset, episode)
        predictor.condition(dataset.observed(support))
        predictions = predictor.predict(dataset.query(query)).frame
        joined = predictions[["location_id", "depth_index", "pred_log_qc"]].merge(
            dataset.truth(query)[["location_id", "depth_index", "log_qc"]],
            on=["location_id", "depth_index"], validate="one_to_one")
        joined["squared_error"] = (joined.pred_log_qc.to_numpy() - joined.log_qc.to_numpy()) ** 2
        # Match formal evaluation: first calculate an RMSE for each CPT, then
        # average CPTs within a scenario. Scenarios have equal weight within a
        # site even when their query sets differ in size; sites are equal too.
        cpt_rmse = np.sqrt(joined.groupby("location_id").squared_error.mean())
        episode_score = float(cpt_rmse.mean())
        site_episode_scores.setdefault(str(episode["site_id"]), []).append(episode_score)
    scores = {site: float(np.mean(episodes)) for site, episodes in site_episode_scores.items()}
    del predictor
    gc.collect()
    if adapter.use_amp:
        import torch
        torch.cuda.empty_cache()
    if not scores or not np.isfinite(list(scores.values())).all():
        raise RuntimeError("Validation must yield finite scores for held-out sites")
    return {"macro_log_rmse": float(np.mean(list(scores.values()))), "site_log_rmse": scores,
            "validation_aggregation": "mean_site(mean_episode(mean_cpt(rmse_depth)))"}


def fit_tabpfn(dataset, train_episodes, validation_episodes, config, output_dir):
    """Train one fold; retain the best *updated* checkpoint even if baseline wins.

    max_steps is an explicit development cap, never enabled by default. Partial
    epochs and development runs are identified in the saved summary.
    """
    train = _records(train_episodes)
    validation = _records(validation_episodes)
    if not train or not validation:
        raise ValueError("Fine-tuning requires nonempty train and validation episodes")
    if any(e.get("role", "train") != "train" for e in train):
        raise ValueError("Only train-role episodes may update the weights")
    if any(e.get("role", "validation") != "validation" for e in validation):
        raise ValueError("Only validation-role episodes may select checkpoints")
    train_sites = {str(e["site_id"]) for e in train}
    validation_sites = {str(e["site_id"]) for e in validation}
    if train_sites & validation_sites:
        raise ValueError("Training and validation sites overlap")
    folds = {int(e["fold_id"]) for e in train + validation if "fold_id" in e}
    if len(folds) > 1:
        raise ValueError("A checkpoint belongs to exactly one outer fold")
    if "investigation_group" in dataset.sites:
        site_groups = dataset.sites.set_index("site_id").investigation_group
        if set(site_groups.loc[list(train_sites)]) & set(site_groups.loc[list(validation_sites)]):
            raise ValueError("Training and validation investigation groups overlap")
    grouped = {}
    for episode in train:
        _check_episode(dataset, episode)
        grouped.setdefault(str(episode["site_id"]), []).append(episode)
    for episodes in grouped.values():
        episodes.sort(key=lambda e: str(e.get("episode_id", "")))
    settings = config.get("tabpfn", {})
    epochs = min(10, int(settings.get("max_epochs", 10)))
    patience = int(settings.get("early_stopping_patience", 3))
    time_limit = float(settings.get("time_limit_seconds", 7200))
    max_steps = settings.get("max_steps")
    if epochs < 1 or patience < 1 or (max_steps is not None and int(max_steps) < 1):
        raise ValueError("Epochs, patience and optional max_steps must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "tabpfn_finetuned.ckpt"
    started = perf_counter()
    adapter = EpisodeAdapter(config)
    import torch
    if adapter.use_amp:
        torch.cuda.reset_peak_memory_stats()
    baseline = _validate(adapter, dataset, validation)
    write_json(output_dir / "tabpfn_pretrained_validation.json", baseline)
    history, step_history = [], []
    best_score, best_epoch, stalled = np.inf, None, 0
    stop_reason = "max_epochs"
    metadata = {"fold_id": next(iter(folds), None), "train_sites": sorted(train_sites),
                "validation_sites": sorted(validation_sites), "model_generation": "v2",
                "base_model_repo": "Prior-Labs/TabPFN-v2-reg", "package_version": "9.2.0",
                "feature_columns": list(FEATURE_COLUMNS), "loss": "native_bar_distribution_nll",
                "preprocessing": "support_only_numeric_16_columns", "n_estimators": 1}
    query_limit = min(8, int(config.get("episodes", {}).get("query_profiles_per_batch", 8)))
    for epoch in range(epochs):
        epoch_losses = []
        site_order = sorted(grouped)
        np.random.default_rng(int(config.get("split", {}).get("seed", 20261013)) + epoch).shuffle(site_order)
        completed_sites = 0
        for site in site_order:
            # Every training site receives one update per complete epoch; source
            # masks alternate across epochs and query profiles rotate per mask.
            choices = grouped[site]
            episode = choices[epoch % len(choices)]
            payload = build_episode(dataset, episode, query_cycle=epoch // len(choices), query_limit=query_limit)
            while True:
                try:
                    step = adapter.step(**{key: payload[key] for key in ("X_support", "y_support", "X_query", "y_query")}, seed=int(episode.get("seed", 0)))
                    break
                except torch.OutOfMemoryError:
                    if query_limit <= 1:
                        raise
                    adapter.optimizer.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    query_limit = max(1, query_limit // 2)
                    payload = build_episode(dataset, episode, query_cycle=epoch // len(choices), query_limit=query_limit)
            completed_sites += 1
            epoch_losses.append(step["nll"])
            step_history.append({**step, "epoch": epoch + 1, "site_id": site, "episode_id": episode.get("episode_id"),
                                 "support_location_ids": payload["support_location_ids"], "query_location_ids": payload["query_location_ids"]})
            if max_steps is not None and adapter.steps >= int(max_steps):
                stop_reason = "development_max_steps"
                break
            if perf_counter() - started >= time_limit:
                stop_reason = "time_limit"
                break
        scores = _validate(adapter, dataset, validation)
        history.append({"epoch": epoch + 1, "mean_train_nll": float(np.mean(epoch_losses)), **scores,
                        "completed_sites": completed_sites, "total_train_sites": len(grouped),
                        "complete_epoch": completed_sites == len(grouped)})
        if scores["macro_log_rmse"] < best_score:
            best_score, best_epoch, stalled = scores["macro_log_rmse"], epoch + 1, 0
            adapter.save(best_path, {**metadata, "epoch": best_epoch, "validation_macro_log_rmse": best_score})
        else:
            stalled += 1
        write_json(output_dir / "tabpfn_training_history.json", {"epochs": history, "steps": step_history})
        if stop_reason != "max_epochs":
            break
        if stalled >= patience:
            stop_reason = "early_stopping"
            break
    if best_epoch is None:
        raise RuntimeError("No updated, validated checkpoint was produced")
    summary = {**metadata, "checkpoint_path": str(best_path.resolve()), "best_epoch": best_epoch,
               "best_validation_macro_log_rmse": float(best_score), "baseline_validation": baseline,
               "validation_preferred_model": "tabpfn_finetuned" if best_score < baseline["macro_log_rmse"] else "tabpfn",
               "gradient_steps": adapter.steps, "epochs_evaluated": len(history), "stop_reason": stop_reason,
               "development_run": max_steps is not None, "elapsed_seconds": perf_counter() - started,
               "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()) if adapter.use_amp else None,
               "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()) if adapter.use_amp else None,
               "device": adapter.device, "torch_version": torch.__version__,
               "query_profiles_per_batch": query_limit, "config": config}
    write_json(output_dir / "tabpfn_training_summary.json", summary)
    return summary
