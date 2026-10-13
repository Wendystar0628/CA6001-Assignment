"""Spatial TabPFN v2: native distributions and site-balanced task adaptation.

The differentiable preprocessing/forward path is inherited from the v0.1
adapter pinned to the installed official TabPFN 9.2.0 implementation. Target
thresholds follow data_util.py's support z-score and the fitted per-estimator
target transform, in that order. Inference uses the official raw-space CDF.
"""
from __future__ import annotations

import copy
import gc
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from cpt_budget.config import write_json
from cpt_budget.evaluate import evaluate, join_evaluation
from cpt_budget.models.distribution import QUANTILES, distribution_prediction_frame
from cpt_budget.models.tabpfn_adapter import (
    EpisodeAdapter, decode_full_output, estimator_options, geometry_only,
    make_estimator,
)
from cpt_budget.models.tabpfn_train import _check_episode, _ids, _records
from cpt_budget.types import PredictionBundle
from .features import FEATURE_COLUMNS_V2, build_features_v2


def feature_array_v2(observed, query, *, leave_location_out=False):
    frame = build_features_v2(observed, query, leave_location_out=leave_location_out)
    if list(frame.columns) != list(FEATURE_COLUMNS_V2):
        raise ValueError("Feature order differs from the spatial v2 schema")
    values = frame.to_numpy(dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError("Features may contain NaN but cannot contain infinity")
    return values


class SpatialTabPFNPredictor:
    def __init__(self, config=None, artifact=None, model_id="tabpfn_spatial"):
        self.config = config or {}
        self.artifact = Path(artifact) if artifact is not None else None
        self.model_id = model_id
        self._estimator = None
        self._observed = None
        self.query_profiles_per_batch = min(8, int(
            self.config.get("episodes", {}).get("query_profiles_per_batch", 8)))
        if self.query_profiles_per_batch < 1:
            raise ValueError("query_profiles_per_batch must be positive")

    def condition(self, observed):
        if observed.empty or observed.site_id.nunique() != 1:
            raise ValueError("A context must contain observations from exactly one site")
        if not np.isfinite(observed.log_qc).all():
            raise ValueError("Context log_qc must be finite")
        self._observed = observed.copy()
        if self._estimator is None:
            self._estimator = make_estimator(self.config, self.artifact)
        X = feature_array_v2(observed, geometry_only(observed), leave_location_out=True)
        self._estimator.fit(X, observed.log_qc.to_numpy(dtype=np.float64))
        for model in self._estimator.models_:
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    def predict(self, query_grid):
        if self._observed is None:
            raise RuntimeError("condition() must be called before predict()")
        query = geometry_only(query_grid).reset_index(drop=True)
        if not query.empty and set(query.site_id) != set(self._observed.site_id):
            raise ValueError("Query and context must belong to the same site")
        if query.empty:
            empty = distribution_prediction_frame(query, np.empty((0, 7)), np.empty((0, 3)))
            return PredictionBundle(empty, self.model_id)
        X = feature_array_v2(self._observed, query)
        ids = list(query.location_id.drop_duplicates())
        parts, position = [], 0
        batch_size = self.query_profiles_per_batch
        import torch
        while position < len(ids):
            selected = ids[position:position + batch_size]
            row_indices = np.flatnonzero(query.location_id.isin(selected).to_numpy())
            try:
                with torch.no_grad():
                    full = self._estimator.predict(
                        X[row_indices], output_type="full", quantiles=list(QUANTILES))
                piece = decode_full_output(full, query.iloc[row_indices])
            except torch.OutOfMemoryError:
                if batch_size <= 1:
                    raise
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
                continue
            piece["_row_order"] = row_indices
            parts.append(piece)
            position += len(selected)
        frame = pd.concat(parts).sort_values("_row_order").drop(columns="_row_order").reset_index(drop=True)
        return PredictionBundle(frame, self.model_id)


def event_thresholds_in_estimator_space(y_support, configs, *, log_threshold=0.0):
    """Return one transformed physical threshold per fitted estimator.

    y_support is in the input label space ln(qc/MPa), NOT the standardized
    support targets in RegressorBatch.y_context. Standard deviation is the
    population statistic (ddof=0), as in official data_util.py.
    """
    import torch
    from tabpfn.finetuning.finetuned_regressor import _targets_in_estimator_space
    support = np.asarray(y_support, dtype=np.float64)
    if support.size < 2 or not np.isfinite(support).all():
        raise ValueError("The physical event needs finite input-space support labels")
    mean, scale = float(support.mean()), max(float(support.std()), 1e-8)
    standardized = torch.tensor([(float(log_threshold) - mean) / scale], dtype=torch.float32)
    thresholds = torch.stack([
        _targets_in_estimator_space(standardized, member[0]).reshape(())
        for member in configs
    ])
    if not torch.isfinite(thresholds).all():
        raise ValueError("Non-finite physical threshold after target preprocessing")
    return thresholds


def native_event_brier(logits_BQL, bardist, thresholds_B, raw_query, *, log_threshold=0.0):
    """Differentiable CDF Brier using transformed thresholds and raw events."""
    import torch
    thresholds = thresholds_B.to(device=logits_BQL.device, dtype=logits_BQL.dtype)
    thresholds = thresholds[:, None, None].expand(-1, logits_BQL.shape[1], 1)
    probabilities = bardist.cdf(logits_BQL, thresholds).squeeze(-1)
    labels = torch.as_tensor(np.asarray(raw_query) < log_threshold,
                             device=logits_BQL.device, dtype=logits_BQL.dtype)
    return (probabilities - labels[None, :]).square().mean()


class SpatialEpisodeAdapter(EpisodeAdapter):
    """Official CRPS + mean MSE, with an optional physical low-qc objective."""
    def __init__(self, config=None, *, variant="crps"):
        if variant not in {"crps", "task"}:
            raise ValueError("variant must be 'crps' or 'task'")
        adapted = copy.deepcopy(config or {})
        adapted.setdefault("tabpfn", {}).update({
            key: value for key, value in adapted.get("v2", {}).get("training", {}).items()
            if key in {"learning_rate", "weight_decay", "gradient_clip"}
        })
        super().__init__(adapted)
        self.variant = variant
        self.model_id = f"tabpfn_{variant}"

    def step(self, X_support, y_support, X_query, y_query, *, seed=0):
        import torch
        from tabpfn.architectures.interface import PerformanceOptions
        from tabpfn.finetuning._torch_compat import autocast, sdpa_kernel_context
        from tabpfn.finetuning.finetuned_regressor import _compute_regression_loss, _targets_in_estimator_space
        started = perf_counter()
        batch = self.preprocess(X_support, y_support, X_query, y_query, seed=seed)
        self.estimator.raw_space_bardist_ = batch.raw_space_bardist
        self.estimator.znorm_space_bardist_ = batch.znorm_space_bardist
        self.estimator.fit_from_preprocessed(
            batch.X_context, batch.y_context, batch.cat_indices, batch.configs,
            performance_options=PerformanceOptions(force_recompute_layer=True, use_chunkwise_inference=False),
            no_refit=True,
        )
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=self.use_amp), sdpa_kernel_context():
            _, per_estimator_logits, _ = self.estimator.forward(batch.X_query, use_inference_mode=False)
            logits_QBEL = torch.stack(per_estimator_logits, dim=2)
            Q, B, E, L = logits_QBEL.shape
            if B != 1 or E != 1 or Q != len(y_query):
                raise RuntimeError("Official forward violated single-episode dimensions")
            # Use float32 for CDF/mean losses even when the model forward uses AMP.
            logits = logits_QBEL.permute(1, 2, 0, 3).reshape(B * E, Q, L).float()
            targets = torch.stack([
                _targets_in_estimator_space(batch.y_query[0], c[0]) for c in batch.configs
            ]).to(self.device)
            bardist = batch.znorm_space_bardist.to(device=logits.device, dtype=logits.dtype)
            regression = _compute_regression_loss(
                logits_BQL=logits, targets_BQ=targets, bardist_loss_fn=bardist,
                ce_loss_weight=0., crps_loss_weight=1., crls_loss_weight=0.,
                mse_loss_weight=1., mae_loss_weight=0.,
            )
            thresholds = event_thresholds_in_estimator_space(y_support, batch.configs)
            brier = native_event_brier(logits, bardist, thresholds, y_query)
            loss = regression + (0.5 * brier if self.variant == "task" else 0.)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite spatial episode loss")
        with sdpa_kernel_context():
            (self.scaler.scale(loss) if self.scaler is not None else loss).backward()
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.settings.get("gradient_clip", 1.0)))
        if not torch.isfinite(norm) or norm.item() <= 0:
            raise RuntimeError(f"Episode has invalid gradient norm: {norm.item()}")
        parameter = next(p for p in self.model.parameters() if p.grad is not None and torch.any(p.grad != 0))
        before = parameter.detach().clone()
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        delta = torch.linalg.vector_norm(parameter.detach() - before).item()
        if delta <= 0:
            raise RuntimeError("No parameter update occurred; do not call this fine-tuning")
        self.steps += 1
        return {"loss": float(loss.detach().cpu()), "crps_plus_mse": float(regression.detach().cpu()),
                "native_event_brier": float(brier.detach().cpu()), "event_threshold_estimator_space": thresholds.tolist(),
                "gradient_norm_before_clip": float(norm.detach().cpu()), "updated_parameter_delta_l2": delta,
                "seconds": perf_counter() - started, "support_rows": len(y_support),
                "query_rows": len(y_query), "step": self.steps}

    def predictor(self):
        from tabpfn import TabPFNRegressor
        from tabpfn.finetuning.train_util import clone_model_for_evaluation
        result = SpatialTabPFNPredictor(self.config, model_id=self.model_id)
        result._estimator = clone_model_for_evaluation(
            self.estimator, {**estimator_options(self.config), "fit_mode": "fit_preprocessors"}, TabPFNRegressor)
        return result

    def save(self, path, metadata):
        from tabpfn.model_loading import save_tabpfn_model
        if not self.steps:
            raise RuntimeError("Refusing to save unchanged pretrained weights as fine-tuned")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_tabpfn_model(self.estimator, path, additional_fields={"cpt_budget": {
            **metadata, "feature_columns": list(FEATURE_COLUMNS_V2),
            "gradient_steps": self.steps, "model_id": self.model_id, "variant": self.variant}})
        return path


def query_blocks(episode, config):
    """Fixed non-overlapping complete-profile chunks; never truncate support."""
    options = config.get("episodes", {})
    max_rows = min(512, int(options.get("max_rows", 512)))
    support_rows = len(_ids(episode["support_location_ids"])) * 24
    limit = min(8, int(options.get("query_profiles_per_batch", 8)), (max_rows - support_rows) // 24)
    if limit < 1:
        raise ValueError("The intact support leaves no room for a complete query CPT")
    query = _ids(episode["query_location_ids"])
    return [query[start:start + limit] for start in range(0, len(query), limit)]


def build_spatial_episode(dataset, episode, query_ids=None):
    support, allowed = _check_episode(dataset, episode)
    query = allowed if query_ids is None else [str(i) for i in query_ids]
    if not query or not set(query).issubset(allowed) or len(set(query)) != len(query):
        raise ValueError("Query block must contain unique locations from its fixed episode")
    observed = dataset.observed(support)
    geometry = dataset.query(query)
    truth = dataset.truth(query)
    for name, frame in (("support", observed), ("query", geometry)):
        if not frame.groupby("location_id").size().eq(24).all() or frame.duplicated(["location_id", "depth_index"]).any():
            raise ValueError(f"Each {name} CPT must contain exactly 24 distinct depth cells")
    labels = geometry[["location_id", "depth_index"]].merge(
        truth[["location_id", "depth_index", "log_qc"]], on=["location_id", "depth_index"],
        validate="one_to_one", how="left").log_qc.to_numpy(dtype=np.float64)
    if not np.isfinite(labels).all():
        raise ValueError("Query loss labels are missing or non-finite")
    return {"X_support": feature_array_v2(observed, geometry_only(observed), leave_location_out=True),
            "y_support": observed.log_qc.to_numpy(dtype=np.float64),
            "X_query": feature_array_v2(observed, geometry), "y_query": labels,
            "support_location_ids": support, "query_location_ids": query}


def make_training_plan(episodes, config):
    """Round-robin episodes within sites, then blocks, keeping site weights equal."""
    grouped = {}
    for index, episode in enumerate(_records(episodes)):
        item = {**episode, "_episode_key": str(episode.get("episode_id", f"episode_{index}"))}
        grouped.setdefault(str(episode["site_id"]), []).append(item)
    result = {}
    for site, members in sorted(grouped.items()):
        members = sorted(members, key=lambda e: e["_episode_key"])
        chunks = [query_blocks(e, config) for e in members]
        result[site] = [(episode, blocks[b], f"{episode['_episode_key']}:{b}")
                        for b in range(max(map(len, chunks)))
                        for episode, blocks in zip(members, chunks) if b < len(blocks)]
    return result


def _optional_mean(values):
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def validate_spatial(adapter, dataset, episodes, config):
    """Macro over sites/scenarios/CPTs, using official inference and native CDF."""
    predictor = adapter.predictor()
    rows = []
    try:
        for episode in episodes:
            support, query = _check_episode(dataset, episode)
            predictor.condition(dataset.observed(support))
            predictions = predictor.predict(dataset.query(query))
            truth = dataset.truth(query)
            metrics = evaluate(predictions, truth, config)
            joined = join_evaluation(predictions, truth)
            joined["event_brier"] = (joined.p_low_1 - (joined.truth_qc_mpa < 1.).astype(float)) ** 2
            brier = float(joined.groupby("location_id").event_brier.mean().mean())
            rows.append({"site_id": str(episode["site_id"]), "episode_id": episode.get("episode_id"),
                         "log_mae": metrics["log_mae"], "native_event_brier": brier,
                         "score": metrics["log_mae"] + .5 * brier,
                         "h_point_mae_m": metrics["h_point_mae_m"], "recall": metrics["recall"],
                         "false_positive_rate": metrics["false_positive_rate"]})
        names = ["score", "log_mae", "native_event_brier", "h_point_mae_m", "recall", "false_positive_rate"]
        site_metrics = {site: {name: _optional_mean([r[name] for r in rows if r["site_id"] == site]) for name in names}
                        for site in sorted({r["site_id"] for r in rows})}
        macro = {f"macro_{name}": _optional_mean([values[name] for values in site_metrics.values()]) for name in names}
        if macro["macro_score"] is None or not np.isfinite(macro["macro_score"]):
            raise RuntimeError("Validation must yield finite scores for held-out sites")
        return {**macro, "site_metrics": site_metrics, "episode_metrics": rows,
                "selection_metric": "mean_site(mean_episode(log_mae + 0.5 * native_event_brier))",
                "event_threshold_mpa": 1., "risk_calibration_used": False}
    finally:
        del predictor
        gc.collect()
        if adapter.use_amp:
            import torch
            torch.cuda.empty_cache()


def fit_spatial_tabpfn(dataset, train_episodes, validation_episodes, config, output_dir, *, variant="crps"):
    """Train an exploratory fold under an explicit <=450 update budget.

    Each round gives every source site one update. Per-site tasks rotate through
    every fixed (episode, disjoint query block) before repeating. Checkpoint
    selection may occur before complete exposure, but early stopping may not.
    The original pretrained validation is retained separately, even if superior.
    """
    train, validation = _records(train_episodes), _records(validation_episodes)
    if not train or not validation:
        raise ValueError("Fine-tuning requires nonempty train and validation episodes")
    if any(e.get("role", "train") != "train" for e in train):
        raise ValueError("Only train-role episodes may update the weights")
    if any(e.get("role", "validation") != "validation" for e in validation):
        raise ValueError("Only validation-role episodes may select checkpoints")
    train_sites, validation_sites = {str(e["site_id"]) for e in train}, {str(e["site_id"]) for e in validation}
    if train_sites & validation_sites:
        raise ValueError("Training and validation sites overlap")
    folds = {int(e["fold_id"]) for e in train + validation if "fold_id" in e}
    if len(folds) > 1:
        raise ValueError("A checkpoint belongs to exactly one outer fold")
    if "investigation_group" in dataset.sites:
        groups = dataset.sites.set_index("site_id").investigation_group
        if set(groups.loc[list(train_sites)]) & set(groups.loc[list(validation_sites)]):
            raise ValueError("Training and validation investigation groups overlap")
    for episode in train + validation:
        _check_episode(dataset, episode)
    if float(config.get("data", {}).get("tau_mpa", 1.)) != 1.:
        raise ValueError("The v2 physical event objective is fixed to 1 MPa")
    settings = {**config.get("tabpfn", {}), **config.get("v2", {}).get("training", {})}
    max_steps = min(450, int(settings.get("max_steps", 450)))
    validation_every = int(settings.get("validation_every", settings.get("validation_every_steps", 50)))
    patience = int(settings.get("early_stopping_patience", 3))
    if min(max_steps, validation_every, patience) < 1:
        raise ValueError("Training budgets, validation interval and patience must be positive")
    plan = make_training_plan(train, config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / f"tabpfn_{variant}.ckpt"
    history_path = output_dir / f"tabpfn_{variant}_training_history.json"
    model_id = f"tabpfn_{variant}"
    metadata = {"fold_id": next(iter(folds), None), "model_id": model_id, "variant": variant,
                "train_sites": sorted(train_sites), "validation_sites": sorted(validation_sites),
                "model_generation": "v2", "package_version": "9.2.0", "n_estimators": 1,
                "feature_columns": list(FEATURE_COLUMNS_V2), "study": "exploratory_v2",
                "loss": "official_crps_plus_mse" + ("_plus_0.5_native_event_brier" if variant == "task" else ""),
                "event_threshold_mpa": 1., "target_threshold_transform": "support_zscore_then_fitted_estimator_target_transform",
                "site_weighting": "one_update_per_site_per_round", "query_rotation": "fixed_disjoint_complete_CPT_blocks",
                "max_steps": max_steps, "validation_every_steps": validation_every,
                "source_commit": config.get("_source_commit", config.get("source_commit")),
                "early_stopping_requires_complete_query_exposure": True}
    started = perf_counter()
    adapter = SpatialEpisodeAdapter(config, variant=variant)
    import torch
    if adapter.use_amp:
        torch.cuda.reset_peak_memory_stats()
    baseline = validate_spatial(adapter, dataset, validation, config)
    write_json(output_dir / f"tabpfn_{variant}_pretrained_validation.json", baseline)
    seen = {site: set() for site in plan}
    site_updates = dict.fromkeys(plan, 0)
    history, steps = [], []
    best_score, best_step, stalled = np.inf, None, 0
    best_exposure = None
    stop_reason, round_index = "max_steps", 0

    def exposure():
        per_site = {site: {"seen_blocks": len(seen[site]), "total_blocks": len(tasks),
                           "block_fraction": len(seen[site]) / len(tasks), "updates": site_updates[site]}
                    for site, tasks in plan.items()}
        seen_count = sum(len(v) for v in seen.values())
        total_count = sum(len(v) for v in plan.values())
        return {"seen_blocks": seen_count, "total_blocks": total_count,
                "query_block_fraction": seen_count / total_count,
                "site_macro_query_block_fraction": float(np.mean([v["block_fraction"] for v in per_site.values()])),
                "all_query_blocks_seen": seen_count == total_count, "per_site": per_site}

    while adapter.steps < max_steps:
        site_order = sorted(plan)
        np.random.default_rng(int(config.get("split", {}).get("seed", 20261013)) + round_index).shuffle(site_order)
        for site in site_order:
            episode, block, key = plan[site][site_updates[site] % len(plan[site])]
            payload = build_spatial_episode(dataset, episode, block)
            # Do not silently drop query rows or redraw blocks on OOM: that would
            # invalidate fixed exposure bookkeeping. A smaller block is a new run.
            step = adapter.step(**{name: payload[name] for name in ("X_support", "y_support", "X_query", "y_query")},
                                seed=int(episode.get("seed", 0)))
            seen[site].add(key)
            site_updates[site] += 1
            steps.append({**step, "round": round_index + 1, "site_id": site,
                          "episode_id": episode.get("episode_id"), "query_block_key": key,
                          "support_location_ids": payload["support_location_ids"], "query_location_ids": block})
            if adapter.steps % validation_every == 0 or adapter.steps == max_steps:
                scores = validate_spatial(adapter, dataset, validation, config)
                current_exposure = exposure()
                history.append({"step": adapter.steps, **scores, "exposure": current_exposure})
                if scores["macro_score"] < best_score:
                    best_score, best_step, stalled = scores["macro_score"], adapter.steps, 0
                    best_exposure = current_exposure
                    adapter.save(best_path, {**metadata, "best_step": best_step, "validation": scores,
                                             "checkpoint_exposure": current_exposure})
                else:
                    stalled += 1
                write_json(history_path, {"validation": history, "steps": steps,
                                          "pretrained_validation": baseline, "metadata": metadata})
                if current_exposure["all_query_blocks_seen"] and stalled >= patience:
                    stop_reason = "early_stopping_after_complete_exposure"
                    break
            if adapter.steps >= max_steps:
                break
        if stop_reason != "max_steps":
            break
        round_index += 1
    if best_step is None:
        raise RuntimeError("No updated, validated checkpoint was produced")
    summary = {**metadata, "checkpoint_path": str(best_path.resolve()), "history_path": str(history_path.resolve()),
               "best_step": best_step, "best_validation_score": float(best_score),
               "selection_metric": baseline["selection_metric"], "baseline_validation": baseline,
               "validation_preferred_model": model_id if best_score < baseline["macro_score"] else "tabpfn_spatial",
               "gradient_steps": adapter.steps, "actual_steps": adapter.steps, "exposure": exposure(),
               "best_checkpoint_exposure": best_exposure, "stop_reason": stop_reason,
               "development_run": max_steps < 450, "elapsed_seconds": perf_counter() - started,
               "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()) if adapter.use_amp else None,
               "device": adapter.device, "torch_version": torch.__version__, "config": config}
    write_json(output_dir / f"tabpfn_{variant}_training_summary.json", summary)
    return summary
