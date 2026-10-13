"""Task-aligned TabPFN residual training on source-only CPT episodes.

Support labels are whole-location-out IDW residuals. Query residuals are
relative to IDW fitted on the revealed support, exactly as at deployment.
Physical low-qc events have a different residual threshold at every query;
they are never approximated by the event ``residual < 0``.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np

from cpt_budget.config import write_json
from cpt_budget.diagnostics.residual_adapter import (
    BaselineResidualTabPFNPredictor, residual_context,
)
from cpt_budget.models.idw import IDWPredictor
from cpt_budget.models.tabpfn_adapter import EpisodeAdapter, estimator_options
from cpt_budget.models.tabpfn_train import _check_episode, _records
from cpt_budget.v2.features import FEATURE_COLUMNS_V2
from cpt_budget.v2.tabpfn import (
    feature_array_v2, optimizer_step_with_recovery, query_blocks, validate_spatial,
)


def residual_event_thresholds_in_estimator_space(
    support_residual, configs, query_baseline_log_qc, *, log_threshold=0.0,
):
    """Return [estimator, query] physical-event thresholds in fitted space.

    R = log(qc) - IDW, so qc < tau iff R < log(tau) - IDW(query).
    The official preprocessing first z-scores support labels (ddof=0), then
    applies each member's fitted target transform. Thresholds follow both.
    """
    import torch
    from tabpfn.finetuning.finetuned_regressor import _targets_in_estimator_space

    support = np.asarray(support_residual, dtype=np.float64).reshape(-1)
    baseline = np.asarray(query_baseline_log_qc, dtype=np.float64).reshape(-1)
    if (len(support) < 2 or not len(baseline) or not np.isfinite(support).all()
            or not np.isfinite(baseline).all() or not np.isfinite(log_threshold)):
        raise ValueError("Residual event thresholds need finite support and query baselines")
    raw_threshold = float(log_threshold) - baseline
    standardized = torch.as_tensor(
        (raw_threshold - support.mean()) / max(float(support.std()), 1e-8),
        dtype=torch.float32,
    )
    thresholds = torch.stack([
        _targets_in_estimator_space(standardized, member[0]) for member in configs
    ])
    if thresholds.ndim != 2 or not torch.isfinite(thresholds).all():
        raise ValueError("Residual physical thresholds became invalid after target transform")
    return thresholds


def residual_native_event_brier(logits_BQL, bardist, thresholds_BQ, raw_query_log_qc,
                                *, log_threshold=0.0):
    """Differentiable physical Brier with strictly row-dependent CDF evaluation."""
    import torch

    if logits_BQL.ndim != 3 or tuple(thresholds_BQ.shape) != tuple(logits_BQL.shape[:2]):
        raise ValueError("Thresholds must have exactly one entry per estimator/query row")
    labels_raw = np.asarray(raw_query_log_qc, dtype=np.float64).reshape(-1)
    if len(labels_raw) != logits_BQL.shape[1] or not np.isfinite(labels_raw).all():
        raise ValueError("Physical Brier needs one finite raw log-qc target per query")
    thresholds = thresholds_BQ.to(device=logits_BQL.device, dtype=logits_BQL.dtype)
    # FullSupportBarDistribution.cdf requires ys.shape[:-1] == logits.shape[:-1].
    probabilities = bardist.cdf(logits_BQL, thresholds.unsqueeze(-1)).squeeze(-1)
    labels = torch.as_tensor(labels_raw < log_threshold,
                             device=logits_BQL.device, dtype=logits_BQL.dtype)
    return (probabilities - labels.unsqueeze(0)).square().mean()


def build_residual_episode(dataset, episode, query_ids=None, *, config=None):
    """Build revealed-only features/baselines; attach query truth solely as labels."""
    support, allowed = _check_episode(dataset, episode)
    query = allowed if query_ids is None else [str(value) for value in query_ids]
    if not query or not set(query).issubset(allowed) or len(set(query)) != len(query):
        raise ValueError("Query block must contain unique locations from its source episode")
    observed, geometry = dataset.observed(support), dataset.query(query)
    for name, frame in (("support", observed), ("query", geometry)):
        if (not frame.groupby("location_id").size().eq(24).all()
                or frame.duplicated(["location_id", "depth_index"]).any()):
            raise ValueError(f"Each {name} CPT must contain exactly 24 distinct depth cells")
    context = residual_context(observed, config)
    baseline = IDWPredictor(config)
    baseline.condition(observed)
    reference = baseline.predict(geometry).frame.pred_log_qc.to_numpy(dtype=np.float64)
    X_query = feature_array_v2(observed, geometry)
    truth = dataset.truth(query)
    raw_targets = geometry[["location_id", "depth_index"]].merge(
        truth[["location_id", "depth_index", "log_qc"]],
        on=["location_id", "depth_index"], validate="one_to_one", how="left",
    ).log_qc.to_numpy(dtype=np.float64)
    if not np.isfinite(raw_targets).all() or not np.isfinite(reference).all():
        raise ValueError("Query labels and revealed IDW reference must be finite")
    return {
        "X_support": context["X_support"], "y_support": context["y_residual"],
        "X_query": X_query, "y_query": raw_targets - reference,
        "query_baseline_log_qc": reference, "raw_query_log_qc": raw_targets,
        "support_baseline_log_qc": context["loo_baseline_log_qc"],
        "support_location_ids": support, "query_location_ids": query,
    }


def _site_seed(seed, site):
    digest = hashlib.sha256(f"{int(seed)}:{site}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def make_residual_training_plan(episodes, config):
    """Seeded episode/block ordering per site; complete CPT blocks stay intact."""
    grouped = {}
    for index, episode in enumerate(_records(episodes)):
        item = {**episode, "_episode_key": str(episode.get("episode_id", f"episode_{index}"))}
        grouped.setdefault(str(item["site_id"]), []).append(item)
    seed = int(config.get("v4", {}).get("training", {}).get(
        "seed", config.get("split", {}).get("seed", 20261014)))
    result = {}
    for site, members in sorted(grouped.items()):
        if len({member["_episode_key"] for member in members}) != len(members):
            raise ValueError("Episode IDs must be unique within each source site")
        rng = np.random.default_rng(_site_seed(seed, site))
        members = sorted(members, key=lambda member: member["_episode_key"])
        rng.shuffle(members)
        member_blocks = []
        for member in members:
            blocks = list(enumerate(query_blocks(member, config)))
            rng.shuffle(blocks)
            member_blocks.append(blocks)
        # Interleave scenarios so a large query pool does not delay every other
        # budget/trajectory until all of that pool's chunks have been consumed.
        result[site] = [
            (member, blocks[index][1], f"{member['_episode_key']}:{blocks[index][0]}")
            for index in range(max(map(len, member_blocks)))
            for member, blocks in zip(members, member_blocks) if index < len(blocks)
        ]
    return result


class ResidualEpisodeAdapter(EpisodeAdapter):
    """Official differentiable TabPFN, trained in IDW residual target space."""

    def __init__(self, config=None, *, variant="crps"):
        if variant not in {"crps", "task"}:
            raise ValueError("variant must be 'crps' or 'task'")
        adapted = copy.deepcopy(config or {})
        adapted.setdefault("tabpfn", {}).update({
            key: value for key, value in adapted.get("v4", {}).get("training", {}).items()
            if key in {"learning_rate", "weight_decay", "gradient_clip"}
        })
        if float(adapted.get("v4", {}).get("training", {}).get("event_brier_weight", .5)) != .5:
            raise ValueError("The registered physical event Brier weight is 0.5")
        super().__init__(adapted)
        self.variant = variant
        self.model_id = f"tabpfn_residual_{variant}"

    def step(self, X_support, y_support, X_query, y_query, *, query_baseline_log_qc,
             raw_query_log_qc=None, seed=0):
        import torch
        from tabpfn.architectures.interface import PerformanceOptions
        from tabpfn.finetuning._torch_compat import autocast, sdpa_kernel_context
        from tabpfn.finetuning.finetuned_regressor import (
            _compute_regression_loss, _targets_in_estimator_space,
        )

        started = perf_counter()
        residual = np.asarray(y_query, dtype=np.float64).reshape(-1)
        reference = np.asarray(query_baseline_log_qc, dtype=np.float64).reshape(-1)
        raw_targets = residual + reference if raw_query_log_qc is None else np.asarray(
            raw_query_log_qc, dtype=np.float64).reshape(-1)
        if (len(reference) != len(residual) or len(raw_targets) != len(residual)
                or not np.isfinite(reference).all() or not np.isfinite(raw_targets).all()
                or not np.allclose(residual + reference, raw_targets, rtol=1e-10, atol=1e-10)):
            raise ValueError("Query residual + revealed baseline must reconstruct raw log qc")
        batch = self.preprocess(X_support, y_support, X_query, y_query, seed=seed)
        self.estimator.raw_space_bardist_ = batch.raw_space_bardist
        self.estimator.znorm_space_bardist_ = batch.znorm_space_bardist
        thresholds = residual_event_thresholds_in_estimator_space(y_support, batch.configs, reference)
        self.model.train()
        original_autocast = self.estimator.use_autocast_

        def compute_losses(amp_enabled):
            self.estimator.use_autocast_ = amp_enabled
            self.estimator.fit_from_preprocessed(
                batch.X_context, batch.y_context, batch.cat_indices, batch.configs,
                performance_options=PerformanceOptions(
                    force_recompute_layer=True, use_chunkwise_inference=False),
                no_refit=True,
            )
            with autocast(enabled=amp_enabled), sdpa_kernel_context():
                _, per_estimator_logits, _ = self.estimator.forward(batch.X_query, use_inference_mode=False)
                logits_QBEL = torch.stack(per_estimator_logits, dim=2)
                Q, B, E, L = logits_QBEL.shape
                if B != 1 or E != 1 or Q != len(y_query):
                    raise RuntimeError("Official forward violated single residual episode dimensions")
                logits = logits_QBEL.permute(1, 2, 0, 3).reshape(B * E, Q, L).float()
                targets = torch.stack([
                    _targets_in_estimator_space(batch.y_query[0], member[0]) for member in batch.configs
                ]).to(self.device)
                bardist = batch.znorm_space_bardist.to(device=logits.device, dtype=logits.dtype)
                regression = _compute_regression_loss(
                    logits_BQL=logits, targets_BQ=targets, bardist_loss_fn=bardist,
                    ce_loss_weight=0., crps_loss_weight=1., crls_loss_weight=0.,
                    mse_loss_weight=1., mae_loss_weight=0.,
                )
                brier = residual_native_event_brier(logits, bardist, thresholds, raw_targets)
                loss = regression + (.5 * brier if self.variant == "task" else 0.)
            return {"loss": loss, "regression": regression, "brier": brier}

        try:
            losses, update = optimizer_step_with_recovery(
                compute_losses, model=self.model, optimizer=self.optimizer, scaler=self.scaler,
                use_amp=self.use_amp, gradient_clip=float(self.settings.get("gradient_clip", 1.0)),
            )
        finally:
            self.estimator.use_autocast_ = original_autocast
        self.steps += 1
        return {
            "loss": float(losses["loss"].detach().cpu()),
            "crps_plus_mse": float(losses["regression"].detach().cpu()),
            "native_event_brier": float(losses["brier"].detach().cpu()),
            "event_residual_threshold_min": float((-reference).min()),
            "event_residual_threshold_max": float((-reference).max()),
            "event_threshold_estimator_min": float(thresholds.min()),
            "event_threshold_estimator_max": float(thresholds.max()),
            **update, "seconds": perf_counter() - started,
            "support_rows": len(y_support), "query_rows": len(y_query), "step": self.steps,
        }

    def predictor(self):
        from tabpfn import TabPFNRegressor
        from tabpfn.finetuning.train_util import clone_model_for_evaluation

        result = BaselineResidualTabPFNPredictor(self.config, model_id=self.model_id, alpha=1.0)
        result._estimator = clone_model_for_evaluation(
            self.estimator, {**estimator_options(self.config), "fit_mode": "fit_preprocessors"},
            TabPFNRegressor,
        )
        return result

    def save(self, path, metadata):
        from tabpfn.model_loading import save_tabpfn_model

        if not self.steps:
            raise RuntimeError("Refusing to label unchanged pretrained weights as fine-tuned")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_tabpfn_model(self.estimator, path, additional_fields={"cpt_budget": {
            **metadata, "feature_columns": list(FEATURE_COLUMNS_V2),
            "gradient_steps": self.steps, "model_id": self.model_id, "variant": self.variant,
            "target": "log_qc_minus_revealed_idw", "context_target": "whole_location_out_idw_residual",
        }})
        return path


def validate_residual(adapter, dataset, episodes, config):
    """Score reconstructed physical predictions, with equal weight per site."""
    scores = validate_spatial(adapter, dataset, _records(episodes), config)
    return {**scores, "prediction_space": "reconstructed_log_qc_and_native_physical_cdf",
            "residual_alpha_for_checkpoint_selection": 1.0}


def fit_residual_tabpfn(dataset, train_episodes, validation_episodes, config, output_dir,
                       *, variant="crps"):
    """Train <=600 successful updates; report exposure without calling updates epochs."""
    train, validation = _records(train_episodes), _records(validation_episodes)
    if not train or not validation:
        raise ValueError("Residual training requires nonempty train and validation episodes")
    if any(episode.get("role", "train") != "train" for episode in train):
        raise ValueError("Only train-role episodes may update weights")
    if any(episode.get("role", "validation") != "validation" for episode in validation):
        raise ValueError("Only validation-role episodes may select checkpoints")
    train_sites = {str(episode["site_id"]) for episode in train}
    validation_sites = {str(episode["site_id"]) for episode in validation}
    if train_sites & validation_sites:
        raise ValueError("Training and validation sites overlap")
    folds = {int(episode["fold_id"]) for episode in train + validation if "fold_id" in episode}
    if len(folds) > 1:
        raise ValueError("A residual checkpoint belongs to exactly one outer fold")
    if "investigation_group" in dataset.sites:
        groups = dataset.sites.set_index("site_id").investigation_group
        if set(groups.loc[list(train_sites)]) & set(groups.loc[list(validation_sites)]):
            raise ValueError("Training and validation investigation groups overlap")
    for episode in train + validation:
        _check_episode(dataset, episode)
    if float(config.get("data", {}).get("tau_mpa", 1.)) != 1.:
        raise ValueError("The registered physical event objective is fixed to 1 MPa")
    settings = {**config.get("tabpfn", {}), **config.get("v4", {}).get("training", {})}
    max_steps = int(settings.get("max_steps", 600))
    validation_every = int(settings.get("validation_every", 100))
    min_steps = max(200, int(settings.get("min_steps", 200)))
    patience = int(settings.get("early_stopping_patience", settings.get("patience", 3)))
    if not 1 <= max_steps <= 600 or min(validation_every, patience) < 1:
        raise ValueError("Use 1..600 successful updates and positive validation interval/patience")
    if float(settings.get("selection_brier_weight", .5)) != .5:
        raise ValueError("The registered checkpoint selection Brier weight is 0.5")
    seed = int(settings.get("seed", config.get("split", {}).get("seed", 20261014)))
    plan = make_residual_training_plan(train, config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_id = f"tabpfn_residual_{variant}"
    best_path = output_dir / f"{model_id}.ckpt"
    history_path = output_dir / f"{model_id}_training_history.json"
    metadata = {
        "fold_id": next(iter(folds), None), "model_id": model_id, "variant": variant,
        "train_sites": sorted(train_sites), "validation_sites": sorted(validation_sites),
        "model_generation": "v2", "package_version": "9.2.0", "n_estimators": 1,
        "feature_columns": list(FEATURE_COLUMNS_V2), "study": "exploratory_v4_residual",
        "loss": "official_crps_plus_mse" + ("_plus_0.5_physical_native_event_brier" if variant == "task" else ""),
        "event_threshold_mpa": 1., "support_target": "log_qc_minus_location_out_idw",
        "query_target": "log_qc_minus_revealed_support_idw",
        "target_threshold_transform": "query_negative_idw_then_support_residual_zscore_then_fitted_target_transform",
        "site_weighting": "one_successful_update_per_site_per_round",
        "query_rotation": "seeded_scenario_and_disjoint_complete_CPT_block_order",
        "training_seed": seed, "max_steps": max_steps, "validation_every_steps": validation_every,
        "min_steps_before_early_stopping": min_steps,
        "source_commit": config.get("_source_commit", config.get("source_commit")),
        "early_stopping_requires_complete_query_exposure": False,
    }
    started = perf_counter()
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    adapter = ResidualEpisodeAdapter(config, variant=variant)
    if adapter.use_amp:
        torch.cuda.reset_peak_memory_stats()
    baseline = validate_residual(adapter, dataset, validation, config)
    write_json(output_dir / f"{model_id}_pretrained_validation.json", baseline)
    seen_blocks, seen_episodes = ({site: set() for site in plan} for _ in range(2))
    site_updates = dict.fromkeys(plan, 0)
    history, steps = [], []
    best_score, best_step, stalled, best_exposure = np.inf, None, 0, None
    stop_reason, round_index = "max_steps", 0

    def exposure():
        per_site = {}
        for site, tasks in plan.items():
            total_episodes = len({episode["_episode_key"] for episode, _, _ in tasks})
            per_site[site] = {
                "seen_blocks": len(seen_blocks[site]), "total_blocks": len(tasks),
                "block_fraction": len(seen_blocks[site]) / len(tasks),
                "seen_episodes": len(seen_episodes[site]), "total_episodes": total_episodes,
                "episode_fraction": len(seen_episodes[site]) / total_episodes,
                "updates": site_updates[site],
            }
        seen_count, total_count = sum(map(len, seen_blocks.values())), sum(map(len, plan.values()))
        episode_seen = sum(value["seen_episodes"] for value in per_site.values())
        episode_total = sum(value["total_episodes"] for value in per_site.values())
        return {
            "seen_blocks": seen_count, "total_blocks": total_count,
            "query_block_fraction": seen_count / total_count,
            "site_macro_query_block_fraction": float(np.mean([v["block_fraction"] for v in per_site.values()])),
            "seen_episodes": episode_seen, "total_episodes": episode_total,
            "episode_fraction": episode_seen / episode_total,
            "all_query_blocks_seen": seen_count == total_count, "per_site": per_site,
        }

    step_names = ("X_support", "y_support", "X_query", "y_query",
                  "query_baseline_log_qc", "raw_query_log_qc")
    while adapter.steps < max_steps:
        site_order = sorted(plan)
        np.random.default_rng(seed + round_index).shuffle(site_order)
        for site in site_order:
            episode, block, key = plan[site][site_updates[site] % len(plan[site])]
            payload = build_residual_episode(dataset, episode, block, config=config)
            result = adapter.step(**{name: payload[name] for name in step_names},
                                  seed=int(episode.get("seed", seed)))
            seen_blocks[site].add(key)
            seen_episodes[site].add(episode["_episode_key"])
            site_updates[site] += 1
            steps.append({
                **result, "round": round_index + 1, "site_id": site,
                "episode_id": episode.get("episode_id"), "query_block_key": key,
                "support_location_ids": payload["support_location_ids"], "query_location_ids": block,
            })
            if adapter.steps % validation_every == 0 or adapter.steps == max_steps:
                scores = validate_residual(adapter, dataset, validation, config)
                current_exposure = exposure()
                history.append({"step": adapter.steps, **scores, "exposure": current_exposure})
                if scores["macro_score"] < best_score:
                    best_score, best_step, stalled = scores["macro_score"], adapter.steps, 0
                    best_exposure = copy.deepcopy(current_exposure)
                    adapter.save(best_path, {**metadata, "best_step": best_step,
                                             "validation": scores, "checkpoint_exposure": current_exposure})
                else:
                    stalled += 1
                write_json(history_path, {"validation": history, "steps": steps,
                                          "pretrained_validation": baseline, "metadata": metadata})
                if adapter.steps >= min_steps and stalled >= patience:
                    stop_reason = "early_stopping_after_minimum_successful_updates"
                    break
            if adapter.steps >= max_steps:
                break
        if stop_reason != "max_steps":
            break
        round_index += 1
    if best_step is None:
        raise RuntimeError("No updated, validated residual checkpoint was produced")
    summary = {
        **metadata, "checkpoint_path": str(best_path.resolve()), "history_path": str(history_path.resolve()),
        "best_step": best_step, "best_validation_score": float(best_score),
        "selection_metric": baseline["selection_metric"], "baseline_validation": baseline,
        "validation_preferred_model": model_id if best_score < baseline["macro_score"] else "tabpfn_residual_pretrained",
        "gradient_steps": adapter.steps, "actual_steps": adapter.steps, "successful_optimizer_updates": adapter.steps,
        "exposure": exposure(), "best_checkpoint_exposure": best_exposure, "stop_reason": stop_reason,
        "development_run": max_steps < 600, "elapsed_seconds": perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()) if adapter.use_amp else None,
        "device": adapter.device, "torch_version": torch.__version__, "config": config,
    }
    write_json(output_dir / f"{model_id}_training_summary.json", summary)
    write_json(output_dir / "summary.json", summary)
    return summary
