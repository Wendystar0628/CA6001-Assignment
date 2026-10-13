"""Version-pinned TabPFN v2 boundary: contexts, episode gradients and raw CDF.

The private calls here follow PriorLabs/TabPFN tag v9.2.0, especially
finetuning/{data_util,finetuned_regressor,finetuned_base}.py. Ordinary fit()
replaces an inference context; only EpisodeAdapter.step() changes weights.
"""
from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from cpt_budget.features import FEATURE_COLUMNS, build_features
from cpt_budget.types import PredictionBundle, QUERY_COLUMNS
from .distribution import QUANTILES, THRESHOLDS, distribution_prediction_frame


def _runtime():
    installed = version("tabpfn")
    if installed != "9.2.0":
        raise RuntimeError(f"This adapter requires tabpfn==9.2.0; found {installed}")
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    return torch, TabPFNRegressor, ModelVersion


def estimator_options(config=None):
    torch, _, _ = _runtime()
    config = config or {}
    settings = config.get("tabpfn", {})
    if str(settings.get("package_version", "9.2.0")) != "9.2.0" or settings.get("model_generation", "v2") != "v2":
        raise ValueError("The registered adapter fixes package 9.2.0 and v2 weights")
    if settings.get("target_context_only", True) is not True:
        raise ValueError("Historical context banks are outside the registered experiment")
    if settings.get("n_estimators", 1) != 1:
        raise ValueError("The registered experiment uses exactly one estimator")
    return dict(
        n_estimators=1,
        device=settings.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        random_state=int(config.get("split", {}).get("seed", 20261013)),
        n_preprocessing_jobs=1,
        show_progress_bar=False,
        categorical_features_indices=[],
        # All sixteen fields are quantitative. Do not infer a categorical feature
        # from a small support set's accidental number of distinct values.
        inference_config={"MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE": 1_000_000},
    )


def make_estimator(config=None, artifact=None, *, training=False):
    _, cls, model_version = _runtime()
    options = estimator_options(config)
    path = artifact or (config or {}).get("tabpfn", {}).get("base_checkpoint_path")
    if path is not None:
        path = Path(path)
        if not path.is_absolute():
            path = Path((config or {}).get("_root", ".")) / path
        if artifact is not None and not path.is_file():
            raise FileNotFoundError(path)
        options["model_path"] = path
    return cls.create_default_for_version(
        model_version.V2, fit_mode="batched" if training else "fit_preprocessors",
        **options,
    )


def feature_array(observed, query, *, leave_location_out=False):
    frame = build_features(observed, query, leave_location_out=leave_location_out)
    if list(frame.columns) != list(FEATURE_COLUMNS):
        raise ValueError("Feature order differs from the registered 16-column schema")
    values = frame.to_numpy(dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError("Features may contain NaN but cannot contain infinity")
    return values


def geometry_only(frame):
    """Avoid propagating measurements through query or prediction frames."""
    return frame[[c for c in QUERY_COLUMNS if c in frame]].copy()


def decode_full_output(full: dict[str, Any], query: pd.DataFrame) -> pd.DataFrame:
    """Decode native tails and bars in the *input label* space ln(qc/MPa)."""
    import torch
    quantiles = np.column_stack(full["quantiles"])
    logits = full["logits"]
    if not torch.is_tensor(logits):
        raise TypeError("TabPFN full output must contain its native tensor logits")
    criterion = full["criterion"].to(device=logits.device, dtype=logits.dtype)
    thresholds = torch.as_tensor(np.log(THRESHOLDS), device=logits.device, dtype=logits.dtype)
    # FullSupportBarDistribution.cdf handles both half-normal tails. The raw
    # criterion already reverses target standardisation: do not normalise twice.
    with torch.no_grad():
        probabilities = criterion.cdf(logits, thresholds).detach().cpu().numpy()
    if quantiles.shape != (len(query), 7) or probabilities.shape != (len(query), 3):
        raise ValueError("Unexpected TabPFN full distribution shapes")
    return distribution_prediction_frame(geometry_only(query), quantiles, probabilities)


class TabPFNPredictor:
    def __init__(self, config=None, artifact=None):
        self.config = config or {}
        self.artifact = Path(artifact) if artifact is not None else None
        self.model_id = "tabpfn_finetuned" if artifact is not None else "tabpfn"
        self._estimator = None
        self._observed = None
        self.query_profiles_per_batch = min(8, int(self.config.get("episodes", {}).get("query_profiles_per_batch", 8)))
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
        X = feature_array(observed, geometry_only(observed), leave_location_out=True)
        # This inference-only estimator is loaded from the checkpoint (or an
        # official ModelSpecs clone); no optimiser ever acts on its parameters.
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
            return PredictionBundle(distribution_prediction_frame(query, np.empty((0, 7)), np.empty((0, 3))), self.model_id)
        X = feature_array(self._observed, query)
        parts = []
        ids = list(query.location_id.drop_duplicates())
        batch_size = self.query_profiles_per_batch
        position = 0
        import torch
        while position < len(ids):
            selected = ids[position:position + batch_size]
            row_indices = np.flatnonzero(query.location_id.isin(selected).to_numpy())
            try:
                with torch.no_grad():
                    full = self._estimator.predict(X[row_indices], output_type="full", quantiles=list(QUANTILES))
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


class EpisodeAdapter:
    """Supply one intact CPT episode to the official differentiable path."""
    def __init__(self, config=None):
        torch, _, _ = _runtime()
        from tabpfn.finetuning._torch_compat import GradScaler
        self.config = config or {}
        self.settings = self.config.get("tabpfn", {})
        self.estimator = make_estimator(config, training=True)
        self.estimator._initialize_model_variables()
        self.device = str(self.estimator.devices_[0])
        self.model = self.estimator.models_[0].to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(self.settings.get("learning_rate", 1e-5)), weight_decay=float(self.settings.get("weight_decay", .01)))
        self.use_amp = self.device.startswith("cuda")
        self.scaler = GradScaler(init_scale=1024.0) if self.use_amp else None
        self.steps = 0

    def preprocess(self, X_support, y_support, X_query, y_query, *, seed):
        from tabpfn.finetuning.data_util import get_preprocessed_dataset_chunks, meta_dataset_collator
        # pandas 3 can expose read-only arrays. Official torch.from_numpy calls
        # require writable storage, so isolate the four inputs explicitly.
        X_support = np.array(X_support, dtype=np.float64, copy=True)
        y_support = np.array(y_support, dtype=np.float64, copy=True)
        X_query = np.array(X_query, dtype=np.float64, copy=True)
        y_query = np.array(y_query, dtype=np.float64, copy=True)
        if len(X_support) + len(X_query) > int(self.config.get("episodes", {}).get("max_rows", 512)):
            raise ValueError("Episode exceeds configured context plus query row limit")
        if len(X_support) < 2 or not len(X_query):
            raise ValueError("An episode needs a nonempty context and query")

        def fixed_split(X_context_clean, y_context_clean, **_):
            # Unlike calling the helper on concatenated labels, this keeps its
            # initial schema/config fitting support-only as well. All columns
            # are numeric; actual ensemble transforms fit support below, and
            # transform query with those same fitted objects in official code.
            if len(X_context_clean) != len(X_support):
                raise RuntimeError("Official preprocessing unexpectedly changed row count")
            return X_context_clean, X_query, y_context_clean, y_query

        chunks = get_preprocessed_dataset_chunks(
            self.estimator, X_support, y_support, split_fn=fixed_split,
            max_data_size=None, model_type="regressor", equal_split_size=False,
            data_shuffle_seed=int(seed), preprocessing_random_state=int(seed),
            shuffle=False, force_no_stratify=True,
        )
        if len(chunks) != 1:
            raise RuntimeError("One CPT episode must produce exactly one batch")
        return meta_dataset_collator([chunks[0]])

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
            targets = torch.stack([_targets_in_estimator_space(batch.y_query[0], c[0]) for c in batch.configs]).to(self.device)
            loss = _compute_regression_loss(
                logits_BQL=logits_QBEL.permute(1, 2, 0, 3).reshape(B * E, Q, L),
                targets_BQ=targets, bardist_loss_fn=batch.znorm_space_bardist,
                ce_loss_weight=1., crps_loss_weight=0., crls_loss_weight=0., mse_loss_weight=0., mae_loss_weight=0.,
            )
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite episode NLL")
        with sdpa_kernel_context():
            (self.scaler.scale(loss) if self.scaler is not None else loss).backward()
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.settings.get("gradient_clip", 1.0)))
        if not torch.isfinite(norm) or norm.item() <= 0:
            raise RuntimeError(f"Episode has invalid gradient norm: {norm.item()}")
        changed_parameter = next(p for p in self.model.parameters() if p.grad is not None and torch.any(p.grad != 0))
        before = changed_parameter.detach().clone()
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        delta = torch.linalg.vector_norm(changed_parameter.detach() - before).item()
        if delta <= 0:
            raise RuntimeError("No parameter update occurred; do not call this fine-tuning")
        self.steps += 1
        return {"nll": float(loss.detach().cpu()), "gradient_norm_before_clip": float(norm.detach().cpu()),
                "updated_parameter_delta_l2": delta, "seconds": perf_counter() - started,
                "support_rows": len(y_support), "query_rows": len(y_query), "step": self.steps}

    def predictor(self):
        from tabpfn import TabPFNRegressor
        from tabpfn.finetuning.train_util import clone_model_for_evaluation
        result = TabPFNPredictor(self.config)
        result.model_id = "tabpfn_finetuned"
        result._estimator = clone_model_for_evaluation(self.estimator, {**estimator_options(self.config), "fit_mode": "fit_preprocessors"}, TabPFNRegressor)
        return result

    def save(self, path, metadata):
        from tabpfn.model_loading import save_tabpfn_model
        if not self.steps:
            raise RuntimeError("Refusing to save unchanged pretrained weights as fine-tuned")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_tabpfn_model(self.estimator, path, additional_fields={"cpt_budget": {**metadata, "feature_columns": list(FEATURE_COLUMNS), "gradient_steps": self.steps}})
        return path
