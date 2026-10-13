"""Frozen, validation-label-free context adaptations for the CPT diagnostic study.

Two hypotheses only: expose real source episodes directly in the inference
context, or construct target context/query features from exactly the same
revealed anchor set. These are new exploratory candidates, not v0.2 reruns.
Official API reference (pinned implementation):
https://github.com/PriorLabs/TabPFN/blob/v9.2.0/src/tabpfn/regressor.py
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cpt_budget.config import write_json
from cpt_budget.models.base import validate_observed, validate_query_site
from cpt_budget.models.distribution import QUANTILES, distribution_prediction_frame
from cpt_budget.models.tabpfn_adapter import _runtime, decode_full_output, geometry_only
from cpt_budget.models.tabpfn_train import _check_episode
from cpt_budget.types import PredictionBundle
from cpt_budget.v2.features import FEATURE_COLUMNS_V2, build_features_v2


DEFAULT_BUDGETS = (5, 7, 9, 11, 13)


def _make_context_estimator(config, artifact=None):
    """Retain official pinned v2 settings without v0.2's target-only contract.

    This is a separate study-local factory. The original registered adapter and
    its prohibition on historical inference contexts remain unchanged.
    """
    torch, cls, model_version = _runtime()
    settings = config.get("tabpfn", {})
    if str(settings.get("package_version", "9.2.0")) != "9.2.0" or settings.get("model_generation", "v2") != "v2":
        raise ValueError("The diagnostic fixes official package 9.2.0 and v2 weights")
    if settings.get("n_estimators", 1) != 1:
        raise ValueError("The context diagnostic fixes one inference estimator")
    options = dict(n_estimators=1, device=settings.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
                   random_state=int(config.get("split", {}).get("seed", 20261013)),
                   n_preprocessing_jobs=1, show_progress_bar=False, categorical_features_indices=[],
                   inference_config={"MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE": 1_000_000})
    path = artifact or settings.get("base_checkpoint_path")
    if path is not None:
        path = Path(path)
        if not path.is_absolute():
            path = Path(config.get("_root", ".")) / path
        if artifact is not None and not path.is_file():
            raise FileNotFoundError(path)
        options["model_path"] = path
    return cls.create_default_for_version(model_version.V2, fit_mode="fit_preprocessors", **options)


def _stable_key(seed, *parts):
    return hashlib.sha256("|".join(map(str, (seed, *parts))).encode()).hexdigest()


@dataclass
class SourceContextBank:
    """One complete labelled source query profile per eligible source site/budget."""

    rows: pd.DataFrame
    provenance: dict

    def for_budget(self, budget):
        rows = self.rows.loc[self.rows.target_budget.eq(int(budget))].copy()
        if rows.empty:
            raise ValueError(f"Source bank has no predeclared budget {budget}")
        return rows[list(FEATURE_COLUMNS_V2)], rows.source_log_qc.to_numpy(float)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.rows.to_parquet(directory / "source_context.parquet", index=False)
        write_json(directory / "source_context.json", self.provenance)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        return cls(pd.read_parquet(directory / "source_context.parquet"),
                   json.loads((directory / "source_context.json").read_text(encoding="utf-8")))


def build_source_context_bank(dataset, train_episodes, *, fold_id, source_sites,
                              budgets=DEFAULT_BUDGETS, seed=20261014):
    """Use only explicitly registered training episodes; never retrieve by score.

    For each source site and requested target budget, choose the source episode
    with the closest support size; resolve ties by a seeded stable hash. Select
    one of its whole query CPTs by another seeded hash. The selection uses IDs
    and counts only, not feature values, source labels, or target measurements.
    Source sites without an existing train episode are listed as ineligible.
    Every eligible site contributes the same complete 24-depth profile count.
    """
    episodes = pd.DataFrame(train_episodes).copy()
    allowed = set(map(str, source_sites))
    if episodes.empty or not {"role", "fold_id", "site_id", "episode_id"}.issubset(episodes):
        raise ValueError("Source context requires explicitly role-labelled training episodes")
    if not episodes.role.eq("train").all() or not episodes.fold_id.eq(fold_id).all():
        raise ValueError("Source bank cannot include validation, test, or another fold")
    if not set(episodes.site_id).issubset(allowed):
        raise ValueError("An episode site is outside the frozen fold training sites")
    if episodes.duplicated(["site_id", "episode_id"]).any():
        raise ValueError("Duplicate source episodes")
    budgets = tuple(sorted(set(map(int, budgets))))
    if not budgets or min(budgets) < 2:
        raise ValueError("Positive predeclared support budgets are required")
    parts, records, expected_depths = [], [], None
    for budget in budgets:
        for site, site_episodes in episodes.groupby("site_id", sort=True):
            candidates = site_episodes.to_dict("records")
            record = min(candidates, key=lambda row: (
                abs(len(row["support_location_ids"]) - budget),
                _stable_key(seed, site, row["episode_id"])))
            support_ids, query_ids = _check_episode(dataset, record)
            selected = min(query_ids, key=lambda loc: _stable_key(seed, site, record["episode_id"], loc))
            observed = dataset.observed(support_ids)
            query = dataset.query([selected])
            truth = dataset.truth([selected])
            keys = ["site_id", "location_id", "depth_index"]
            labels = query[keys].merge(truth[keys + ["log_qc"]], on=keys,
                                       how="left", validate="one_to_one", sort=False)
            if not set(observed.site_id).union(query.site_id).issubset({site}):
                raise ValueError("Source episode spans sites")
            depths = tuple(query.sort_values("depth_index").depth_index.tolist())
            if expected_depths is None:
                expected_depths = depths
            if len(depths) != 24 or depths != expected_depths or not np.isfinite(labels.log_qc).all():
                raise ValueError("Balanced source bank requires complete common-depth profiles")
            features = build_features_v2(observed, query).reset_index(drop=True)
            features["source_log_qc"] = labels.log_qc.to_numpy(float)
            features["target_budget"] = budget
            features["source_site_id"] = str(site)
            features["source_episode_id"] = str(record["episode_id"])
            features["source_query_location_id"] = str(selected)
            features["source_depth_index"] = query.depth_index.to_numpy()
            parts.append(features)
            records.append({"target_budget": budget, "site_id": str(site),
                            "episode_id": str(record["episode_id"]),
                            "support_location_ids": list(map(str, support_ids)),
                            "query_location_id": str(selected), "n_support": len(support_ids),
                            "rows": len(query), "role": "train"})
    rows = pd.concat(parts, ignore_index=True)
    counts = rows.groupby(["target_budget", "source_site_id"]).size()
    if counts.nunique() != 1:
        raise RuntimeError("Each eligible source site must contribute equal row counts")
    site_groups = (dataset.sites.set_index("site_id").investigation_group.to_dict()
                   if "investigation_group" in dataset.sites else {})
    provenance = {"schema_version": "diagnostic-context-1", "fold_id": int(fold_id),
                  "seed": int(seed), "budgets": list(budgets), "feature_columns": FEATURE_COLUMNS_V2,
                  "allowed_train_sites": sorted(allowed), "eligible_train_sites": sorted(set(episodes.site_id)),
                  "ineligible_train_sites_no_episode": sorted(allowed - set(episodes.site_id)),
                  "source_investigation_groups": sorted({str(site_groups[s]) for s in allowed if s in site_groups}),
                  "selection_rule": "nearest_support_count_then_seeded_episode_hash_one_seeded_whole_query_per_site",
                  "target_labels_used": False, "source_rows_per_site_per_budget": int(counts.iloc[0]),
                  "native_probability_calibration": "none", "records": records,
                  "official_api": "https://github.com/PriorLabs/TabPFN/blob/v9.2.0/src/tabpfn/regressor.py"}
    return SourceContextBank(rows, provenance)


def honest_anchor_split(observed, *, anchor_fraction=0.6):
    """Fixed geometry-only maximin partition; all inputs are already revealed.

    Start nearest the revealed-set centroid, then add the farthest location
    from existing anchors. Stable location order resolves all distance ties.
    Context labels and query features both use the same anchor subset.
    """
    observed = validate_observed(observed)
    locations = observed.drop_duplicates("location_id").sort_values("location_id", kind="stable")
    n = len(locations)
    if n < 3 or not 0 < anchor_fraction < 1:
        raise ValueError("Honest context needs at least three revealed CPTs and a fraction in (0, 1)")
    n_anchors = min(n - 1, max(2, int(np.ceil(n * anchor_fraction))))
    coords = locations[["x_m", "y_m"]].to_numpy(float)
    selected = [int(np.argmin(np.linalg.norm(coords - coords.mean(axis=0), axis=1)))]
    while len(selected) < n_anchors:
        distance = np.linalg.norm(coords[:, None, :] - coords[selected][None, :, :], axis=2).min(axis=1)
        distance[selected] = -np.inf
        selected.append(int(np.argmax(distance)))
    anchor_ids = locations.iloc[selected].location_id.tolist()
    mask = observed.location_id.isin(anchor_ids)
    return observed.loc[mask].copy(), observed.loc[~mask].copy()


def context_feature_diagnostics(observed, query_grid):
    """Measure raw feature shift without ever inspecting query measurements.

    Constant raw features can be removed by official preprocessing. Thus this
    documents a construction difference, not evidence of its causal harm.
    """
    observed = validate_observed(observed)
    query = geometry_only(query_grid)
    support = build_features_v2(observed, geometry_only(observed), leave_location_out=True)
    query_features = build_features_v2(observed, query)
    anchors, labelled = honest_anchor_split(observed)
    honest_context = build_features_v2(anchors, geometry_only(labelled))
    honest_query = build_features_v2(anchors, query)
    statistics = []
    for name in FEATURE_COLUMNS_V2:
        a, b = support[name].dropna().to_numpy(), query_features[name].dropna().to_numpy()
        statistics.append({"feature": name, "support_mean": float(a.mean()) if len(a) else None,
                           "query_mean": float(b.mean()) if len(b) else None,
                           "support_unique": int(len(np.unique(a))),
                           "query_outside_support_range_fraction": float(((b < a.min()) | (b > a.max())).mean()) if len(a) and len(b) else None})
    return {"revealed_cpts": int(observed.location_id.nunique()),
            "standard_support_rows": len(support), "query_rows": len(query),
            "standard_support_counts": sorted(support.support_count.unique().tolist()),
            "standard_query_counts": sorted(query_features.support_count.unique().tolist()),
            "honest_anchor_ids": anchors.location_id.drop_duplicates().tolist(),
            "honest_labelled_ids": labelled.location_id.drop_duplicates().tolist(),
            "honest_context_counts": sorted(honest_context.support_count.unique().tolist()),
            "honest_query_counts": sorted(honest_query.support_count.unique().tolist()),
            "statistics": statistics,
            "interpretation_limit": "Raw feature shift only; constant columns may be removed by preprocessing; no causal performance conclusion."}


class ContextAdapterPredictor:
    """Official pretrained TabPFN with either source bank or honest target context."""

    def __init__(self, config=None, *, source_bank=None, mode="source_bank", artifact=None):
        if mode not in {"source_bank", "honest"}:
            raise ValueError("mode must be source_bank or honest")
        if (mode == "source_bank") != (source_bank is not None):
            raise ValueError("A source bank is required only for source_bank mode")
        self.config = copy.deepcopy(config or {})
        self.config.setdefault("tabpfn", {})["target_context_only"] = mode == "honest"
        self.config.setdefault("diagnostics", {})["context_adapter_mode"] = mode
        self.mode, self.source_bank, self.artifact = mode, source_bank, artifact
        self.model_id = "tabpfn_source_context" if mode == "source_bank" else "tabpfn_honest_context"
        self._estimator = self._observed = self._anchors = None
        self.context_manifest = None

    def context_arrays(self, observed):
        """Build inspectable fit data without loading weights or touching a GPU."""
        observed = validate_observed(observed)
        self._observed = observed
        if self.mode == "source_bank":
            site = str(observed.site_id.iloc[0])
            if site in self.source_bank.provenance["allowed_train_sites"]:
                raise ValueError("Target site overlaps source training sites")
            groups = set(self.source_bank.provenance.get("source_investigation_groups", []))
            if "investigation_group" in observed and set(observed.investigation_group.astype(str)) & groups:
                raise ValueError("Target investigation group overlaps source training groups")
            target_x = build_features_v2(observed, geometry_only(observed), leave_location_out=True)
            source_x, source_y = self.source_bank.for_budget(observed.location_id.nunique())
            x = pd.concat([source_x, target_x], ignore_index=True)
            y = np.concatenate([source_y, observed.log_qc.to_numpy(float)])
            self._anchors = observed
            manifest = {"mode": self.mode, "target_site": site, "source_rows": len(source_y),
                        "target_label_rows": len(observed), "source_fold_id": self.source_bank.provenance["fold_id"],
                        "source_sites": self.source_bank.provenance["eligible_train_sites"],
                        "target_label_location_ids": observed.location_id.drop_duplicates().tolist()}
        else:
            anchors, labelled = honest_anchor_split(observed)
            x = build_features_v2(anchors, geometry_only(labelled))
            y = labelled.log_qc.to_numpy(float)
            self._anchors = anchors
            manifest = {"mode": self.mode, "target_site": str(observed.site_id.iloc[0]), "source_rows": 0,
                        "anchor_fraction": 0.6, "target_label_rows": len(labelled),
                        "anchor_location_ids": anchors.location_id.drop_duplicates().tolist(),
                        "target_label_location_ids": labelled.location_id.drop_duplicates().tolist()}
        if np.isinf(x.to_numpy(float)).any() or not np.isfinite(y).all():
            raise ValueError("Context features/labels are invalid")
        self.context_manifest = {**manifest, "feature_columns": FEATURE_COLUMNS_V2,
                                 "fit_rows": len(y), "query_truth_used": False,
                                 "target_context_only": self.mode == "honest",
                                 "factory": "diagnostics.context_adapter._make_context_estimator",
                                 "inference_weight_updates": 0}
        return x.to_numpy(float), y

    def condition(self, observed):
        x, y = self.context_arrays(observed)
        if self._estimator is None:
            self._estimator = _make_context_estimator(self.config, self.artifact)
        self._estimator.fit(x, y)
        for model in self._estimator.models_:
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    def query_features(self, query_grid):
        validate_query_site(query_grid, self._observed)
        return build_features_v2(self._anchors, geometry_only(query_grid)).to_numpy(float)

    def predict(self, query_grid):
        if self._estimator is None:
            raise RuntimeError("condition() must be called before predict()")
        query = geometry_only(query_grid).reset_index(drop=True)
        x = self.query_features(query)
        if query.empty:
            return PredictionBundle(distribution_prediction_frame(query, np.empty((0, 7)), np.empty((0, 3))), self.model_id)
        import torch
        ids = query.location_id.drop_duplicates().tolist()
        batch_size = max(1, min(8, int(self.config.get("episodes", {}).get("query_profiles_per_batch", 8))))
        parts, position = [], 0
        while position < len(ids):
            selected = ids[position:position + batch_size]
            row_indices = np.flatnonzero(query.location_id.isin(selected).to_numpy())
            try:
                with torch.no_grad():
                    full = self._estimator.predict(x[row_indices], output_type="full", quantiles=list(QUANTILES))
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
