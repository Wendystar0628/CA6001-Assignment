"""Whole-profile, query-relative DeepSets conditional neural regressor.

This is a deliberately small CNP-inspired conditional field, not a Transformer
or a baseline residual model. It encodes complete observed CPT profiles and
uses a permutation-invariant set summary conditioned on each query position.
It exposes point predictions only: a mean/max pooled network does not itself
establish a calibrated stochastic process or spatial joint distribution.

Reference: Garnelo et al. (2018), Conditional Neural Processes,
https://proceedings.mlr.press/v80/garnelo18a.html
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..config import PROJECT_ROOT, load_config, write_json
from ..models.base import validate_observed, validate_query_site
from ..models.distribution import point_prediction_frame
from ..types import PredictionBundle, load_dataset


MODEL_ID = "cnp_profile"
DEFAULTS = {
    "hidden_width": 64, "xy_scale_m": 50.0, "max_steps": 2000,
    "validation_every": 100, "min_steps": 400,
    "early_stopping_patience": 4, "learning_rate": 1e-3,
    "weight_decay": 1e-4, "gradient_clip": 1.0, "seed": 20261014,
}


class ProfileCNP(nn.Module):
    """Query-relative set encoder and shared depth-conditioned decoder."""

    def __init__(self, n_depth: int = 24, hidden_width: int = 64, xy_scale_m: float = 50.0):
        super().__init__()
        if n_depth < 1 or hidden_width < 1 or xy_scale_m <= 0:
            raise ValueError("Depth count, width and metric coordinate scale must be positive")
        self.n_depth = int(n_depth)
        self.hidden_width = int(hidden_width)
        self.xy_scale_m = float(xy_scale_m)
        h = self.hidden_width
        self.profile_encoder = nn.Sequential(nn.Linear(n_depth, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.relative_encoder = nn.Sequential(nn.Linear(h + 3, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(2 * h + 1, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, support_xy, support_profiles, query_xy, depth_positions):
        """Return [query profiles, depths], with no query measurement input."""
        if support_profiles.ndim != 2 or support_profiles.shape[1] != self.n_depth:
            raise ValueError("The context must contain complete fixed-grid profiles")
        if len(support_profiles) != len(support_xy) or not len(support_profiles):
            raise ValueError("Support coordinates and complete profiles must correspond")
        if len(depth_positions) != self.n_depth:
            raise ValueError("Depth coordinates must match the trained complete profile grid")
        if not len(query_xy):
            return support_profiles.new_empty((0, self.n_depth))
        encoded = self.profile_encoder(support_profiles)
        relative = (support_xy[None, :, :] - query_xy[:, None, :]) / self.xy_scale_m
        signed_xy = relative.sign() * relative.abs().log1p()
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True).log1p()
        inputs = torch.cat([encoded[None].expand(len(query_xy), -1, -1), signed_xy, distance], dim=-1)
        local = self.relative_encoder(inputs)
        pooled = torch.cat([local.mean(dim=1), local.amax(dim=1)], dim=-1)
        decoder_input = torch.cat([
            pooled[:, None, :].expand(-1, self.n_depth, -1),
            depth_positions[None, :, None].expand(len(query_xy), -1, -1),
        ], dim=-1)
        return self.decoder(decoder_input).squeeze(-1)


def _profile_arrays(rows, depth_indices, *, measurements):
    """Read labels only from explicitly observed/source-training frames."""
    required = ["site_id", "location_id", "depth_index", "x_m", "y_m"]
    if measurements:
        required.append("log_qc")
    missing = set(required) - set(rows)
    if missing:
        raise ValueError(f"Profile data missing {sorted(missing)}")
    if rows.empty or rows.site_id.nunique() != 1:
        raise ValueError("Profiles must belong to one nonempty site")
    if rows.duplicated(["location_id", "depth_index"]).any():
        raise ValueError("Duplicate profile/depth keys")
    ids = sorted(rows.location_id.unique().tolist())
    coordinates = []
    for location_id in ids:
        block = rows[rows.location_id.eq(location_id)]
        xy = block[["x_m", "y_m"]].to_numpy(float)
        if not np.isfinite(xy).all() or not np.all(xy == xy[0]):
            raise ValueError("Each profile must have one finite horizontal coordinate")
        coordinates.append(xy[0])
        if measurements and set(block.depth_index) != set(depth_indices):
            raise ValueError("Every observed/source profile must contain all trained depths")
    profiles = None
    if measurements:
        profiles = rows.pivot(index="location_id", columns="depth_index", values="log_qc").reindex(
            index=ids, columns=depth_indices).to_numpy(float)
        if not np.isfinite(profiles).all():
            raise ValueError("Complete profile measurements must be finite")
    return ids, np.asarray(coordinates), profiles


@dataclass
class EpisodeTensors:
    site_id: str
    episode_id: str
    support_xy: torch.Tensor
    support_profiles: torch.Tensor
    query_xy: torch.Tensor
    truth: torch.Tensor


def _tensor_episode(dataset, row, depth_indices, mean, scale, device):
    support_ids, query_ids = list(row.support_location_ids), list(row.query_location_ids)
    if set(support_ids) & set(query_ids):
        raise ValueError("Source episode support and query profiles overlap")
    observed = dataset.observed(support_ids)
    target = dataset.truth(query_ids)
    if set(observed.near_group_id) & set(target.near_group_id):
        raise ValueError("Source episode near groups overlap")
    if set(observed.site_id) != {row.site_id} or set(target.site_id) != {row.site_id}:
        raise ValueError("Source episode includes other-site measurements")
    _, xy, profiles = _profile_arrays(observed, depth_indices, measurements=True)
    _, query_xy, truth = _profile_arrays(target, depth_indices, measurements=True)
    # Subtract the context centre in float64 before casting projected coordinates
    # to float32; retaining large raw map coordinates would lose metre precision.
    origin = xy.mean(axis=0)
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    return EpisodeTensors(str(row.site_id), str(row.episode_id), tensor(xy-origin),
                          tensor((profiles-mean)/scale), tensor(query_xy-origin),
                          tensor((truth-mean)/scale))


def _validate_plans(train, validation):
    if train.empty or validation.empty:
        raise ValueError("Training and validation episode plans are both required")
    for plan, role in [(train, "train"), (validation, "validation")]:
        if not plan.role.eq(role).all() or plan.fold_id.nunique() != 1:
            raise ValueError("Episode role/fold mismatch; outer test plans are forbidden")
        if plan.episode_id.duplicated().any():
            raise ValueError("Duplicate source episode identifiers")
    if int(train.fold_id.iloc[0]) != int(validation.fold_id.iloc[0]):
        raise ValueError("Training and validation belong to different folds")
    if set(train.site_id) & set(validation.site_id):
        raise ValueError("Training and validation sites overlap")
    if set(train.investigation_group) & set(validation.investigation_group):
        raise ValueError("Training and validation investigation groups overlap")


def fit_neural_process(dataset, train, validation, config, output_dir, *, device="cpu"):
    """Fit one frozen candidate with equal-site updates and source-only scaling."""
    _validate_plans(train, validation)
    options = {**DEFAULTS, **config.get("v5", {}).get("neural_process", {})}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(options["seed"]) + int(train.fold_id.iloc[0])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    source_ids = sorted({location for ids in train.support_location_ids for location in ids}
                        | {location for ids in train.query_location_ids for location in ids})
    source = dataset.observed(source_ids)
    if set(source.site_id) != set(train.site_id):
        raise ValueError("Source normalization sites disagree with train plan")
    depth_grid = source[["depth_index", "z_center_m"]].drop_duplicates().sort_values("depth_index")
    if depth_grid.depth_index.duplicated().any():
        raise ValueError("Depth coordinates differ across source profiles")
    depth_indices = list(map(int, depth_grid.depth_index))
    depths = depth_grid.z_center_m.to_numpy(float)
    depth_positions = (depths-depths.mean()) / max(float(np.ptp(depths))/2, 1e-6)
    # Each source site has equal influence on fixed normalization statistics.
    moments = source.groupby("site_id").log_qc.agg(["mean", lambda y: np.mean(np.square(y))])
    mean = float(moments.iloc[:, 0].mean())
    scale = max(float(np.sqrt(max(moments.iloc[:, 1].mean()-mean*mean, 0))), 1e-3)
    network = ProfileCNP(len(depths), int(options["hidden_width"]), float(options["xy_scale_m"])).to(device)
    z = torch.as_tensor(depth_positions, dtype=torch.float32, device=device)
    packed_train = [_tensor_episode(dataset, row, depth_indices, mean, scale, device)
                    for row in train.itertuples(index=False)]
    packed_validation = [_tensor_episode(dataset, row, depth_indices, mean, scale, device)
                         for row in validation.itertuples(index=False)]
    by_site = {site: [p for p in packed_train if p.site_id == site]
               for site in sorted(train.site_id.unique())}
    orders = {site: rng.permutation(len(values)).tolist() for site, values in by_site.items()}
    cursors = {site: 0 for site in by_site}
    optimizer = torch.optim.AdamW(network.parameters(), lr=float(options["learning_rate"]),
                                 weight_decay=float(options["weight_decay"]))

    def validate():
        network.eval()
        values = {}
        with torch.no_grad():
            for item in packed_validation:
                pred = network(item.support_xy, item.support_profiles, item.query_xy, z)
                values.setdefault(item.site_id, []).append(float((pred-item.truth).abs().mean())*scale)
        return float(np.mean([np.mean(v) for v in values.values()])), {k: float(np.mean(v)) for k, v in values.items()}

    initial_validation, _ = validate()
    history = []
    best_score, best_step, bad, best_exposed = float("inf"), 0, 0, 0
    started = time.perf_counter()
    exposure = {item.episode_id: 0 for item in packed_train}
    checkpoint = output_dir/"best.pt"
    metadata = {"model_id": MODEL_ID, "architecture": "query_relative_profile_deepsets",
                "n_depth": len(depths), "depth_indices": depth_indices,
                "depth_positions": depth_positions.tolist(), "depth_m": depths.tolist(),
                "target_mean": mean, "target_scale": scale,
                "hidden_width": int(options["hidden_width"]), "xy_scale_m": float(options["xy_scale_m"]),
                "fold_id": int(train.fold_id.iloc[0]), "seed": seed,
                "execution_device": str(device), "torch_version": str(torch.__version__),
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "options": options, "train_sites": sorted(train.site_id.unique().tolist()),
                "validation_sites": sorted(validation.site_id.unique().tolist()),
                "source_normalizer_sites": sorted(source.site_id.unique().tolist()),
                "loss": "equal_site_episode_mean_absolute_log_error", "native_distribution": False}
    try:
        metadata["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        metadata["source_commit"] = "unavailable"
    sites = list(by_site)
    for step in range(1, int(options["max_steps"])+1):
        site = sites[(step-1) % len(sites)]
        cursor = cursors[site]
        if cursor == len(orders[site]):
            orders[site] = rng.permutation(len(by_site[site])).tolist()
            cursor = 0
        item = by_site[site][orders[site][cursor]]
        cursors[site] = cursor+1
        network.train()
        optimizer.zero_grad(set_to_none=True)
        pred = network(item.support_xy, item.support_profiles, item.query_xy, z)
        loss = (pred-item.truth).abs().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite CNP training loss")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(network.parameters(), float(options["gradient_clip"]), error_if_nonfinite=True)
        optimizer.step()
        exposure[item.episode_id] += 1
        if step % int(options["validation_every"]) == 0 or step == int(options["max_steps"]):
            score, site_scores = validate()
            improved = score < best_score-1e-8
            history.append({"step": step, "train_episode_log_mae": float(loss.detach())*scale,
                            "validation_log_mae": score, "validation_site_scores": site_scores,
                            "gradient_norm": float(grad), "improved": improved,
                            "elapsed_seconds": time.perf_counter()-started})
            if improved:
                best_score, best_step, bad = score, step, 0
                best_exposed = sum(v > 0 for v in exposure.values())
                torch.save({"metadata": metadata, "state_dict": network.state_dict(),
                            "best_step": best_step, "validation_log_mae": best_score,
                            "exposure": exposure.copy()}, checkpoint)
            else:
                bad += 1
            write_json(output_dir/"history.json", history)
            print(f"cnp_profile fold={metadata['fold_id']} step={step} val_log_mae={score:.6f} best={best_score:.6f}", flush=True)
            if step >= int(options["min_steps"]) and bad >= int(options["early_stopping_patience"]):
                break
    summary = {**metadata, "checkpoint_path": str(checkpoint.resolve()), "actual_steps": step,
               "best_step": best_step, "best_validation_log_mae": best_score,
               "initial_validation_log_mae": initial_validation,
               "n_parameters": sum(p.numel() for p in network.parameters()),
               "n_train_episodes": len(train), "n_validation_episodes": len(validation),
               "n_train_sites": len(sites), "n_validation_sites": validation.site_id.nunique(),
               "elapsed_seconds": time.perf_counter()-started,
               "final_exposure": exposure, "exposed_episodes": sum(v > 0 for v in exposure.values()),
               "best_exposed_episodes": best_exposed,
               "training_note": "Profiles/depths/trajectories are correlated; they are not independent sites.",
               "reference": "https://proceedings.mlr.press/v80/garnelo18a.html"}
    write_json(output_dir/"summary.json", summary)
    return summary


class NeuralProcessPredictor:
    model_id = MODEL_ID

    def __init__(self, config=None, artifact=None, *, device="cpu"):
        if artifact is None:
            raise ValueError("A source-trained CNP checkpoint is required")
        path = Path(artifact)
        if path.is_dir():
            path = path/"best.pt"
        saved = torch.load(path, map_location=device, weights_only=False)
        self.metadata = saved["metadata"]
        m = self.metadata
        self.device = device
        self.network = ProfileCNP(m["n_depth"], m["hidden_width"], m["xy_scale_m"]).to(device)
        self.network.load_state_dict(saved["state_dict"])
        self.network.eval()
        self.depth_indices = m["depth_indices"]
        self.z = torch.as_tensor(m["depth_positions"], dtype=torch.float32, device=device)
        self.observed = None

    def condition(self, observed):
        self.observed = validate_observed(observed)
        ids, xy, profiles = _profile_arrays(self.observed, self.depth_indices, measurements=True)
        self.origin = xy.mean(axis=0)
        self.support_xy = torch.as_tensor(xy-self.origin, dtype=torch.float32, device=self.device)
        self.support_profiles = torch.as_tensor((profiles-self.metadata["target_mean"])/self.metadata["target_scale"],
                                               dtype=torch.float32, device=self.device)
        self.context_manifest = {"model_id": self.model_id, "observed_location_ids": ids,
                                 "n_complete_profiles": len(ids), "weights_updated_on_target": False,
                                 "native_distribution": False}

    def predict(self, query_grid):
        validate_query_site(query_grid, self.observed)
        if query_grid.empty:
            return PredictionBundle(point_prediction_frame(query_grid, np.empty(0)), self.model_id)
        ids, xy, _ = _profile_arrays(query_grid, self.depth_indices, measurements=False)
        if not set(query_grid.depth_index) <= set(self.depth_indices):
            raise ValueError("Query depth is outside the trained grid")
        query_xy = torch.as_tensor(xy-self.origin, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            prediction = self.network(self.support_xy, self.support_profiles, query_xy, self.z).cpu().numpy()
        prediction = prediction*self.metadata["target_scale"]+self.metadata["target_mean"]
        lookup = {(loc, depth): prediction[i, j] for i, loc in enumerate(ids)
                  for j, depth in enumerate(self.depth_indices)}
        values = np.asarray([lookup[(row.location_id, row.depth_index)] for row in query_grid.itertuples(index=False)])
        return PredictionBundle(point_prediction_frame(query_grid, values), self.model_id)


def run(config, output, *, device="cpu"):
    """Fit registered folds once, then predict the same sixty validation states."""
    from ..diagnostics.common import run_predictor, source_folds
    output = Path(output)
    dataset = load_dataset(config)
    summaries = []
    episode_root = PROJECT_ROOT/"runs/v4-residual-20261014/episodes"
    for fold in source_folds():
        fid = int(fold["fold_id"])
        folder = output/f"fold_{fid}"
        train = pd.read_parquet(episode_root/f"fold_{fid}_train.parquet")
        validation = pd.read_parquet(episode_root/f"fold_{fid}_validation.parquet")
        if set(train.site_id) - set(fold["train_sites"]) or set(validation.site_id) - set(fold["validation_sites"]):
            raise ValueError("Saved source episodes are outside frozen fold roles")
        if (folder/"summary.json").exists():
            summary = json.loads((folder/"summary.json").read_text(encoding="utf-8"))
            expected = {**DEFAULTS, **config.get("v5", {}).get("neural_process", {})}
            if summary["options"] != expected or not (folder/"best.pt").is_file():
                raise ValueError("Existing checkpoint settings differ; use a new output path")
        else:
            summary = fit_neural_process(dataset, train, validation, config, folder, device=device)
        summaries.append(summary)
    write_json(output/"training_summary.json", summaries)
    result = run_predictor(lambda fid: NeuralProcessPredictor(config, output/f"fold_{fid}", device=device),
                           output/"validation", config=config)
    for name, frame in result.items():
        frame["run_id"] = "v5-exploration-20261014"
        frame["study_version"] = "0.5-exploration"
        if name == "predictions":
            frame.to_parquet(output/"validation/predictions.parquet", index=False)
        else:
            frame.to_csv(output/"validation/metrics.csv", index=False)
    for path in (output/"validation/states").glob("*.parquet"):
        frame = pd.read_parquet(path)
        frame["run_id"] = "v5-exploration-20261014"
        frame["study_version"] = "0.5-exploration"
        frame.to_parquet(path, index=False)
    protocol_path = output/"validation/evaluation_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol.update(run_id="v5-exploration-20261014", study_version="0.5-exploration",
                    model_id=MODEL_ID, execution_device=device)
    write_json(protocol_path, protocol)
    return summaries


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v4.yaml")
    parser.add_argument("--output", default="runs/v5-exploration-20261014/neural_process")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(load_config(args.config), PROJECT_ROOT/args.output, device=args.device)
