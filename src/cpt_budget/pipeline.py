"""Stage orchestration; inputs, folds and result provenance stay explicit."""
from __future__ import annotations
import copy
import hashlib
import importlib.metadata
import json
import subprocess
import shutil
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import write_json
from .types import load_dataset

MODELS = ("idw", "gp", "xgb", "tabpfn", "tabpfn_finetuned")
TABLES = ("predictions", "metrics", "recommendations", "acquisition_scores", "states")

def _public_config(config):
    return {k: v for k, v in config.items() if not k.startswith("_")}

def _config_signature(config):
    return hashlib.sha256(json.dumps(_public_config(config), sort_keys=True).encode()).hexdigest()

def _git(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None

def prepare(config):
    from .data.normalize import prepare_data
    dataset = prepare_data(config)
    return {"profiles": len(dataset.profiles), "locations": dataset.grid.location_id.nunique(),
            "sites": len(dataset.sites), "grid_rows": len(dataset.grid)}

def split(config):
    from .data.splits import make_splits
    result = make_splits(load_dataset(config), config)
    return {"folds": len(result["folds"]), "episodes": len(result["episodes"]),
            "replay_splits": len(result["replay_splits"]),
            "skipped_replay": result["skipped_replay"]}

def _ensure_prepared(config):
    directory = Path(config["_root"]) / config["project"]["prepared_dir"]
    signature = _input_signature(config)
    stamp = directory / "pipeline_input.json"
    existing = json.loads(stamp.read_text(encoding="utf-8")) if stamp.exists() else {}
    if not (directory / "grid.parquet").exists() or existing.get("signature") != signature:
        prepare(config)
        split(config)
        write_json(stamp, {"signature": signature, "inputs": _input_inventory(config)})
    elif not (directory / "folds.json").exists():
        split(config)
    return load_dataset(config)

def _input_inventory(config):
    root = Path(config["_root"])
    source = root / config["project"]["input_dir"]
    files = [source / config["data"]["manifest"], source / config["data"]["arrays"],
             source / config["data"].get("source_archive", "downloads/Global_CPT1196.zip"),
             root / config.get("split", {}).get("sites_config", "configs/sites.yaml")]
    inventory = {}
    for path in files:
        stat = path.stat() if path.exists() else None
        inventory[str(path.relative_to(root))] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} if stat else None
    return {"files": inventory, "settings": {key: config.get(key, {}) for key in ("data", "split", "episodes", "replay")}}

def _input_signature(config):
    return hashlib.sha256(json.dumps(_input_inventory(config), sort_keys=True).encode()).hexdigest()

def _bind_run_inputs(directory, config):
    """Pin the effective split and source metadata before consuming model results."""
    signature = _input_signature(config)
    path = directory / "inputs" / "input_inventory.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["signature"] != signature:
            raise ValueError("Input data or grouping changed. Create a new run.")
    else:
        write_json(path, {"signature": signature, **_input_inventory(config)})
        prepared = Path(config["_root"]) / config["project"]["prepared_dir"]
        for name in ("folds.json", "replay_splits.json", "preparation.json", "episodes.parquet", "sites.parquet"):
            shutil.copy2(prepared / name, path.parent / name)
        folds = json.loads((path.parent / "folds.json").read_text(encoding="utf-8"))
        _meta_update(directory, "unverified_grouping_sites", folds.get("unverified_grouping_sites", []))

def open_run(config, run_id=None, *, resume=False):
    root = Path(config["_root"])
    output = root / config["project"]["output_dir"]
    signature = _config_signature(config)
    if run_id is None and resume and (output / "latest.json").exists():
        latest = json.loads((output / "latest.json").read_text(encoding="utf-8"))
        candidate = output / latest["run_id"]
        meta = json.loads((candidate / "run.json").read_text(encoding="utf-8"))
        if meta["config_signature"] == signature:
            run_id = latest["run_id"]
    if run_id is None:
        mode = config.get("run", {}).get("mode", "main")
        run_id = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S") + f"-{mode}"
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run_id must be a directory name, not a path")
    directory = output / run_id
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "run.json").exists():
        meta = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        if meta["config_signature"] != signature:
            raise ValueError("Configuration changed. Use a new run rather than reusing old outputs.")
    else:
        versions = {}
        for package in ("numpy", "pandas", "scikit-learn", "xgboost", "torch", "tabpfn"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        meta = {"run_id": run_id, "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "mode": config.get("run", {}).get("mode", "main"), "config": _public_config(config),
                "config_signature": signature, "code_commit": _git(root, "rev-parse", "HEAD"),
                "working_tree_dirty": bool(_git(root, "status", "--porcelain")),
                "versions": versions, "stages": {}, "completed_tasks": [], "failures": []}
        write_json(directory / "run.json", meta)
    write_json(output / "latest.json", {"run_id": run_id})
    return directory

def _meta_update(directory, key, value):
    path = directory / "run.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta[key] = value
    write_json(path, meta)

def _stage(directory, name, detail):
    meta = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    meta["stages"][name] = {"completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), **detail}
    write_json(directory / "run.json", meta)

def _folds(config):
    path = Path(config["_root"]) / config["project"]["prepared_dir"] / "folds.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    return value["folds"]

def _episodes(config, fold_id, role):
    path = Path(config["_root"]) / config["project"]["prepared_dir"] / "episodes.parquet"
    frame = pd.read_parquet(path)
    return frame.loc[(frame.fold_id == fold_id) & (frame.role == role)].to_dict("records")

def fit(config, directory, fold_id=None, models=None):
    dataset = _ensure_prepared(config)
    _bind_run_inputs(directory, config)
    models = list(models or config.get("run", {}).get("models", MODELS))
    summaries = []
    for fold in _folds(config):
        fid = fold["fold_id"]
        if fold_id is not None and fid != fold_id:
            continue
        output = directory / "checkpoints" / f"fold_{fid}"
        output.mkdir(parents=True, exist_ok=True)
        artifact_path = output / "artifact.json"
        if artifact_path.exists():
            existing = json.loads(artifact_path.read_text(encoding="utf-8"))
            if set(models).issubset(existing.get("models", [])):
                required = [output / "gp_trend.json"]
                if "xgb" in models:
                    required.append(output / "xgb_model.json")
                if "tabpfn_finetuned" in models:
                    required.append(Path(existing["tabpfn_training"]["checkpoint_path"]))
                if all(p.exists() for p in required):
                    summaries.append(existing)
                    continue
        from .models.gp import fit_gp_trend
        trend = fit_gp_trend(dataset, fold["train_sites"])
        write_json(output / "gp_trend.json", {str(k): float(v) for k, v in trend.items()})
        train = _episodes(config, fid, "train")
        validation = _episodes(config, fid, "validation")
        if not train:
            raise ValueError(f"No valid training episodes in fold {fid}")
        info = {"fold_id": fid, "train_sites": fold["train_sites"],
                "validation_sites": fold["validation_sites"], "models": ["gp", "idw"]}
        if "xgb" in models:
            from .models.xgb import fit_xgb
            artifact = fit_xgb(dataset, train + validation, config, output)
            info["xgb_artifact"] = str(Path(artifact).relative_to(directory))
            info["models"].append("xgb")
        if "tabpfn_finetuned" in models:
            from .models.tabpfn_train import fit_tabpfn
            result = fit_tabpfn(dataset, train, validation, config, output)
            info["tabpfn_training"] = result
            info["models"].append("tabpfn_finetuned")
        if "tabpfn" in models:
            info["models"].append("tabpfn")
        write_json(output / "artifact.json", info)
        summaries.append(info)
        _stage(directory, f"fit_fold_{fid}", info)
    return summaries

def load_predictor(model_id, config, directory, fold_id):
    output = directory / "checkpoints" / f"fold_{fold_id}"
    if model_id == "idw":
        from .models.idw import IDWPredictor
        return IDWPredictor(config)
    if model_id == "gp":
        from .models.gp import GPPredictor
        trend_path = output / "gp_trend.json"
        if not trend_path.exists():
            raise FileNotFoundError("Source GP trend missing; run fit first")
        trend = pd.Series({int(k): v for k, v in json.loads(trend_path.read_text()).items()})
        return GPPredictor(config, trend=trend)
    if model_id == "xgb":
        from .models.xgb import XGBPredictor
        return XGBPredictor(config, artifact=output / "xgb_model.json")
    if model_id in ("tabpfn", "tabpfn_finetuned"):
        from .models.tabpfn_adapter import TabPFNPredictor
        artifact = None
        if model_id == "tabpfn_finetuned":
            info = json.loads((output / "artifact.json").read_text(encoding="utf-8"))
            artifact = Path(info["tabpfn_training"]["checkpoint_path"])
            if not artifact.is_absolute():
                artifact = Path(config["_root"]) / artifact
        return TabPFNPredictor(config, artifact=artifact)
    raise ValueError(f"Unknown model {model_id}")

def _replay_splits(config):
    path = Path(config["_root"]) / config["project"]["prepared_dir"] / "replay_splits.json"
    return json.loads(path.read_text(encoding="utf-8"))

def _append_tables(directory, tables, run_id, task_key):
    """Replace only this task's rows, making interrupted consolidation idempotent."""
    for name, frame in tables.items():
        if frame is None or frame.empty:
            continue
        frame = frame.copy()
        frame["run_id"] = run_id
        frame["task_key"] = task_key
        path = directory / f"{name}.parquet"
        if path.exists():
            old = pd.read_parquet(path)
            if "task_key" in old:
                old = old.loc[old.task_key != task_key]
            frame = pd.concat([old, frame], ignore_index=True)
        temp = path.with_suffix(".tmp.parquet")
        frame.to_parquet(temp, index=False)
        temp.replace(path)

def replay(config, directory, *, fold_id=None, models=None, policies=None, role="test", site_id=None, budgets=None, seeds=None):
    from .replay import run_replay
    dataset = _ensure_prepared(config)
    _bind_run_inputs(directory, config)
    models = list(models or config.get("run", {}).get("models", MODELS))
    policies = list(policies or config["replay"]["policies"])
    splits = _replay_splits(config)
    task_count = 0
    for fold in _folds(config):
        fid = fold["fold_id"]
        if fold_id is not None and fid != fold_id:
            continue
        site_role = "validation" if role == "pilot" else role
        allowed_sites = fold[f"{site_role}_sites"]
        selected_splits = [s for s in splits if s["site_id"] in allowed_sites
                           and (site_id is None or s["site_id"] == site_id)
                           and (seeds is None or s["seed"] in seeds)]
        if not selected_splits:
            continue
        for model_id in models:
            predictor = None
            model_policies = policies if model_id in ("gp", "tabpfn", "tabpfn_finetuned") else ["maximin"]
            for spec in selected_splits:
                for policy in model_policies:
                    local = copy.deepcopy(spec)
                    local["role"] = role
                    if budgets is not None:
                        if not set(budgets).issubset(local["budgets"]):
                            raise ValueError("Requested budget not in frozen split")
                        local["budgets"] = list(budgets)
                    key = f"{role}:{fid}:{spec['site_id']}:{spec['seed']}:{model_id}:{policy}:{local['budgets']}"
                    meta = json.loads((directory / "run.json").read_text(encoding="utf-8"))
                    if key in meta["completed_tasks"]:
                        continue
                    if predictor is None:
                        predictor = load_predictor(model_id, config, directory, fid)
                    started = time.perf_counter()
                    print(f"Replay {key}", flush=True)
                    try:
                        tables = run_replay(dataset, local, predictor, config, policy=policy, fold_id=fid)
                        _append_tables(directory, tables, directory.name, key)
                    except Exception as exc:
                        meta["failures"].append({"task": key, "error": f"{type(exc).__name__}: {exc}"})
                        write_json(directory / "run.json", meta)
                        raise
                    meta["completed_tasks"].append(key)
                    meta.setdefault("task_seconds", {})[key] = time.perf_counter() - started
                    write_json(directory / "run.json", meta)
                    task_count += 1
            del predictor
    if not (directory / "metrics.parquet").exists():
        raise ValueError("No replay tasks matched the requested fold/site/seed")
    _stage(directory, f"replay_{role}", {"new_tasks": task_count})
    return {"new_tasks": task_count, "run": directory.name}

def freeze_quality(config, directory, fold_id=None):
    from .economics import freeze_quality_targets
    replay(config, directory, fold_id=fold_id, models=["gp"], policies=["maximin"], role="validation")
    metrics = pd.read_parquet(directory / "metrics.parquet")
    targets = freeze_quality_targets(metrics, config)
    write_json(directory / "summary" / "quality_targets.json", targets)
    _stage(directory, "freeze_quality", {"targets": targets})
    return targets

def summarize(config, directory):
    from .economics import summarize_results
    metrics = pd.read_parquet(directory / "metrics.parquet")
    target_path = directory / "summary" / "quality_targets.json"
    targets = json.loads(target_path.read_text(encoding="utf-8")) if target_path.exists() else {"folds": {}}
    tables = summarize_results(metrics, targets, config)
    summary = directory / "summary"
    summary.mkdir(exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(summary / f"{name}.csv", index=False, encoding="utf-8-sig")
    _stage(directory, "summarize", {"tables": list(tables)})
    return {name: len(frame) for name, frame in tables.items()}

def export(config, directory, preset="both", figure=None):
    from .figures import export_figures
    paths = export_figures(directory, load_dataset(config), config, preset=preset, figure=figure)
    _stage(directory, "export", {"files": [str(p.relative_to(directory)) for p in paths]})
    return [str(p) for p in paths]

def pilot(config, directory, *, site_id=None, models=None):
    _ensure_prepared(config)
    candidates = _replay_splits(config)
    seed = config.get("run", {}).get("seeds", [17])[0]
    choice = None
    for fold in _folds(config):
        for spec in candidates:
            if spec["seed"] == seed and spec["site_id"] in fold["validation_sites"]:
                if site_id is None or spec["site_id"] == site_id:
                    choice = (fold, spec)
                    break
        if choice:
            break
    if choice is None:
        raise ValueError("No matching source-validation replay site for pilot")
    fold, spec = choice
    fid = fold["fold_id"]
    chosen_models = models or config.get("run", {}).get("models", ["idw", "gp"])
    fit(config, directory, fid, models=chosen_models)
    result = replay(config, directory, fold_id=fid, models=chosen_models, role="pilot",
                    site_id=spec["site_id"], budgets=config.get("run", {}).get("budgets", [5, 7]), seeds=[seed])
    summarize(config, directory)
    export(config, directory)
    _stage(directory, "pilot", {"site_id": spec["site_id"], "fold_id": fid, "models": chosen_models})
    return result
