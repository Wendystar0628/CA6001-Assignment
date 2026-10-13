"""Project-relative configuration and serializable experiment provenance."""
import copy
import json
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def merge_dict(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result

def load_config(path="configs/main.yaml"):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    parent = config.pop("extends", None)
    if parent:
        config = merge_dict(load_config(path.parent / parent), config)
    config["_root"] = str(PROJECT_ROOT)
    config["_config_path"] = str(path.resolve())
    d = config["data"]
    if d["depth_step_m"] <= 0 or d["depth_end_m"] <= d["depth_start_m"]:
        raise ValueError("Invalid depth window")
    if d["tau_mpa"] <= 0:
        raise ValueError("Low-qc threshold must be positive")
    return config

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(value):
        if hasattr(value, "item"):
            return value.item()
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=convert), encoding="utf-8")
