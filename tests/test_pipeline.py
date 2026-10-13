"""Only integration risks: result replacement and altered input provenance."""
from pathlib import Path
import pandas as pd
import pytest
from cpt_budget.pipeline import _append_tables, _bind_run_inputs
from cpt_budget.config import write_json

def test_replayed_task_replaces_rows_instead_of_duplicating(tmp_path):
    _append_tables(tmp_path, {"metrics": pd.DataFrame({"value": [1, 2]})}, "run", "a")
    _append_tables(tmp_path, {"metrics": pd.DataFrame({"value": [8]})}, "run", "b")
    _append_tables(tmp_path, {"metrics": pd.DataFrame({"value": [3, 4]})}, "run", "a")
    got = pd.read_parquet(tmp_path / "metrics.parquet")
    assert got.value.tolist() == [8, 3, 4]
    assert got.groupby("task_key").size().to_dict() == {"a": 2, "b": 1}

def test_changed_raw_input_cannot_resume_old_run(tmp_path):
    root = tmp_path / "project"
    source = root / "Data"
    prepared = root / "data_processed"
    source.mkdir(parents=True)
    prepared.mkdir()
    (source / "registry.json").write_text("[]")
    (source / "array.npz").write_bytes(b"original")
    for name in ("folds.json", "replay_splits.json", "preparation.json"):
        write_json(prepared / name, {})
    for name in ("episodes.parquet", "sites.parquet"):
        pd.DataFrame({"x": [1]}).to_parquet(prepared / name)
    config = {"_root": str(root), "project": {"input_dir": "Data", "prepared_dir": "data_processed"},
              "data": {"manifest": "registry.json", "arrays": "array.npz"}}
    run = root / "runs" / "trial"
    write_json(run / "run.json", {})
    _bind_run_inputs(run, config)
    (source / "array.npz").write_bytes(b"changed and longer")
    with pytest.raises(ValueError, match="Input data or grouping changed"):
        _bind_run_inputs(run, config)
