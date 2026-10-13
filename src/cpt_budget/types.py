"""Shared data boundaries. Only observed() and truth() reveal measurements."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

QUERY_COLUMNS = ["site_id", "profile_id", "location_id", "near_group_id", "x_m", "y_m",
                 "geometry_scale_m", "depth_index", "z_center_m"]

@dataclass
class PredictionBundle:
    frame: pd.DataFrame
    model_id: str

    def __post_init__(self):
        required = {"location_id", "depth_index", "pred_log_qc", "pred_qc_mpa"}
        if not required.issubset(self.frame):
            raise ValueError(f"Prediction missing columns: {required - set(self.frame)}")
        if self.frame.duplicated(["location_id", "depth_index"]).any():
            raise ValueError("Duplicate prediction keys")
        if not np.isfinite(self.frame[["pred_log_qc", "pred_qc_mpa"]].to_numpy()).all():
            raise ValueError("Non-finite point predictions")

@dataclass
class PreparedDataset:
    profiles: pd.DataFrame
    grid: pd.DataFrame
    sites: pd.DataFrame
    root: Path

    def locations(self, site_id=None):
        p = self.profiles.copy()
        if "is_representative" in p:
            p = p.loc[p.is_representative.fillna(False)]
        if "usable" in p:
            p = p.loc[p.usable.fillna(False)]
        p = p.loc[p.location_id.isin(self.grid.location_id.unique())]
        p = p.drop_duplicates("location_id")
        if site_id is not None:
            p = p.loc[p.site_id == site_id]
        extras = [c for c in ["geometry_scale_m", "investigation_group", "display_name"]
                  if c not in p and c in self.sites]
        if extras:
            p = p.merge(self.sites[["site_id", *extras]], on="site_id", validate="many_to_one")
        return p.sort_values("location_id").reset_index(drop=True)

    def _rows(self, ids):
        ids = list(ids)
        if len(ids) != len(set(ids)):
            raise ValueError("Location IDs must be unique")
        locations = self.locations()
        missing = set(ids) - set(locations.location_id)
        if missing:
            raise KeyError(f"Unknown/unusable locations: {sorted(missing)}")
        g = self.grid.loc[self.grid.location_id.isin(ids)].copy()
        extra = [c for c in QUERY_COLUMNS if c not in g and c in locations]
        if extra:
            g = g.merge(locations[["location_id", *extra]], on="location_id", validate="many_to_one")
        order = {loc: i for i, loc in enumerate(ids)}
        g["_order"] = g.location_id.map(order)
        return g.sort_values(["_order", "depth_index"]).drop(columns="_order").reset_index(drop=True)

    def observed(self, ids):
        return self._rows(ids)

    def query(self, ids):
        rows = self._rows(ids)
        return rows[[c for c in QUERY_COLUMNS if c in rows]].copy()

    def truth(self, ids):
        return self._rows(ids)

def load_dataset(config):
    root = Path(config["_root"])
    directory = root / config["project"]["prepared_dir"]
    return PreparedDataset(*(pd.read_parquet(directory / f"{name}.parquet")
                             for name in ("profiles", "grid", "sites")), root=root)
