"""Exploratory CPU-only, validation-state GP block-ALC one-step probe.

Never writes main results. Selection sees initial observations and candidate
geometry only; heldout truth is accessed only after both choices are frozen.
ALC integration uses candidate geometry, never evaluation labels or locations.
"""
from pathlib import Path
import json
import time
import warnings
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sklearn.exceptions import ConvergenceWarning
from cpt_budget.config import load_config
from cpt_budget.types import load_dataset
from cpt_budget.models.gp import GPPredictor, fit_gp_trend
from cpt_budget.acquisition import select_batch
from cpt_budget.evaluate import evaluate

warnings.filterwarnings("ignore", category=ConvergenceWarning)
OUT = Path(__file__).parent
ROOT = OUT.parents[2]
cfg = load_config(ROOT / "configs/main.yaml")
dataset = load_dataset(cfg)
folds = json.loads((ROOT / "results/main-20261013/folds.json").read_text())["folds"]
splits = json.loads((ROOT / "runs/main-20261013/inputs/replay_splits.json").read_text())
states = pd.read_parquet(ROOT / "runs/main-20261013/states.parquet")
states = states[(states.role == "validation") & (states.seed == 17) & (states.budget_n == 5)]
rows, scores_rows = [], []
t0 = time.perf_counter()

def gains(cov, groups, noise):
    values = []
    for ix in groups:
        cross = cov[:, ix]
        obs = cov[np.ix_(ix, ix)] + np.eye(len(ix)) * (noise + 1e-8)
        gain = np.einsum("ij,ji->i", cross, cho_solve(cho_factor(obs), cross.T))
        values.append(float(gain.mean()))
    return np.asarray(values)

for state in states.itertuples():
    split = next(s for s in splits if s["site_id"] == state.site_id and s["seed"] == state.seed)
    observed_ids = list(state.observed_ids)
    candidate_ids = sorted(state.remaining_ids)
    query = dataset.query(candidate_ids)
    coords = dataset.locations(state.site_id)[["location_id", "x_m", "y_m"]]
    candidate_coords = coords.set_index("location_id").loc[candidate_ids].reset_index()
    trend = fit_gp_trend(dataset, folds[state.fold_id]["train_sites"])
    gp = GPPredictor(cfg, trend=trend)
    gp.condition(dataset.observed(observed_ids))
    bundle = gp.predict(query)
    _, total_cov = gp.regressor.predict(gp._coordinates(query), return_cov=True)
    noise = float(gp.regressor.kernel_.k2.noise_level)
    cov = total_cov - np.eye(len(query)) * noise
    groups = [np.flatnonzero(query.location_id.eq(k)) for k in candidate_ids]
    initial = gains(cov, groups, noise)
    first = int(np.argmax(initial))
    ix = groups[first]
    cross = cov[:, ix]
    after = cov - cross @ cho_solve(cho_factor(cov[np.ix_(ix, ix)] + np.eye(len(ix))*(noise+1e-8)), cross.T)
    second_gain = gains(after, groups, noise)
    xy = candidate_coords[["x_m", "y_m"]].to_numpy()
    distances = np.linalg.norm(xy - xy[first], axis=1)
    eligible = (distances >= split["min_spacing_m"])
    eligible[first] = False
    if eligible.any():
        second = int(np.argmax(np.where(eligible, second_gain, -np.inf)))
    else:
        remaining = [i for i in range(len(candidate_ids)) if i != first]
        second = sorted(remaining, key=lambda i: (-distances[i], -second_gain[i], candidate_ids[i]))[0]
    choices = {"block_alc": [candidate_ids[first], candidate_ids[second]]}
    for policy in ("maximin", "uncertainty", "threshold_uncertainty"):
        choices[policy], _ = select_batch(coords[coords.location_id.isin(observed_ids)], candidate_coords,
                                         bundle, cfg, policy=policy,
                                         min_spacing_m=split["min_spacing_m"])
    for i, key in enumerate(candidate_ids):
        scores_rows.append(dict(fold_id=state.fold_id, site_id=state.site_id, seed=state.seed,
                                location_id=key, first_gain=initial[i], second_gain=second_gain[i],
                                selected=key in choices["block_alc"], noise_level=noise))
    for policy, selected in choices.items():
        ix = np.flatnonzero(query.location_id.isin(selected))
        cross = cov[:, ix]
        reduction = (cross * cho_solve(cho_factor(cov[np.ix_(ix, ix)] + np.eye(len(ix))*(noise+1e-8)), cross.T).T).sum(axis=1).mean()
        gp7 = GPPredictor(cfg, trend=trend)
        gp7.condition(dataset.observed(observed_ids + selected))
        metrics = evaluate(gp7.predict(dataset.query(state.evaluation_ids)), dataset.truth(state.evaluation_ids), cfg)
        rows.append(dict(fold_id=state.fold_id, site_id=state.site_id, seed=state.seed,
                         policy=policy, selected_ids=json.dumps(selected),
                         expected_integrated_variance_reduction=reduction,
                         initial_integrated_latent_variance=np.diag(cov).mean(),
                         **{k: metrics[k] for k in ["log_mae", "h_point_mae_m", "recall"]}))
    print(state.fold_id, state.site_id, "done", round(time.perf_counter()-t0,1), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "one_step_validation.csv", index=False)
    pd.DataFrame(scores_rows).to_csv(OUT / "one_step_alc_scores.csv", index=False)
print(pd.DataFrame(rows).groupby("policy")[["log_mae", "h_point_mae_m", "recall", "expected_integrated_variance_reduction"]].mean().to_string())
print("elapsed_seconds", round(time.perf_counter()-t0, 2))
