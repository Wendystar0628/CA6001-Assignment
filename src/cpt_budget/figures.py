"""Evidence-driven figures and replay frames. Exporting never calls a predictor.

Figure contract: G1 describes geometry and measured variation; G2 explains the
information boundary; G3 shows reconstruction and marginal reliability; G4
explains the saved decision; G5 separates model and policy comparisons; G6
retains censoring. Quantitative panels use saved observations, predictions and
paired run identifiers only. Incomplete experiments remain explicitly labelled.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

from .plot_style import (ALERT, INK, MUTED, TEAL, MODEL_COLORS, MODEL_NAMES,
                         POLICY_LINES, POLICY_NAMES, canvas, figure_style,
                         finish_axis, heading, method_name, save_figure, scale_bar)


KEYS = ["fold_id", "site_id", "seed", "model_id", "policy_id", "budget_n"]


def _read(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _ids(value):
    if value is None:
        return []
    if isinstance(value, str):
        try:
            return list(json.loads(value))
        except (ValueError, TypeError):
            return [value]
    if np.isscalar(value):
        return []
    return list(value)


def _column(frame, names):
    return next((name for name in names if name in frame), None)


def _select(frame, **values):
    result = frame
    for name, value in values.items():
        if name in result and value is not None:
            result = result.loc[result[name].astype(str) == str(value)]
    return result.copy()


def _evaluation(frame):
    if "prediction_role" in frame:
        return frame[frame.prediction_role == "evaluation"].copy()
    if "role" in frame and frame.role.isin(["candidate", "evaluation"]).any():
        return frame[frame.role == "evaluation"].copy()
    if "truth_qc_mpa" in frame:
        return frame[frame.truth_qc_mpa.notna()].copy()
    return pd.DataFrame()


def _candidate(frame):
    for key in ["prediction_role", "role"]:
        if key in frame and frame[key].isin(["candidate", "evaluation"]).any():
            return frame[frame[key] == "candidate"].copy()
    if "truth_qc_mpa" in frame:
        return frame[frame.truth_qc_mpa.isna()].copy()
    return pd.DataFrame()


def _depth(frame):
    name = _column(frame, ["depth_m", "depth_center_m", "z_center_m", "z_m"])
    return frame[name].to_numpy(float) if name else 2.125 + .25 * frame.depth_index.to_numpy(float)


def _site_name(site):
    return {"newlock": "NL", "hollywood": "HW", "christchurch_stadium": "CS",
            "christchurch_metro_sports_facility": "CMS", "lower_hutt_town_hall": "LH",
            "marshland": "ML", "tauranga_farmers_building": "TF", "wellington_airport": "WA"}.get(
                str(site), "".join(part[:1].upper() for part in str(site).split("_")))


def _full_site_name(site):
    return "New Lock" if str(site) == "newlock" else str(site).replace("_", " ").title()


def _location_labels(locations):
    if "source_file" in locations:
        return {row.location_id: "CPT " + Path(str(row.source_file)).stem[-14:]
                for row in locations.itertuples()}
    return {loc: f"CPT {i + 1:02d}" for i, loc in enumerate(sorted(locations.location_id))}


def _case(predictions, dataset, config):
    evaluation = _evaluation(predictions)
    choices = evaluation.site_id.astype(str).unique() if not evaluation.empty else dataset.locations().site_id.astype(str).unique()
    preferred = str(config.get("presentation", {}).get("preferred_site", "new_lock"))
    site = next((x for x in choices if preferred.lower().replace("_", "") in x.lower().replace("_", "")), sorted(choices)[0])
    rows = _select(evaluation, site_id=site)
    seed = config.get("presentation", {}).get("representative_seed", 17)
    if not rows.empty and "seed" in rows and seed not in rows.seed.values:
        seed = sorted(rows.seed.unique())[0]
    rows = _select(rows, seed=seed)
    models = rows.model_id.unique() if "model_id" in rows else []
    model = next((x for x in ["tabpfn_finetuned", "gp", "tabpfn", "xgb", "idw"] if x in models), None)
    rows = _select(rows, model_id=model)
    policies = rows.policy_id.unique() if "policy_id" in rows else []
    policy = next((x for x in ["threshold_uncertainty", "maximin", "uncertainty"] if x in policies), None)
    rows = _select(rows, policy_id=policy)
    fold = sorted(rows.fold_id.unique())[0] if "fold_id" in rows and not rows.empty else None
    rows = _select(rows, fold_id=fold)
    geometry = dataset.locations(site)
    location = None
    if not rows.empty:
        available = geometry[geometry.location_id.isin(rows.location_id.unique())]
        if not available.empty:
            center = geometry[["x_m", "y_m"]].mean().to_numpy()
            distance = ((available[["x_m", "y_m"]].to_numpy() - center) ** 2).sum(axis=1)
            location = available.iloc[int(np.argmin(distance))].location_id
    return {"site_id": site, "seed": seed, "model_id": model, "policy_id": policy,
            "fold_id": fold, "location_id": location}


def _select_result_scope(metrics, predictions, states, scores):
    """Choose one scientific role before case selection, calibration or averaging.

    Prediction ``role`` describes evaluation/candidate information, so its
    independent ``split_role`` must be used for scientific-scope filtering.
    Mixing source validation and held-out test rows would duplicate sites and
    give the GP reference an incompatible population; never silently do that.
    """
    tables = {"metrics": metrics, "predictions": predictions, "states": states, "scores": scores}
    columns = {"metrics": "role", "predictions": "split_role", "states": "role", "scores": "role"}
    available = set()
    for name, frame in tables.items():
        column = columns[name]
        if column in frame:
            available.update(frame[column].dropna().astype(str))
    scope = next((role for role in ("test", "pilot", "validation") if role in available), None)
    if scope is None:
        return "unlabelled", metrics, predictions, states, scores
    filtered = {}
    for name, frame in tables.items():
        column = columns[name]
        if frame.empty:
            filtered[name] = frame.copy()
        elif column not in frame:
            raise ValueError(f"Cannot safely export {scope} figures: {name} has no {column} scope column")
        else:
            filtered[name] = frame.loc[frame[column].astype(str).eq(scope)].copy()
    return scope, filtered["metrics"], filtered["predictions"], filtered["states"], filtered["scores"]


def _run_label(run_dir, metrics, scope="unlabelled"):
    meta = {}
    if (run_dir / "run.json").exists():
        meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    role_columns = [x for x in ["role", "split_role"] if x in metrics]
    pilot = "pilot" in str(meta.get("stage", "")).lower() or "pilot" in run_dir.name.lower()
    pilot = pilot or any(metrics[c].astype(str).eq("pilot").any() for c in role_columns)
    incomplete = metrics.empty or metrics.site_id.nunique() < 2 or metrics.budget_n.max() < 13
    if scope == "validation":
        return "VALIDATION ONLY · quality-target reference; not test evidence"
    if scope == "pilot" or (scope == "unlabelled" and pilot):
        return "PILOT · observed results only"
    if scope == "test":
        return "TEST · partial run, observed results only" if incomplete else "TEST · held-out results only"
    return "SCOPE UNRECORDED · descriptive results only"


def _state(states, case, budget):
    rows = _select(states, **{k: v for k, v in case.items() if k != "location_id"}, budget_n=budget)
    return rows.iloc[0] if len(rows) else pd.Series(dtype=object)


def _geometry_limits(locations):
    xy = locations[["x_m", "y_m"]].to_numpy(float)
    low, high = xy.min(axis=0), xy.max(axis=0)
    span = np.maximum(high - low, 1.)
    extent = (low[0] - span[0] * .10, high[0] + span[0] * .10,
              low[1] - span[1] * .10, high[1] + span[1] * .10)
    # The densest geometry-only neighbourhood defines one persistent core view.
    from scipy.spatial import cKDTree
    count = max(3, int(np.ceil(len(xy) * .55)))
    distances, indices = cKDTree(xy).query(xy, k=min(count, len(xy)))
    if distances.ndim == 1:
        return extent, extent
    center = int(np.argmin(distances[:, -1]))
    core_xy = xy[indices[center]]
    core_low, core_high = core_xy.min(0), core_xy.max(0)
    core_span = np.maximum(core_high - core_low, span * .03)
    core = (core_low[0] - core_span[0] * .16, core_high[0] + core_span[0] * .16,
            core_low[1] - core_span[1] * .16, core_high[1] + core_span[1] * .16)
    return extent, core


def _map(ax, locations, state=None, selected=None, *, limits=None, scores=None, score_range=None):
    state = {} if state is None else state
    selected = [] if selected is None else list(selected)
    observed = set(_ids(state.get("observed_ids", [])))
    evaluation = set(_ids(state.get("evaluation_ids", [])))
    candidates = locations[~locations.location_id.isin(observed | evaluation)]
    size = 27 if plt.rcParams["font.size"] > 12 else 13
    if scores is not None and not scores.empty:
        colored = candidates.merge(scores[["location_id", "score"]].drop_duplicates("location_id"), on="location_id", how="left")
        lower, upper = score_range if score_range is not None else (None, None)
        ax.scatter(colored.x_m, colored.y_m, c=colored.score, cmap="cividis", vmin=lower, vmax=upper, s=size, zorder=2)
    else:
        ax.scatter(candidates.x_m, candidates.y_m, c="#C7CED2", s=size, label="Candidate", zorder=2)
    for ids, marker, fill, edge, label, factor in [
        (evaluation, "s", "white", MUTED, "Fixed evaluation", 1.7),
        (observed, "o", INK, "white", "Measured", 2.8)]:
        rows = locations[locations.location_id.isin(ids)]
        ax.scatter(rows.x_m, rows.y_m, s=size * factor, marker=marker, facecolor=fill,
                   edgecolor=edge, linewidth=.8, zorder=4, label=label)
    for rank, identifier in enumerate(selected, 1):
        rows = locations[locations.location_id == identifier]
        if rows.empty:
            continue
        row = rows.iloc[0]
        measured = identifier in observed
        ax.scatter(row.x_m, row.y_m, s=size * 7, marker="D", facecolor="none" if measured else TEAL,
                   edgecolor=TEAL if measured else "white", linewidth=1.4, zorder=6)
        ax.annotate(str(rank), (row.x_m, row.y_m), xytext=(7, 7), textcoords="offset points",
                    color=TEAL, fontweight="bold", zorder=7,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": .7, "alpha": .9})
    if limits:
        ax.set_xlim(limits[:2]); ax.set_ylim(limits[2:])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Local x (m)"); ax.set_ylabel("Local y (m)")
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3)); ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    scale_bar(ax)


def _profile_limits(rows, config):
    bounds = list(config.get("presentation", {}).get("profile_axis_seed_range_mpa", [.05, 100.]))
    values = []
    for col in ["truth_qc_mpa", "pred_qc_mpa"]:
        if col in rows:
            values.extend(rows[col].to_numpy(float))
    for col in ["q05_log", "q95_log"]:
        if col in rows:
            values.extend(np.exp(np.clip(rows[col].to_numpy(float), -40, 40)))
    finite = np.asarray(values)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if len(finite):
        bounds = [min(bounds[0], finite.min() / 1.12), max(bounds[1], finite.max() * 1.12)]
    return bounds


def _raw_profile(dataset, identifier, config):
    """Original fine sampling is retrospective context, never a prediction input."""
    rows = dataset.locations()
    rows = rows[rows.location_id == identifier]
    if rows.empty or "array_key" not in rows:
        return None
    path = Path(config.get("_root", dataset.root)) / config.get("project", {}).get("input_dir", "Data") / config.get("data", {}).get("arrays", "cpt_screened_depth_qc.npz")
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as arrays:
        key = str(rows.iloc[0].array_key)
        if key not in arrays:
            return None
        values = np.asarray(arrays[key], dtype=float)
    valid = np.isfinite(values[:, :2]).all(axis=1) & (values[:, 1] > 0) & (values[:, 0] >= 2) & (values[:, 0] <= 8)
    return values[valid, :2]


def _profile(ax, rows, model, limits, *, truth=True, probability_ax=None, title=None, raw=None):
    rows = rows.sort_values("depth_index")
    z = _depth(rows)
    color = MODEL_COLORS.get(model, TEAL)
    if raw is not None and truth:
        ax.plot(raw[:, 1], raw[:, 0], color="#CDD3D7", lw=.7, zorder=1, label="Raw observations")
    for lower, upper, alpha in [("q05_log", "q95_log", .12), ("q10_log", "q90_log", .19), ("q25_log", "q75_log", .32)]:
        if lower in rows and upper in rows and rows[lower].notna().any():
            ax.fill_betweenx(z, np.exp(rows[lower]), np.exp(rows[upper]), color=color, alpha=alpha, lw=0)
    prediction = np.exp(rows.q50_log) if "q50_log" in rows and rows.q50_log.notna().any() else rows.pred_qc_mpa
    ax.plot(prediction, z, color=color, lw=2, label="Prediction")
    if truth and "truth_qc_mpa" in rows and rows.truth_qc_mpa.notna().any():
        ax.step(rows.truth_qc_mpa, z, color=INK, lw=1.1, where="mid", label="Grid observation")
        ax.scatter(rows.truth_qc_mpa, z, s=8, color=INK, zorder=5)
        if "q05_log" in rows:
            outside = (np.log(rows.truth_qc_mpa) < rows.q05_log) | (np.log(rows.truth_qc_mpa) > rows.q95_log)
            ax.scatter(rows.loc[outside, "truth_qc_mpa"], z[outside], marker="D", s=16, color=ALERT, zorder=6)
    ax.axvline(1, color=MUTED, ls="--", lw=.8)
    ax.set_xscale("log"); ax.set_xlim(limits); ax.set_ylim(8, 2)
    ax.set_yticks([2, 4, 6, 8]); ax.set_xticks([.1, 1, 10, 100])
    ax.set_xticklabels(["0.1", "1", "10", "100"])
    ax.set_xlabel("$q_c$ (MPa)"); ax.set_ylabel("Depth (m)")
    ax.grid(axis="x", alpha=.7)
    if title:
        ax.set_title(title, loc="left")
    pcol = _column(rows, ["p_low_1", "p_low_10", "p_low_1_0"])
    if probability_ax is not None and pcol:
        probability_ax.imshow(rows[pcol].to_numpy()[:, None], cmap="cividis", vmin=0, vmax=1,
                              extent=[0, 1, 8, 2], aspect="auto", interpolation="nearest")
        probability_ax.set_ylim(8, 2); probability_ax.set_xticks([]); probability_ax.set_yticks([])
        probability_ax.set_title("P(low)", fontsize=plt.rcParams["xtick.labelsize"], pad=8)
        if truth and "truth_qc_mpa" in rows:
            for depth, is_low in zip(z, rows.truth_qc_mpa < 1):
                if is_low:
                    probability_ax.add_patch(Rectangle((1.22, depth - .125), .30, .25, color=INK, clip_on=False))


def _g1(dataset, states, case, preset, label):
    locations = dataset.locations(case["site_id"])
    extent, core = _geometry_limits(locations)
    budgets = _select(states, site_id=case["site_id"]).get("budget_n", pd.Series([5]))
    state = _state(states, case, int(budgets.min()) if len(budgets) else 5)
    fig = canvas(preset, 108)
    gs = fig.add_gridspec(1, 2, left=.10, right=.96, top=.77, bottom=.22, wspace=.38)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    _map(axes[0], locations, state, limits=extent)
    _map(axes[1], locations, state, limits=core)
    axes[0].set_title("a   Full site", loc="left")
    axes[1].set_title("b   Fixed geometry-defined core", loc="left")
    axes[0].add_patch(Rectangle((core[0], core[2]), core[1] - core[0], core[3] - core[2], fill=False, ec=TEAL, lw=1.1))
    fig.add_artist(ConnectionPatch((core[1], core[3]), (0, 1), "data", "axes fraction", axesA=axes[0], axesB=axes[1], color=TEAL, lw=.6))
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.0, -.17), ncol=3, handletextpad=.4, columnspacing=1.2)
    heading(fig, "Geometry sets the investigation problem", f"{_full_site_name(case['site_id'])} · {len(locations)} distinct locations · {label}", preset,
            footer="Equal metre scales within each map. Core selection uses coordinates only.")
    return [("G1_site_geometry", fig)]


def _g1_thickness(dataset, config, preset, label):
    grid = dataset.grid.copy()
    if grid.empty or "qc_mpa" not in grid:
        return []
    step = float(config.get("data", {}).get("depth_step_m", .25))
    grid["thickness"] = (grid.qc_mpa < 1).astype(float) * step
    thickness = grid.groupby(["site_id", "location_id"], sort=True).thickness.sum().reset_index()
    counts = thickness.groupby("site_id").size()
    # Primary replay sites have >=20 positions; smaller source sites are retained in data.
    sites = sorted(counts[counts >= config.get("replay", {}).get("min_site_locations", 20)].index)
    if not sites:
        sites = sorted(counts.index)
    fig = canvas(preset, max(90, 10 * len(sites) + 35))
    ax = fig.add_axes([.30, .20, .62, .56])
    for y, site in enumerate(sites):
        values = thickness.loc[thickness.site_id == site].sort_values("location_id").thickness.to_numpy()
        jitter = np.sin(np.arange(len(values)) * 2.399963) * .16
        ax.scatter(values, y + jitter, s=15 if preset == "report" else 35, color=TEAL, alpha=.4, edgecolor="none")
        ax.scatter(np.median(values), y, s=50 if preset == "report" else 100, color=INK, marker="|")
    ax.set_yticks(range(len(sites)), [f"{_site_name(s)} (n={counts[s]})" for s in sites])
    ax.set_xlim(-.1, 6.1); ax.set_xticks([0, 2, 4, 6]); ax.invert_yaxis()
    ax.set_xlabel("Measured low-cone-resistance thickness (m)")
    finish_axis(ax, grid="x")
    heading(fig, "Low-resistance thickness varies between sites", "Retrospective data description · grid observations, 2–8 m depth", preset,
            footer="Each point is a distinct location; dark ticks mark site medians. Threshold: 1 MPa.")
    return [("G1_measured_thickness", fig)]


def _g2(preset):
    fig = canvas(preset, 140)
    ax = fig.add_axes([.045, .09, .91, .71]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fontsize = 17 if preset == "slides" else 8.5
    def box(x, y, w, h, text, color=TEAL):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.009,rounding_size=.012", facecolor=color, alpha=.085, edgecolor=color, linewidth=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, linespacing=1.4)
    def arrow(start, end, color=MUTED, ls="-"):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, color=color, lw=1, linestyle=ls))
    box(.015, .70, .20, .17, "Measured support row\nx, y, z, ... | log(qc)")
    box(.015, .45, .20, .17, "Unmeasured query row\nx, y, z, ... | ?")
    box(.29, .56, .21, .24, "16 shared features\nSupport: exclude itself\nQuery: observed only")
    box(.56, .56, .19, .24, "TabPFN Transformer\nTarget context only\nNo target gradients", "#CC79A7")
    box(.81, .67, .17, .19, "Depthwise\n$q_c$ distribution", "#0072B2")
    box(.81, .40, .17, .19, "P($q_c$ < 1 MPa)\nExpected thickness", "#0072B2")
    arrow((.215, .785), (.29, .71)); arrow((.215, .53), (.29, .64))
    arrow((.50, .68), (.56, .68)); arrow((.75, .68), (.81, .76)); arrow((.895, .67), (.895, .59))
    ax.text(.525, .45, "Width + threshold crossing\n→ next batch of two positions", ha="center", va="center", color=TEAL, fontsize=fontsize)
    arrow((.81, .72), (.72, .48), TEAL)
    ax.text(.89, .34, r"$H_{expected}=\sum p_{low}\times0.25$ m", ha="center", va="center", fontsize=fontsize)
    ax.axhline(.28, color="#DCE3E7", lw=1)
    for y, tag, text, color in [(.22, "01  PRETRAIN", "Provider weights: general tabular prior", MUTED),
                               (.135, "02  FINE-TUNE", "Source episodes: update weights with source labels", "#CC79A7"),
                               (.05, "03  INFER", "Target site: replace observed context; weights stay fixed", TEAL)]:
        ax.text(.02, y, tag, color=color, va="center", fontweight="bold", fontsize=fontsize)
        ax.text(.28, y, text, va="center", fontsize=fontsize)
    heading(fig, "Historical learning and current observations have different roles", "Method diagram · planned information flow, not a performance claim", preset,
            footer="Evaluation truth → error calculation only. Expected thickness is not a joint confidence interval.")
    return [("G2_information_flow", fig)]


def _g3(predictions, case, config, preset, label, dataset=None):
    rows = _select(_evaluation(predictions), **case)
    if rows.empty:
        return []
    budgets = sorted(rows.budget_n.unique())
    budgets = [budgets[0], budgets[-1]] if len(budgets) > 1 else [budgets[0]]
    fig = canvas(preset, 127)
    gs = fig.add_gridspec(1, len(budgets) * 2, width_ratios=[1, .055] * len(budgets), left=.11, right=.89, top=.78, bottom=.29, wspace=.38)
    limits = _profile_limits(rows, config)
    raw = _raw_profile(dataset, case["location_id"], config) if dataset is not None else None
    if raw is not None and len(raw):
        limits = [min(limits[0], raw[:, 1].min() / 1.12), max(limits[1], raw[:, 1].max() * 1.12)]
    for i, budget in enumerate(budgets):
        ax = fig.add_subplot(gs[0, 2 * i]); pa = fig.add_subplot(gs[0, 2 * i + 1])
        _profile(ax, rows[rows.budget_n == budget], case["model_id"], limits, probability_ax=pa, raw=raw,
                 title=f"{chr(97 + i)}   {int(budget)} measured CPTs")
        if i > 0:
            ax.set_ylabel("")
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="cividis"), cax=fig.add_axes([.38, .135, .25, .02]), orientation="horizontal")
    colorbar.set_ticks([0, .5, 1]); colorbar.set_label("P(low): probability of $q_c$ < 1 MPa")
    colorbar.ax.xaxis.set_label_position("top")
    heading(fig, "One fixed profile, compared at observed budgets", f"{_full_site_name(case['site_id'])} · seed {case['seed']} · {method_name(case['model_id'], case['policy_id'])}", preset,
            footer=(("Grey: raw samples; black: grid + low-cell strip. Steps do not mark layer boundaries.\n" if raw is not None else
                     "Black: grid + low-cell strip. Steps do not mark layer boundaries.\n") +
                    "50/80/90% depthwise intervals; no simultaneous guarantee. " + label))
    return [("G3_profile_comparison", fig)]


def _calibration(predictions):
    rows = _evaluation(predictions)
    if rows.empty or "q05_log" not in rows or "truth_qc_mpa" not in rows:
        return pd.DataFrame()
    rows = _select(rows, policy_id="maximin")
    records = []
    keys = [k for k in KEYS if k in rows]
    for values, group in rows.groupby(keys, dropna=False):
        truth = np.log(group.truth_qc_mpa.to_numpy(float))
        for nominal, lo, hi in [(.5, "q25_log", "q75_log"), (.8, "q10_log", "q90_log"), (.9, "q05_log", "q95_log")]:
            if lo not in group or hi not in group:
                continue
            valid = np.isfinite(truth) & group[lo].notna().to_numpy() & group[hi].notna().to_numpy()
            if not valid.any():
                continue
            lower, upper = group[lo].to_numpy()[valid], group[hi].to_numpy()[valid]
            records.append({**dict(zip(keys, values)), "nominal_coverage": nominal,
                            "empirical_coverage": np.mean((truth[valid] >= lower) & (truth[valid] <= upper)),
                            "mean_width_log": np.mean(upper - lower), "n_cells": int(valid.sum())})
    return pd.DataFrame(records)


def _g3_calibration(predictions, preset, label):
    calibration = _calibration(predictions)
    if calibration.empty:
        return []
    budget = 13 if 13 in calibration.budget_n.values else calibration.budget_n.max()
    data = calibration[calibration.budget_n == budget]
    site = data.groupby(["site_id", "model_id", "nominal_coverage"], as_index=False)[["empirical_coverage", "mean_width_log"]].mean()
    fig = canvas(preset, 110)
    gs = fig.add_gridspec(1, 2, left=.11, right=.95, bottom=.25, top=.75, wspace=.44)
    left, right = [fig.add_subplot(gs[0, i]) for i in range(2)]
    left.plot([.4, 1], [.4, 1], color=MUTED, ls="--", lw=.8)
    right.axhline(.9, color=MUTED, ls="--", lw=.8)
    for model in sorted(site.model_id.unique()):
        series = site[site.model_id == model]; color = MODEL_COLORS.get(model, TEAL)
        for _, group in series.groupby("site_id"):
            left.plot(group.nominal_coverage, group.empirical_coverage, "o-", color=color, alpha=.22, markersize=3)
        average = series.groupby("nominal_coverage").empirical_coverage.mean()
        left.plot(average.index, average.values, "D-", color=color, markersize=5, label=MODEL_NAMES.get(model, model))
        ninety = series[series.nominal_coverage == .9]
        right.scatter(ninety.mean_width_log, ninety.empirical_coverage, color=color, alpha=.35, s=24)
        right.scatter(ninety.mean_width_log.mean(), ninety.empirical_coverage.mean(), marker="D", color=color, s=55)
    left.set(xlabel="Nominal coverage", ylabel="Empirical coverage", xlim=(.45, .95), ylim=(0, 1.02), xticks=[.5, .8, .9])
    right.set(xlabel="Mean 90% width (log MPa)", ylabel="Empirical 90% coverage", ylim=(0, 1.02))
    left.set_title("a   Marginal interval calibration", loc="left"); right.set_title("b   Coverage–width trade-off", loc="left")
    left.legend(loc="upper center", bbox_to_anchor=(1.15, -.30), ncol=min(3, site.model_id.nunique()))
    heading(fig, "Coverage must be read together with interval width", f"Common maximin policy · budget {int(budget)} · {label}", preset,
            footer="Faint marks: site means across seeds. Diamonds: equal-weight site means; no confidence intervals.")
    return [("G3_interval_reliability", fig)]


def _ordered_selected(rows):
    if rows.empty:
        return []
    column = _column(rows, ["selection_order", "rank_in_batch", "selected_rank"])
    selected = rows[rows.selected.fillna(False).astype(bool)] if "selected" in rows else rows
    return selected.sort_values(column).location_id.tolist() if column else selected.location_id.tolist()


def _g4(predictions, scores, states, dataset, case, config, preset, label):
    subset = _select(scores, **{k: v for k, v in case.items() if k not in ["location_id", "policy_id"]})
    if subset.empty or "score" not in subset:
        return []
    first = subset.budget_n.min()
    subset = subset[subset.budget_n == first]
    primary = _select(subset, policy_id=case["policy_id"])
    if primary.empty:
        return []
    state = _state(states, case, first)
    selected = _ordered_selected(primary)
    locations = dataset.locations(case["site_id"])
    extent, _ = _geometry_limits(locations)
    figures = []
    fig = canvas(preset, 116)
    gs = fig.add_gridspec(1, 2, left=.10, right=.96, top=.75, bottom=.23, wspace=.43, width_ratios=[1, 1.15])
    map_ax, score_ax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    _map(map_ax, locations, state, selected, limits=extent)
    map_ax.set_title("a   Planned next positions", loc="left")
    top = primary.sort_values(["score", "location_id"], ascending=[False, True]).head(10)
    top = pd.concat([top, primary[primary.location_id.isin(selected) & ~primary.location_id.isin(top.location_id)]]).drop_duplicates("location_id")
    top = top.sort_values("score", ascending=True)
    base = top[_column(top, ["base_score", "uncertainty_term", "score"])].fillna(0).to_numpy(float)
    bonus_col = _column(top, ["threshold_bonus", "threshold_term"])
    bonus = top[bonus_col].fillna(0).to_numpy(float) if bonus_col else np.zeros(len(top))
    y = np.arange(len(top))
    bars = score_ax.barh(y, base, color="#A7CBD5", height=.64, label="Base term")
    if np.any(bonus):
        score_ax.barh(y, bonus, left=base, color=TEAL, height=.64, label="Threshold bonus")
    labels = []; location_labels = _location_labels(locations)
    for i, row in enumerate(top.itertuples()):
        identifier = row.location_id
        excluded = not getattr(row, "eligible_for_second", True) and identifier not in selected
        labels.append((f"{selected.index(identifier) + 1}  " if identifier in selected else "× " if excluded else "") + location_labels[identifier])
        if excluded:
            bars[i].set_hatch("///"); bars[i].set_edgecolor("#718A95")
    score_ax.set_yticks(y, labels)
    score_ax.set_xlabel("Geometric distance (m)" if case["policy_id"] == "maximin" else "Acquisition score")
    score_ax.set_title("b   Candidate score decomposition", loc="left")
    score_ax.legend(loc="upper left", bbox_to_anchor=(0, -.17), ncol=2)
    finish_axis(score_ax, grid="x")
    heading(fig, "The saved scores explain the next measurement", f"{_full_site_name(case['site_id'])} · seed {case['seed']} · {int(first)} measured · {method_name(case['model_id'], case['policy_id'])}", preset,
            footer="Numbers: selected positions. × / hatching: second-point spacing exclusion. Candidate truth stays hidden.\n" + label)
    figures.append(("G4_first_round_selection", fig))
    # Same-state comparisons are only valid when the saved observed IDs coincide.
    alternate = _select(subset, policy_id="maximin")
    alt_state = _state(states, {**case, "policy_id": "maximin"}, first)
    if case["policy_id"] != "maximin" and not alternate.empty and set(_ids(state.get("observed_ids"))) == set(_ids(alt_state.get("observed_ids"))):
        fig = canvas(preset, 108)
        gs = fig.add_gridspec(1, 2, left=.10, right=.95, top=.76, bottom=.21, wspace=.38)
        for i, (policy, rows) in enumerate([("maximin", alternate), (case["policy_id"], primary)]):
            ax = fig.add_subplot(gs[0, i]); _map(ax, locations, state, _ordered_selected(rows), limits=extent)
            ax.set_title(POLICY_NAMES.get(policy, policy), loc="left")
        heading(fig, "Two rules, one shared information state", f"Same model, pool, seed and measured set · {_full_site_name(case['site_id'])}", preset,
                footer="Diamond numbers indicate selection order. Different later trajectories are not same-state comparisons.")
        figures.append(("G4_same_state_policy_maps", fig))
    candidate_rows = _select(_candidate(predictions), **{k: v for k, v in case.items() if k != "location_id"}, budget_n=first)
    candidate_rows = candidate_rows[candidate_rows.location_id.isin(selected)]
    if not candidate_rows.empty:
        fig = canvas(preset, 115)
        gs = fig.add_gridspec(1, len(selected), left=.12, right=.94, top=.76, bottom=.24, wspace=.40)
        limits = _profile_limits(candidate_rows, config)
        for i, identifier in enumerate(selected):
            ax = fig.add_subplot(gs[0, i])
            _profile(ax, _select(candidate_rows, location_id=identifier), case["model_id"], limits, truth=False, title=f"Recommendation {i + 1}")
        heading(fig, "Recommended profiles before measurement", f"Saved decision-time predictions · budget {int(first)} · {label}", preset,
                footer="Shading: marginal 50/80/90% intervals. Dashed line: 1 MPa threshold. No candidate truth revealed.")
        figures.append(("G4_recommendation_profiles", fig))
    return figures


def _metric_name(metrics):
    return _column(metrics, ["log_mae", "mae_log", "log_qc_mae", "mae_log_qc"])


def _g5(metrics, preset, label):
    metric = _metric_name(metrics)
    if not metric:
        return []
    valid = metrics[metrics[metric].notna()].copy()
    if valid.empty:
        return []
    figures = []
    comparisons = []
    maximin = _select(valid, policy_id="maximin")
    models = [m for m in ["gp", "tabpfn", "tabpfn_finetuned", "idw", "xgb"] if m in maximin.model_id.values]
    if models:
        comparisons.append(("model", "Common maximin policy", maximin[maximin.model_id.isin(models[:3])]))
    for model in ["tabpfn_finetuned", "gp"]:
        data = _select(valid, model_id=model)
        if data.policy_id.nunique() > 1:
            comparisons.append((f"policy_{model}", f"Fixed {MODEL_NAMES[model]} model", data))
    for suffix, title, data in comparisons:
        fig = canvas(preset, 107)
        ax = fig.add_axes([.13, .23, .81, .52])
        for (model, policy), rows in data.groupby(["model_id", "policy_id"], sort=False):
            site_means = rows.groupby(["site_id", "budget_n"], as_index=False)[metric].mean()
            color = MODEL_COLORS.get(model, TEAL); style = POLICY_LINES.get(policy, "-")
            for _, group in site_means.groupby("site_id"):
                ax.plot(group.budget_n, group[metric], color=color, linestyle=style, alpha=.17, linewidth=.8)
            aggregate = site_means.groupby("budget_n")[metric].mean()
            name = MODEL_NAMES.get(model, model) if suffix == "model" else POLICY_NAMES.get(policy, policy)
            ax.plot(aggregate.index, aggregate, marker="o", color=color, linestyle=style, label=name, lw=2.2)
        ax.set_xticks(sorted(data.budget_n.unique())); ax.set_xlabel("Measured CPT budget")
        ax.set_ylabel("Absolute log-MAE"); finish_axis(ax)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -.20), ncol=min(3, data[["model_id", "policy_id"]].drop_duplicates().shape[0]))
        heading(fig, "Prediction error at the budgets actually observed", title + f" · {data.site_id.nunique()} sites, {data.seed.nunique()} seeds · " + label, preset,
                footer="Faint lines: site means across seeds. Bold: equal-weight site mean. Non-monotonic changes retained.")
        figures.append((f"G5_budget_{suffix}", fig))
    baseline = _select(valid, model_id="gp", policy_id="maximin")
    if baseline.empty:
        return figures
    requested = [("tabpfn_finetuned", "maximin", "Model comparison"), ("tabpfn", "maximin", "Model comparison"),
                 ("gp", "threshold_uncertainty", "Policy comparison"), ("tabpfn_finetuned", "threshold_uncertainty", "Overall system comparison")]
    pair_keys = ["fold_id", "site_id", "seed", "budget_n"]
    pair_keys = [k for k in pair_keys if k in valid]
    for model, policy, scope in requested:
        method = _select(valid, model_id=model, policy_id=policy)
        if method.empty:
            continue
        paired = method[pair_keys + [metric]].merge(baseline[pair_keys + [metric]], on=pair_keys, suffixes=("_method", "_reference"), validate="one_to_one")
        if paired.empty:
            continue
        budget = paired.budget_n.max(); paired = paired[paired.budget_n == budget].copy()
        paired["difference"] = paired[f"{metric}_method"] - paired[f"{metric}_reference"]
        sites = sorted(paired.site_id.unique()); means = paired.groupby("site_id").difference.mean()
        fig = canvas(preset, max(98, len(sites) * 10 + 50))
        ax = fig.add_axes([.30, .24, .63, .53]); ax.axvline(0, color=MUTED, ls="--", lw=.8)
        for y, site in enumerate(sites):
            differences = paired[paired.site_id == site].sort_values("seed").difference.to_numpy()
            jitter = np.linspace(-.13, .13, len(differences)) if len(differences) > 1 else [0]
            ax.scatter(differences, y + np.asarray(jitter), color=MODEL_COLORS[model], alpha=.35, s=19)
            ax.scatter(means[site], y, color=MODEL_COLORS[model], s=50, zorder=3)
        ax.axhline(len(sites) - .45, color="#E1E7EA", lw=1)
        ax.scatter(means.mean(), len(sites), marker="D", color=INK, s=65)
        ax.set_yticks(range(len(sites) + 1), [_site_name(s) for s in sites] + ["Site macro mean"]); ax.invert_yaxis()
        ax.set_xlabel("Log-MAE difference: method − GP / Maximin"); finish_axis(ax, grid="x")
        heading(fig, "Paired differences retain both gains and regressions", f"{scope} · {method_name(model, policy)} · budget {int(budget)}", preset,
                footer=f"{int((means < 0).sum())}/{len(sites)} sites have lower error. Faint points: paired seeds. Negative = lower error.\n" + label)
        figures.append((f"G5_paired_{model}_{policy}", fig))
    return figures


def _g6(attainment, preset, label):
    if attainment.empty or "n_stable" not in attainment:
        return []
    baseline = _select(attainment, model_id="gp", policy_id="maximin")
    if baseline.empty:
        return []
    candidates = [("tabpfn_finetuned", "threshold_uncertainty"), ("gp", "threshold_uncertainty"), ("tabpfn", "maximin")]
    model_policy = next(((m, p) for m, p in candidates if not _select(attainment, model_id=m, policy_id=p).empty), None)
    if model_policy is None:
        return []
    model, policy = model_policy
    method = _select(attainment, model_id=model, policy_id=policy)
    keys = [x for x in ["fold_id", "site_id", "seed"] if x in attainment]
    paired = method.merge(baseline, on=keys, suffixes=("_method", "_reference"), validate="one_to_one")
    if paired.empty:
        return []
    missing = pd.Series(False, index=paired.index)
    for suffix in ["method", "reference"]:
        status = f"status_{suffix}"
        if status in paired:
            missing |= paired[status].astype(str).str.contains("missing|incomplete|unavailable", case=False, regex=True)
    valid = paired[~missing].copy()
    if valid.empty:
        return []
    last_col = _column(valid, ["last_budget_method", "budget_max_method", "last_budget_reference"])
    last = int(valid[last_col].max()) if last_col else 13
    censor = last + 3
    sites = sorted(valid.site_id.unique())
    fig = canvas(preset, max(118, 9 * len(sites) + 60))
    ax = fig.add_axes([.25, .25, .69, .49])
    categories = {x: 0 for x in ["Both: earlier", "Both: same", "Both: later", "Method only", "Baseline only", "Neither"]}
    for y, site in enumerate(sites):
        instances = valid[valid.site_id == site].sort_values("seed")
        jitter = np.linspace(-.23, .23, len(instances)) if len(instances) > 1 else [0]
        for offset, (_, row) in zip(jitter, instances.iterrows()):
            a, b = row.n_stable_reference, row.n_stable_method
            a_ok, b_ok = pd.notna(a), pd.notna(b)
            key = ("Both: earlier" if b < a else "Both: later" if b > a else "Both: same") if a_ok and b_ok else "Method only" if b_ok else "Baseline only" if a_ok else "Neither"
            categories[key] += 1
            a_x, b_x = float(a) if a_ok else censor, float(b) if b_ok else censor
            ax.plot([a_x, b_x], [y + offset, y + offset], color="#C7CED2", lw=1)
            ax.scatter(a_x, y + offset, facecolor="#0072B2" if a_ok else "white", edgecolor="#0072B2", s=35, marker="o", zorder=3)
            ax.scatter(b_x, y + offset, facecolor=MODEL_COLORS[model] if b_ok else "white", edgecolor=MODEL_COLORS[model], s=38, marker="D", zorder=4)
    ax.axvspan(last + 1, censor + 1, facecolor="#F1F3F5")
    ax.set_yticks(range(len(sites)), [_site_name(s) for s in sites]); ax.invert_yaxis()
    budgets = list(range(5, last + 1, 2))
    ax.set_xticks(budgets + [censor], [str(x) for x in budgets] + [f">{last}\ncensored"])
    ax.set_xlim(4, censor + 1); ax.set_xlabel("Stable quality-attainment budget (CPTs)")
    ax.plot([], [], "o", color="#0072B2", label="GP / Maximin")
    ax.plot([], [], "D", color=MODEL_COLORS[model], label=method_name(model, policy))
    ax.legend(loc="upper left", bbox_to_anchor=(0, -.23), ncol=2)
    heading(fig, "Stable attainment is discrete; failure remains visible", f"Paired seeds · {label}", preset,
            footer=f"Open markers: target not attained within observed budget. Missing targets: {int(missing.sum())} instances.\nCost interpretation requires both methods to attain the same target.")
    figures = [("G6_stable_attainment", fig)]
    fig = canvas(preset, 112); ax = fig.add_axes([.29, .22, .62, .53])
    colors = ["#72ABA9", "#BCC4CA", "#D7B5A7", "#237C82", "#99BACA", "#77828B"]
    ax.barh(list(categories), list(categories.values()), color=colors)
    ax.invert_yaxis(); ax.set_xlabel("Paired instances (site × seed)")
    for i, value in enumerate(categories.values()):
        ax.text(value + .06, i, str(value), va="center")
    ax.set_xlim(0, max(categories.values(), default=1) * 1.2 + 1); finish_axis(ax, grid="x")
    heading(fig, "Attainment composition makes the usable evidence explicit", method_name(model, policy) + " versus GP / Maximin", preset,
            footer=f"All {len(valid)} valid pairs retained; {int(missing.sum())} pairs excluded only because targets are incomplete.")
    figures.append(("G6_attainment_composition", fig))
    return figures


def _g6_costs(costs, preset, label, config):
    if costs.empty or "pair_status" not in costs:
        return []
    pairs = costs[costs.pair_status == "both_reached"].copy()
    if pairs.empty:
        return []
    choice = next(((m, p) for m, p in [("tabpfn_finetuned", "threshold_uncertainty"), ("gp", "threshold_uncertainty"), ("tabpfn", "maximin")]
                   if not _select(pairs, model_id=m, policy_id=p).empty), None)
    if choice is None:
        return []
    model, policy = choice; pairs = _select(pairs, model_id=model, policy_id=policy)
    series = [("saved_cpt", "CPTs saved relative to baseline", "Shared-target investigation savings")]
    monetary = pairs.get("monetary_status", pd.Series(index=pairs.index, dtype=str)).eq("conditional_scenario")
    if monetary.all() and "net_savings" in pairs and pairs.net_savings.notna().all() and pairs.currency.nunique() == 1:
        series.append(("net_savings", f"Conditional net saving ({pairs.currency.iloc[0]})", "Cost depends on explicit scenario assumptions"))
    figures = []; sites = sorted(pairs.site_id.unique())
    count_total = int(pairs.n_all_instances.iloc[0]) if "n_all_instances" in pairs else len(_select(costs, model_id=model, policy_id=policy))
    for field, xlabel, title in series:
        if field not in pairs:
            continue
        fig = canvas(preset, 110); ax = fig.add_axes([.23, .24, .69, .52])
        ax.axvline(0, color=MUTED, ls="--", lw=.8)
        for y, site in enumerate(sites):
            group = pairs[pairs.site_id == site].sort_values("seed")
            jitter = np.linspace(-.15, .15, len(group)) if len(group) > 1 else [0]
            ax.scatter(group[field], y + np.asarray(jitter), color=MODEL_COLORS[model], marker="D", s=45)
        ax.set_yticks(range(len(sites)), [_site_name(s) for s in sites]); ax.invert_yaxis()
        ax.set_xlabel(xlabel); finish_axis(ax, grid="x")
        if field == "saved_cpt":
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        if field == "net_savings":
            economics = config.get("economics", {})
            foot = (f"Scenario: marginal CPT {economics.get('marginal_cost_per_cpt')}; analysis {economics.get('extra_analysis_cost')}; "
                    f"mobilization {economics.get('extra_mobilization_cost')} {pairs.currency.iloc[0]}.\n")
        else:
            foot = "Each point is one paired seed. Positive = fewer CPTs; negative = more CPTs.\n"
        heading(fig, title, method_name(model, policy) + " versus GP / Maximin", preset,
                footer=foot + f"Both attained the same target: {len(pairs)}/{count_total} instances. " + label)
        figures.append((f"G6_conditional_{field}", fig))
    return figures


def _demo_frames(run_dir, dataset, predictions, metrics, scores, states, case, config, label):
    """Fixed geometry, profile, axes and true reveal order; no synthetic tweening."""
    if "new" not in str(case["site_id"]).lower() or "lock" not in str(case["site_id"]).lower():
        return [], "New Lock is not present in this run; no substituted demo site."
    selected_case = {k: v for k, v in case.items() if k != "location_id"}
    saved = _select(states, **selected_case)
    if saved.empty or case["location_id"] is None:
        return [], "No saved New Lock replay states/evaluation profile."
    saved = saved.sort_values("budget_n")
    geometry = dataset.locations(case["site_id"]); extent, _ = _geometry_limits(geometry)
    all_predictions = _select(predictions, **selected_case)
    representative = _select(_evaluation(all_predictions), location_id=case["location_id"])
    limits = _profile_limits(all_predictions, config)
    curve = _select(metrics, **selected_case)
    if not curve.empty:
        curve = curve.sort_values("budget_n")
    all_scores = _select(scores, **selected_case)
    score_range = (all_scores.score.min(), all_scores.score.max()) if not all_scores.empty else None
    metric = _metric_name(curve)
    folder = run_dir / "demo"; folder.mkdir(parents=True, exist_ok=True)
    frames, narrative = [], []
    frame_index = 0
    for _, state in saved.iterrows():
        budget = int(state.budget_n)
        score = _select(scores, **selected_case, budget_n=budget)
        selected = _ordered_selected(score)
        phases = ["Updated prediction"] if frame_index else ["Initial information"]
        if selected:
            phases += ["Candidate scoring", "Batch selected", "New measurements revealed"]
        for phase in phases:
            frame_index += 1
            fig = canvas("slides")
            gs = fig.add_gridspec(2, 3, left=.07, right=.96, bottom=.16, top=.78, wspace=.52, hspace=.60, width_ratios=[1.3, .75, .90])
            map_ax = fig.add_subplot(gs[:, 0]); profile_ax = fig.add_subplot(gs[0, 2]); curve_ax = fig.add_subplot(gs[1, 2])
            visible = selected if phase in ["Batch selected", "New measurements revealed"] else []
            map_state = state.copy()
            if phase == "New measurements revealed":
                map_state["observed_ids"] = _ids(state.observed_ids) + selected
            _map(map_ax, geometry, map_state, visible, limits=extent, scores=score if phase == "Candidate scoring" else None, score_range=score_range)
            map_ax.set_title("Investigation state", loc="left")
            show_batch = bool(selected) and phase in ["Batch selected", "New measurements revealed"]
            if show_batch:
                for rank, identifier in enumerate(selected):
                    decision_ax = fig.add_subplot(gs[rank, 1])
                    candidate_prediction = _select(_candidate(all_predictions), budget_n=budget, location_id=identifier)
                    if phase == "New measurements revealed":
                        truth = dataset.truth([identifier])[["location_id", "depth_index", "qc_mpa"]].rename(columns={"qc_mpa": "truth_qc_mpa"})
                        candidate_prediction = candidate_prediction.drop(columns=["truth_qc_mpa"], errors="ignore").merge(truth, on=["location_id", "depth_index"], validate="one_to_one")
                    _profile(decision_ax, candidate_prediction, case["model_id"], limits,
                             truth=phase == "New measurements revealed", title=f"Recommendation {rank + 1}")
            else:
                decision_ax = fig.add_subplot(gs[:, 1])
            if phase == "Candidate scoring" and not score.empty:
                top = score.sort_values("score", ascending=False).head(6).iloc[::-1]
                decision_ax.barh(np.arange(len(top)), top.score, color=TEAL)
                location_labels = _location_labels(geometry)
                decision_ax.set_yticks(np.arange(len(top)), [location_labels[x] for x in top.location_id])
                decision_ax.set_xlabel("Saved score"); decision_ax.set_title("Candidate ranking", loc="left")
            elif not show_batch:
                decision_ax.axis("off")
                decision_ax.text(.1, .70, f"{budget}\nmeasured CPTs", va="top", fontsize=25, fontweight="bold")
                decision_ax.text(.1, .42, "Fixed evaluation set\nCandidate truth hidden\nBatch size: 2", va="top", fontsize=18, linespacing=1.6, color=MUTED)
            _profile(profile_ax, _select(representative, budget_n=budget), case["model_id"], limits, title="Fixed evaluation profile")
            profile_ax.set_ylabel("Depth (m)")
            if metric and len(curve):
                curve_ax.plot(curve.budget_n, curve[metric], color=MODEL_COLORS[case["model_id"]], marker="o")
                curve_ax.axvline(budget, color=TEAL, lw=2)
                curve_ax.set_xticks(curve.budget_n.unique()); curve_ax.set_xlabel("Prediction budget")
                curve_ax.set_ylabel("Log-MAE"); finish_axis(curve_ax)
            count = budget + len(selected) if phase == "New measurements revealed" else budget
            subtitle = (f"New Lock · {method_name(case['model_id'], case['policy_id'])} · seed {case['seed']}\n"
                        f"Measured {count} CPTs · prediction conditioned on {budget}")
            heading(fig, phase, subtitle, "slides", footer="Replay from saved states; evaluation truth is shown only for retrospective assessment.\n" + label)
            path = folder / f"frame_{frame_index:03d}.png"; fig.savefig(path, dpi=120); plt.close(fig)
            frames.append(path)
            narrative.append({"file": path.name, "phase": phase, "measured_cpts": count,
                              "prediction_budget": budget, "selected_ids": selected,
                              "duration_seconds": config.get("presentation", {}).get("demo_frame_seconds", 3)})
    (folder / "frames.json").write_text(json.dumps(narrative, indent=2), encoding="utf-8")
    (folder / "narration.md").write_text("# New Lock replay\n\n" + "\n".join(f"- {item['file']}: {item['phase']}; measured {item['measured_cpts']}, prediction budget {item['prediction_budget']}." for item in narrative), encoding="utf-8")
    if config.get("presentation", {}).get("demo_encode_mp4", True):
        try:
            import imageio.v2 as imageio
            output = folder / "new_lock_replay.mp4"
            with imageio.get_writer(output, fps=1, codec="libx264", quality=8, macro_block_size=1) as writer:
                for path, item in zip(frames, narrative):
                    pixels = imageio.imread(path)
                    for _ in range(max(1, int(item["duration_seconds"]))):
                        writer.append_data(pixels)
            frames.append(output)
        except (ImportError, OSError, RuntimeError) as error:
            return frames, f"Frames complete; video encoding unavailable: {error}"
    return frames, None


def export_figures(run_dir, dataset, config, *, preset="both", figure=None):
    """Export supported G1–G6 panels and optional New Lock frames from one run.

    ``figure`` accepts G1...G6, ``demo``, or profile/calibration/selection aliases.
    Aliases regenerate their complete family. The export manifest records the case,
    source files, labels and skipped families, so a pilot cannot masquerade as
    the completed experiment. Returns all generated image/vector/video paths.
    """
    run_dir = Path(run_dir)
    presentation = config.get("presentation", {})
    style_path = Path(config.get("_root", ".")) / presentation.get("style_config", "configs/presentation.yaml")
    if style_path.exists():
        import yaml
        config = {**config, "presentation": {**(yaml.safe_load(style_path.read_text(encoding="utf-8")) or {}), **presentation}}
    if preset not in ["report", "slides", "both"]:
        raise ValueError("preset must be report, slides or both")
    requested_selection = str(figure).upper() if figure else None
    aliases = {"PROFILE": "G3", "CALIBRATION": "G3", "SELECTION": "G4"}
    selection = aliases.get(requested_selection, requested_selection)
    if selection not in [None, "G1", "G2", "G3", "G4", "G5", "G6", "DEMO"]:
        raise ValueError("figure must be G1–G6, demo, profile, calibration or selection")
    predictions = _read(run_dir / "predictions.parquet")
    if predictions.empty:
        parts = []
        for name, role in [("evaluation", "evaluation"), ("candidates", "candidate")]:
            data = _read(run_dir / "predictions" / f"{name}.parquet")
            if not data.empty:
                data["prediction_role"] = role; parts.append(data)
        predictions = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    metrics = _read(run_dir / "metrics.parquet")
    scores = _read(run_dir / "acquisition_scores.parquet")
    states = _read(run_dir / "states.parquet")
    scope, metrics, predictions, states, scores = _select_result_scope(metrics, predictions, states, scores)
    attainment = _read(run_dir / "summary" / "attainment.csv")
    if attainment.empty:
        attainment = _read(run_dir / "summary" / "attainment.parquet")
    costs = _read(run_dir / "summary" / "costs.csv")
    case = _case(predictions, dataset, config); label = _run_label(run_dir, metrics, scope)
    exports, skipped = [], {}
    presets = ["report", "slides"] if preset == "both" else [preset]
    functions = {
        "G1": lambda style: _g1(dataset, states, case, style, label) + _g1_thickness(dataset, config, style, label),
        "G2": lambda style: _g2(style),
        "G3": lambda style: _g3(predictions, case, config, style, label, dataset) + _g3_calibration(predictions, style, label),
        "G4": lambda style: _g4(predictions, scores, states, dataset, case, config, style, label),
        "G5": lambda style: _g5(metrics, style, label),
        "G6": lambda style: _g6(attainment, style, label) + _g6_costs(costs, style, label, config),
    }
    for style in presets:
        with figure_style(style):
            for family, draw in functions.items():
                if selection and selection != family:
                    continue
                plots = draw(style)
                if not plots:
                    skipped[family] = "Required saved predictions, paired methods, or valid frozen quality targets are unavailable."
                for name, fig in plots:
                    exports.extend(save_figure(fig, run_dir / "figures" / style, name, style))
    if selection in [None, "DEMO"] and config.get("presentation", {}).get("export_demo_frames", True):
        with figure_style("slides"):
            frames, reason = _demo_frames(run_dir, dataset, predictions, metrics, scores, states, case, config, label)
        exports.extend(frames)
        if reason:
            skipped["demo"] = reason
    manifest = {"label": label, "result_scope": scope,
                "requested_figure": requested_selection, "resolved_figure": selection,
                "case_selection": {k: str(v) if isinstance(v, np.generic) else v for k, v in case.items()},
                "case_rule": "Seed 17 when present; evaluation location nearest full-site geometry centroid. Fallback seed is explicit.",
                "sources": [str(x.relative_to(run_dir)) for x in [run_dir / n for n in ["predictions.parquet", "metrics.parquet", "states.parquet", "acquisition_scores.parquet"]] if x.exists()],
                "site_names": {str(s): {"abbreviation": _site_name(s), "full_name": _full_site_name(s)} for s in dataset.locations().site_id.unique()},
                "files": [str(x.relative_to(run_dir)) for x in exports], "skipped": skipped,
                "notes": ["No model trained or reconditioned during export.", "Site macro means weight each site equally.",
                          "Quantiles are marginal depthwise intervals, not simultaneous profile intervals.",
                          "Only measured reveal-frame candidates may acquire truth during demo export."]}
    output = run_dir / "figures" / "export_manifest.json"; output.parent.mkdir(parents=True, exist_ok=True)
    manifest["file_scopes"] = {name: scope for name in manifest["files"]}
    if selection and output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        retained = [name for name in previous.get("files", []) if (run_dir / name).exists()]
        manifest["last_export_files"] = manifest["files"].copy()
        manifest["last_export_selection"] = selection
        manifest["files"] = list(dict.fromkeys(retained + manifest["files"]))
        manifest["file_scopes"] = {
            **{name: previous.get("file_scopes", {}).get(name, previous.get("result_scope", "unrecorded")) for name in retained},
            **manifest["file_scopes"],
        }
        manifest["skipped"] = {**{k: v for k, v in previous.get("skipped", {}).items()
                                   if k.upper() != selection}, **skipped}
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    location_table = dataset.locations()
    names = pd.DataFrame([{"site_id": s, "abbreviation": _site_name(s), "full_name": _full_site_name(s),
                           "n_locations": int((location_table.site_id == s).sum())} for s in sorted(location_table.site_id.unique())])
    names.to_csv(output.parent / "site_legend.csv", index=False)
    return exports
