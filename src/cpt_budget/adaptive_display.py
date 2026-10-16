"""CSV-driven figures, experiment notebook and local adaptive-investigation display.

The six figures form an evidence chain: method, measured locations, model
quality, stopping, site outcomes, and economic consequences.  Site is the
replicate unit; seeds are averaged within site.  No synthetic observations are
used.  All plotted text is dark and at least 8 pt at a 183 mm export width.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

INK = "#142D43"
BLUE = "#376896"
TEAL = "#087F83"
ORANGE = "#CF691F"
RED = "#AA3854"
GRID = "#DCE6EE"
MODEL_NAMES = {"gp": "GP", "xgb": "XGBoost residual", "cnp": "CNP",
               "tabpfn_joint": "TabPFN joint", "tabpfn_graph": "TabPFN + graph"}
MODEL_COLOURS = {"gp": BLUE, "xgb": "#718F45", "cnp": "#8B6899",
                 "tabpfn_joint": ORANGE, "tabpfn_graph": "#B94E55"}
POLICY_NAMES = {"B_full": "Full plan", "B_seq": "Sequential + stopping", "P_adapt": "Adaptive + stopping"}
POLICY_COLOURS = {"B_full": INK, "B_seq": BLUE, "P_adapt": TEAL}
SITE_NAMES = {"christchurch_metro_sports_facility": "Christchurch Metro Sports",
              "lower_hutt_town_hall": "Lower Hutt Town Hall", "marshland": "Marshland",
              "tauranga_farmers_building": "Tauranga Farmers", "newlock": "New Lock Terneuzen",
              "hollywood": "Hollywood",
              "wellington_airport": "Wellington Airport", "wellington_college": "Wellington College",
              "port_nelson": "Port Nelson"}
FIGURES = ["fig01_workflow", "fig02_selection_map", "fig03_five_models",
           "fig04_quality_trajectories", "fig05_site_outcomes", "fig06_economics"]
CAPTIONS = [
    "The investigation loop separates online decisions from hidden-label quality assessment. A frozen full candidate plan is the engineering comparator; the audit CPTs are excluded from that plan and from selection.",
    "Matched sequential and adaptive investigations start from the same initial CPTs. The maps use the common observation count nearest 13, irrespective of quality; panel labels also give the actual final stopping count. Filled circles show revealed candidates; diamonds identify hidden audit locations. The plotted site and seed are stated in the panel.",
    "Five AI methods on a common observation sequence. Budgets and seeds are averaged within site before the equal-site mean; marks show individual site means. Recall is undefined at sites without low-resistance cells and is omitted only from that metric.",
    "Quality on the same hidden audit set as CPTs are added. Filled stop markers are the actual online triggers; crosses identify triggers that fail the frozen audit criteria. Full-plan endpoints remain visible. Audit labels do not feed the controller.",
    "Every evaluation site and policy is retained. Left: planned-CPT reduction against the full plan, with completed/planned counts. A cross indicates an incorrect early stop; a square indicates no early stop. Right: the additional adaptive saving equals sequential minus adaptive completed CPTs, paired within seed. Points and ranges are site means and seed ranges; they are not confidence intervals.",
    "New Lock economic scenario assessment, not observed project cash flow. The decomposition uses α = 1 and means over three seeds; whiskers and sensitivity envelopes show their range, not a confidence interval. Direct investigation savings, project-time costs, earlier-use benefits and AI/verification costs are separated. Sensitivity varies the time-transfer coefficient and daily net benefit; all signed outcomes are retained.",
]


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.edgecolor": INK, "axes.linewidth": .7,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "legend.frameon": False, "lines.linewidth": 1.8,
    })


def truth(value) -> bool:
    return str(value).lower() in {"true", "1", "yes", "1.0"}


def site_name(value: str) -> str:
    return SITE_NAMES.get(str(value), str(value).replace("_", " ").capitalize())


def read_csv(directory: Path, name: str, required: bool = True) -> pd.DataFrame:
    path = directory / name
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required experiment result: {path}")
        return pd.DataFrame()
    try:
        result = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    aliases = {"h_point_mae_m": "thickness_mae_m", "false_positive_rate": "fpr",
               "strategy": "policy_id", "n_ai": "n_stop", "model": "model_id"}
    return result.rename(columns={k: v for k, v in aliases.items() if k in result and v not in result})


def load_results(root: Path) -> dict[str, pd.DataFrame]:
    directory = root / "results" / "adaptive_v1"
    tables = {key: read_csv(directory, key + ".csv", key in {
        "model_comparison", "trajectories", "quality_audit", "decisions", "locations", "selections"})
        for key in ["model_comparison", "trajectories", "quality_audit", "decisions",
                    "locations", "selections", "economic_results", "sensitivity", "data_manifest", "economic_inputs", "social_user_benefit", "ablation", "full_plan_model_sensitivity"]}
    online, audit = tables["trajectories"], tables["quality_audit"]
    keys = [c for c in ["run_id", "policy_id", "round"] if c in online and c in audit]
    if not keys:
        raise ValueError("Online trajectories and hidden audit results require matched keys")
    audit_columns = keys + [c for c in audit if c not in online and c not in keys]
    tables["joined"] = online.merge(audit[audit_columns], on=keys, how="left", validate="one_to_one")
    frozen_path = directory / "frozen_config.json"
    if frozen_path.exists():
        quality = json.loads(frozen_path.read_text(encoding="utf-8")).get("quality", {})
        for column, key in [("target_log_mae", "absolute_log_mae_max"),
                            ("target_thickness_mae_m", "absolute_thickness_mae_max_m")]:
            if key in quality:
                tables["joined"][column] = quality[key]
    for name, frame in tables.items():
        for column in ["audit_pass", "is_early_stop", "false_early_stop", "stopped", "coverage_ok"]:
            if column in frame:
                frame[column] = frame[column].map(truth)
    return tables


def evaluation(frame: pd.DataFrame) -> pd.DataFrame:
    if "role" in frame and (frame.role == "evaluation").any():
        return frame.loc[frame.role == "evaluation"].copy()
    return frame.copy()


def model_summary(frame: pd.DataFrame, restricted: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = evaluation(frame)
    if restricted:
        data = data.loc[data.model_id.isin(MODEL_NAMES)]
    metrics = [c for c in ["log_mae", "thickness_mae_m", "recall", "fpr"] if c in data]
    seed = data.groupby(["model_id", "site_id", "seed"], sort=False)[metrics].mean()
    site = seed.groupby(["model_id", "site_id"], sort=False).mean().reset_index()
    macro = site.groupby("model_id", sort=False)[metrics].mean()
    if restricted:
        macro = macro.reindex(MODEL_NAMES)
    return site, macro


def save(fig: plt.Figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel(ax, label: str, title: str) -> None:
    ax.set_title(title, loc="left", pad=12, fontweight="bold")
    ax.text(-.12, 1.06, label, transform=ax.transAxes, fontweight="bold", fontsize=11)


def workflow(tables, directory: Path) -> None:
    fig, ax = plt.subplots(figsize=(183/25.4, 87/25.4))
    ax.set(xlim=(0, 10), ylim=(0, 4.2))
    ax.axis("off")
    boxes = [(.1, 2.7, "Initial CPTs", "Same dispersed\ninitial set"),
             (2.1, 2.7, "AI prediction", "Profiles + low-qc\nthickness"),
             (4.1, 2.7, "Select + reveal", "Uncertainty\ndispute, distance"),
             (6.1, 2.7, "Online stop", "Stability, support\nand coverage"),
             (8.1, 2.7, "Economics", "CPTs, days\nand net benefit")]
    for i, (x, y, title, desc) in enumerate(boxes):
        colour = TEAL if i in [2, 3] else BLUE
        ax.add_patch(FancyBboxPatch((x, y), 1.75, 1.15, boxstyle="round,pad=0.06,rounding_size=.06", facecolor="#ECF6F7" if i in [2, 3] else "#F0F4FA", edgecolor=colour, linewidth=1))
        ax.text(x+.875, y+.8, title, ha="center", fontsize=10, weight="bold")
        ax.text(x+.875, y+.31, desc, ha="center", va="center", fontsize=9)
        if i < 4:
            ax.add_patch(FancyArrowPatch((x+1.82, y+.58), (x+1.96, y+.58), arrowstyle="-|>", color=INK, mutation_scale=10))
    ax.annotate("Continue: update context", xy=(2.96, 2.62), xytext=(5.4, 2.1), ha="center", fontsize=9,
                arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=-90,angleB=180,rad=7", color=TEAL))
    ax.plot([.05, 9.96], [1.68, 1.68], color=RED, lw=1, linestyle=(0, (4, 3)))
    ax.text(.1, 1.38, "Information boundary", color=RED, weight="bold", fontsize=10)
    ax.text(.1, 1.08, "Hidden CPTs do not enter selection or stopping", fontsize=9)
    ax.add_patch(FancyBboxPatch((5.9, .22), 3.95, .77, boxstyle="round,pad=.05,rounding_size=.06", facecolor="#FCF0F3", edgecolor=RED))
    ax.text(7.875, .65, "Post-stop quality audit", ha="center", weight="bold", fontsize=10)
    ax.text(7.875, .35, "Pass · incorrect early stop · completed plan", ha="center", fontsize=8.5)
    ax.annotate("", xy=(7.1, 1.05), xytext=(7.1, 2.65), arrowprops=dict(arrowstyle="->", color=RED, linewidth=1))
    ax.text(.1, .65, "Comparators", fontweight="bold", fontsize=10)
    ax.text(.1, .3, "Full plan  |  Sequential + stop  |  Adaptive + stop", fontsize=9)
    fig.subplots_adjust(left=.02, right=.98, top=.98, bottom=.05)
    save(fig, directory, FIGURES[0])


def representative(tables) -> tuple[str, object]:
    decisions = evaluation(tables["decisions"])
    adaptive = decisions.loc[decisions.policy_id == "P_adapt"]
    # New Lock is the pre-specified engineering case, irrespective of outcome.
    case = adaptive.loc[adaptive.site_id == "newlock"]
    row = (case.sort_values("seed") if len(case) else adaptive.sort_values(
        ["n_full", "site_id", "seed"], ascending=[False, True, True])).iloc[0]
    return str(row.site_id), row.seed


def id_set(value) -> set[str]:
    if isinstance(value, str):
        try:
            return set(map(str, json.loads(value)))
        except (ValueError, TypeError):
            return set()
    return set()


def selection_map(tables, directory: Path) -> None:
    site, seed = representative(tables)
    loc = tables["locations"]
    loc = loc.loc[(loc.site_id.astype(str) == site) & (loc.seed == seed)].copy()
    loc["location_id"] = loc.location_id.astype(str)
    all_tracks = tables["joined"]
    fig, axes = plt.subplots(1, 2, figsize=(183/25.4, 92/25.4), sharex=True, sharey=True)
    all_decisions = tables["decisions"]
    case_tracks = all_tracks.loc[(all_tracks.site_id.astype(str) == site) & (all_tracks.seed == seed)]
    common = set(case_tracks.loc[case_tracks.policy_id == "B_seq", "n_observed"]) & set(case_tracks.loc[case_tracks.policy_id == "P_adapt", "n_observed"])
    snapshot_n = min(common, key=lambda n: (abs(n-13), n))
    for ax, policy, letter in zip(axes, ["B_seq", "P_adapt"], "ab"):
        rows = all_tracks.loc[(all_tracks.site_id.astype(str) == site) & (all_tracks.seed == seed) & (all_tracks.policy_id == policy)].sort_values("round")
        decision = all_decisions.loc[(all_decisions.site_id.astype(str) == site) & (all_decisions.seed == seed) & (all_decisions.policy_id == policy)].iloc[0]
        n_stop = int(decision.n_stop)
        state = rows.loc[rows.n_observed <= snapshot_n].iloc[-1]
        known = id_set(state.known_ids)
        candidate = loc.loc[loc.role == "candidate"]
        ax.scatter(candidate.x_m, candidate.y_m, facecolors="white", edgecolors=BLUE, s=25, linewidths=.7)
        revealed = loc.loc[(loc.role == "candidate") & loc.location_id.isin(known)]
        ax.scatter(revealed.x_m, revealed.y_m, color=POLICY_COLOURS[policy], s=34, zorder=3)
        initial = loc.loc[loc.role == "initial"]
        ax.scatter(initial.x_m, initial.y_m, marker="s", color=INK, s=30, zorder=4)
        audit = loc.loc[loc.role == "audit"]
        ax.scatter(audit.x_m, audit.y_m, marker="D", facecolor="#FBE8EE", edgecolor=RED, s=30, linewidths=.9, zorder=4)
        sel = tables["selections"]
        sel = sel.loc[(sel.site_id.astype(str) == site) & (sel.seed == seed) & (sel.policy_id == policy) & (sel["round"] <= state["round"])].copy()
        if len(sel):
            sel["location_id"] = sel.location_id.astype(str)
            route = sel.loc[sel.location_id.isin(known)].merge(loc[["location_id", "x_m", "y_m"]], on="location_id").sort_values("round")
            ax.plot(route.x_m, route.y_m, color=POLICY_COLOURS[policy], alpha=.55, lw=.8, zorder=1)
        panel(ax, letter, f"{POLICY_NAMES[policy]}\nSnapshot {int(state.n_observed)}; final {n_stop}/{int(decision.n_full)}")
        ax.set_xlabel("Easting (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.margins(.14)
    axes[0].set_ylabel("Northing (m)")
    handles = [Line2D([], [], marker="s", color=INK, ls="", label="Initial"),
               Line2D([], [], marker="o", markerfacecolor="white", color=BLUE, ls="", label="Candidate"),
               Line2D([], [], marker="o", color=TEAL, ls="", label="Revealed"),
               Line2D([], [], marker="D", markerfacecolor="#FBE8EE", color=RED, ls="", label="Hidden audit")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(.5, -.01))
    fig.suptitle(f"{site_name(site)} · seed {int(seed)} · matched observation count", x=.08, ha="left", fontsize=11, weight="bold")
    fig.subplots_adjust(left=.1, right=.97, bottom=.23, top=.78, wspace=.26)
    save(fig, directory, FIGURES[1])


def five_models(tables, directory: Path) -> None:
    site, macro = model_summary(tables["model_comparison"])
    models = list(MODEL_NAMES)
    fig, axes = plt.subplots(1, 3, figsize=(183/25.4, 100/25.4), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for ax, metric, title, letter in zip(axes[:2], ["log_mae", "thickness_mae_m"], ["Profile error", "Thickness error"], "ab"):
        y = np.arange(len(models))
        for i, model in enumerate(models):
            values = site.loc[site.model_id == model, metric].dropna()
            ax.hlines(i, values.min(), values.max(), color=MODEL_COLOURS[model], lw=1.1)
            ax.scatter(values, np.repeat(i, len(values)), s=13, facecolors="white", edgecolors=MODEL_COLOURS[model], lw=.8, zorder=3)
            ax.scatter([macro.loc[model, metric]], [i], marker="D", s=30, color=MODEL_COLOURS[model], zorder=4)
        ax.set_yticks(y, [MODEL_NAMES[m] for m in models] if metric == "log_mae" else [""]*len(models))
        ax.invert_yaxis()
        ax.set_xlabel("log-MAE" if metric == "log_mae" else "MAE (m)")
        ax.grid(axis="x", color=GRID, lw=.6)
        panel(ax, letter, title)
    ax = axes[2]
    for model in models:
        r = macro.loc[model]
        raw = site.loc[site.model_id == model]
        ax.scatter(raw.fpr*100, raw.recall*100, s=17, facecolors="white", edgecolors=MODEL_COLOURS[model], lw=.8, alpha=.65)
        ax.scatter(r.fpr*100, r.recall*100, s=48, color=MODEL_COLOURS[model], edgecolors="white", lw=.6, label=MODEL_NAMES[model], zorder=3)
    ax.set(xlabel="False-positive rate (%)", ylabel="Recall (%)")
    ax.grid(color=GRID, lw=.6)
    panel(ax, "c", "Low-qc trade-off")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.55, .015), ncol=3, columnspacing=1.1, handletextpad=.4, fontsize=9)
    fig.subplots_adjust(left=.18, right=.98, bottom=.3, top=.84, wspace=.52)
    save(fig, directory, FIGURES[2])


def quality_trajectories(tables, directory: Path) -> None:
    site, seed = representative(tables)
    frame = tables["joined"]
    frame = frame.loc[(frame.site_id.astype(str) == site) & (frame.seed == seed)]
    decisions = tables["decisions"]
    decisions = decisions.loc[(decisions.site_id.astype(str) == site) & (decisions.seed == seed)]
    fig, axes = plt.subplots(1, 2, figsize=(183/25.4, 91/25.4))
    for ax, metric, title, letter in zip(axes, ["log_mae", "thickness_mae_m"], ["Profile reconstruction", "Low-qc thickness"], "ab"):
        for policy in POLICY_NAMES:
            rows = frame.loc[frame.policy_id == policy].sort_values("n_observed")
            if rows.empty:
                continue
            ax.plot(rows.n_observed, rows[metric], color=POLICY_COLOURS[policy], marker="o", ms=3, label=POLICY_NAMES[policy], linestyle=":" if policy == "B_full" else "-")
            dec = decisions.loc[decisions.policy_id == policy]
            if len(dec):
                stop = dec.iloc[0]
                point = rows.iloc[(rows.n_observed-float(stop.n_stop)).abs().argmin()]
                failed = truth(stop.get("false_early_stop", False))
                ax.scatter(point.n_observed, point[metric], marker="X" if failed else "s" if policy == "B_full" or not truth(stop.get("is_early_stop", False)) else "D", s=65, color=RED if failed else POLICY_COLOURS[policy], edgecolors="white", linewidths=.5, zorder=5)
        target = None
        for candidate in (["target_log_mae", "log_mae_target"] if metric == "log_mae" else ["target_thickness_mae_m", "thickness_mae_target"]):
            if candidate in frame:
                target = frame[candidate].dropna().iloc[0] if frame[candidate].notna().any() else None
        if target is not None:
            ax.axhline(target, color=RED, lw=.9, linestyle="--", label="Frozen quality target")
        panel(ax, letter, title)
        ax.set(xlabel="Observed planned CPTs", ylabel="log-MAE" if metric == "log_mae" else "Thickness MAE (m)")
        ax.grid(color=GRID, lw=.6)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -.23), ncol=2)
    fig.suptitle(f"{site_name(site)} · seed {int(seed)} · same hidden audit CPTs", x=.09, ha="left", fontsize=11, weight="bold")
    fig.text(.09, .015, "Diamond: online stop. Cross: incorrect early stop. Square: full-plan endpoint.", fontsize=8)
    fig.subplots_adjust(left=.09, right=.98, bottom=.3, top=.77, wspace=.36)
    save(fig, directory, FIGURES[3])


def site_outcomes(tables, directory: Path) -> None:
    decisions = evaluation(tables["decisions"])
    decisions = decisions.loc[decisions.policy_id.isin(["B_seq", "P_adapt"])].copy()
    decisions["saving_pct"] = 100*(decisions.n_full-decisions.n_stop)/decisions.n_full
    sites = list(dict.fromkeys(decisions.site_id))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(183/25.4, max(98, 13*len(sites)+40)/25.4),
                                 gridspec_kw={"width_ratios": [1.65, 1.0], "wspace": .36}, sharey=True)
    for j, policy in enumerate(["B_seq", "P_adapt"]):
        subset = decisions.loc[decisions.policy_id == policy]
        for i, site in enumerate(sites):
            group = subset.loc[subset.site_id == site]
            if group.empty:
                continue
            y = i+(j-.5)*.26
            values = group.saving_pct
            has_error = group.false_early_stop.any()
            full = not group.is_early_stop.any()
            ax.hlines(y, values.min(), values.max(), color=POLICY_COLOURS[policy], linewidth=1.4)
            ax.scatter(values.mean(), y, marker="X" if has_error else "s" if full else "o", s=62, color=RED if has_error else POLICY_COLOURS[policy], zorder=3)
            count = f"{group.n_stop.mean():.1f}".rstrip("0").rstrip(".")
            ax.text(max(values.max(), values.mean())+1.4, y, f"{count} / {group.n_full.mean():g}", va="center", fontsize=9, color=INK)
    ax.set_yticks(np.arange(len(sites)), [site_name(s) for s in sites])
    ax.invert_yaxis()
    ax.set_xlabel("Planned CPT reduction (%)")
    ax.set_xlim(left=min(-2, decisions.saving_pct.min()-2), right=max(15, decisions.saving_pct.max()+18))
    ax.grid(axis="x", color=GRID, lw=.6)
    ax.axvline(0, color=INK, lw=.7)
    paired = decisions.pivot(index=["site_id", "seed"], columns="policy_id", values="n_stop")
    paired["additional"] = paired.B_seq - paired.P_adapt
    for i, site in enumerate(sites):
        values = paired.loc[site, "additional"]
        if np.isscalar(values):
            values = pd.Series([values])
        colour = TEAL if values.mean() >= 0 else ORANGE
        bx.hlines(i, values.min(), values.max(), color=colour, lw=1.3)
        bx.scatter(values.mean(), i, color=colour, s=38, zorder=3)
        bx.annotate(f"{values.mean():+.1f}", (values.mean(), i), xytext=(8, -3), textcoords="offset points", fontsize=9, color=INK)
    extent = max(1, float(paired.additional.abs().max())*1.4)
    bx.set_xlim(-extent, extent*1.4)
    bx.axvline(0, color=INK, lw=.8)
    bx.grid(axis="x", color=GRID, lw=.6)
    bx.set_xlabel("Additional CPTs saved")
    panel(ax, "a", "Versus full plan")
    panel(bx, "b", "Beyond sequential")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="Sequential"), Line2D([], [], marker="o", ls="", color=TEAL, label="Adaptive"), Line2D([], [], marker="X", ls="", color=RED, label="Incorrect early stop"), Line2D([], [], marker="s", ls="", color=INK, label="No early stop")], loc="lower center", bbox_to_anchor=(.6, .015), ncol=2, fontsize=9)
    fig.subplots_adjust(left=.26, right=.98, bottom=.26, top=.86)
    save(fig, directory, FIGURES[4])


def economics(tables, directory: Path) -> None:
    econ, sensitivity = tables["economic_results"], tables["sensitivity"]
    if econ.empty:
        return
    econ = evaluation(econ)
    case_site = "newlock" if (econ.site_id == "newlock").any() else None
    if case_site:
        econ = econ.loc[econ.site_id == case_site]
    if "alpha" in econ:
        econ = econ.loc[np.isclose(econ.alpha, 1)]
    if "policy_id" in econ:
        econ = econ.loc[econ.policy_id == "P_adapt"]
    if econ.empty:
        return
    aliases = {
        "direct_saving": ["direct_saving", "direct_saving_eur", "direct_saving_currency", "benefit_direct"],
        "schedule_saving": ["project_time_saving", "schedule_saving", "schedule_saving_eur", "time_cost_saving", "benefit_schedule"],
        "early_use_benefit": ["early_opening_benefit", "early_use_benefit", "early_use_benefit_eur", "benefit_early_use"],
        "ai_cost": ["ai_incremental_cost", "ai_cost", "ai_cost_eur", "ai_and_verification_cost", "cost_ai"],
        "net_benefit": ["net_benefit", "net_benefit_eur", "net_benefit_currency"],
    }
    for key, choices in aliases.items():
        for choice in choices:
            if choice in econ:
                econ[key] = econ[choice]
                break
    if "net_benefit" not in econ:
        return
    fig = plt.figure(figsize=(183/25.4, 100/25.4))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=.48)
    ax, bx = fig.add_subplot(outer[0]), fig.add_subplot(outer[1])
    components = [c for c in ["direct_saving", "schedule_saving", "early_use_benefit", "ai_cost"] if c in econ]
    labels = {"direct_saving": "Investigation", "schedule_saving": "Project time", "early_use_benefit": "Earlier use", "ai_cost": "AI + checks"}
    site_means = econ.groupby("site_id")[components+["net_benefit"]].mean()
    heights = [site_means[c].mean()/1000*(-1 if c == "ai_cost" else 1) for c in components]
    starts = np.cumsum([0]+heights[:-1])
    signed = econ[components].copy()/1000
    if "ai_cost" in signed:
        signed["ai_cost"] *= -1
    cumulative = signed.cumsum(axis=1)
    label_margin = max(.6, max(map(abs, heights))*.045)
    for i, (height, start) in enumerate(zip(heights, starts)):
        ax.bar(i, height, bottom=start, width=.62, color=[BLUE, "#619B9A", TEAL, ORANGE][i])
        values = cumulative.iloc[:, i]
        small = abs(height) <= max(map(abs, heights))*.1
        label_y = values.min()-label_margin if small and height < 0 else values.max()+label_margin if small else start+height/2
        ax.text(i, label_y, f"{height:+,.1f}", ha="center", va="top" if small and height < 0 else "bottom" if small else "center", color=INK if small else "white", fontsize=9, weight="bold")
        ax.errorbar([i], [start+height], yerr=[[max(0, start+height-values.min())], [max(0, values.max()-start-height)]], color=INK, fmt="none", capsize=2, linewidth=.9)
    net = site_means.net_benefit.mean()/1000
    ax.bar(len(heights), net, width=.62, color=INK)
    ax.errorbar([len(heights)], [net], yerr=[[max(0, net-econ.net_benefit.min()/1000)], [max(0, econ.net_benefit.max()/1000-net)]], color=ORANGE, fmt="none", capsize=2, linewidth=.9)
    net_label = econ.net_benefit.max()/1000+label_margin if net >= 0 else econ.net_benefit.min()/1000-label_margin
    ax.text(len(heights), net_label, f"{net:,.1f}", ha="center", va="bottom" if net >= 0 else "top", fontsize=9, fontweight="bold")
    ax.set_ylim(bottom=min(float(cumulative.min().min()), float(econ.net_benefit.min()/1000))-5*label_margin,
                top=max(0., float(cumulative.max().max()), float(econ.net_benefit.max()/1000))+3*label_margin)
    ax.set_xticks(range(len(heights)+1), [labels[c] for c in components]+["Net"], rotation=28, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Scenario benefit (thousand EUR, 2025)")
    ax.axhline(0, color=INK, lw=.7)
    panel(ax, "a", "Benefit decomposition: α = 1")
    if not sensitivity.empty and "alpha" in sensitivity:
        data = evaluation(sensitivity)
        if case_site:
            data = data.loc[data.site_id == case_site]
        if "sensitivity_kind" in data and (data.sensitivity_kind == "alpha_opening_grid").any():
            data = data.loc[data.sensitivity_kind == "alpha_opening_grid"]
        if "policy_id" in data:
            data = data.loc[data.policy_id == "P_adapt"]
        net_col = next((c for c in aliases["net_benefit"] if c in data), None)
        daily_col = next((c for c in ["daily_net_benefit", "daily_net_benefit_eur", "daily_net_revenue"] if c in data), None)
        if net_col:
            if daily_col:
                for alpha, group in data.groupby("alpha"):
                    curve = group.groupby(daily_col)[net_col].agg(["mean", "min", "max"]).sort_index()
                    colour = {.5: BLUE, .75: ORANGE, 1.: TEAL}.get(alpha, INK)
                    bx.fill_between(curve.index/1000, curve["min"]/1000, curve["max"]/1000, color=colour, alpha=.1, linewidth=0)
                    bx.plot(curve.index/1000, curve["mean"]/1000, marker="o", ms=3, label=f"α = {alpha:g}", color=colour)
                bx.set_xlabel("Daily net benefit (thousand EUR/day)")
            else:
                curve = data.groupby("alpha")[net_col].mean().sort_index()
                bx.plot(curve.index, curve.values/1000, marker="o", color=TEAL)
                bx.set_xlabel("Time-transfer coefficient α")
            bx.set_ylabel("Net benefit (thousand EUR)")
            bx.axhline(0, color=INK, lw=.7)
            bx.grid(color=GRID, lw=.6)
            if daily_col:
                bx.legend(loc="best")
    panel(bx, "b", "Economic sensitivity")
    fig.text(.08, .01, "New Lock counterfactual owner-financial case; EUR 2025; mean over the three seeds.", fontsize=9)
    fig.subplots_adjust(left=.1, right=.98, bottom=.3, top=.83)
    save(fig, directory, FIGURES[5])


def json_rows(frame: pd.DataFrame) -> list:
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def build_html(root: Path, tables) -> None:
    template = Path(__file__).with_name("adaptive_status_template.html").read_text(encoding="utf-8")
    columns = {
        "model_comparison": ["model_id", "site_id", "seed", "budget_n", "role", "log_mae", "thickness_mae_m", "recall", "fpr"],
        "joined": ["run_id", "site_id", "seed", "role", "model_id", "policy_id", "round", "n_observed", "n_full", "known_ids", "revealed_ids", "selected_ids", "delta_q90", "delta_h90", "support_log_mae", "uncertainty_q90", "coverage_ok", "stopped", "stop_reason", "log_mae", "thickness_mae_m", "recall", "fpr", "audit_pass", "low_qc_audit_pass", "target_log_mae", "target_thickness_mae_m"],
        "decisions": ["run_id", "site_id", "seed", "role", "policy_id", "n_full", "n_stop", "audit_pass", "low_qc_audit_pass", "is_early_stop", "false_early_stop"],
        "locations": ["site_id", "seed", "location_id", "x_m", "y_m", "role"],
        "selections": ["site_id", "seed", "location_id", "policy_id", "round", "selection_order"],
    }
    economic_columns = ["site_id", "seed", "role", "policy_id", "alpha", "n_full", "n_stop", "saved_cpt", "economic_status", "audit_pass", "false_early_stop", "investigation_days_saved", "project_days_saved", "direct_saving", "project_time_saving", "early_opening_benefit", "ai_incremental_cost", "net_benefit", "currency", "price_year"]
    columns["economic_results"] = economic_columns
    columns["sensitivity"] = economic_columns+["sensitivity_kind", "daily_net_benefit"]
    payload = {}
    for key, selected in columns.items():
        frame = tables[key]
        if key == "sensitivity" and "sensitivity_kind" in frame:
            frame = frame.loc[frame.sensitivity_kind.isin(["alpha", "alpha_opening_grid"])]
        payload[key] = json_rows(frame[[c for c in selected if c in frame]])
    payload["generated_date"] = "2026-10-17"
    payload["site_names"] = SITE_NAMES
    frozen_path = root / "results" / "adaptive_v1" / "frozen_config.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        payload["frozen"] = {k: frozen.get(k) for k in ["primary_model", "acquisition_weights", "stop_rule", "quality", "selection_source"]}
    full = evaluation(tables["full_plan_model_sensitivity"])
    payload["full_plan_models"] = json_rows(full.groupby(["site_id", "model_id"])[["log_mae", "thickness_mae_m"]].mean().reset_index()) if len(full) else []
    payload["figures"] = []
    directory = root / "results" / "adaptive_v1" / "figures"
    for name, caption in zip(FIGURES, CAPTIONS):
        path = directory / (name+".svg")
        if path.exists():
            payload["figures"].append({"name": name, "caption": caption,
                "src": "data:image/svg+xml;base64,"+base64.b64encode(path.read_bytes()).decode("ascii")})
    content = template.replace("__RESULT_DATA__", json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("</", "<\\/"))
    (root / "CA6001_项目状态.html").write_text(content, encoding="utf-8")


def build_notebook(root: Path) -> None:
    import nbformat as nbf
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata.language_info = {"name": "python", "version": "3.11"}
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb.cells = [md("""# Machine learning-guided adaptive CPT investigation and economic benefit assessment

This experiment notebook follows one question: **can a model-guided investigation complete fewer planned CPTs while retaining the required prediction quality?** It compares five AI methods, evaluates sequential and adaptive stopping against the full candidate plan, and translates the observed work reduction into explicitly defined economic scenarios.

The public CPT measurements come from the [ISSMGE TC304 CPT database](https://tc304.org/cpt_databases.html). The prediction task covers 2–8 m, using 24 depth cells of 0.25 m. The research threshold qc < 1 MPa defines cumulative low-cone-resistance thickness; it is not a soil classification or a design safety criterion. The previous fixed-budget investigation remains available in the [v0.9.1 notebook](archive/CA6001_report_figures_v0.9.1.ipynb).

All findings below are read from the saved experimental CSV files. The independent unit is the site, with seeds averaged within site. These sites have a development history: the new results are a site-separated re-evaluation, rather than a newly collected blind test."""),
        code("""from pathlib import Path
import sys, json
import pandas as pd
from IPython.display import display, Markdown, Image

ROOT = Path.cwd()
if not (ROOT / 'src' / 'cpt_budget').exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
from cpt_budget.adaptive_display import (load_results, model_summary, evaluation,
                                        FIGURES, CAPTIONS, MODEL_NAMES, build)
tables = load_results(ROOT)
RESULTS = ROOT / 'results' / 'adaptive_v1'
FIG = RESULTS / 'figures'
pd.set_option('display.max_colwidth', 80)
pd.set_option('display.precision', 4)
"""),
        md("## Summary\n\nThe summary is computed from the saved evaluation decisions. An online trigger and a successful hidden audit are distinct outcomes; incorrect early stops remain in every decision and economic table. The main audit covers profile and cumulative-thickness targets; low-qc recall and false-positive targets are assessed separately."),
        code("""decisions = evaluation(tables['decisions'])
adaptive = decisions.query("policy_id == 'P_adapt'").copy()
adaptive['reduction_pct'] = 100 * (adaptive.n_full - adaptive.n_stop) / adaptive.n_full
site_summary = adaptive.groupby('site_id').agg(
    planned_CPTs=('n_full', 'mean'), completed_CPTs=('n_stop', 'mean'),
    reduction_pct=('reduction_pct', 'mean'), audit_pass_rate=('audit_pass', 'mean'),
    incorrect_early_stops=('false_early_stop', 'sum'))
display(site_summary.round(3))
display(Markdown(f"The adaptive policy was evaluated at **{len(site_summary)} sites**. "
    f"The equal-site mean planned-CPT reduction was **{site_summary.reduction_pct.mean():.1f}%**. "
    f"There were **{int(adaptive.false_early_stop.sum())} incorrect early stops** among "
    f"{len(adaptive)} site–seed trajectories. Savings are interpreted together with these quality verdicts."))
"""),
        md("""## Context and methods

The same initial CPTs, candidate pool and hidden audit set are used for all policies. The full plan completes every initial and selectable CPT; the hidden audit CPTs are excluded from the plan count. Sequential investigation adds maximin locations. Adaptive investigation ranks uncertainty, model disagreement and distance. Both incremental policies apply the same frozen stopping rule.

Online stability is measured by the 90th-percentile changes in log(qc) and cumulative thickness on a fixed query grid. The rule also checks held-out support-group error, the uncertainty quantile, the minimum observation count and spatial coverage. All conditions must hold for two consecutive rounds; the hidden audit is not one of these signals.

### Key assumptions

Only revealed CPT curves and pre-known coordinates enter selection and stopping. Hidden audit labels are read by a separate evaluator. The economic comparator is the frozen complete candidate plan, not an assumed 13-CPT endpoint. The main economic scenario uses α = 1; α = 0.5 and 0.75 are also assessed. A daily scenario benefit is not an observed financial return."""),
        code("build(ROOT, notebook=False)\ndisplay(Image(filename=str(FIG / (FIGURES[0]+'.png'))))\ndisplay(Markdown('**Figure 1.** '+CAPTIONS[0]))\nfrozen = json.loads((RESULTS / 'frozen_config.json').read_text(encoding='utf-8'))\ndisplay(pd.Series(frozen['stop_rule'], name='Frozen online stopping setting').to_frame())"),
        md("""## Data and separation

The data manifest reports the actual usable records and the location-group split. Depth cells are repeated measurements within a CPT, not independent observations. Development and evaluation roles are retained in the tables; model selection and stopping thresholds use development data only."""),
        code("""site_manifest = pd.read_csv(RESULTS / 'site_manifest.csv')
display(site_manifest[['display_name', 'n_profiles', 'n_planned_locations',
                       'n_locations', 'n_near_groups', 'adaptive_role']])
display(tables['locations'].groupby(['site_id', 'seed', 'role']).size().unstack(fill_value=0))
representation = pd.read_csv(RESULTS / 'representation_summary.csv')
overall = representation.query("site_id == 'ALL'").iloc[0]
display(Markdown(f"Across **{int(overall.n_profiles)} usable profiles**, the 0.25 m representation "
    f"had a mean absolute cumulative-thickness difference of **{overall.grid_025_mae_m:.3f} m** "
    f"from measurement-derived integration. A 0.1 m diagnostic grid reduced this representation "
    f"difference to **{overall.grid_010_diagnostic_mae_m:.3f} m**, but the present five-method "
    f"comparison consistently retains the 0.25 m grid. This is a representation check, not a model "
    f"accuracy claim or an independent geological truth."))
display(representation.loc[representation.site_id.isin(['ALL'] + tables['locations'].site_id.unique().tolist()),
    ['site_id', 'n_profiles', 'grid_025_mae_m', 'grid_025_p90_abs_error_m', 'grid_010_diagnostic_mae_m']])"""),
        md("""## AI methods and common-sequence comparison

**GP** models spatial correlations and provides a probabilistic uncertainty signal. **XGBoost residual** learns nonlinear corrections to the spatial base from observed-neighbour features. **CNP** aggregates the revealed support set into a conditional neural representation. **TabPFN joint** combines a trained residual component with a spatial base and profile-level decisions. **TabPFN + graph** combines residual prediction with robust graph smoothing. These hybrid methods retain their spatial components: the whole improvement cannot be attributed to the Transformer alone.

For the five-method comparison, all methods receive the same CPT sequence. Site means include the common budgets and seeds; the overall comparison gives each site equal weight."""),
        code("site_models, model_means = model_summary(tables['model_comparison'])\ndisplay(model_means.round(4))\nprimary = tables['joined'].model_id.iloc[0]\nbest_profile = model_means.log_mae.idxmin()\nbest_thickness = model_means.thickness_mae_m.idxmin()\ndisplay(Markdown(f'The development-selected primary method was **{MODEL_NAMES.get(primary, primary)}**. In the evaluation comparison, the lowest profile error was obtained by **{MODEL_NAMES.get(best_profile, best_profile)}**, whereas the lowest thickness error was obtained by **{MODEL_NAMES.get(best_thickness, best_thickness)}**. The primary method is retained after evaluation; it is not re-selected from these outcomes.'))\ndisplay(Image(filename=str(FIG / (FIGURES[2]+'.png'))))\ndisplay(Markdown('**Figure 3.** '+CAPTIONS[2]))"),
        md("### Weight fine-tuning and component ablation\n\nThe matched ablation separates gradient-updated weights from the spatial guard and joint profile decision. Context fitting alone is not described as weight fine-tuning. The results are compared on the same evaluation observation states; a benefit is reported only where it is observed."),
        code("""if not tables['ablation'].empty:
    _, ablation_means = model_summary(tables['ablation'], restricted=False)
    ablation_means.loc['tabpfn_joint'] = model_means.loc['tabpfn_joint']
    display(ablation_means.round(4))
    if 'tabpfn_joint_unfinetuned' in ablation_means.index:
        unfine = ablation_means.loc['tabpfn_joint_unfinetuned', 'log_mae']
        fine = ablation_means.loc['tabpfn_joint', 'log_mae']
        display(Markdown(f'The joint method had evaluation log-MAE **{fine:.4f}** with the stored '
            f'fine-tuned weights and **{unfine:.4f}** with the matched non-fine-tuned weights. '
            f'This comparison is retained even when the non-fine-tuned variant performs better.'))"""),
        md("## Selection and stopping\n\nThe maps retain the distinction between initial, selectable and hidden audit locations. New Lock is the pre-specified engineering case, irrespective of its result; the seed is printed on the figure. The accompanying quality curves use the same audit CPTs at every round."),
        code("display(Image(filename=str(FIG / (FIGURES[1]+'.png'))))\ndisplay(Markdown('**Figure 2.** '+CAPTIONS[1]))\ndisplay(Image(filename=str(FIG / (FIGURES[3]+'.png'))))\ndisplay(Markdown('**Figure 4.** '+CAPTIONS[3]))\nfull_case = evaluation(tables['full_plan_model_sensitivity']).query(\"site_id == 'newlock'\")\nif not full_case.empty:\n    display(full_case.groupby('model_id')[['log_mae', 'thickness_mae_m', 'recall', 'fpr']].mean().round(4))\n    display(Markdown('The full-plan New Lock comparison retains all five methods. Complete candidate-plan observations do not automatically guarantee accurate prediction at the separate hidden audit CPTs.'))"),
        md("## Site outcomes\n\nEarly stopping is counted as successful only when the hidden audit passes. Failure cases and complete-plan outcomes remain visible. An advantage over the full plan may arise from early stopping itself; the matched sequential policy identifies any additional contribution from adaptive location choice."),
        code("display(Image(filename=str(FIG / (FIGURES[4]+'.png'))))\ndisplay(Markdown('**Figure 5.** '+CAPTIONS[4]))\ndisplay(decisions[[c for c in ['site_id', 'seed', 'policy_id', 'n_full', 'n_stop', 'audit_pass', 'low_qc_audit_pass', 'false_early_stop'] if c in decisions]])\npaired = decisions.pivot(index=['site_id', 'seed'], columns='policy_id', values='n_stop')\npaired['Additional_adaptive_saving'] = paired.B_seq - paired.P_adapt\ndisplay(paired.groupby('site_id').mean().round(3))") ,
        md("""## Economic consequences and sensitivity

The economic calculation starts from experimentally observed CPT counts. It separates direct investigation savings, project-time costs, earlier-use benefits and AI/verification costs. The project-time transfer coefficient α connects investigation time saved to project time saved. These values are scenario assessments with explicit cost assumptions; an incorrect early stop is retained and can incur verification or return-visit costs."""),
        code("""cols = [c for c in ['site_id', 'policy_id', 'n_full', 'n_stop', 'saved_cpt', 'investigation_days_saved', 'project_days_saved', 'direct_saving', 'project_time_saving', 'early_opening_benefit', 'ai_incremental_cost', 'net_benefit'] if c in tables['economic_results']]
display(evaluation(tables['economic_results'])[cols])
if not tables['economic_inputs'].empty:
    display(tables['economic_inputs'][[c for c in ['parameter', 'value', 'unit', 'source_type', 'source_url'] if c in tables['economic_inputs']]])
if (FIG / (FIGURES[5]+'.png')).exists():
    display(Image(filename=str(FIG / (FIGURES[5]+'.png'))))
    display(Markdown('**Figure 6.** '+CAPTIONS[5]))
display(tables['sensitivity'].groupby(['sensitivity_kind', 'alpha']).net_benefit.agg(['count', 'mean', 'min', 'max']).round(2))"""),
        md("""## Interpretation

The engineering result has three parts: prediction quality, the completed planned-CPT count, and the hidden-audit verdict. An investigation that stops early but fails audit does not establish a deployable saving. Where sequential and adaptive outcomes are similar, the gain is mainly due to staged investigation and stopping. Where adaptive sampling passes audit with fewer CPTs, it supports an additional benefit from selecting informative locations.

Low-qc detection is difficult when thin or laterally variable features are sparsely observed. The reported recall, false-positive rate and thickness error therefore accompany profile error throughout the comparison. The evidence applies to the measured 2–8 m reconstruction task and the defined economic scenarios.

## Reproducibility and sources

The figures are generated by `cpt_budget.adaptive_display` from `results/adaptive_v1/*.csv`. SVG and PDF files retain editable text; PNG previews are saved for this notebook. Original measurements: ISSMGE TC304 CPT database. Model references and the project background and economic parameter sources are recorded in the execution plan and economic parameter document. The previous investigation's notebook, HTML and result files are retained as historical evidence."""),
    ]
    path = root / "notebooks" / "CA6001_report_figures.ipynb"
    nbf.write(nb, path)


def build(root: Path, notebook: bool = False) -> dict:
    style()
    tables = load_results(root)
    directory = root / "results" / "adaptive_v1" / "figures"
    for fn in [workflow, selection_map, five_models, quality_trajectories, site_outcomes, economics]:
        fn(tables, directory)
    build_html(root, tables)
    notes = "# Figure evidence and export notes\n\n"
    for name, caption in zip(FIGURES, CAPTIONS):
        notes += f"- **{name}:** {caption}\n"
    notes += "\nAll figures use actual saved CSV observations. Figure 3 averages budgets and seeds within site, then sites equally; site marks and ranges show between-site variation, not confidence intervals. Figures 2 and 4 show the pre-specified New Lock case and one stated seed without an uncertainty interval. Figure 5 shows within-site seed ranges. Figure 6 shows New Lock means and ranges over the three seeds; these are descriptive ranges in a deterministic parameter scenario, not financial confidence intervals. No significance tests are used. Exports are 183 mm wide before tight bounding-box trimming; source expressions divide mm by 25.4 to obtain inches. PNG uses the requested 300 dpi figure resolution; editable SVG/PDF are supplied instead of a submission TIFF.\n"
    metrics = evaluation(tables["model_comparison"])
    notes += "\n## Defined metrics and missing observations\n\n"
    for metric in ["log_mae", "thickness_mae_m", "recall", "fpr"]:
        notes += f"- {metric}: {metrics[metric].notna().sum()} defined values of {len(metrics)} source rows. Undefined values are not imputed; recall is undefined where no low-qc target exists. No valid observation is removed for plotting convenience.\n"
    (directory / "figure_notes.md").write_text(notes, encoding="utf-8")
    if notebook:
        build_notebook(root)
    return {"figures": [name for name in FIGURES if (directory/(name+".svg")).exists()], "rows": {key: len(frame) for key, frame in tables.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--notebook", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.root.resolve(), notebook=args.notebook), indent=2))


if __name__ == "__main__":
    main()
