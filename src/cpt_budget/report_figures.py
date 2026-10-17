"""Report figures for the adaptive CPT investigation study.

Every figure is drawn from the saved CSV/JSON outputs in ``results/``; nothing is
hand-entered. Run from the project root:

    .venv/Scripts/python.exe -m cpt_budget.report_figures

Outputs go to ``results/adaptive_v1/figures/`` as 300 dpi PNG plus vector PDF and SVG,
sized for an A4 page with 1-inch margins (6.3 in wide).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch

# ---------------------------------------------------------------- style
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FULL, SEQ, ADAPT = "#52514e", "#2a78d6", "#eb6834"
GOOD, BAD, AMBER = "#0ca30c", "#d03b3b", "#c98500"
EVAL_C, DEV_C, PROFILE_C = "#3d3c39", "#b5b3ab", "#5b7a99"
MODEL_ORDER = ["gp", "xgb", "cnp", "tabpfn_joint", "tabpfn_graph"]
MODEL_NAME = {"gp": "GP", "xgb": "XGBoost residual", "cnp": "CNP",
              "tabpfn_joint": "TabPFN joint", "tabpfn_graph": "TabPFN + graph"}
MODEL_COLOR = {"gp": "#2a78d6", "xgb": "#008300", "cnp": MUTED,
               "tabpfn_joint": "#eda100", "tabpfn_graph": "#e34948"}
MODEL_MARK = {"gp": "o", "xgb": "s", "cnp": "v", "tabpfn_joint": "^", "tabpfn_graph": "D"}
EVAL_SITES = ["wellington_airport", "christchurch_stadium", "hollywood", "newlock"]
DEV_SITES = ["christchurch_metro_sports_facility", "lower_hutt_town_hall", "marshland",
             "tauranga_farmers_building"]
SITE_NAME = {"wellington_airport": "Wellington Airport", "christchurch_stadium": "Christchurch Stadium",
             "hollywood": "Hollywood", "newlock": "New Lock Terneuzen",
             "christchurch_metro_sports_facility": "Chch Metro Sports",
             "lower_hutt_town_hall": "Lower Hutt Town Hall", "marshland": "Marshland",
             "tauranga_farmers_building": "Tauranga Farmers"}
SITE_MARK = {"wellington_airport": "o", "christchurch_stadium": "s", "hollywood": "^", "newlock": "D"}
POLICY_COLOR = {"B_full": FULL, "B_seq": SEQ, "P_adapt": ADAPT}
TEXT_W = 6.3  # inches: A4 text width with 1-inch margins

FIGURES = ["fig01_workflow", "fig02_dataset", "fig03_selection_maps", "fig04_five_models",
           "fig05_quality_vs_cpts", "fig06_site_outcomes", "fig07_economics"]
CAPTIONS = [
    "Investigation loop. Revealed CPTs feed prediction, batch selection and an online stop rule; the hidden "
    "audit set A is read only by a separate evaluator after the stop. All policies share S0, C and A.",
    "Data. (a) Usable CPT positions at the eight replay sites and their share of qc < 1 MPa cells. "
    "(b, c) All 2-8 m profiles at the easiest (Wellington Airport) and hardest (New Lock) evaluation sites.",
    "Selection maps, seed 17. Wellington Airport (a, b) at the online stop; New Lock (c, d) after the first "
    "12-13 CPTs. Numbers give the round in which a position was revealed.",
    "Five AI methods on identical observation states at the four evaluation sites. Open marks are site means; "
    "bars and filled circles are the equal-site mean. Recall is undefined at Wellington Airport (no low-qc cells).",
    "Hidden-audit quality as CPTs are added (three seeds per policy). Stars mark online early stops; squares "
    "the full-plan endpoint; the red line the frozen target; the dotted line the oracle earliest passing count "
    "(uses hidden labels, diagnostic only).",
    "Decision outcomes for all 48 sequential and adaptive trajectories. (a) CPTs used at the stop and the "
    "full-plan audit verdict per site; (b) CPTs saved by adaptive relative to sequential, paired by seed.",
    "Owner-financial scenario, EUR 2025, adaptive policy (mean of three seeds). (a) Benefit components relative "
    "to the full plan at alpha = 1. (b) Net benefit as alpha and the daily earlier-opening benefit vary (vertical "
    "line: base case). (c) Net benefit against the decision delay per batch.",
]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.6, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", name)


def panel(ax, letter, title=""):
    ax.set_title(f"{letter}   {title}".rstrip(), loc="left", pad=5)


def site_equal(df, cols, by="model_id"):
    """Mean within seed, then within site; returns per-site table and equal-site mean."""
    a = df.groupby(["site_id", "seed", by])[cols].mean().groupby(["site_id", by]).mean()
    return a, a.groupby(by).mean()



# ---------------------------------------------------------------- load
OUT = Path()
decisions = quality = models = locations = selections = econ = sens = sites = oracle = grid = None
LOG_TARGET = H_TARGET = None


def _load(root: Path) -> None:
    """Read every input table once; figures read these module-level frames."""
    global OUT, decisions, quality, models, locations, selections, econ, sens, sites, oracle, grid
    global LOG_TARGET, H_TARGET
    res = root / "results" / "adaptive_v1"
    OUT = res / "figures"
    OUT.mkdir(parents=True, exist_ok=True)
    decisions = pd.read_csv(res / "decisions.csv")
    quality = pd.read_csv(res / "quality_audit.csv")
    models = pd.read_csv(res / "model_comparison.csv")
    locations = pd.read_csv(res / "locations.csv")
    selections = pd.read_csv(res / "selections.csv")
    econ = pd.read_csv(res / "economic_results.csv")
    sens = pd.read_csv(res / "sensitivity.csv")
    sites = pd.read_csv(res / "site_manifest.csv")
    oracle = pd.read_csv(root / "results" / "adaptive_v2_diagnostics" / "stopping_oracle_diagnostic.csv")
    grid = pd.read_parquet(root / "data_processed_v2" / "grid.parquet")
    frozen = json.loads((res / "frozen_config.json").read_text(encoding="utf-8"))
    LOG_TARGET = frozen["quality"]["absolute_log_mae_max"]
    H_TARGET = frozen["quality"]["absolute_thickness_mae_max_m"]


# ================================================================ Fig 1 workflow
def fig_workflow():
    """Method overview: site split -> online loop -> offline audit and economics."""
    fig = plt.figure(figsize=(TEXT_W, 2.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 41)
    ax.axis("off")
    line = "#3d3c39"
    blue_fill, red_fill, region = "#eef4fc", "#fdf1f1", "#f3f2ee"

    def box(x, y, w, h, num, title, body, edge=line, face="white", num_color=line):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.1",
                                    lw=0.9, ec=edge, fc=face, zorder=2))
        ax.add_patch(Circle((x + 2.4, y + h - 2.3), 1.35, color=num_color, zorder=3))
        ax.text(x + 2.4, y + h - 2.35, num, ha="center", va="center", fontsize=6.2, color="white",
                fontweight="bold", zorder=4)
        ax.text(x + 4.6, y + h - 2.3, title, ha="left", va="center", fontsize=7.2, fontweight="bold",
                color=INK, zorder=4)
        ax.text(x + w / 2, y + h - 5.0, body, ha="center", va="top", fontsize=6.2, color=INK2,
                linespacing=1.35, zorder=4)

    def arrow(p, q, color=line, ls="-", lw=0.9, conn="arc3"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=lw, color=color, ls=ls,
                                     connectionstyle=conn, shrinkA=0, shrinkB=0, zorder=1))

    # ---- 1 site split: a schematic plan using the same symbols as the selection maps
    ax.add_patch(FancyBboxPatch((1.0, 12.5), 18.5, 22.0, boxstyle="round,pad=0,rounding_size=1.1",
                                lw=0.9, ec=line, fc="white", zorder=2))
    ax.add_patch(Circle((3.4, 32.2), 1.35, color=line, zorder=3))
    ax.text(3.4, 32.15, "1", ha="center", va="center", fontsize=6.2, color="white", fontweight="bold", zorder=4)
    ax.text(5.6, 32.2, "Split the site", ha="left", va="center", fontsize=7.2, fontweight="bold", zorder=4)
    pts_s0 = [(4.5, 27.5), (15.5, 28.5), (6.0, 16.5), (16.0, 17.0)]
    pts_c = [(8.5, 29.0), (12.0, 26.5), (4.0, 22.5), (9.5, 22.0), (14.0, 22.5), (11.5, 18.0), (8.0, 15.5)]
    pts_a = [(17.0, 30.0), (6.5, 19.5), (13.5, 15.0)]
    ax.scatter(*zip(*pts_c), s=13, facecolor="white", edgecolor=MUTED, lw=0.8, zorder=3)
    ax.scatter(*zip(*pts_s0), s=15, marker="s", color=INK, zorder=3)
    ax.scatter(*zip(*pts_a), s=17, marker="D", facecolor="#fbeaea", edgecolor=BAD, lw=0.8, zorder=3)
    legend = [("s", INK, INK, r"$S_0$  initial ($\geq$5)"), ("o", "white", MUTED, r"$C$  candidate pool"),
              ("D", "#fbeaea", BAD, r"$A$  hidden audit (~20%)")]
    for k, (mk, fc, ec, label) in enumerate(legend):
        yy = 10.0 - 2.9 * k
        ax.scatter([2.4], [yy], s=13, marker=mk, facecolor=fc, edgecolor=ec, lw=0.8)
        ax.text(4.0, yy, label, fontsize=6.0, va="center", color=BAD if mk == "D" else INK2)
    ax.text(1.0, 0.6, r"Full plan $N_{full} = |S_0 \cup C|$", fontsize=5.9, color=INK2, va="bottom")

    # ---- online loop region
    ax.add_patch(FancyBboxPatch((23.0, 5.0), 50.0, 31.0, boxstyle="round,pad=0,rounding_size=1.6",
                                lw=0, fc=region, zorder=0))
    ax.text(48.0, 6.4, "Online loop: sees coordinates and revealed CPTs only", ha="center", va="center",
            fontsize=6.0, style="italic", color=INK2)
    bw, bh = 21.0, 11.5
    xl, xr, yt, yb = 25.0, 50.0, 22.5, 9.0
    box(xl, yt, bw, bh, "2", "Predict", "TabPFN + graph\n" + r"$\ln q_c$ and $H(x)$ everywhere",
        edge=SEQ, face=blue_fill, num_color=SEQ)
    box(xr, yt, bw, bh, "3", "Stop rule met?", "4 stability and error signals\n" + r"$\geq$9 CPTs, 2 rounds in a row")
    box(xr, yb, bw, bh, "4", "Select batch", r"Adaptive: $0.5U + 0.3D + 0.2d$" + "\nSequential: maximin",
        edge=SEQ, face=blue_fill, num_color=SEQ)
    box(xl, yb, bw, bh, "5", "Reveal 2 CPTs", "add the measured curves\nto the site context")
    arrow((xl + bw, yt + bh / 2), (xr, yt + bh / 2))
    arrow((xr + bw / 2, yt), (xr + bw / 2, yb + bh))
    ax.text(xr + bw / 2 + 0.8, (yt + yb + bh) / 2, "no", fontsize=6.0, color=INK2, va="center")
    arrow((xr, yb + bh / 2), (xl + bw, yb + bh / 2))
    arrow((xl + bw / 2, yb + bh), (xl + bw / 2, yt))
    arrow((19.5, yt + bh / 2), (xl, yt + bh / 2))

    # ---- offline outputs
    box(77.5, yt, 21.5, bh, "6", "Audit on A", "pass / false early stop /\nno early stop, vs full plan",
        edge=BAD, face=red_fill, num_color=BAD)
    box(77.5, yb, 21.5, bh, "7", "Economics", "CPTs, field + wait days\n" + r"$\alpha = 1$, net owner benefit")
    arrow((xr + bw, yt + bh / 2), (77.5, yt + bh / 2))
    ax.text((xr + bw + 77.5) / 2, yt + bh / 2 + 1.0, "yes", fontsize=6.0, color=INK2, ha="center", va="bottom")
    arrow((88.25, yt), (88.25, yb + bh))

    # ---- information boundary: hidden labels travel around the loop, straight to the evaluator
    path_x = [17.0, 17.0, 88.25, 88.25]
    path_y = [31.3, 39.0, 39.0, 34.6]
    ax.plot(path_x[:3], path_y[:3], color=BAD, lw=0.9, ls=(0, (3, 2)), zorder=2.5)
    arrow((88.25, 39.0), (88.25, yt + bh), color=BAD, ls=(0, (3, 2)))
    ax.text(48.0, 39.7, "hidden labels of A go to the evaluator only; they never enter the loop",
            ha="center", va="bottom", fontsize=6.0, color=BAD)
    save(fig, "fig01_workflow")


# ================================================================ Fig 2 dataset
def fig_dataset():
    fig = plt.figure(figsize=(TEXT_W, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1, 1], wspace=0.38)
    ax = fig.add_subplot(gs[0])
    replay = sites[sites.replay_eligible].copy()
    low = grid.groupby("site_id").low_1.mean()
    replay["low_frac"] = replay.site_id.map(low)
    replay["order"] = replay.adaptive_role.map({"evaluation": 0, "development": 1})
    replay = replay.sort_values(["order", "n_locations"], ascending=[False, True])
    ys = np.arange(len(replay))
    colors = [EVAL_C if r == "evaluation" else DEV_C for r in replay.adaptive_role]
    ax.barh(ys, replay.n_locations, color=colors, height=0.62)
    for yy, (_, r) in zip(ys, replay.iterrows()):
        ax.text(r.n_locations + 1.2, yy, f"{r.n_locations}  ({100 * r.low_frac:.0f}% low)",
                va="center", fontsize=6.6, color=INK2)
    ax.set_yticks(ys, [SITE_NAME[s] for s in replay.site_id])
    ax.set_xlim(0, 118)
    ax.set_xlabel("Usable CPT positions  (% low: cells with $q_c$ < 1 MPa)", fontsize=7)
    ax.grid(axis="y", visible=False)
    for yy, label, colr in [(5.5, "Evaluation sites", EVAL_C), (1.5, "Development sites", INK2)]:
        ax.text(117, yy, label, ha="right", va="center", fontsize=7, fontweight="bold", color=colr)
    panel(ax, "a", "Replay sites")

    for k, sid in enumerate(["wellington_airport", "newlock"]):
        a = fig.add_subplot(gs[k + 1])
        g = grid[grid.site_id == sid]
        for _, p in g.groupby("profile_id"):
            a.plot(p.qc_mpa, p.z_center_m, color=PROFILE_C, alpha=0.12, lw=0.5)
        q = g.groupby("z_center_m").qc_mpa.quantile([0.1, 0.5, 0.9]).unstack()
        a.fill_betweenx(q.index, q[0.1], q[0.9], color=PROFILE_C, alpha=0.3, lw=0, label="10–90% range")
        a.plot(q[0.5].values, q.index, color=INK, lw=1.6, label="Site median")
        a.axvline(1.0, color=BAD, lw=0.9, ls=(0, (4, 3)))
        a.text(0.93, 7.9, "1 MPa", color=BAD, fontsize=6.4, va="bottom", ha="right")
        a.set_xscale("log")
        a.set_xlim(0.1, 60)
        a.set_ylim(8, 2)
        a.set_xlabel("$q_c$ (MPa, log scale)")
        if k == 0:
            a.set_ylabel("Depth (m)")
            a.legend(loc="center left", fontsize=6.2)
        a.text(0.97, 0.98, f"{g.profile_id.nunique()} CPTs", transform=a.transAxes, fontsize=6.4, color=INK2,
               ha="right", va="top")
        panel(a, "bc"[k], SITE_NAME[sid].replace(" Terneuzen", ""))
    save(fig, "fig02_dataset")


# ================================================================ Fig 3 selection maps
def _revealed(site, seed, policy, max_n):
    s = selections[(selections.site_id == site) & (selections.seed == seed) & (selections.policy_id == policy)]
    q = quality[(quality.site_id == site) & (quality.seed == seed) & (quality.policy_id == policy)]
    counts = q.set_index("round").n_observed
    keep = counts[counts <= max_n].index
    return s[s["round"].isin(keep)], int(counts.loc[keep].max())


def fig_maps():
    seed = 17
    rows = [("wellington_airport", 11, True), ("newlock", 13, False)]
    fig = plt.figure(figsize=(TEXT_W, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.45], hspace=0.4, wspace=0.08)
    for r, (site, max_n, is_final) in enumerate(rows):
        loc = locations[(locations.site_id == site) & (locations.seed == seed)].drop_duplicates("location_id")
        n_full = int(loc.n_full.iloc[0])
        x0, y0 = loc.x_m.min(), loc.y_m.min()
        for c, policy in enumerate(["B_seq", "P_adapt"]):
            ax = fig.add_subplot(gs[r, c])
            sel, n_obs = _revealed(site, seed, policy, max_n)
            new = sel[sel["round"] > 0].merge(loc[["location_id", "x_m", "y_m"]], on="location_id")
            cand = loc[loc.partition == "C"]
            ax.scatter(cand.x_m - x0, cand.y_m - y0, s=15, facecolor="white", edgecolor=MUTED, lw=0.7, zorder=2)
            aud = loc[loc.partition == "A"]
            ax.scatter(aud.x_m - x0, aud.y_m - y0, s=20, marker="D", facecolor="#fbeaea", edgecolor=BAD,
                       lw=0.8, zorder=3)
            ini = loc[loc.partition == "S0"]
            ax.scatter(ini.x_m - x0, ini.y_m - y0, s=24, marker="s", color=INK, zorder=4)
            col = POLICY_COLOR[policy]
            ax.scatter(new.x_m - x0, new.y_m - y0, s=30, color=col, edgecolor="white", lw=0.8, zorder=5)
            for _, p in new.drop_duplicates("location_id").iterrows():
                ax.annotate(str(int(p["round"])), (p.x_m - x0, p.y_m - y0), xytext=(-3, 3),
                            textcoords="offset points", fontsize=6.2, color=col, fontweight="bold",
                            ha="right", zorder=6)
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(labelsize=6.5)
            if c == 0:
                ax.set_ylabel("Northing offset (m)")
            else:
                ax.tick_params(labelleft=False)
            ax.set_xlabel("Easting offset (m)")
            final = decisions[(decisions.site_id == site) & (decisions.seed == seed)
                              & (decisions.policy_id == policy)].n_stop.iloc[0]
            name = "Sequential (maximin)" if policy == "B_seq" else "Adaptive (UDS score)"
            note = (f"online stop at {n_obs} of {n_full} CPTs" if is_final
                    else f"first {n_obs} of {n_full} CPTs shown; final {final}/{n_full}")
            ax.set_title(f"{'abcd'[2 * r + c]}   {SITE_NAME[site].replace(' Terneuzen', '')}: {name}",
                         loc="left", pad=13)
            ax.text(0.0, 1.015, note, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.3, color=INK2)
    handles = [Line2D([], [], marker="s", ls="", color=INK, ms=5, label="Initial S0"),
               Line2D([], [], marker="o", ls="", mfc="white", mec=MUTED, ms=5, label="Candidate (unrevealed)"),
               Line2D([], [], marker="o", ls="", color=SEQ, ms=5, label="Revealed, sequential"),
               Line2D([], [], marker="o", ls="", color=ADAPT, ms=5, label="Revealed, adaptive"),
               Line2D([], [], marker="D", ls="", mfc="#fbeaea", mec=BAD, ms=5, label="Hidden audit A")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0), fontsize=6.5,
               handletextpad=0.3, columnspacing=1.0)
    save(fig, "fig03_selection_maps")


# ================================================================ Fig 4 five models
def fig_models():
    ev = models[(models.role == "evaluation") & models.model_id.isin(MODEL_ORDER)]
    per_site, mean = site_equal(ev, ["log_mae", "thickness_mae_m", "recall", "fpr"])
    fig = plt.figure(figsize=(TEXT_W, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.14)
    ys = {m: len(MODEL_ORDER) - 1 - i for i, m in enumerate(MODEL_ORDER)}
    for k, (col, xl, title) in enumerate([("log_mae", "log-MAE of ln $q_c$", "Profile error"),
                                          ("thickness_mae_m", "Thickness MAE (m)", "Low-cell thickness error")]):
        ax = fig.add_subplot(gs[k])
        for m in MODEL_ORDER:
            for s in EVAL_SITES:
                ax.scatter(per_site.loc[(s, m), col], ys[m], marker=SITE_MARK[s], s=14, facecolor="white",
                           edgecolor=MODEL_COLOR[m], lw=0.8, zorder=3)
            ax.scatter(mean.loc[m, col], ys[m], marker="|", s=260, color=MODEL_COLOR[m], lw=2.4, zorder=4)
            ax.text(mean.loc[m, col], ys[m] + 0.3, f"{mean.loc[m, col]:.3f}", ha="center", va="bottom",
                    fontsize=6.0, color=INK)
        ax.set_yticks(list(ys.values()), [MODEL_NAME[m] for m in ys] if k == 0 else [])
        ax.set_ylim(-0.6, len(MODEL_ORDER) - 0.25)
        ax.set_xlabel(xl)
        ax.grid(axis="y", visible=False)
        panel(ax, "ab"[k], title)
    ax = fig.add_subplot(gs[2])
    for m in MODEL_ORDER:
        sub = per_site.xs(m, level="model_id").dropna(subset=["recall"])
        for site, row in sub.iterrows():
            ax.scatter(100 * row.fpr, 100 * row.recall, s=12, marker=SITE_MARK[site], facecolor="white",
                       edgecolor=MODEL_COLOR[m], lw=0.7, alpha=0.85, zorder=2)
        ax.scatter(100 * mean.loc[m, "fpr"], 100 * mean.loc[m, "recall"], s=50, marker="o",
                   color=MODEL_COLOR[m], edgecolor="white", lw=0.8, zorder=4)
    offs = {"gp": (6, -2), "xgb": (7, -7), "cnp": (6, 3), "tabpfn_joint": (7, -1), "tabpfn_graph": (-7, 6)}
    for m in MODEL_ORDER:
        ax.annotate(MODEL_NAME[m].replace(" residual", ""), (100 * mean.loc[m, "fpr"], 100 * mean.loc[m, "recall"]),
                    xytext=offs[m], textcoords="offset points", fontsize=6.2, va="center",
                    ha="right" if offs[m][0] < 0 else "left", color=INK)
    ax.set_xlabel("False-positive rate (%)")
    ax.set_ylabel("Recall (%)", labelpad=1)
    ax.set_xlim(-1.5, 38)
    ax.set_ylim(-3, 55)
    panel(ax, "c", "Low-cell detection")
    site_h = [Line2D([], [], marker=SITE_MARK[s], ls="", mfc="white", mec=INK2, ms=4.5, label=SITE_NAME[s])
              for s in EVAL_SITES]
    fig.legend(handles=site_h + [Line2D([], [], marker="|", ls="", color=INK2, ms=9, mew=2,
                                        label="Equal-site mean (filled circle in c)")],
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.14), fontsize=6.4, handletextpad=0.2)
    save(fig, "fig04_five_models")


# ================================================================ Fig 5 quality vs CPTs
def fig_quality():
    orc = oracle[oracle.role == "evaluation"]
    fig, axes = plt.subplots(2, 4, figsize=(TEXT_W, 3.7), gridspec_kw={"hspace": 0.42, "wspace": 0.38})
    for c, site in enumerate(EVAL_SITES):
        q = quality[(quality.site_id == site) & (quality.role == "evaluation")]
        for r, (col, tgt, yl) in enumerate([("log_mae", LOG_TARGET, "Audit log-MAE"),
                                            ("thickness_mae_m", H_TARGET, "Audit thickness MAE (m)")]):
            ax = axes[r, c]
            for policy in ["B_seq", "P_adapt"]:
                for seed in [17, 29, 43]:
                    t = q[(q.policy_id == policy) & (q.seed == seed)].sort_values("n_observed")
                    ax.plot(t.n_observed, t[col], color=POLICY_COLOR[policy], lw=1.0, alpha=0.8,
                            ls="-" if policy == "B_seq" else (0, (3, 1.5)))
                    d = decisions[(decisions.site_id == site) & (decisions.seed == seed)
                                  & (decisions.policy_id == policy)].iloc[0]
                    if d.is_early_stop:
                        last = t[t.n_observed == d.n_stop].iloc[-1]
                        ax.scatter(last.n_observed, last[col], marker="*", s=95 if policy == "B_seq" else 45,
                                   color=POLICY_COLOR[policy], edgecolor="white", lw=0.5,
                                   zorder=6 if policy == "B_seq" else 7)
            full = q[q.policy_id == "B_full"]
            ax.scatter(full.n_observed, full[col], marker="s", s=12, color=FULL, zorder=5)
            ax.axhline(tgt, color=BAD, lw=0.9, ls=(0, (4, 3)))
            ob = orc[orc.site_id == site].oracle_earliest_passing_budget.dropna()
            if len(ob):
                ax.axvline(ob.min(), color=INK2, lw=0.9, ls=(0, (1, 2)))
            ax.tick_params(labelsize=6.3)
            if r == 0:
                panel(ax, "abcd"[c], SITE_NAME[site].replace(" Terneuzen", "").replace("Christchurch", "Chch"))
            if c == 0:
                ax.set_ylabel(yl, fontsize=7)
            if r == 1:
                ax.set_xlabel("Observed CPTs", fontsize=7)
            top = max(q[col].max(), tgt) * 1.12
            ax.set_ylim(-0.04 * top, top)
    handles = [Line2D([], [], color=SEQ, lw=1.2, label="Sequential + stop"),
               Line2D([], [], color=ADAPT, lw=1.2, ls=(0, (3, 1.5)), label="Adaptive + stop"),
               Line2D([], [], marker="*", ls="", color=INK2, ms=7, label="Online early stop"),
               Line2D([], [], marker="s", ls="", color=FULL, ms=4, label="Full-plan endpoint"),
               Line2D([], [], color=BAD, lw=0.9, ls=(0, (4, 3)), label="Frozen quality target"),
               Line2D([], [], color=INK2, lw=0.9, ls=(0, (1, 2)), label="Oracle earliest pass")]
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.045), fontsize=6.2,
               handletextpad=0.3, columnspacing=0.9)
    save(fig, "fig05_quality_vs_cpts")


# ================================================================ Fig 6 site outcomes
def fig_outcomes():
    d = decisions[decisions.policy_id.isin(["B_seq", "P_adapt"])].copy()
    order = EVAL_SITES + DEV_SITES
    fig = plt.figure(figsize=(TEXT_W, 3.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.75, 0.42, 0.95], wspace=0.06)
    ypos, y = {}, 0.0
    for s in order[::-1]:
        ypos[s] = y
        y += 1.0 if s != DEV_SITES[0] else 1.7
    offs = {"B_seq": 0.18, "P_adapt": -0.18}
    jit = {17: -0.07, 29: 0.0, 43: 0.07}
    labels = [f"{SITE_NAME[s]}  ($N_{{full}}$ = {int(d[d.site_id == s].n_full.iloc[0])})" for s in order]
    sep = ypos[DEV_SITES[0]] + 0.85

    ax = fig.add_subplot(gs[0])
    for _, r in d.iterrows():
        yy = ypos[r.site_id] + offs[r.policy_id] + jit[r.seed]
        used = 100 * r.n_stop / r.n_full
        if r.false_early_stop:
            ax.scatter(used, yy, marker="X", s=30, color=BAD, zorder=5)
        else:
            ax.scatter(used, yy, marker="o", s=24, lw=1.0, edgecolor=POLICY_COLOR[r.policy_id],
                       facecolor=POLICY_COLOR[r.policy_id] if r.is_early_stop else "white", zorder=4)
    ax.set_yticks([ypos[s] for s in order], labels)
    ax.set_xlim(45, 103)
    ax.set_ylim(-0.7, y - 0.3)
    ax.set_xlabel("CPTs used at stop (% of full plan)")
    ax.axhline(sep, color=AXIS, lw=0.8)
    ax.text(46, sep + 0.08, "Evaluation sites (frozen rule)", fontsize=6.6, color=INK, va="bottom")
    ax.text(46, sep - 0.08, "Development sites (rule tuned here)", fontsize=6.6, color=INK, va="top")
    ax.grid(axis="y", visible=False)
    panel(ax, "a", "Stopping point of all 48 trajectories")

    ax = fig.add_subplot(gs[1])
    for s in order:
        full = decisions[(decisions.site_id == s) & (decisions.policy_id == "B_full")].audit_pass
        n_ok = int(full.sum())
        ax.text(0.5, ypos[s], f"{'pass' if n_ok == 3 else 'FAIL' if n_ok == 0 else 'mixed'} {n_ok}/3",
                ha="center", va="center", fontsize=6.3, fontweight="bold",
                color=GOOD if n_ok == 3 else (BAD if n_ok == 0 else AMBER))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, y - 0.3)
    ax.axis("off")
    ax.axhline(sep, color=AXIS, lw=0.8)
    ax.set_title("Full plan\npasses audit", fontsize=7, pad=2)

    ax = fig.add_subplot(gs[2])
    piv = d.pivot_table(index=["site_id", "seed"], columns="policy_id", values="n_stop").reset_index()
    piv["diff"] = piv.B_seq - piv.P_adapt  # positive: adaptive used fewer CPTs
    for _, r in piv.iterrows():
        col = GOOD if r["diff"] > 0 else (BAD if r["diff"] < 0 else MUTED)
        ax.scatter(r["diff"], ypos[r.site_id] + 2.5 * jit[r.seed], s=22, color=col, edgecolor="white",
                   lw=0.6, zorder=4)
    ax.axvline(0, color=AXIS, lw=0.9)
    ax.set_yticks([ypos[s] for s in order], [])
    ax.set_ylim(-0.7, y - 0.3)
    ax.axhline(sep, color=AXIS, lw=0.8)
    ax.set_xlim(-8.5, 8.5)
    ax.set_xlabel("Fewer CPTs with adaptive")
    ax.grid(axis="y", visible=False)
    panel(ax, "b", "Adaptive vs sequential")
    handles = [Line2D([], [], marker="o", ls="", mfc=SEQ, mec=SEQ, ms=5, label="Sequential, early stop, audit pass"),
               Line2D([], [], marker="o", ls="", mfc=ADAPT, mec=ADAPT, ms=5, label="Adaptive, early stop, audit pass"),
               Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=5, label="No early stop (full plan)"),
               Line2D([], [], marker="o", ls="", color=GOOD, ms=4.5, label="b: adaptive fewer"),
               Line2D([], [], marker="o", ls="", color=BAD, ms=4.5, label="b: adaptive more"),
               Line2D([], [], marker="o", ls="", color=MUTED, ms=4.5, label="b: equal")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.11), fontsize=6.2,
               handletextpad=0.3, columnspacing=1.2)
    save(fig, "fig06_site_outcomes")


# ================================================================ Fig 7 economics
def fig_economics():
    e = econ[(econ.role == "evaluation") & (econ.policy_id == "P_adapt") & (econ.alpha == 1.0)]
    comp = ["direct_saving", "project_time_saving", "early_opening_benefit"]
    m = e.groupby("site_id")[comp + ["ai_incremental_cost", "net_benefit", "n_stop", "n_full",
                                     "project_days_saved"]].mean()
    m["ai"] = -m.ai_incremental_cost
    parts = [("direct_saving", "Direct CPT cost", "#2a78d6"),
             ("project_time_saving", "Project-time cost", "#1baf7a"),
             ("early_opening_benefit", "Earlier-opening benefit", "#eda100"),
             ("ai", "AI deployment + review", "#e87ba4")]
    fig = plt.figure(figsize=(TEXT_W, 3.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], wspace=0.32, hspace=0.62)
    ax = fig.add_subplot(gs[:, 0])
    for i, s in enumerate(EVAL_SITES):
        pos, neg = 0.0, 0.0
        for key, _, colr in parts:
            v = m.loc[s, key] / 1000
            ax.bar(i, v, bottom=pos if v >= 0 else neg, width=0.58, color=colr, edgecolor="white", lw=0.8)
            pos, neg = (pos + v, neg) if v >= 0 else (pos, neg + v)
        net = m.loc[s, "net_benefit"] / 1000
        ax.scatter(i, net, marker="D", s=28, color=INK, zorder=5)
        label = f"{net:+.1f}k".replace("-", "\u2212")
        ax.text(i, net + (2.5 if net > 0 else -2.5), label, ha="center", va="bottom" if net > 0 else "top",
                fontsize=6.6, fontweight="bold")
    ax.axhline(0, color=INK2, lw=0.8)
    short = {"wellington_airport": "Airport", "christchurch_stadium": "Chch Stadium", "hollywood": "Hollywood",
             "newlock": "New Lock"}
    ticks = [f"{short[s]}\n{m.loc[s, 'n_stop']:.0f}/{m.loc[s, 'n_full']:.0f} CPTs" for s in EVAL_SITES]
    ax.set_xticks(range(len(EVAL_SITES)), ticks, fontsize=6.4)
    ax.set_ylim(-66, 32)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylabel("vs full plan (EUR thousand)")
    ax.grid(axis="x", visible=False)
    panel(ax, "a", r"Adaptive policy, $\alpha = 1$")
    ax.legend(handles=[Patch(color=c, label=l) for _, l, c in parts]
              + [Line2D([], [], marker="D", ls="", color=INK, ms=4.5, label="Net benefit")],
              loc="lower left", fontsize=6.2, borderaxespad=0.3)

    ax = fig.add_subplot(gs[0, 1])
    g = sens[(sens.sensitivity_kind == "alpha_opening_grid") & (sens.role == "evaluation")
             & (sens.policy_id == "P_adapt")]
    styles = {0.5: (0, (1, 1.5)), 0.75: (0, (4, 2)), 1.0: "-"}
    for s, colr in [("wellington_airport", SEQ), ("newlock", ADAPT)]:
        for a in [0.5, 0.75, 1.0]:
            t = g[(g.site_id == s) & (g.alpha == a)].groupby("opening_net_per_calendar_day").net_benefit.mean()
            ax.plot(t.index / 1000, t.values / 1000, color=colr, ls=styles[a], lw=1.4, marker="o", ms=3)
        ax.text(10.4, t.values[-1] / 1000, SITE_NAME[s].replace(" Terneuzen", "").replace("Wellington ", ""),
                color=colr, fontsize=6.4, fontweight="bold", va="center")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylim(-135, 50)
    ax.axvline(3, color=AXIS, lw=0.9)
    ax.text(3.2, ax.get_ylim()[1] * 0.98, "base case", fontsize=5.8, color=INK2, va="top")
    ax.set_xlim(-0.4, 13.8)
    ax.set_xlabel("Earlier-opening benefit (k€/day)")
    ax.set_ylabel("Net (k€)")
    ax.legend(handles=[Line2D([], [], color=INK2, ls=styles[a], lw=1.3, label=rf"$\alpha$ = {a:g}")
                       for a in [1.0, 0.75, 0.5]], loc="lower left", fontsize=6.0, ncol=3,
              columnspacing=0.8, handlelength=1.6)
    panel(ax, "b", r"$\alpha$ and daily benefit")

    ax = fig.add_subplot(gs[1, 1])
    o = sens[(sens.sensitivity_kind == "one_at_a_time") & (sens.role == "evaluation")
             & (sens.policy_id == "P_adapt") & (sens.varied_parameter == "batch_wait_calendar_days")]
    lat = o.groupby(["site_id", "parameter_value"]).net_benefit.mean().unstack()
    lat[0.25] = m.net_benefit
    lat = lat[sorted(lat.columns)]
    site_c = {"wellington_airport": SEQ, "christchurch_stadium": "#1baf7a", "hollywood": "#eda100",
              "newlock": ADAPT}
    label_y = {"wellington_airport": 12, "christchurch_stadium": -32, "hollywood": -62, "newlock": -222}
    label_text = {"christchurch_stadium": "Chch Stadium", "hollywood": "Hollywood"}
    for site in EVAL_SITES:
        ax.plot(lat.columns, lat.loc[site] / 1000, color=site_c[site], marker="o", ms=3, lw=1.4)
        ax.text(1.05, label_y[site], label_text.get(site, short[site]), color=site_c[site], fontsize=6.0,
                va="center", fontweight="bold")
    ax.axvline(0.25, color=AXIS, lw=0.9)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlim(-0.05, 1.5)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("Decision delay per batch (days)")
    ax.set_ylabel("Net (k€)")
    panel(ax, "c", "Batch decision latency")
    save(fig, "fig07_economics")


FIGURE_BUILDERS = [fig_workflow, fig_dataset, fig_maps, fig_models, fig_quality, fig_outcomes, fig_economics]


def build_figures(root: Path) -> list[str]:
    """Regenerate every report figure from the saved results under ``root``."""
    _load(root)
    for builder in FIGURE_BUILDERS:
        builder()
    return FIGURES


if __name__ == "__main__":
    build_figures(Path.cwd())
