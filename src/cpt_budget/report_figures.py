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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

# ---------------------------------------------------------------- style
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FULL, SEQ, ADAPT = "#52514e", "#2a78d6", "#eb6834"
GOOD, BAD = "#0ca30c", "#d03b3b"
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
    "bars and filled marks are the equal-site mean. Recall is undefined at Wellington Airport (no low-qc cells).",
    "Hidden-audit quality as CPTs are added (three seeds per policy). Stars mark online early stops; squares "
    "the full-plan endpoint; the red line the frozen target; the green line the oracle earliest passing count "
    "(uses hidden labels, diagnostic only).",
    "Decision outcomes for all 48 sequential and adaptive trajectories. (a) CPTs used at the stop and the "
    "full-plan audit verdict per site; (b) CPTs saved by adaptive relative to sequential, paired by seed.",
    "Owner-financial scenario, EUR 2025. (a) Benefit components of the adaptive policy relative to the full "
    "plan at alpha = 1 (mean of three seeds). (b) Net benefit as alpha and the daily earlier-opening benefit vary.",
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
    fig = plt.figure(figsize=(TEXT_W, 2.55))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, title, body, edge, face):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                    lw=1.0, ec=edge, fc=face))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=7.6, fontweight="bold")
        ax.text(x + w / 2, y + h - 6.6, body, ha="center", va="top", fontsize=6.4, color=INK2,
                linespacing=1.3)

    def arrow(p, q, color=INK2, ls="-", conn="arc3"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=0.9, color=color,
                                     ls=ls, connectionstyle=conn, shrinkA=0, shrinkB=0))

    blue_f, org_f, gray_f = "#eaf2fc", "#fdeee7", "#f3f2ee"
    w, h, y = 17.6, 15.5, 28.5
    xs = [1.2, 21.0, 40.8, 60.6, 80.4]
    box(xs[0], y, w, h, "1  Initial CPTs", "S0: >= 5 spatially\ndispersed positions\n(same for all policies)", INK2, gray_f)
    box(xs[1], y, w, h, "2  AI prediction", "TabPFN + graph\nln qc on 24 cells, 2-8 m\nlow-qc thickness H(x)", SEQ, blue_f)
    box(xs[2], y, w, h, "3  Choose next batch", "a = 0.5 U + 0.3 D + 0.2 S\nU GP std, D disagreement,\nS distance; 2 groups", ADAPT, org_f)
    box(xs[3], y, w, h, "4  Stop rule met?", "stability, support error,\nGP uncertainty, coverage;\n>= 9 CPTs, 2 rounds running", ADAPT, org_f)
    box(xs[4], y, w, h, "5  Economics", "CPTs, field + wait days,\nalpha = 1 time transfer,\nnet owner benefit", INK2, gray_f)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow((a + w + 0.4, y + h / 2), (b - 0.4, y + h / 2))
    ax.text(xs[4] - 1.0, y + h + 0.7, "yes", fontsize=6.2, color=INK2, ha="center", va="bottom")
    arrow((xs[3] + w / 2 - 3, y - 0.4), (xs[1] + w / 2 + 3, y - 0.4), color=ADAPT, conn="arc3,rad=-0.22")
    ax.text((xs[1] + xs[3] + w) / 2, 21.6, "no: reveal the batch, update the site context, predict again",
            ha="center", va="center", fontsize=6.4, color=ADAPT)

    ax.plot([1, 99], [15.6, 15.6], color=BAD, lw=0.9, ls=(0, (4, 3)))
    ax.text(1.2, 13.6, "Information boundary: hidden audit CPTs (A) never enter prediction, selection or stopping",
            fontsize=6.4, color=BAD, va="top")
    parts = [("S0  initial", 7.0, INK2), ("C  candidate pool", 13.0, SEQ), ("A  hidden audit (~20%)", 14.5, BAD)]
    x0 = 1.2
    for label, width, col in parts:
        ax.add_patch(FancyBboxPatch((x0, 3.4), width * 1.4, 5.0, boxstyle="round,pad=0.1,rounding_size=0.6",
                                    lw=0.8, ec=col, fc="white"))
        ax.text(x0 + width * 0.7, 5.9, label, ha="center", va="center", fontsize=6.2, color=col)
        x0 += width * 1.4 + 0.8
    ax.text(1.2, 0.6, "N_full = |S0 + C| (audit excluded); near-duplicate groups always move together",
            fontsize=5.9, color=INK2, va="bottom")
    box(66.0, 1.0, 33.0, 10.6, "Independent audit after the stop",
        "pass  /  false early stop  /  no early stop\ncompared with B_full on the same A", BAD, "#fbeaea")
    arrow((xs[3] + w - 2.0, y - 0.4), (xs[3] + w - 2.0, 12.0), color=BAD, ls=(0, (3, 2)))
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
    colors = [ADAPT if r == "evaluation" else SEQ for r in replay.adaptive_role]
    ax.barh(ys, replay.n_locations, color=colors, height=0.62)
    for yy, (_, r) in zip(ys, replay.iterrows()):
        ax.text(r.n_locations + 1.2, yy, f"{r.n_locations}  ({100 * r.low_frac:.0f}% low)",
                va="center", fontsize=6.2, color=INK2)
    ax.set_yticks(ys, [SITE_NAME[s] for s in replay.site_id])
    ax.set_xlim(0, 118)
    ax.set_xlabel("Usable CPT positions  (% low: cells with qc < 1 MPa)", fontsize=7)
    ax.grid(axis="y", visible=False)
    for yy, label, colr in [(5.5, "Evaluation sites", ADAPT), (1.5, "Development sites", SEQ)]:
        ax.text(117, yy, label, ha="right", va="center", fontsize=6.6, fontweight="bold", color=colr)
    panel(ax, "a", "Replay sites")

    for k, (sid, col) in enumerate([("wellington_airport", SEQ), ("newlock", ADAPT)]):
        a = fig.add_subplot(gs[k + 1])
        g = grid[grid.site_id == sid]
        for _, p in g.groupby("profile_id"):
            a.plot(p.qc_mpa, p.z_center_m, color=col, alpha=0.2, lw=0.6)
        med = g.groupby("z_center_m").qc_mpa.median()
        a.plot(med.values, med.index, color=INK, lw=1.6, label="Site median")
        a.axvline(1.0, color=BAD, lw=0.9, ls=(0, (4, 3)))
        a.text(0.92, 2.1, "1 MPa", color=BAD, fontsize=6.2, va="top", ha="right")
        a.set_xscale("log")
        a.set_xlim(0.1, 60)
        a.set_ylim(8, 2)
        a.set_xlabel("qc (MPa)")
        if k == 0:
            a.set_ylabel("Depth (m)")
            a.legend(loc="lower left", fontsize=6.2, bbox_to_anchor=(0.0, 0.08))
        a.text(0.03, 0.02, f"{g.profile_id.nunique()} CPTs", transform=a.transAxes, fontsize=6.2, color=INK2)
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
    for k, (col, xl, title) in enumerate([("log_mae", "log-MAE of ln qc", "Profile error"),
                                          ("thickness_mae_m", "Thickness MAE (m)", "Low-qc thickness error")]):
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
        ax.scatter(100 * sub.fpr, 100 * sub.recall, s=10, marker=MODEL_MARK[m], facecolor="white",
                   edgecolor=MODEL_COLOR[m], lw=0.7, alpha=0.8, zorder=2)
        ax.scatter(100 * mean.loc[m, "fpr"], 100 * mean.loc[m, "recall"], s=46, marker=MODEL_MARK[m],
                   color=MODEL_COLOR[m], edgecolor="white", lw=0.8, zorder=4)
    offs = {"gp": (6, -2), "xgb": (7, -7), "cnp": (6, 2), "tabpfn_joint": (7, -1), "tabpfn_graph": (7, 5)}
    for m in MODEL_ORDER:
        ax.annotate(MODEL_NAME[m].replace(" residual", ""), (100 * mean.loc[m, "fpr"], 100 * mean.loc[m, "recall"]),
                    xytext=offs[m], textcoords="offset points", fontsize=6.2, va="center",
                    ha="right" if offs[m][0] < 0 else "left", color=INK)
    ax.set_xlabel("False-positive rate (%)")
    ax.set_ylabel("Recall (%)", labelpad=1)
    ax.set_xlim(-1.5, 38)
    ax.set_ylim(-3, 55)
    panel(ax, "c", "Low-qc detection")
    site_h = [Line2D([], [], marker=SITE_MARK[s], ls="", mfc="white", mec=INK2, ms=4.5, label=SITE_NAME[s])
              for s in EVAL_SITES]
    fig.legend(handles=site_h + [Line2D([], [], marker="|", ls="", color=INK2, ms=9, mew=2,
                                        label="Site-equal mean (filled in c)")],
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
                        ax.scatter(last.n_observed, last[col], marker="*", s=70, color=POLICY_COLOR[policy],
                                   edgecolor="white", lw=0.5, zorder=6)
            full = q[q.policy_id == "B_full"]
            ax.scatter(full.n_observed, full[col], marker="s", s=12, color=FULL, zorder=5)
            ax.axhline(tgt, color=BAD, lw=0.9, ls=(0, (4, 3)))
            ob = orc[orc.site_id == site].oracle_earliest_passing_budget.dropna()
            if len(ob):
                ax.axvline(ob.min(), color=GOOD, lw=0.9, ls=(0, (1, 2)))
            ax.tick_params(labelsize=6.3)
            if r == 0:
                panel(ax, "abcd"[c], SITE_NAME[site].replace(" Terneuzen", "").replace("Christchurch", "Chch"))
            if c == 0:
                ax.set_ylabel(yl, fontsize=7)
            if r == 1:
                ax.set_xlabel("Observed CPTs", fontsize=7)
            ax.set_ylim(0, max(q[col].max(), tgt) * 1.12)
    handles = [Line2D([], [], color=SEQ, lw=1.2, label="Sequential + stop"),
               Line2D([], [], color=ADAPT, lw=1.2, ls=(0, (3, 1.5)), label="Adaptive + stop"),
               Line2D([], [], marker="*", ls="", color=INK2, ms=7, label="Online early stop"),
               Line2D([], [], marker="s", ls="", color=FULL, ms=4, label="Full-plan endpoint"),
               Line2D([], [], color=BAD, lw=0.9, ls=(0, (4, 3)), label="Frozen quality target"),
               Line2D([], [], color=GOOD, lw=0.9, ls=(0, (1, 2)), label="Oracle earliest pass")]
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
    labels = [f"{SITE_NAME[s]}  (N_full {int(d[d.site_id == s].n_full.iloc[0])})" for s in order]
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
    ax.text(46, sep + 0.08, "Evaluation sites (frozen rule)", fontsize=6.2, color=INK2, va="bottom")
    ax.text(46, sep - 0.08, "Development sites (rule tuned here)", fontsize=6.2, color=INK2, va="top")
    ax.grid(axis="y", visible=False)
    panel(ax, "a", "Stopping point of all 48 trajectories")

    ax = fig.add_subplot(gs[1])
    for s in order:
        full = decisions[(decisions.site_id == s) & (decisions.policy_id == "B_full")].audit_pass
        n_ok = int(full.sum())
        ax.text(0.5, ypos[s], f"{'pass' if n_ok == 3 else 'FAIL' if n_ok == 0 else 'mixed'} {n_ok}/3",
                ha="center", va="center", fontsize=6.3, fontweight="bold",
                color=GOOD if n_ok == 3 else BAD)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, y - 0.3)
    ax.axis("off")
    ax.axhline(sep, color=AXIS, lw=0.8)
    ax.set_title("Full plan\naudit", fontsize=7, pad=2)

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
               Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=5, label="No early stop"),
               Line2D([], [], marker="X", ls="", color=BAD, ms=5, label="False early stop (none)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07), fontsize=6.2,
               handletextpad=0.3, columnspacing=0.9)
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
    fig = plt.figure(figsize=(TEXT_W, 2.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.3)
    ax = fig.add_subplot(gs[0])
    for i, s in enumerate(EVAL_SITES):
        pos, neg = 0.0, 0.0
        for key, _, colr in parts:
            v = m.loc[s, key] / 1000
            ax.bar(i, v, bottom=pos if v >= 0 else neg, width=0.58, color=colr, edgecolor="white", lw=0.8)
            pos, neg = (pos + v, neg) if v >= 0 else (pos, neg + v)
        net = m.loc[s, "net_benefit"] / 1000
        ax.scatter(i, net, marker="D", s=28, color=INK, zorder=5)
        ax.text(i + 0.34, net, f"{net:+.1f}k", va="center", fontsize=6.4, fontweight="bold")
    ax.axhline(0, color=INK2, lw=0.8)
    short = {"wellington_airport": "Airport", "christchurch_stadium": "Chch Stadium", "hollywood": "Hollywood",
             "newlock": "New Lock"}
    ticks = [f"{short[s]}\n{m.loc[s, 'n_stop']:.0f}/{m.loc[s, 'n_full']:.0f} CPTs\n"
             f"{m.loc[s, 'project_days_saved']:+.1f} days" for s in EVAL_SITES]
    ax.set_xticks(range(len(EVAL_SITES)), ticks, fontsize=6.0)
    ax.set_ylim(-62, 30)
    ax.set_xlim(-0.5, 3.8)
    ax.set_ylabel("vs full plan (EUR thousand, 2025)")
    ax.grid(axis="x", visible=False)
    panel(ax, "a", "Adaptive policy, alpha = 1, mean of 3 seeds")
    ax.legend(handles=[Patch(color=c, label=l) for _, l, c in parts]
              + [Line2D([], [], marker="D", ls="", color=INK, ms=4.5, label="Net benefit")],
              loc="lower left", fontsize=5.9, borderaxespad=0.2)

    ax = fig.add_subplot(gs[1])
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
    ax.set_xlim(-0.4, 13.2)
    ax.set_xlabel("Owner benefit of earlier opening (EUR thousand/day)")
    ax.set_ylabel("Net benefit (EUR thousand)")
    ax.legend(handles=[Line2D([], [], color=INK2, ls=styles[a], lw=1.3, label=f"alpha = {a:g}")
                       for a in [1.0, 0.75, 0.5]], loc="lower left", fontsize=6.2)
    panel(ax, "b", "Sensitivity")
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
