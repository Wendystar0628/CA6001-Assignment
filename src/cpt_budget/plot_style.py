"""Shared visual semantics for real CPT results; report and slide layouts."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#263442"
MUTED = "#64727D"
GRID = "#E7EBED"
TEAL = "#237C82"
ALERT = "#C5563E"
MODEL_COLORS = {"idw": "#7C848B", "gp": "#0072B2", "xgb": "#009E73",
                "tabpfn": "#E69F00", "tabpfn_finetuned": "#CC79A7"}
MODEL_NAMES = {"idw": "IDW", "gp": "GP", "xgb": "XGBoost", "tabpfn": "TabPFN v2",
               "tabpfn_finetuned": "Fine-tuned TabPFN"}
POLICY_NAMES = {"maximin": "Maximin", "uncertainty": "Uncertainty",
                "threshold_uncertainty": "Threshold-sensitive"}
POLICY_LINES = {"maximin": "-", "uncertainty": "--", "threshold_uncertainty": ":"}


@contextmanager
def figure_style(preset="report"):
    """Keep fonts editable; slides are drawn afresh at exactly 1920 x 1080."""
    slides = preset == "slides"
    with plt.rc_context({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 18 if slides else 9,
        "axes.titlesize": 20 if slides else 10,
        "axes.labelsize": 18 if slides else 9,
        "xtick.labelsize": 16 if slides else 8,
        "ytick.labelsize": 16 if slides else 8,
        "legend.fontsize": 16 if slides else 8,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#ADB7BD",
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .65, "lines.linewidth": 1.6 if slides else 1.1,
        "grid.color": GRID, "grid.linewidth": .6,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "svg.fonttype": "none", "pdf.fonttype": 42,
        "legend.frameon": False, "axes.unicode_minus": True,
    }):
        yield


def canvas(preset="report", height_mm=108):
    return plt.figure(figsize=(16, 9) if preset == "slides" else (175 / 25.4, height_mm / 25.4))


def heading(fig, title, subtitle, preset="report", *, footer=""):
    fig.text(.065, .955, title, ha="left", va="top", fontweight="bold",
             fontsize=28 if preset == "slides" else 12)
    fig.text(.065, .90, subtitle, ha="left", va="top", color=MUTED,
             fontsize=17 if preset == "slides" else 8)
    if footer:
        fig.text(.065, .025, footer, ha="left", va="bottom", color=MUTED,
                 fontsize=16 if preset == "slides" else 8)


def panel_label(ax, letter, title=None):
    ax.text(-.02, 1.08, letter, transform=ax.transAxes, fontweight="bold", va="bottom")
    if title:
        ax.set_title(title, loc="left", pad=12)


def method_name(model, policy=None):
    name = MODEL_NAMES.get(str(model), str(model))
    return name if policy is None else f"{name} / {POLICY_NAMES.get(str(policy), str(policy))}"


def finish_axis(ax, *, grid="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=grid, zorder=0)


def nice_scale(length):
    if length <= 0 or not np.isfinite(length):
        return 1.
    magnitude = 10 ** np.floor(np.log10(length))
    return min([1., 2., 5., 10.], key=lambda k: abs(k * magnitude - length)) * magnitude


def scale_bar(ax):
    """Use the actual metre axis; no unsupported north arrow."""
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    length = nice_scale((right - left) * .22)
    x, y = left + (right - left) * .06, bottom + (top - bottom) * .08
    ax.plot([x, x + length], [y, y], color=INK, lw=2.5, solid_capstyle="butt")
    ax.text(x + length / 2, y + (top - bottom) * .035, f"{length:g} m", ha="center",
            va="bottom", fontsize=plt.rcParams["xtick.labelsize"],
            bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none", "pad": .5})


def save_figure(fig, out_dir: Path, name: str, preset="report"):
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = ("svg", "pdf", "png") if preset == "report" else ("svg", "png")
    paths = []
    for extension in formats:
        path = out_dir / f"{name}.{extension}"
        fig.savefig(path, dpi=120 if preset == "slides" else 300)
        paths.append(path)
    plt.close(fig)
    return paths
