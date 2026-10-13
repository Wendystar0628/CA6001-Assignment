"""Compact, source-backed v0.2 figures; plotting never calls a predictor.

Figure contract (Python, quantitative grids, 175 mm report / 16:9 slides):
V1 separates unchanged point estimates from independently calibrated risk.
V2 shows the validation promotion decision and each numerical guardrail.
V3 isolates policy changes under one fixed prediction model.
V4 describes quality-target attainability, never an algorithm improvement.

Only explicitly labelled validation / test tables are eligible. Validation risk
is displayed only when its calibration is marked out-of-fold. Incomplete runs
produce only supported panels; no mock data or placeholder figures are made.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from ..plot_style import (ALERT, GRID, INK, MUTED, TEAL, canvas, figure_style,
                          finish_axis, save_figure)

MODEL_ORDER = ("idw", "gp", "idw_residual", "tabpfn_spatial", "tabpfn_crps", "tabpfn_task")
MODEL_NAMES = {"idw": "IDW", "gp": "GP", "idw_residual": "IDW + residual",
               "tabpfn_spatial": "TabPFN spatial", "tabpfn_crps": "TabPFN CRPS",
               "tabpfn_task": "TabPFN task"}
MODEL_COLORS = {"idw": "#8B939A", "gp": "#477EA4", "idw_residual": "#C49A62",
                "tabpfn_spatial": "#A998B8", "tabpfn_crps": "#86729E",
                "tabpfn_task": "#237C82"}
POLICY_ORDER = ("maximin", "gp_threshold", "gp_alc", "gp_sur")
POLICY_NAMES = {"maximin": "Maximin", "gp_threshold": "GP threshold",
                "gp_alc": "Block ALC", "gp_sur": "Risk SUR"}
POLICY_COLORS = {"maximin": "#8B939A", "gp_threshold": "#B58B61",
                 "gp_alc": "#727EA9", "gp_sur": "#237C82"}
POLICY_STYLES = {"maximin": "--", "gp_threshold": ":", "gp_alc": "-.", "gp_sur": "-"}
POLICY_MARKERS = {"maximin": "s", "gp_threshold": "v", "gp_alc": "o", "gp_sur": "D"}
RISK_COLUMNS = ("risk_recall", "risk_precision", "risk_fpr", "risk_brier", "h_risk_expected_mae_m")
KEYS = ["fold_id", "site_id", "seed", "model_id", "policy_id", "budget_n"]


def _table(directory, names):
    for name in names:
        for folder in (directory, directory / "summary", directory / "tables"):
            for extension in ("parquet", "csv"):
                path = folder / f"{name}.{extension}"
                if path.exists():
                    try:
                        frame = pd.read_parquet(path) if extension == "parquet" else pd.read_csv(path)
                    except pd.errors.EmptyDataError:
                        continue
                    if not frame.empty:
                        return frame, path
    return pd.DataFrame(), None


def _json(directory, names):
    for name in names:
        for folder in (directory, directory / "summary"):
            path = folder / name
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8-sig"))
    return {}


def _role(frame):
    return next((c for c in ("split_role", "role") if c in frame), None)


def _is_oof(frame):
    result = pd.Series(False, index=frame.index)
    for column in ("calibration_scope", "risk_calibration_scope", "calibration_method", "risk_calibration", "risk_calibration_mode"):
        if column in frame:
            value = frame[column].astype(str).str.lower()
            result |= value.str.contains("oof|crossfit|cross.fit|leave.*group", regex=True, na=False)
    if "risk_oof" in frame:
        result |= frame.risk_oof.eq(True)
    return result


def _load_metrics(directory):
    raw, raw_path = _table(directory, ["metrics"])
    oof, oof_path = _table(directory, ["metrics_oof", "validation_metrics_oof", "validation_oof_metrics"])
    if not oof.empty:
        if _role(oof) is None or not oof[_role(oof)].eq("validation").all():
            raise ValueError("OOF figure input must explicitly contain only validation rows")
        oof = oof.copy()
        oof["calibration_scope"] = "oof"
        if not raw.empty:
            common = [k for k in KEYS if k in raw and k in oof]
            if len(common) != len(KEYS):
                raise ValueError("Cannot replace validation metrics without complete replay keys")
            marker = oof[common].drop_duplicates().assign(_has_oof=True)
            raw = raw.merge(marker, on=common, how="left")
            # Only validation rows are replaceable; identical test keys stay test.
            mask = raw._has_oof.eq(True) & raw[_role(raw)].eq("validation")
            raw = raw.loc[~mask].drop(columns="_has_oof")
        raw = pd.concat([raw, oof], ignore_index=True)
    if raw.empty:
        return raw, []
    role = _role(raw)
    if role is None:
        raise ValueError("v0.2 figures require explicit validation/test role metadata")
    raw = raw.loc[raw[role].isin(["validation", "test", "pilot"])].copy()
    raw["figure_scope"] = raw[role]
    meta = _json(directory, ["metadata.json", "run.json"])
    stage = " ".join(str(meta.get(k, "")) for k in ("study", "mode", "stage")).lower()
    if any(word in stage for word in ("smoke", "pilot")):
        raw.loc[raw.figure_scope.eq("validation"), "figure_scope"] = "pilot"
    # Fitted-in-sample calibration may remain in tables but cannot masquerade
    # as evidence of validation performance in an exported quantitative panel.
    unsafe = raw.figure_scope.isin(["validation", "pilot"]) & ~_is_oof(raw)
    for column in RISK_COLUMNS:
        if column in raw:
            raw.loc[unsafe, column] = np.nan
    raw["figure_risk_oof"] = _is_oof(raw)
    duplicated = raw.duplicated(["figure_scope", *KEYS])
    if duplicated.any():
        raise ValueError("Duplicate scope/replay keys would overweight v0.2 figure observations")
    return raw, [str(p.relative_to(directory)) for p in (raw_path, oof_path) if p]


def _load_policy_metrics(directory):
    frame, path = _table(directory, ["policy_metrics"])
    if frame.empty:
        return frame, None
    role = _role(frame)
    if role is None:
        raise ValueError("Policy figures require explicit validation/test role metadata")
    frame = frame.loc[frame[role].isin(["validation", "test", "pilot"])].copy()
    frame["figure_scope"] = frame[role]
    frame["figure_risk_oof"] = _is_oof(frame)
    unsafe = frame.figure_scope.isin(["validation", "pilot"]) & ~frame.figure_risk_oof
    for column in RISK_COLUMNS:
        if column in frame:
            frame.loc[unsafe, column] = np.nan
    if frame.duplicated(["figure_scope", *KEYS]).any():
        raise ValueError("Duplicate policy replay keys would overweight figure observations")
    return frame, str(path.relative_to(directory))


def _models(frame):
    available = set(frame.model_id.astype(str))
    return [m for m in MODEL_ORDER if m in available] + sorted(available.difference(MODEL_ORDER))


def _site_means(frame, columns, group_columns):
    """Seed mean -> fold mean within site -> distinct-site macro mean later."""
    columns = [c for c in columns if c in frame]
    if not columns:
        return pd.DataFrame()
    keys = list(dict.fromkeys([*group_columns, "site_id"]))
    first = frame.groupby([*keys, "fold_id"], dropna=False)[columns].mean().reset_index()
    return first.groupby(keys, dropna=False)[columns].mean().reset_index()


def _layout(preset, title, subtitle, *, height=143, footer=""):
    fig = canvas(preset, height_mm=height)
    slide = preset == "slides"
    fig.text(.075, .967, title, va="top", weight="bold", fontsize=27 if slide else 11.5)
    fig.text(.075, .91, subtitle, va="top", color=MUTED, fontsize=15 if slide else 8)
    if footer:
        fig.text(.075, .023, footer, va="bottom", color=MUTED, fontsize=12.5 if slide else 8)
    return fig


def _scope_label(scope, risk=False):
    if scope == "test":
        return "EXPLORATORY ROUND-2 TEST"
    if scope == "pilot":
        return "VALIDATION PILOT · partial results"
    return "SOURCE VALIDATION · group-OOF risk" if risk else "SOURCE VALIDATION"


def _panel(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold", pad=11)
    finish_axis(ax, grid="x")


def _model_dots(ax, site, models, column, *, second=None, preset="report"):
    """Every small dot is a distinct-site mean, never an independent seed."""
    slide = preset == "slides"
    for i, model in enumerate(models):
        block = site.loc[site.model_id.eq(model)]
        has_pair = second in block and block[second].notna().any() if second else False
        for col, marker, offset, fill in ((column, "o", -.11 if has_pair else 0., True),
                                          (second, "D", .11, False)):
            if not col or col not in block:
                continue
            vals = block[col].dropna().to_numpy(float)
            if not len(vals):
                continue
            jitter = np.linspace(-.04, .04, len(vals)) if len(vals) > 1 else np.zeros(1)
            color = MODEL_COLORS.get(model, TEAL)
            ax.scatter(vals, i + offset + jitter, color=color, s=13 if slide else 5,
                       alpha=.30, zorder=3, clip_on=False)
            ax.scatter([vals.mean()], [i + offset], marker=marker, s=72 if slide else 30,
                       facecolors=color if fill else "white", edgecolors=color,
                       linewidths=1.4 if slide else 1, zorder=5, clip_on=False)
        if second and column in block and second in block:
            left, right = block[column].mean(), block[second].mean()
            if np.isfinite(left) and np.isfinite(right):
                ax.plot([left, right], [i - .11, i + .11], color=MODEL_COLORS.get(model, TEAL),
                        alpha=.5, linewidth=.8, zorder=2)
    ax.set_yticks(range(len(models)), [MODEL_NAMES.get(m, m) for m in models])
    ax.set_ylim(len(models) - .5, -.5)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(left=0)


def _point_risk(metrics, out, preset, scope):
    selected = metrics.loc[metrics.policy_id.eq("maximin")].copy()
    if selected.empty:
        return [], None
    budget = int(selected.budget_n.max())
    selected = selected.loc[selected.budget_n.eq(budget)]
    fields = ["log_mae", "h_point_mae_m", "h_risk_expected_mae_m", "recall", "risk_recall",
              "false_positive_rate", "risk_fpr"]
    site = _site_means(selected, fields, ["model_id"])
    models = _models(selected)
    risk = any(c in site and site[c].notna().any() for c in RISK_COLUMNS)
    n_sites = selected.site_id.nunique()
    n_tasks = selected[["fold_id", "site_id"]].drop_duplicates().shape[0]
    with figure_style(preset):
        fig = _layout(preset, "Point accuracy and risk sensitivity remain separate",
                      f"{_scope_label(scope, risk)}  |  Maximin, {budget} measured CPTs",
                      footer=f"Large symbols: macro mean; small dots: {n_sites} named sites ({n_tasks} fold-site tasks).\n"
                             + ("Circles: original point rule. Open diamonds: calibrated risk rule (FN:FP = 4:1)." if risk else
                                "Circles: original point rule. Risk is hidden until OOF validation is available."))
        axes = fig.subplots(2, 2)
        fig.subplots_adjust(left=.22, right=.98, top=.80, bottom=.19, hspace=.58, wspace=.60)
        settings = [("log_mae", None, "Profile error", "log-MAE (lower is better)"),
                    ("h_point_mae_m", "h_risk_expected_mae_m", "Low-qc thickness", "MAE (m; lower is better)"),
                    ("recall", "risk_recall", "Low-qc recall", "Recall (higher is better)"),
                    ("false_positive_rate", "risk_fpr", "False-positive rate", "FPR (lower is better)")]
        for index, (ax, (col, second, title, label)) in enumerate(zip(axes.flat, settings)):
            _panel(ax, "abcd"[index], title)
            _model_dots(ax, site, models, col, second=second, preset=preset)
            ax.set_xlabel(label)
            if col in ("recall", "false_positive_rate"):
                ax.set_xlim(0, 1)
                ax.set_xticks([0, .5, 1])
        if not risk:
            fig.text(.22, .105, "OOF risk is not available yet; only original point metrics are shown.",
                     color=ALERT, fontsize=12 if preset == "slides" else 8)
        paths = save_figure(fig, out, "v2_01_point_and_risk", preset)
    return paths, site.assign(budget_n=budget, figure_scope=scope)


def _selection_value(selection, names):
    for key in names:
        value = selection.get(key)
        if isinstance(value, str):
            return value
    return None


def _promotion(metrics, selection, config, out, preset, fold_id=None):
    del metrics  # Gate aggregation belongs exclusively to the selection stage.
    selected_model = _selection_value(selection, ["selected_model", "selected_model_id", "chosen_model", "model_id"])
    baseline = _selection_value(selection, ["baseline_model", "baseline_model_id", "reference_model", "baseline"])
    if not selected_model or not baseline:
        return [], None
    plot_cols = ["error_auc_gain", "delta_h_point_mae_m", "delta_risk_recall", "delta_risk_fpr"]
    saved_gate = selection.get("baseline_metrics", {})
    candidate_gates = selection.get("candidate_gates", [])
    use_saved = bool(candidate_gates) and all(k in saved_gate for k in ("aubc_log_mae", "aubc_h_point_mae", "risk_recall", "risk_fpr"))
    if not use_saved:
        return [], None
    if use_saved:
        values = []
        for candidate in candidate_gates:
            record = {**candidate}
            for field in ("aubc_log_mae", "aubc_h_point_mae", "risk_recall", "risk_fpr"):
                record[f"baseline_{field}"] = saved_gate[field]
            for source, destination, direction in [("aubc_log_mae", "error_auc_gain", -1),
                                                     ("aubc_h_point_mae", "delta_h_point_mae_m", 1),
                                                     ("risk_recall", "delta_risk_recall", 1),
                                                     ("risk_fpr", "delta_risk_fpr", 1)]:
                estimate, reference = candidate.get(source), saved_gate.get(source)
                record[destination] = direction * (float(estimate) - float(reference)) if estimate is not None and reference is not None else np.nan
            values.append(record)
        site = pd.DataFrame(values)
        models = [m for m in _models(site) if m != baseline]
    if not models:
        return [], None
    fallback = selected_model == baseline
    result = f"Fallback retained: {MODEL_NAMES.get(baseline, baseline)}" if fallback else f"Selected prediction model: {MODEL_NAMES.get(selected_model, selected_model)}"
    selected_metrics = saved_gate if fallback else next((g for g in candidate_gates if g["model_id"] == selected_model), {})
    base_fpr, selected_fpr = saved_gate.get("risk_fpr"), selected_metrics.get("risk_fpr")
    fpr_note = (f"Absolute risk FPR: baseline {base_fpr:.1%}; selected {selected_fpr:.1%} (budget-averaged)."
                if base_fpr is not None and selected_fpr is not None else "Absolute risk FPR: unavailable.")
    with figure_style(preset):
        fold_label = f"Fold {fold_id} | " if fold_id is not None else ""
        fig = _layout(preset, "Model promotion requires error gain and risk protection",
                      f"SOURCE VALIDATION · OOF risk  |  {fold_label}Budget-averaged metrics versus {MODEL_NAMES.get(baseline, baseline)}",
                      footer=f"{result}\n{fpr_note}\nSaved gate values; all four metrics use normalized budget AUC. Shading: acceptable condition.", height=135)
        axes = fig.subplots(1, 4, sharey=True)
        fig.subplots_adjust(left=.23, right=.98, top=.76, bottom=.24, wspace=.43)
        gate = config.get("v2", {}).get("gate", {})
        definitions = [(plot_cols[0], "Error gain", "log-MAE AUC gain", 0., True),
                       (plot_cols[1], "Thickness", "Delta MAE (m)", gate.get("h_tolerance_m", .05), False),
                       (plot_cols[2], "Risk recall", "Delta recall", -gate.get("recall_tolerance", .02), True),
                       (plot_cols[3], "Risk FPR", "Delta FPR", gate.get("fpr_tolerance", .02), False)]
        for i, (ax, (col, title, label, limit, greater)) in enumerate(zip(axes, definitions)):
            finite = site.loc[site.model_id.isin(models), col].dropna().to_numpy(float)
            low, high = min([limit, 0., *finite]), max([limit, 0., *finite])
            pad = max((high - low) * .18, .005)
            ax.set_xlim(low - pad, high + pad)
            ax.axvspan(limit if greater else low - pad, high + pad if greater else limit, color=TEAL, alpha=.07)
            ax.axvline(limit, color=TEAL, linestyle="--", linewidth=1)
            ax.axvline(0, color=GRID, linewidth=.7)
            _panel(ax, "abcd"[i], title)
            for j, model in enumerate(models):
                vals = site.loc[site.model_id.eq(model), col].dropna().to_numpy(float)
                if not len(vals):
                    continue
                ax.scatter([vals.mean()], [j], marker="D", s=80 if preset == "slides" else 32,
                           color=MODEL_COLORS.get(model, TEAL), edgecolor="white", linewidth=.5, zorder=5)
            ax.set_xlabel(label)
            ax.set_ylim(len(models) - .5, -.5)
            ax.set_yticks(range(len(models)), [MODEL_NAMES.get(m, m) for m in models])
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
        suffix = f"_fold_{fold_id}" if fold_id is not None else ""
        paths = save_figure(fig, out, f"v2_02_validation_gate{suffix}", preset)
    return paths, site


def _policy_curves(metrics, selection, out, preset, scope, fold_id=None):
    selected_model = _selection_value(selection, ["selected_model", "selected_model_id", "chosen_model", "model_id"])
    if not selected_model:
        count = metrics.groupby("model_id").policy_id.nunique()
        eligible = count[count > 1].index.tolist()
        if len(eligible) != 1:
            return [], None
        selected_model = eligible[0]
    rows = metrics.loc[metrics.model_id.eq(selected_model) & metrics.policy_id.isin(POLICY_ORDER)]
    if rows.empty or rows.policy_id.nunique() < 2:
        return [], None
    cols = ["log_mae", "h_point_mae_m", "risk_recall", "risk_fpr"]
    site = _site_means(rows, cols, ["policy_id", "budget_n"])
    risk_available = any(c in site and site[c].notna().any() for c in ("risk_recall", "risk_fpr"))
    variability = ("Lines: site macro means. Ribbons: +/- 1 SD across sites, not confidence intervals."
                   if rows.site_id.nunique() > 1 else "One named site: lines average available seeds; no across-site variability interval.")
    with figure_style(preset):
        fold_label = f"Fold {fold_id} | " if fold_id is not None else ""
        fig = _layout(preset, "Selection policies compared under one fixed predictor",
                      f"{_scope_label(scope, risk_available)}  |  {fold_label}{MODEL_NAMES.get(selected_model, selected_model)} prediction; GP acquisition",
                      footer=variability + "\n"
                             + ("GP selects locations for every non-geometric policy; prediction model stays fixed." if risk_available else
                                "OOF risk is not available yet; only original point metrics are shown."),
                      height=143 if risk_available else 112)
        axes = fig.subplots(2, 2) if risk_available else fig.subplots(1, 2)
        fig.subplots_adjust(left=.105, right=.97, top=.72 if risk_available else .68, bottom=.18 if risk_available else .26,
                            hspace=.58, wspace=.30)
        labels = [("log_mae", "Profile reconstruction", "log-MAE; lower is better"),
                  ("h_point_mae_m", "Point thickness error", "MAE (m); lower is better"),
                  ("risk_recall", "Calibrated risk recall", "Recall; higher is better"),
                  ("risk_fpr", "Calibrated false-positive rate", "FPR; lower is better")]
        if not risk_available:
            labels = labels[:2]
        for i, (ax, (col, title, ylabel)) in enumerate(zip(axes.flat, labels)):
            ax.set_title(f"{'abcd'[i]}  {title}", loc="left", weight="bold", pad=11)
            extents = []
            for policy in POLICY_ORDER:
                subset = site.loc[site.policy_id.eq(policy)]
                if subset.empty or col not in subset or not subset[col].notna().any():
                    continue
                stats = subset.groupby("budget_n")[col].agg(["mean", "std", "count"])
                x = stats.index.to_numpy(float)
                y, sd = stats["mean"].to_numpy(), stats["std"].to_numpy()
                color = POLICY_COLORS[policy]
                ax.plot(x, y, color=color, ls=POLICY_STYLES[policy], marker=POLICY_MARKERS[policy],
                        markersize=(7 if policy == "gp_alc" else 4) if preset == "slides" else (4.5 if policy == "gp_alc" else 2.7),
                        markerfacecolor="white" if policy == "gp_alc" else color,
                        linewidth=2.4 if policy == "gp_sur" and preset == "slides" else 1.4,
                        label=POLICY_NAMES[policy], zorder=4)
                lower, upper = np.maximum(0, y - sd), y + sd
                if col in ("risk_recall", "risk_fpr"):
                    upper = np.minimum(1, upper)
                ax.fill_between(x, lower, upper, where=np.isfinite(sd), color=color, alpha=.07, linewidth=0)
                extents.extend(y[np.isfinite(y)])
                extents.extend(lower[np.isfinite(lower)])
                extents.extend(upper[np.isfinite(upper)])
            ax.set_xticks(sorted(rows.budget_n.unique()))
            ax.set_xlabel("Measured CPT locations")
            ax.set_ylabel(ylabel)
            if col in ("risk_recall", "risk_fpr"):
                ax.set_ylim(0, 1)
                ax.set_yticks([0, .5, 1])
            elif extents:
                lo, hi = min(extents), max(extents)
                pad = max((hi - lo) * .2, hi * .03, .01)
                ax.set_ylim(max(0, lo - pad), hi + pad)
            finish_axis(ax)
        handles = [Line2D([0], [0], color=POLICY_COLORS[p], ls=POLICY_STYLES[p],
                          marker=POLICY_MARKERS[p], markerfacecolor="white" if p == "gp_alc" else POLICY_COLORS[p],
                          markersize=5, label=POLICY_NAMES[p])
                   for p in POLICY_ORDER if p in rows.policy_id.values]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .855), ncol=4,
                   handlelength=2.5, columnspacing=1.2)
        suffix = f"_fold_{fold_id}" if fold_id is not None else ""
        paths = save_figure(fig, out, f"v2_03_policy_budget_curves{suffix}", preset)
    return paths, site.assign(model_id=selected_model, figure_scope=scope)


def _fold_records(targets):
    folds = targets.get("folds", {})
    return folds.values() if isinstance(folds, dict) else folds


def _attainment_data(metrics, targets):
    """Prefer saved reference diagnostics; use target thresholds only if explicit."""
    output = []
    for spec in _fold_records(targets):
        fold = spec.get("fold_id")
        if fold is None:
            continue
        reference = spec.get("reference", targets.get("reference", {}))
        rows = metrics.loc[metrics.fold_id.eq(fold) & metrics.model_id.eq(reference.get("model_id", "gp"))
                           & metrics.policy_id.eq(reference.get("policy_id", "maximin"))
                           & metrics.budget_n.eq(reference.get("budget_n", 13))]
        if rows.empty:
            continue
        old_global = targets.get("v1_marginal_targets", {}).get("folds", {}).get(str(fold), {})
        old = spec.get("v1_marginal_targets", spec.get("legacy_targets", spec.get("old_targets", old_global)))
        for label, target, risk in [("Legacy marginal rule", old, False), ("V2 joint rule", spec, True)]:
            if not target.get("complete", False) or not all(target.get(k) is not None for k in ("E_star", "H_star")):
                continue
            reference_rows = rows.loc[~rows.site_id.astype(str).isin(target.get("excluded_sites", {}))].copy()
            if reference_rows.empty:
                continue
            recall = "risk_recall" if risk else "recall"
            if recall not in reference_rows:
                continue
            if risk:
                if "risk_fpr" not in reference_rows or not reference_rows.figure_risk_oof.all():
                    continue
                from .quality import _joint_hits
                passed, valid = _joint_hits(reference_rows, target)
            else:
                valid = reference_rows.log_mae.notna() & reference_rows.h_point_mae_m.notna()
                has_positive = reference_rows.n_true_positive.gt(0) if "n_true_positive" in reference_rows else reference_rows[recall].notna()
                valid &= ~has_positive | reference_rows[recall].notna()
                passed = reference_rows.log_mae.le(target["E_star"]) & reference_rows.h_point_mae_m.le(target["H_star"])
                passed &= ~has_positive | reference_rows[recall].ge(target["R_star"] if target.get("R_star") is not None else np.inf)
            for site, block in reference_rows.assign(_valid=valid, _pass=passed).groupby("site_id"):
                if not block._valid.all():
                    continue
                output.append({"fold_id": fold, "site_id": site, "target_version": label,
                               "pass_rate": float(block._pass.mean()), "n_instances": len(block),
                               "q": spec.get("q", spec.get("selected_q"))})
    return pd.DataFrame(output)


def _quality(metrics, targets, out, preset):
    site = _attainment_data(metrics, targets)
    if site.empty:
        return [], None
    folds = sorted(site.fold_id.unique())
    versions = [v for v in ["Legacy marginal rule", "V2 joint rule"] if v in site.target_version.values]
    with figure_style(preset):
        fig = _layout(preset, "Quality targets must be attainable by their reference",
                      "SOURCE VALIDATION · GP + Maximin at 13 CPTs  |  Target-definition comparison",
                      height=112, footer="Same current reference rows, changed quality definitions: this is not algorithm gain.\n"
                                         "Bars: site macro mean; dots: site rates. Legacy uses point recall; V2 uses OOF risk.")
        ax = fig.add_axes([.14, .25, .80, .51])
        for j, version in enumerate(versions):
            for i, fold in enumerate(folds):
                values = site.loc[site.fold_id.eq(fold) & site.target_version.eq(version), "pass_rate"].to_numpy()
                if not len(values):
                    continue
                x = i + (j - (len(versions) - 1) / 2) * .28
                color = MUTED if version.startswith("Legacy") else TEAL
                ax.bar(x, values.mean(), width=.23, color=color, alpha=.72,
                       label=version if i == 0 else None, zorder=2)
                ax.scatter(np.full(len(values), x) + np.linspace(-.045, .045, len(values)), values,
                           s=45 if preset == "slides" else 22, facecolor="white", edgecolor=color,
                           linewidth=1, zorder=4)
                ax.annotate(f"{values.mean():.0%}", (x, values.max()), xytext=(0, 7),
                            textcoords="offset points", ha="center", color=color,
                            fontsize=17 if preset == "slides" else 9)
        ax.axhline(.5, color=TEAL, linestyle="--", linewidth=1, alpha=.6)
        ax.set_xticks(range(len(folds)), [f"Fold {int(f)}" for f in folds])
        ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_ylim(0, 1.17)
        ax.set_ylabel("Joint reference pass rate")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.20), ncol=2)
        finish_axis(ax)
        paths = save_figure(fig, out, "v2_04_joint_target_attainability", preset)
    return paths, site


def export_v2(run_dir, dataset, config) -> list[Path]:
    """Render supported v0.2 evidence and source data without refitting anything.

    ``dataset`` is accepted for the common export API; numerical figures consume
    only saved metrics. Output is isolated under ``figures_v2/{report,slides}``.
    Validation and exploratory test receive separate subdirectories, and no
    in-sample calibrated validation metric enters a plot.
    """
    del dataset
    directory = Path(run_dir)
    metrics, sources = _load_metrics(directory)
    policies, policy_source = _load_policy_metrics(directory)
    if metrics.empty and policies.empty:
        return []
    if policy_source:
        sources.append(policy_source)
    if metrics.empty:
        metrics = policies.iloc[:0].copy()
    destination = directory / "figures_v2"
    destination.mkdir(parents=True, exist_ok=True)
    selection = _json(directory, ["model_selection.json", "selection.json", "validation_selection.json"])
    targets = _json(directory, ["quality_targets_v2.json", "joint_quality_targets.json", "quality_targets.json"])
    paths, source_paths = [], []
    available_scopes = set(metrics.figure_scope)
    if not policies.empty:
        available_scopes.update(policies.figure_scope)
    scopes = [s for s in ("validation", "pilot", "test") if s in available_scopes]
    for scope in scopes:
        frame = metrics.loc[metrics.figure_scope.eq(scope)].copy()
        policy_frame = policies.loc[policies.figure_scope.eq(scope)].copy() if not policies.empty else frame
        for preset in ("report", "slides"):
            out = destination / scope / preset
            jobs = [lambda: _point_risk(frame, out, preset, scope)] if not frame.empty else []
            fold_selection = selection.get("folds", {})
            if fold_selection:
                selected_values = {_selection_value(chosen, ["selected_model", "selected_model_id", "chosen_model", "model_id"])
                                   for chosen in fold_selection.values()}
                common_predictor = len(selected_values) == 1 and None not in selected_values
                if common_predictor and not policy_frame.empty:
                    chosen = next(iter(fold_selection.values()))
                    jobs.append(lambda chosen=chosen: _policy_curves(policy_frame, chosen, out, preset, scope))
                for fid, chosen in fold_selection.items():
                    fid = int(fid)
                    # Bind defaults now: these jobs execute after this loop.
                    pf = policy_frame.loc[policy_frame.fold_id.eq(fid)]
                    if not common_predictor and not pf.empty:
                        jobs.append(lambda pf=pf, chosen=chosen, fid=fid: _policy_curves(pf, chosen, out, preset, scope, fid))
                    if scope == "validation":
                        mf = frame.loc[frame.fold_id.eq(fid)]
                        if not mf.empty:
                            jobs.append(lambda mf=mf, chosen=chosen, fid=fid: _promotion(mf, chosen, config, out, preset, fid))
            else:
                if not policy_frame.empty:
                    jobs.append(lambda: _policy_curves(policy_frame, selection, out, preset, scope))
                if scope == "validation" and not frame.empty:
                    jobs.append(lambda: _promotion(frame, selection, config, out, preset))
            if scope == "validation" and not frame.empty:
                jobs.append(lambda: _quality(frame, targets, out, preset))
            for job in jobs:
                produced, data = job()
                paths.extend(produced)
                if produced and data is not None and preset == "report":
                    source = destination / scope / "source_data" / f"{produced[0].stem}.csv"
                    source.parent.mkdir(parents=True, exist_ok=True)
                    data.to_csv(source, index=False, encoding="utf-8-sig")
                    source_paths.append(source)
    notes = {
        "study": "exploratory_v2", "source_tables": sources,
        "files": [str(p.relative_to(directory)) for p in paths],
        "source_data": [str(p.relative_to(directory)) for p in source_paths],
        "scope_counts": metrics.groupby("figure_scope").size().to_dict(),
        "risk_rows_masked": int((metrics.figure_scope.isin(["validation", "pilot"]) & ~metrics.figure_risk_oof).sum()),
        "aggregation": "Mean within site/fold over seeds, then mean over folds for each distinct site; site macro mean.",
        "spread": "Policy ribbons are +/- one across-site SD, not a confidence interval; other panels show all site means.",
        "inference": "No hypothesis tests or new independent replication claims; repeated outer sites remain exploratory.",
        "filters": "Point/risk panel: maximin at largest observed budget; promotion: matched complete maximin curves; policy panel: fixed selected predictor; targets: validation GP+maximin reference.",
        "risk_boundary": "Validation risk requires explicit OOF provenance; unsafe risk cells are masked. Missing recall/FPR is not zero.",
        "target_boundary": "V1 and V2 targets use different quality definitions; pass-rate differences are not algorithm gains.",
    }
    manifest = destination / "figure_manifest.json"
    manifest.write_text(json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8")
    return [*paths, *source_paths, manifest]
