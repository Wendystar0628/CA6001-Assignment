"""Experiment notebook and local adaptive-investigation display.

Figures come from :mod:`cpt_budget.report_figures`, the single source of the
report charts; this module embeds them in the offline HTML page and notebook.
Site is the replicate unit; seeds are averaged within site.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import pandas as pd

from .report_figures import CAPTIONS, FIGURES, build_figures

MODEL_NAMES = {"gp": "GP", "xgb": "XGBoost residual", "cnp": "CNP",
               "tabpfn_joint": "TabPFN joint", "tabpfn_graph": "TabPFN + graph"}
SITE_NAMES = {"christchurch_metro_sports_facility": "Christchurch Metro Sports",
              "lower_hutt_town_hall": "Lower Hutt Town Hall", "marshland": "Marshland",
              "tauranga_farmers_building": "Tauranga Farmers", "newlock": "New Lock Terneuzen",
              "hollywood": "Hollywood",
              "wellington_airport": "Wellington Airport", "wellington_college": "Wellington College",
              "port_nelson": "Port Nelson"}


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
    ['site_id', 'n_profiles', 'grid_025_mae_m', 'grid_025_p90_abs_error_m', 'grid_010_diagnostic_mae_m']])
display(Image(filename=str(FIG / (FIGURES[1]+'.png'))))
display(Markdown('**Figure 2.** '+CAPTIONS[1]))"""),
        md("""## AI methods and common-sequence comparison

**GP** models spatial correlations and provides a probabilistic uncertainty signal. **XGBoost residual** learns nonlinear corrections to the spatial base from observed-neighbour features. **CNP** aggregates the revealed support set into a conditional neural representation. **TabPFN joint** combines a trained residual component with a spatial base and profile-level decisions. **TabPFN + graph** combines residual prediction with robust graph smoothing. These hybrid methods retain their spatial components: the whole improvement cannot be attributed to the Transformer alone.

For the five-method comparison, all methods receive the same CPT sequence. Site means include the common budgets and seeds; the overall comparison gives each site equal weight."""),
        code("site_models, model_means = model_summary(tables['model_comparison'])\ndisplay(model_means.round(4))\nprimary = tables['joined'].model_id.iloc[0]\nbest_profile = model_means.log_mae.idxmin()\nbest_thickness = model_means.thickness_mae_m.idxmin()\ndisplay(Markdown(f'The development-selected primary method was **{MODEL_NAMES.get(primary, primary)}**. In the evaluation comparison, the lowest profile error was obtained by **{MODEL_NAMES.get(best_profile, best_profile)}**, whereas the lowest thickness error was obtained by **{MODEL_NAMES.get(best_thickness, best_thickness)}**. The primary method is retained after evaluation; it is not re-selected from these outcomes.'))\ndisplay(Image(filename=str(FIG / (FIGURES[3]+'.png'))))\ndisplay(Markdown('**Figure 4.** '+CAPTIONS[3]))"),
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
        md("## Selection and stopping\n\nThe maps retain the distinction between initial, selectable and hidden audit locations. Wellington Airport shows an actual online stop; New Lock is the pre-specified engineering case, irrespective of its result. The accompanying quality curves use the same audit CPTs at every round."),
        code("display(Image(filename=str(FIG / (FIGURES[2]+'.png'))))\ndisplay(Markdown('**Figure 3.** '+CAPTIONS[2]))\ndisplay(Image(filename=str(FIG / (FIGURES[4]+'.png'))))\ndisplay(Markdown('**Figure 5.** '+CAPTIONS[4]))\nfull_case = evaluation(tables['full_plan_model_sensitivity']).query(\"site_id == 'newlock'\")\nif not full_case.empty:\n    display(full_case.groupby('model_id')[['log_mae', 'thickness_mae_m', 'recall', 'fpr']].mean().round(4))\n    display(Markdown('The full-plan New Lock comparison retains all five methods. Complete candidate-plan observations do not automatically guarantee accurate prediction at the separate hidden audit CPTs.'))"),
        md("## Site outcomes\n\nEarly stopping is counted as successful only when the hidden audit passes. Failure cases and complete-plan outcomes remain visible. An advantage over the full plan may arise from early stopping itself; the matched sequential policy identifies any additional contribution from adaptive location choice."),
        code("display(Image(filename=str(FIG / (FIGURES[5]+'.png'))))\ndisplay(Markdown('**Figure 6.** '+CAPTIONS[5]))\ndisplay(decisions[[c for c in ['site_id', 'seed', 'policy_id', 'n_full', 'n_stop', 'audit_pass', 'low_qc_audit_pass', 'false_early_stop'] if c in decisions]])\npaired = decisions.pivot(index=['site_id', 'seed'], columns='policy_id', values='n_stop')\npaired['Additional_adaptive_saving'] = paired.B_seq - paired.P_adapt\ndisplay(paired.groupby('site_id').mean().round(3))") ,
        md("""## Economic consequences and sensitivity

The economic calculation starts from experimentally observed CPT counts. It separates direct investigation savings, project-time costs, earlier-use benefits and AI/verification costs. The project-time transfer coefficient α connects investigation time saved to project time saved. These values are scenario assessments with explicit cost assumptions; an incorrect early stop is retained and can incur verification or return-visit costs."""),
        code("""cols = [c for c in ['site_id', 'policy_id', 'n_full', 'n_stop', 'saved_cpt', 'investigation_days_saved', 'project_days_saved', 'direct_saving', 'project_time_saving', 'early_opening_benefit', 'ai_incremental_cost', 'net_benefit'] if c in tables['economic_results']]
display(evaluation(tables['economic_results'])[cols])
if not tables['economic_inputs'].empty:
    display(tables['economic_inputs'][[c for c in ['parameter', 'value', 'unit', 'source_type', 'source_url'] if c in tables['economic_inputs']]])
if (FIG / (FIGURES[6]+'.png')).exists():
    display(Image(filename=str(FIG / (FIGURES[6]+'.png'))))
    display(Markdown('**Figure 7.** '+CAPTIONS[6]))
display(tables['sensitivity'].groupby(['sensitivity_kind', 'alpha']).net_benefit.agg(['count', 'mean', 'min', 'max']).round(2))"""),
        md("""## Interpretation

The engineering result has three parts: prediction quality, the completed planned-CPT count, and the hidden-audit verdict. An investigation that stops early but fails audit does not establish a deployable saving. Where sequential and adaptive outcomes are similar, the gain is mainly due to staged investigation and stopping. Where adaptive sampling passes audit with fewer CPTs, it supports an additional benefit from selecting informative locations.

Low-qc detection is difficult when thin or laterally variable features are sparsely observed. The reported recall, false-positive rate and thickness error therefore accompany profile error throughout the comparison. The evidence applies to the measured 2–8 m reconstruction task and the defined economic scenarios.

## Reproducibility and sources

The figures are generated by `cpt_budget.report_figures` from `results/adaptive_v1/*.csv`. SVG and PDF files retain editable text; PNG previews are saved for this notebook. Original measurements: ISSMGE TC304 CPT database. Model references and the project background and economic parameter sources are recorded in the execution plan and economic parameter document. The previous investigation's notebook, HTML and result files are retained as historical evidence."""),
    ]
    path = root / "notebooks" / "CA6001_report_figures.ipynb"
    nbf.write(nb, path)


def build(root: Path, notebook: bool = False) -> dict:
    tables = load_results(root)
    directory = root / "results" / "adaptive_v1" / "figures"
    build_figures(root)
    build_html(root, tables)
    notes = "# Figure evidence and export notes\n\n"
    for name, caption in zip(FIGURES, CAPTIONS):
        notes += f"- **{name}:** {caption}\n"
    notes += ("\nAll figures use saved CSV observations and are generated by `cpt_budget.report_figures`. "
              "Site is the replicate unit: metrics are averaged within seed, then site, then equally across "
              "sites. Seed spreads are descriptive, not confidence intervals; no significance tests are used. "
              "Exports are 6.3 in wide (A4 text width): 300 dpi PNG plus editable PDF and SVG.\n")
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
