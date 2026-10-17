"""Build the CA6001 technical report (Word) on the course template.

Tables are computed from ``results/``; figures are the files written by
``python -m cpt_budget.report_figures``. Run from the project root:

    .venv/Scripts/python.exe report/build_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import docx
import pandas as pd
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "adaptive_v1"
FIG = RES / "figures"
OUT = ROOT / "report" / "CA6001_Adaptive_CPT_Report.docx"

INK = RGBColor(0x0B, 0x0B, 0x0B)
MUTED = RGBColor(0x52, 0x51, 0x4E)
HEAD = RGBColor(0x1C, 0x3D, 0x5A)

# ------------------------------------------------------------------ data
dec = pd.read_csv(RES / "decisions.csv")
mod = pd.read_csv(RES / "model_comparison.csv")
abl = pd.read_csv(RES / "ablation.csv")
eco = pd.read_csv(RES / "economic_results.csv")
dev_rank = pd.read_csv(RES / "development_model_ranking.csv").set_index("model_id")
orc = pd.read_csv(ROOT / "results" / "adaptive_v2_diagnostics" / "stopping_oracle_diagnostic.csv")
frozen = json.loads((RES / "frozen_config.json").read_text(encoding="utf-8"))
MODELS = ["gp", "xgb", "cnp", "tabpfn_joint", "tabpfn_graph"]
NAMES = {"gp": "GP", "xgb": "XGBoost residual", "cnp": "CNP", "tabpfn_joint": "TabPFN joint",
         "tabpfn_graph": "TabPFN + graph (primary)"}
EVAL = ["wellington_airport", "christchurch_stadium", "hollywood", "newlock"]
SITE = {"wellington_airport": "Wellington Airport", "christchurch_stadium": "Christchurch Stadium",
        "hollywood": "Hollywood", "newlock": "New Lock Terneuzen"}


def site_equal(df, cols):
    a = df.groupby(["site_id", "seed", "model_id"])[cols].mean().groupby(["site_id", "model_id"]).mean()
    return a.groupby("model_id").mean()


# ------------------------------------------------------------------ document helpers
doc = docx.Document(str(ROOT / "CA6001 Report Template.docx"))
body = doc.element.body
for child in list(body):
    if child.tag != qn("w:sectPr"):
        body.remove(child)
sec = doc.sections[0]
for side in ("left_margin", "right_margin"):
    setattr(sec, side, Cm(2.3))
sec.top_margin = sec.bottom_margin = Cm(2.0)

normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
normal.font.size = Pt(10.5)
normal.font.color.rgb = INK
normal.paragraph_format.space_after = Pt(4)
normal.paragraph_format.line_spacing = 1.08
for name, size, before in [("Heading 1", 13, 10), ("Heading 2", 11, 6)]:
    st = doc.styles[name]
    st.font.name = "Calibri"
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = HEAD
    st.paragraph_format.space_before = Pt(before)
    st.paragraph_format.space_after = Pt(3)
    st.paragraph_format.keep_with_next = True

fig_no = [0]
tab_no = [0]


def para(text="", bold_lead=None, italic=False, align=None, size=None, color=None, after=None):
    p = doc.add_paragraph()
    if bold_lead:
        r = p.add_run(bold_lead)
        r.bold = True
    _runs(p, text, italic=italic, size=size, color=color)
    if align:
        p.alignment = align
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def _runs(p, text, italic=False, size=None, color=None):
    """Very small markup: *italic* segments and _{sub} / ^{sup} are supported."""
    import re
    text = text.replace("`", "")
    for token in re.split(r"(\*[^*]+\*|_\{[^}]+\}|\^\{[^}]+\})", text):
        if not token:
            continue
        if token.startswith("*") and token.endswith("*"):
            r = p.add_run(token[1:-1])
            r.italic = True
        elif token.startswith("_{"):
            r = p.add_run(token[2:-1])
            r.font.subscript = True
        elif token.startswith("^{"):
            r = p.add_run(token[2:-1])
            r.font.superscript = True
        else:
            r = p.add_run(token)
            r.italic = italic
        if size:
            r.font.size = Pt(size)
        if color is not None:
            r.font.color.rgb = color


def bullet(text, lead=None):
    p = doc.add_paragraph(style="List Paragraph")
    p.paragraph_format.left_indent = Cm(0.5)
    p.paragraph_format.first_line_indent = Cm(-0.35)
    p.paragraph_format.space_after = Pt(2)
    p.add_run("•  ")
    if lead:
        p.add_run(lead).bold = True
    _runs(p, text)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def h1(text):
    doc.add_heading(text, level=1)


def h2(text):
    doc.add_heading(text, level=2)


def equation(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    _runs(p, text, italic=False)
    for r in p.runs:
        r.font.name = "Cambria Math"
    return p


def figure(name, caption, width_cm=16.4):
    fig_no[0] += 1
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(FIG / f"{name}.png"), width=Cm(width_cm))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    c.paragraph_format.space_after = Pt(6)
    r = c.add_run(f"Figure {fig_no[0]}. ")
    r.bold = True
    r.font.size = Pt(9)
    _runs(c, caption, size=9, color=MUTED)
    return fig_no[0]


def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def table(caption, header, rows, widths_cm, align_right_from=1, bold_rows=(), left_cols=()):
    tab_no[0] += 1
    c = doc.add_paragraph()
    c.paragraph_format.space_after = Pt(2)
    c.paragraph_format.keep_with_next = True
    r = c.add_run(f"Table {tab_no[0]}. ")
    r.bold = True
    r.font.size = Pt(9)
    _runs(c, caption, size=9, color=MUTED)
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = doc.styles["Table Grid"] if "Table Grid" in [s.name for s in doc.styles] else None
    for i, row in enumerate([header] + rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.width = Cm(widths_cm[j])
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            p.alignment = (WD_ALIGN_PARAGRAPH.RIGHT if (j >= align_right_from and j not in left_cols and i > 0)
                           else WD_ALIGN_PARAGRAPH.LEFT)
            run = p.add_run(str(val))
            run.font.size = Pt(8.5)
            if i == 0:
                run.bold = True
                shade(cell, "E8EEF4")
            if i - 1 in bold_rows:
                run.bold = True
    t.autofit = False
    for j, w in enumerate(widths_cm):
        t.columns[j].width = Cm(w)
    _borders(t)
    gap = doc.add_paragraph()
    gap.paragraph_format.space_after = Pt(0)
    for r in gap.runs:
        r.font.size = Pt(4)
    return tab_no[0]


def _borders(t):
    tbl = t._tbl
    tblPr = tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom", "insideH"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "4")
        e.set(qn("w:color"), "B8B6AE")
        b.append(e)
    for edge in ("left", "right", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "nil")
        b.append(e)
    tblPr.append(b)


def code_block(lines):
    for i, line in enumerate(lines):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.left_indent = Cm(0.4)
        r = p.add_run(line if line else " ")
        r.font.name = "Consolas"
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor(0x1F, 0x2D, 0x3A)
        pPr = p._p.get_or_add_pPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:fill"), "F3F2EE")
        pPr.append(shd)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


# ================================================================== cover page
for _ in range(5):
    doc.add_paragraph()
para("CA6001 – Applied AI Algorithms Project", align=WD_ALIGN_PARAGRAPH.CENTER, size=12, color=MUTED)
para("Technical Report", align=WD_ALIGN_PARAGRAPH.CENTER, size=12, color=MUTED, after=30)
t = para("Machine Learning–Guided Adaptive CPT Site Investigation and Economic Benefit Assessment",
         align=WD_ALIGN_PARAGRAPH.CENTER, size=20, color=HEAD)
for r in t.runs:
    r.bold = True
para("Predicting shallow cone-resistance profiles, choosing the next cone penetration tests, deciding when to "
     "stop, and pricing the result", align=WD_ALIGN_PARAGRAPH.CENTER, size=11, color=MUTED, after=60)
para("Project Title: Machine Learning–Guided Adaptive CPT Site Investigation and Economic Benefit Assessment",
     align=WD_ALIGN_PARAGRAPH.LEFT, size=11)
para("Student Names: ______________________________________________", align=WD_ALIGN_PARAGRAPH.LEFT, size=11)
para("Course: CA6001 AI Algorithms Fundamentals & Application", align=WD_ALIGN_PARAGRAPH.LEFT, size=11)
para("Date: October 2026", align=WD_ALIGN_PARAGRAPH.LEFT, size=11)
doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

# ================================================================== 1 Introduction
h1("1. Introduction")
para("Cone penetration tests (CPTs) push an instrumented cone into the ground and record cone resistance "
     "*q*_{c} with depth. They are the main source of shallow ground information on most civil projects, but "
     "each test measures only one vertical line; between tests the ground is inferred. A conventional "
     "investigation fixes the number and positions of CPTs in advance and completes them all. Some of those "
     "tests add little new information, while the decision to add more is often made by judgement.")
para("This project asks a practical question: *with only a few CPTs measured, can an AI model predict the "
     "unmeasured profiles well enough to choose the next most useful CPTs and to stop once quality is "
     "adequate – and what is that worth?* The motivation is twofold. Fewer CPTs reduce direct cost; finishing "
     "the investigation earlier can bring the whole programme forward, which on a large asset such as a "
     "navigation lock has a value far exceeding the cost of the tests themselves.")
para("The objectives are to (i) compare five AI predictors under identical information, (ii) build an "
     "adaptive selection and online stopping loop around the selected predictor, (iii) audit every stop "
     "against hidden CPTs that the loop never sees, and (iv) translate the actual CPT counts into an "
     "owner-financial economic scenario. The scope is the 2–8 m depth window of public CPT data. The "
     "threshold *q*_{c} < 1 MPa is a research indicator of low-resistance material, not a soil "
     "classification, bearing-capacity or liquefaction criterion, and the economics are counterfactual "
     "scenarios rather than observed contract savings. Figure 1 summarises the complete loop.")

figure("fig01_workflow", "Investigation loop. Revealed CPTs feed prediction, batch selection and an online "
       "stop rule; the hidden audit set *A* is read only after the stop. All policies share *S*_{0}, *C* and *A*.")
# ================================================================== 2 Problem definition
h1("2. Problem Definition")
para("At a site with planned positions *x* and the 24 depth cells *z*_{k} = 2.125, 2.375, …, 7.875 m "
     "(0.25 m wide), the regression target is *y*(*x*, *z*_{k}) = ln(*q*_{c}/1 MPa). The cumulative "
     "low-resistance thickness at a position is")
equation("H(x) = 0.25 · Σ_{k} 1[ y(x, z_{k}) < 0 ]   (metres)")
para("Given a revealed set *S* of CPTs, a model *f* predicts *ŷ*(*x*, *z*) and *Ĥ*(*x*) for every unrevealed "
     "position. The investigation is a sequential decision problem. Each site is split into an initial set "
     "*S*_{0}, a candidate pool *C* and a hidden audit set *A*. Starting from *S* = *S*_{0}, a policy chooses "
     "a batch *B* ⊂ *C* \\ *S*, reveals it, and decides whether to stop. The aim is the smallest stopping "
     "count *N*_{stop} for which the hidden-audit quality still meets frozen targets, compared with the full "
     "plan *N*_{full} = |*S*_{0} ∪ *C*|:")
equation("minimise  N_{stop}   subject to   log-MAE_{A} ≤ τ_{log},   thickness-MAE_{A} ≤ τ_{H},   loss vs B_{full} ≤ tolerance")
para("The task therefore combines *spatial regression* (profile prediction), *binary event detection* "
     "(cells below 1 MPa, evaluated with precision, recall, F1 and false-positive rate), *active learning* "
     "(choosing informative locations) and *optimal stopping*. Key constraints are:")
bullet("only coordinates, planned depths and already-revealed curves may enter prediction, selection or "
       "stopping; hidden labels are read by a separate evaluator after the stop;", "Information: ")
bullet("near-duplicate or repeated soundings form indivisible groups, so a nominal batch of two can be "
       "larger and counts are always actual positions;", "Grouping: ")
bullet("sites (not depth cells or random seeds) are the replicate unit; with four evaluation sites the "
       "results are descriptive and no significance tests are claimed;", "Small data: ")
bullet("all thresholds are selected on development sites and frozen before evaluation; an online stop that "
       "fails audit stays a failure.", "No hindsight: ")

# ================================================================== 3 Data
h1("3. Dataset Description")
para("The data come from the ISSMGE TC304 public CPT databases (ISSMGE TC304, n.d.). The Global-CPT "
     "collection holds 1,196 soundings from 59 sites or regions (Ching et al., 2023), and the "
     "NL-CPTU/98/2km² set contains the 98 soundings of the New Lock Terneuzen project in the Netherlands, "
     "fully operational since 1 August 2025 (Rijkswaterstaat, 2025). Only the subset that supports spatial "
     "prediction is used (Table 1).")
table("Data hierarchy after screening. Depth cells are repeated measurements within a CPT, not independent samples.",
      ["Unit", "Count", "Meaning"],
      [["Source profiles", "503", "CPT curves with coordinates covering 2–8 m"],
       ["Distinct planned positions", "495", "Repeated soundings at one coordinate merged"],
       ["Usable representative positions", "492", "Complete 24-cell window (8 duplicates, 3 gaps removed)"],
       ["Near-duplicate groups", "456", "435 singletons; groups of 2–5 kept together"],
       ["Sites", "31", "8 replay sites (≥ 20 positions) + 23 source-only sites"],
       ["Depth-cell records", "11,808", "492 × 24 cells of 0.25 m"]],
      [4.6, 1.6, 9.0], align_right_from=1, left_cols=(2,))
para("Each record carries the site, position (local *x*, *y* in metres), near-duplicate group, depth and "
     "*q*_{c}. Preprocessing interpolates each raw curve to the 24 cell mid-points (gaps > 0.5 m rejected), "
     "takes ln *q*_{c}, removes non-representative repeats and assigns groups. For New Lock, 80 of the 98 "
     "original files are retained: 15 start below 2 m and 3 end above 8 m. Neighbour features "
     "(distances, local medians, low quantiles, low-cell fractions, vertical gradients) are rebuilt from "
     "revealed CPTs only at every round.")
para("The 0.25 m cells lose some thin layers. Against exact threshold crossings of the raw curves, the "
     "cell representation has a thickness MAE of 0.092 m (P90 0.256 m); 498 of 805 raw low intervals "
     "thinner than 0.25 m contain no cell mid-point. A 0.1 m grid would cut this to 0.033 m, but all "
     "models keep the same 0.25 m grid so that comparisons remain fair. Figure 2 shows the contrast that "
     "drives the results: Wellington Airport is uniformly stiff with no low cells, whereas New Lock has "
     "24% low cells and strong lateral variability.")
figure("fig02_dataset", "Replay sites and profiles. (a) Usable CPT positions at the eight replay sites and the "
       "share of cells with *q*_{c} < 1 MPa. (b, c) All 2–8 m profiles at the most homogeneous (Wellington "
       "Airport) and most variable (New Lock) evaluation sites; the black line is the site median.")

# ================================================================== 4 Algorithms
h1("4. Algorithm Selection and Rationale")
para("Five predictors span the main families for small-data spatial prediction: a probabilistic kernel "
     "method, gradient-boosted trees, a neural set model and a tabular foundation model in two hybrid forms "
     "(Table 2). All receive identical revealed CPTs at every state, and all thresholds for thickness and "
     "low-cell labels are applied to the one predicted curve of each method.")
table("The five AI methods: principle, strengths and limitations.",
      ["Method", "Principle and configuration", "Strength", "Limitation"],
      [["GP", "Matérn-3/2 anisotropic kernel on (x, y, z) plus white noise, fitted each round to revealed "
              "residuals from a source-site depth trend (Rasmussen & Williams, 2006)",
        "Native predictive std. used for acquisition", "Stationary; smooths thin low layers"],
       ["XGBoost residual", "3-NN inverse-distance base + α × boosted-tree residual on 24 neighbour "
                            "features; α ∈ {0, 0.25, 0.5, 1} chosen per fold (Chen & Guestrin, 2016)",
        "Non-linear local corrections", "Trees fixed after source training"],
       ["CNP", "Encodes revealed profiles with relative coordinates, mean/max pooling, decodes a full "
               "profile (26,817 parameters; Garnelo et al., 2018)", "Learns site-to-site transfer",
        "L1 loss favours the median; never predicts < 1 MPa"],
       ["TabPFN joint", "Transformer pre-trained on synthetic tables (Hollmann et al., 2025), fine-tuned "
                        "500–600 steps on source sites (CRPS + MSE); 0.5 × triangulation + 0.5 × "
                        "sign-guarded residual; whole-profile low-cell decision",
        "In-context learning from few CPTs", "Hybrid: TabPFN supplies half the base"],
       ["TabPFN + graph", "Residual branch averaged with a k-NN graph pseudo-Huber interpolation; a cell is "
                          "low if either branch says so", "Highest low-cell recall", "Union raises false positives"]],
      [2.4, 7.4, 3.2, 3.2], align_right_from=9)
para("*Why these fit.* The data are small (tens of CPTs per site), spatially correlated and heterogeneous "
     "across sites. GP is the classical geostatistical choice and the only method with a calibrated-style "
     "variance; XGBoost is the strongest conventional learner on tabular neighbour features; CNP and TabPFN "
     "test whether meta-learning across sites helps when a new site has only five CPTs. TabPFN performs "
     "inference by conditioning on the current site context, so the loop needs no retraining per round. "
     "*Alternatives considered:* ordinary kriging and inverse-distance weighting are retained only as "
     "internal bases; entropy-based Bayesian compressive sampling (Guan et al., 2022) targets liquefaction "
     "rather than *q*_{c} profiles; reinforcement learning or full Bayesian optimisation was rejected because "
     "four evaluation sites cannot support a learned policy without over-fitting.")
para("The primary model was chosen by a pre-registered rule – the lowest development-site-equal log-MAE "
     "over registered observation states – which selected TabPFN + graph (0.578) ahead of XGBoost (0.597) "
     "and TabPFN joint (0.598). It is used for prediction and stopping; GP supplies the uncertainty signal, "
     "and XGBoost the disagreement partner, in a division of labour rather than a stacked ensemble.")

# ================================================================== 5 Methodology
h1("5. Methodology")
h2("5.1 Partitions, policies and evaluation design")
para("The loop in Figure 1 is replayed on historical data. Eight sites with at least 20 usable positions are replayed: four development sites (Lower Hutt Town "
     "Hall, Tauranga Farmers, Christchurch Metro Sports, Marshland) and four evaluation sites (Wellington "
     "Airport, Christchurch Stadium, Hollywood, New Lock). For each site and seed (17, 29, 43) whole groups "
     "are assigned to *A* (≈20%), a spatially dispersed *S*_{0} (≥ 5 positions) and *C*. Every evaluation "
     "site is predicted with the source-fold weights that excluded it from both training and checkpoint "
     "selection. Three policies share the same partition:")
bullet("complete every position in *S*_{0} ∪ *C* – the engineering reference;", "B_full: ")
bullet("add the two candidates furthest from revealed CPTs (maximin) and apply the stop rule;", "B_seq: ")
bullet("add the two candidates with the highest acquisition score and apply the same stop rule.", "P_adapt: ")
para("B_full versus P_adapt measures the total benefit; B_seq versus P_adapt isolates the contribution of "
     "intelligent location choice from that of simply stopping early.")
h2("5.2 Acquisition score")
para("For each candidate group, *U* is the mean GP predictive standard deviation of ln *q*_{c}, *D* the "
     "fraction of cells where the primary model and XGBoost disagree about *q*_{c} < 1 MPa, and *S* the "
     "distance to the nearest revealed CPT. Each is converted to a 0–1 rank within the pool:")
equation("a(x) = 0.5 · rank(U) + 0.3 · rank(D) + 0.2 · rank(S)")
para("Within a batch only *S* is updated after each pick, so the second choice is pushed away from the "
     "first without peeking at its curve. The only alternative tried was *U* + *S* (0.7, 0, 0.3).")
h2("5.3 Online stopping and independent audit")
para("On a fixed grid of all planned positions × 24 depths, the controller computes four signals each "
     "round: the 90th percentile of the change in predicted ln *q*_{c} (Δ*Q*_{90}) and in thickness "
     "(Δ*H*_{90}), a leave-group-out log-MAE on up to five initial groups, and the 90th percentile of GP "
     "standard deviation. It stops when, after at least nine CPTs with every populated quadrant covered, all "
     "four pass for two consecutive rounds. Thresholds were calibrated as quantiles (0.50/0.75/0.90) of the "
     "signals at acceptable development states; the selection score (verified saving minus false-stop rate) "
     "chose the 0.50 quantile with the full score – Δ*Q*_{90} ≤ 0.082, Δ*H*_{90} ≤ 0 m, support log-MAE ≤ "
     "0.584, GP std P90 ≤ 0.461. The 0.75 and 0.90 quantiles produced false-stop rates of 17% and 58% on "
     "development sites. The audit requires log-MAE ≤ 0.626 and thickness MAE ≤ 0.881 m on *A* (development "
     "full-plan 75th percentiles plus margins) and no more than 0.05 / 0.25 m worse than B_full. Low-cell "
     "recall ≥ 0.5 with FPR ≤ 0.1 is audited separately as a diagnostic.")
h2("5.4 Economic model")
para("The owner-financial benefit of a policy relative to the full plan is")
equation("B_{net} = (C_{full} − C_{policy}) + α · ΔT_{inv} · (c_{day} + b_{day}) − C_{AI,inc}")
para("where *C* is investigation cost (€750 per CPT, all-in, scenario value), Δ*T*_{inv} the saving in "
     "calendar days (5 CPTs per rig-day, one rig, 7/5 calendar conversion, plus 0.25 day decision latency per "
     "batch transition), *c*_{day} = €5,000 avoidable project cost per day, *b*_{day} = €3,000 earlier-opening "
     "benefit per day, and *C*_{AI,inc} = €600 for deployment and review. The main scenario uses α = 1 "
     "(investigation days transfer fully to completion); α = 0.5 and 0.75 and one-at-a-time parameter ranges "
     "are sensitivities. A failed early stop would be repaired to the full plan with remobilisation (€560) and "
     "two days' delay. Parameters without a recovered contract are labelled as assumptions; the only quoted "
     "anchors are a 2025 Dutch CPT contractor quotation for remobilisation and review rates (MOS "
     "Grondmechanica, 2025), following cost-estimating practice (U.S. GAO, 2015, 2020).")
h2("5.5 Tools")
para("Python 3.11, NumPy 2.4, pandas 3.0, SciPy 1.17, scikit-learn 1.9 (GP, Pedregosa et al., 2011), "
     "XGBoost 3.2, PyTorch 2.11 (CNP) and TabPFN 9.2.0 with v2 weights (Prior Labs, n.d.), run on an "
     "NVIDIA RTX 4080 (16 GB); matplotlib for figures.")

# ================================================================== 6 Implementation
h1("6. Implementation")
para("The code is a Python package (`cpt_budget`) with separate modules whose boundaries mirror the "
     "information boundary: `adaptive_data` (partitions and manifests), `adaptive_models` (one interface "
     "`condition(revealed)` / `predict(query)` for all five models; queries are stripped of every measured "
     "column), `adaptive_policy` (acquisition and stop rule), `adaptive_replay` (the loop), an independent "
     "audit evaluator, and `adaptive_economics`. A single command runs each stage (prepare, calibrate, freeze, "
     "evaluate, publish, economics), and the frozen configuration stores SHA-256 fingerprints of data, "
     "weights and settings. The core of batch selection is shown below.")
code_block([
    "group['rank_U'] = rank01(group.uncertainty)          # GP std, fixed within batch",
    "group['rank_D'] = rank01(group.disagreement)         # low-cell disagreement",
    "while remaining and chosen_count < batch_size:",
    "    dist = [min_distance(g, observed_xy) for g in remaining]",
    "    score = 0.5*rank_U + 0.3*rank_D + 0.2*rank01(dist)",
    "    gid = argmax(score, ties=(group_id, location_id))",
    "    chosen += members(gid)                           # whole near-duplicate group",
    "    observed_xy = vstack([observed_xy, xy(gid)])     # coordinates only, no curves",
])
para("Key implementation decisions were: (i) TabPFN weights are frozen during replay and only the site "
     "context changes, so each round costs seconds rather than a retraining run; (ii) grouping is enforced in "
     "selection, counting and auditing; (iii) the CNP and TabPFN checkpoints were verified to differ from "
     "initialisation (all 14 CNP tensors and 129 TabPFN tensors changed), so weak CNP results cannot be "
     "blamed on an untrained model. Four automated checks guard correctness: replacing hidden audit labels "
     "leaves every decision unchanged (24 paired contexts, maximum prediction difference 0); group counts and "
     "audit exclusion are exact; models receive identical inputs; and every economic component recomputes "
     "from its parts. The project's 95 unit tests pass.")

# ================================================================== 7 Results
h1("7. Results and Evaluation")
h2("7.1 Prediction quality of the five methods")
ev = site_equal(mod[(mod.role == "evaluation") & mod.model_id.isin(MODELS)],
                ["log_mae", "thickness_mae_m", "recall", "fpr", "log_error_p90"])
rows = []
for m in MODELS:
    rows.append([NAMES[m], f"{dev_rank.loc[m, 'log_mae']:.3f}", f"{ev.loc[m, 'log_mae']:.3f}",
                 f"{ev.loc[m, 'log_error_p90']:.3f}", f"{ev.loc[m, 'thickness_mae_m']:.3f}",
                 f"{100 * ev.loc[m, 'recall']:.1f}", f"{100 * ev.loc[m, 'fpr']:.1f}"])
table("Prediction quality. Development log-MAE selected the primary model; evaluation columns are "
      "equal-site means over 63 registered states (recall excludes Wellington Airport, which has no low cells).",
      ["Method", "Dev log-MAE", "Eval log-MAE", "Eval P90 |error|", "Thickness MAE (m)", "Recall (%)", "FPR (%)"],
      rows, [4.4, 1.9, 1.9, 2.1, 2.2, 1.7, 1.6], bold_rows=())
para("TabPFN joint has the lowest evaluation profile error (0.497, 2.9% below XGBoost), and the "
     "development-selected TabPFN + graph is second (0.503). GP has the lowest thickness error (0.400 m) "
     "because it rarely predicts low cells: its recall is only 14%. The graph union buys the highest recall "
     "(36%) at the cost of the highest false-positive rate (9.7%) and thickness error (0.595 m). CNP never "
     "predicts *q*_{c} below 1 MPa, so its recall is zero – a genuine negative result for a median-trained "
     "set model on rare events. Figure 3 shows that site difficulty dominates model differences: every method "
     "scores ≈0.1 at Wellington Airport and ≈1.0 at New Lock.")
figure("fig04_five_models", "Five AI methods on identical observation states at the four evaluation sites. "
       "Open marks are site means; bars and filled marks are equal-site means.")
abl_ids = ["tabpfn_residual_unfinetuned", "tabpfn_residual_raw", "tabpfn_residual_guarded",
           "tabpfn_joint_unfinetuned", "tabpfn_joint"]
abl_names = {"tabpfn_residual_unfinetuned": "Residual, pretrained weights",
             "tabpfn_residual_raw": "Residual, fine-tuned weights",
             "tabpfn_residual_guarded": "  + sign guard",
             "tabpfn_joint_unfinetuned": "Joint, pretrained weights", "tabpfn_joint": "Joint, fine-tuned weights"}
rows = []
means = {}
for role in ["development", "evaluation"]:
    x = pd.concat([mod[mod.role == role], abl[abl.role == role]])
    means[role] = site_equal(x[x.model_id.isin(abl_ids)], ["log_mae", "thickness_mae_m"])
for m in abl_ids:
    rows.append([abl_names[m]] + [f"{means[r].loc[m, c]:.3f}" for r in ["development", "evaluation"]
                                  for c in ["log_mae", "thickness_mae_m"]])
table("TabPFN ablation on identical states (equal-site means).",
      ["Variant", "Dev log-MAE", "Dev thick. (m)", "Eval log-MAE", "Eval thick. (m)"],
      rows, [5.6, 2.6, 2.6, 2.6, 2.6])
para("The ablation (Table 4) separates the contributions. Source-site fine-tuning reduced the raw residual's "
     "development log-MAE from 0.627 to 0.598, but on evaluation sites the pretrained and fine-tuned joint "
     "models differ by less than 0.002, with the pretrained one slightly better. Most of the thickness gain "
     "comes from the spatial blend and whole-profile decision (development thickness 0.626 → 0.483 m), not "
     "from the Transformer weights. Improvements must therefore be attributed to the hybrid, not to TabPFN "
     "alone.")

h2("7.2 Selection, stopping and audit")
figure("fig03_selection_maps", "Selection maps, seed 17. Wellington Airport (a, b) at the online stop; New "
       "Lock (c, d) after the first 12–13 CPTs. Numbers give the round in which a position was revealed.",
       width_cm=12.0)
para("Figure 4 shows that the adaptive score does choose different locations from maximin: at New Lock it "
     "concentrates on the dense, variable central cluster where GP uncertainty and model disagreement are "
     "high, whereas maximin spreads out. At Wellington Airport both policies stopped after three batches with "
     "11 of 21 CPTs. Figure 5 tracks audit quality as CPTs are added. At the three easier sites quality is "
     "below the targets from the first few CPTs and changes little afterwards; at New Lock the error falls "
     "steeply over the first 15 CPTs and then plateaus at log-MAE ≈ 0.9 and thickness MAE ≈ 1.1–1.5 m, "
     "above both targets, even with all 64 CPTs.")
dd = dec[dec.role == "evaluation"]
o = orc[orc.role == "evaluation"]
e1 = eco[(eco.role == "evaluation") & (eco.alpha == 1.0)]
rows = []
for s in EVAL:
    g = dd[dd.site_id == s]
    nf = int(g.n_full.iloc[0])
    seq = g[g.policy_id == "B_seq"].n_stop.mean()
    ada = g[g.policy_id == "P_adapt"].n_stop.mean()
    ob = o[o.site_id == s].oracle_earliest_passing_budget
    oracle = "none" if ob.isna().all() else f"{int(ob.min())}–{int(ob.max())}" if ob.min() != ob.max() else f"{int(ob.min())}"
    full_pass = int(g[g.policy_id == "B_full"].audit_pass.sum())
    net = e1[(e1.site_id == s) & (e1.policy_id == "P_adapt")].net_benefit.mean()
    rows.append([SITE[s], nf, f"{seq:.1f}", f"{ada:.1f}", f"{100 * (nf - ada) / nf:.1f}", f"{full_pass}/3",
                 oracle, f"{net:+,.0f}"])
table("Decision outcomes at the evaluation sites (means of three seeds). No policy produced a false early stop.",
      ["Site", "N_full", "B_seq N", "P_adapt N", "Saving (%)", "Full-plan audit", "Oracle N", "Net € (P_adapt)"],
      rows, [3.6, 1.4, 1.6, 1.8, 1.7, 2.1, 1.6, 2.4])
para("Across the 24 evaluation trajectories there were no false early stops (Table 5, Figure 6). Sequential "
     "stopping saved CPTs in 4 of 12 trajectories and adaptive stopping in 3 of 12, giving equal-site savings "
     "of 13.6% and 11.9%. All the adaptive saving comes from Wellington Airport (21 → 11 CPTs, 47.6%), "
     "where sequential stopping saved exactly the same. In the paired comparison adaptive selection was never "
     "better than maximin at evaluation sites and once worse (Hollywood, seed 17: sequential stopped at 16, "
     "adaptive ran to 20). On development sites adaptive was better once and worse once. The full plan "
     "itself fails the audit at New Lock in all three seeds, and no method passes there even at its "
     "full-plan endpoint, so no stopping rule could have produced a reliable saving. Low-cell detection "
     "(recall ≥ 0.5, FPR ≤ 0.1) failed at all three sites that contain low cells.")
figure("fig05_quality_vs_cpts", "Hidden-audit quality as CPTs are added (three seeds per policy). Stars mark "
       "online early stops, squares the full plan, the red line the frozen target, and the green line the "
       "earliest passing count found with hindsight (oracle, diagnostic only).")
figure("fig06_site_outcomes", "Decision outcomes for all 48 sequential and adaptive trajectories. (a) CPTs used "
       "at the stop and the full-plan audit verdict; (b) CPTs saved by adaptive relative to sequential, paired by seed.")
para("The rule is safe but conservative. The oracle column shows that Christchurch Stadium already met the "
     "targets with 9–10 of 18 CPTs and Hollywood with 9–14 of 20, yet both ran to the full plan. The "
     "blocking signal is mainly Δ*H*_{90} ≤ 0, which demands that predicted thickness does not change at all "
     "between rounds; at sites with a few low cells one revealed CPT can flip a cell and reset the streak. "
     "The GP uncertainty signal was also only weakly related to actual audit error at evaluation sites "
     "(mean Spearman ρ = 0.22, versus 0.60 on development sites).")

h2("7.3 Economic value")
para("Applying the observed counts to the scenario (Figure 7), Wellington Airport gives a net owner "
     "benefit of +€23,300 for the adaptive policy: €7,500 from ten avoided CPTs, €10,250 of avoided project "
     "cost and €6,150 of earlier-opening benefit from finishing 2.05 days earlier, less €600 of AI cost. "
     "The other three sites are negative (−€11,900 to −€55,900) even though they used no extra CPTs, because "
     "the staged campaign adds a 0.25-day decision delay at every batch transition: 28–29 transitions at "
     "New Lock delay completion by 6.9 days. Setting this latency to zero reduces those losses to the €600 "
     "AI cost, whereas a one-day latency would make New Lock −€222,000. At α = 0.5 the Airport benefit falls "
     "to €15,100 and the New Lock loss to −€28,300; the sign of each result is unchanged across the tested "
     "ranges except through the latency parameter.")
figure("fig07_economics", "Owner-financial scenario, EUR 2025. (a) Benefit components of the adaptive policy "
       "relative to the full plan at α = 1 (mean of three seeds). (b) Net benefit as α and the daily "
       "earlier-opening benefit vary.")

# ================================================================== 8 Discussion
h1("8. Discussion and Insights")
para("*What worked.* The pipeline is leak-free and auditable: hidden labels never reach the controller, "
     "every stop is audited, and all 72 trajectories – including failures – are reported. The stopping rule "
     "made zero false early stops on unseen sites and halved the investigation where the ground is simple. "
     "Modern tabular AI is competitive: TabPFN-based hybrids reduced profile error by 2–3% relative to "
     "XGBoost and gave the best low-cell recall, while GP remained the best thickness estimator and the most "
     "useful uncertainty source.")
para("*What the results reveal.* First, site predictability, not the acquisition rule, decides whether CPTs "
     "can be saved: savings appeared only at the site where five CPTs already describe the ground, and "
     "vanished at New Lock where even 64 CPTs leave log-MAE near 0.9. Second, adaptive location choice added "
     "nothing over maximin in this study; the benefit came from staged investigation plus a stopping rule. "
     "Third, the economics are dominated by time, not CPT price: the earlier-finish terms are larger than "
     "the direct saving, and per-batch decision latency can turn a neutral outcome strongly negative. A "
     "deployed system must therefore return decisions within the working day and should switch to the full "
     "plan quickly when the site is clearly unpredictable.")
para("*Difficulties.* Four evaluation sites is a small sample, and these sites had been inspected in an "
     "earlier project stage, so this is a site-held-out re-evaluation rather than a new blind test. Low "
     "*q*_{c} cells are rare (≈7–10% of source cells), which explains the CNP failure and the 23–36% recall "
     "of the other models. A post-evaluation exploration (not used for any claim above) tried integrated-"
     "variance-reduction acquisition at New Lock: it lowered the GP's budget-averaged profile error by 10–13% "
     "relative to maximin but worsened thickness error, and none of 432 explored states met the targets.")
para("*Limitations.* The 0.25 m grid misses some thin layers; the 1 MPa threshold is a research indicator; "
     "economic inputs other than the quoted anchors are explicit assumptions; α = 1 assumes that "
     "investigation time lies on the project's critical path; and seeds are not independent sites.")
para("*Future improvements.* (i) Replace the zero-tolerance thickness-change test with a one-cell tolerance "
     "and use quality-predictive signals so that the stop at Christchurch Stadium and Hollywood is not "
     "missed; (ii) add a site-predictability check after the initial CPTs that recommends the full plan "
     "immediately for New Lock-like sites; (iii) adopt an event-weighted loss and a 0.1 m grid for low-cell "
     "detection; (iv) evaluate on genuinely new sites and real contract and programme data.")

# ================================================================== 9 Conclusion
h1("9. Conclusion")
para("This project built and audited a complete AI-guided CPT investigation loop: five AI predictors, an "
     "uncertainty–disagreement–distance acquisition score, a multi-signal stopping rule and an owner-financial "
     "economic model. A TabPFN hybrid gave the lowest profile error (0.497 log-MAE) and the graph variant the "
     "highest low-cell recall (36%), while GP remained best for thickness. With rules frozen on development "
     "sites, the system made no false early stops, saved 47.6% of CPTs at the simplest site (+€23,300 in the "
     "main scenario) and correctly ran to the full plan elsewhere; adaptive selection did not beat simple "
     "maximin ordering, and New Lock could not reach the quality targets with any number of CPTs. The main "
     "lessons are that algorithm choice should be judged by the decision it supports, with an independent "
     "audit and honest negative results, and that in investigation planning, time and site predictability "
     "matter more than the marginal accuracy of the predictor.")

# ================================================================== 10 References
h1("10. References")
refs = [
    "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM "
    "SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). https://doi.org/10.1145/2939672.2939785",
    "Ching, J., Uzielli, M., Phoon, K. K., & Xu, X. J. (2023). Characterization of autocovariance parameters of "
    "detrended cone tip resistance from a global CPT database. *Journal of Geotechnical and Geoenvironmental "
    "Engineering, 149*(10), 04023090. https://doi.org/10.1061/JGGEFK.GTENG-11214",
    "Garnelo, M., Rosenbaum, D., Maddison, C., Ramalho, T., Saxton, D., Shanahan, M., Teh, Y. W., Rezende, D., "
    "& Eslami, S. M. A. (2018). Conditional neural processes. *Proceedings of Machine Learning Research, 80*, "
    "1704–1713. https://proceedings.mlr.press/v80/garnelo18a.html",
    "Guan, Z., Wang, Y., & Zhao, T. (2022). Adaptive sampling strategy for characterizing spatial distribution of "
    "soil liquefaction potential using cone penetration test. *Journal of Rock Mechanics and Geotechnical "
    "Engineering, 14*(4), 1221–1231. https://doi.org/10.1016/j.jrmge.2022.01.011",
    "Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M., Hoo, S. B., Schirrmeister, R. T., & "
    "Hutter, F. (2025). Accurate predictions on small data with a tabular foundation model. *Nature, 637*, "
    "319–326. https://doi.org/10.1038/s41586-024-08328-6",
    "ISSMGE Technical Committee 304. (n.d.). *CPT databases*. Retrieved October 17, 2026, from "
    "https://tc304.org/cpt_databases.html",
    "MOS Grondmechanica B.V. (2025, May 1). *Offerte A2501409: Realisatie appartementencomplex De Spiegelhoek, "
    "Bussum* [Quotation]. https://www.spieghelhoek.nl/wp-content/uploads/2025/08/offerte-Mos-Realisatie-appartementencomplex-De-Spiegelhoek-010525-003.pdf",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, "
    "P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & "
    "Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, "
    "2825–2830.",
    "Prior Labs. (n.d.). *TabPFN* (Version 9.2.0) [Computer software]. https://github.com/PriorLabs/TabPFN",
    "Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian processes for machine learning*. MIT Press.",
    "Rijkswaterstaat. (2025, September 1). *New Lock Terneuzen fully operational from 1 August 2025*. "
    "https://www.rijkswaterstaat.nl/en/news/archive/2025/09/new-lock-terneuzen-fully-operational-from-1-august-2025",
    "U.S. Government Accountability Office. (2015). *GAO schedule assessment guide: Best practices for project "
    "schedules* (GAO-16-89G). https://www.gao.gov/products/gao-16-89g",
    "U.S. Government Accountability Office. (2020). *Cost estimating and assessment guide: Best practices for "
    "developing and managing program costs* (GAO-20-195G). https://www.gao.gov/products/gao-20-195g",
]
for ref in refs:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.8)
    p.paragraph_format.first_line_indent = Cm(-0.8)
    p.paragraph_format.space_after = Pt(2)
    _runs(p, ref, size=9)

# page numbers in footer (none on the cover)
sec.different_first_page_header_footer = True
footer = sec.footer.paragraphs[0] if sec.footer.paragraphs else sec.footer.add_paragraph()
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = footer.add_run()
for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
    if kind:
        el = OxmlElement("w:fldChar")
        el.set(qn("w:fldCharType"), kind)
    else:
        el = OxmlElement("w:instrText")
        el.set(qn("xml:space"), "preserve")
        el.text = text
    run._r.append(el)
run.font.size = Pt(9)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print("saved", OUT)
