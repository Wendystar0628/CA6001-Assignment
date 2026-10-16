# Adaptive CPT formal report

The formal report follows the ten sections in the retained CA6001 Report Template. It uses British academic English and is limited to ten pages after the cover, including references. The report explains how prediction quality becomes a reproducible investigation decision and then an economic scenario.

Completed on 17 October 2026. The final PDF has one cover page and ten content pages, including six scientific figures, three tables and APA references. The Word and PDF were rendered and visually checked. The editable cover retains a student-name field to complete before submission.

## Report structure

| Body page | Content | Main evidence |
|---:|---|---|
| 1 | Introduction and Problem Definition | End-to-end decision workflow and target definitions |
| 2 | Dataset Description | Current data manifest, grouping, depth representation and source acknowledgement |
| 3 | Algorithm Selection and Rationale | GP, XGBoost residual, CNP, TabPFN joint and TabPFN graph |
| 4 | Methodology | Frozen initial set, candidate pool, audit set and matched policy comparison |
| 5 | Methodology and Implementation | Online stopping, independent quality audit, economic equations and reproducible implementation |
| 6 | Results and Evaluation | Five-model results and necessary adaptation ablations |
| 7 | Results and Evaluation | Quality against number of CPTs, policy savings and failed stops |
| 8 | Results and Evaluation | Economic decomposition and sensitivity |
| 9 | Discussion and Conclusion | Geotechnical interpretation, useful improvements, failed cases and supported conclusion |
| 10 | References | APA references to data, methods, implementation and economic sources |

The exact pagination may change with the final evidence while preserving the page limit and template section order. All figures use black, legible labels. Unavailable student identities are left as editable cover fields.

## Evidence rules

The five-model comparison, selected main model, stopping thresholds, trajectory counts and monetary results must come from the new adaptive run. Archived v0.9.1 values are not substituted for new results. Site means are the aggregation unit; seeds and depth cells are not independent sites. The full-plan budget excludes hidden audit positions. Economic scenarios distinguish measured CPT counts from assumed prices, daily costs and project schedule transmission.

The report focuses on prediction quality, stopping outcomes, CPT counts and scenario economics. Hardware and package versions support reproduction. Economic field production, inter-batch waiting and engineering-review time remain business-scenario inputs.

## Deliverables and editable sources

- `reports/CA6001_Adaptive_CPT_Report.docx`
- `reports/CA6001_Adaptive_CPT_Report.pdf`
- `reports/CA6001_Adaptive_CPT_Report.md`
- `reports/build_adaptive_report.py`
- `reports/report_facts.json`
- `reports/report_template.md`
- `reports/references.md`

The builder retains the original template as its source, renders the final Word document, and checks that the formal content occupies at most ten pages after the cover. The original assignment files and v0.9.1 experiment archive remain the reference material.
