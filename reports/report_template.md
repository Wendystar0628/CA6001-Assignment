# Machine Learning Guided Adaptive CPT Investigation and Economic Benefit Assessment

CA6001 Applied AI Algorithms Project

Technical report

Student names {{student_names}}

17 October 2026

{{cover_summary}}

<!-- PAGE -->

## 1 Introduction

Cone penetration testing (CPT) measures vertical ground variability but leaves uncertainty between soundings. Further measurements require site operations, interpretation time and expenditure. The practical question is how many supplementary CPTs are needed, and where they provide the most useful information.

This study compares five AI predictors, then links a selected model to adaptive location selection, online stopping and an independent quality audit. Measured replay workloads feed an economic assessment of investigation cost and programme effects. {{headline_result}}

Guan et al. (2022) demonstrated adaptive CPT numbers and locations for liquefaction assessment using information entropy and Bayesian compressive sampling. Here the targets are shallow cone resistance and cumulative low-resistance thickness. The contribution connects modern AI prediction to investigation and economic decisions, while separating the value of adaptive ordering from the value of early stopping.

<!-- FIGURE workflow 6.15 -->

## 2 Problem Definition

At position x and depth z, the regression target is y(x,z) = ln(qc/MPa). The 2–8 m window contains 24 cells of width 0.25 m. The event qc < 1 MPa has cumulative thickness H(x) = 0.25 Σk I[y(x,zk) < 0], in metres, including disconnected low-resistance intervals.

The decision task selects positions and a stopping time from a fixed pool, seeking fewer CPTs at pre-specified profile and thickness quality. Hidden locations provide an independent audit. The 1 MPa event is a research target, not a soil-classification or design-safety criterion. Economics concerns only the supplementary task validated here.

<!-- PAGE -->

## 3 Dataset Description

The public Global-CPT collection contains 1,196 soundings from 59 sites or regions; 35 have multiple soundings with known coordinates (ISSMGE Technical Committee 304 [TC304], n.d.; Ching et al., 2023). The separately listed New Lock collection is already included and is not counted twice.

{{data_summary}} Table 1 distinguishes the sampling units.

<!-- TABLE dataset -->

Inputs comprise coordinates, query depth and revealed CPT curves. Neighbour summaries, distances, vertical changes and low-value fractions are derived only from revealed data. Hidden resistance and thickness are excluded.

Resistance is expressed in MPa and log transformed. Curves must cover 2–8 m, with interpolation gaps no larger than 0.5 m. Deterministic representatives remove eight repeated-coordinate profiles; three further profiles fail midpoint completeness. Nearby and duplicate groups stay together across support, candidate and audit roles.

Source spreadsheets are standardised into Parquet profile and depth-grid tables. CSV manifests preserve site, location, group and replay-role identifiers.

{{data_roles}}

{{representation_analysis}}

Errors are averaged within soundings and then equally across sites. This prevents dense sites from dominating. Recall is not applicable to sites without low events. Depth rows and replay seeds are not independent field sites.

<!-- PAGE -->

## 4 Algorithm Selection and Rationale

Five methods compare spatial probability, tree boosting, neural set learning and a tabular foundation model under matched revealed data. Residual models retain an explicitly identified spatial trend.

**Gaussian process.** The GP models departures from a source-site depth trend using a Matérn 3/2 covariance, separate horizontal and vertical length scales, and observation noise. Initial scales are 20, 20 and 1 m, optimised on revealed support. Conditioning gives a prediction mean and standard deviation. Spatial correlation makes this a strong comparator and acquisition model (Rasmussen & Williams, 2006).

**XGBoost residual.** Gradient-boosted trees predict the residual after a local inverse-distance-weighted estimate. Neighbour values, distances and vertical summaries support site-specific correction. Source training uses depth-four trees, learning rate 0.05, at most 500 estimators and validation early stopping. The trees capture nonlinear interactions, making this the principal strong machine learning comparator (Chen & Guestrin, 2016).

**Conditional neural process.** The CNP encodes support profiles with query-relative positions through 64-unit hidden layers, pools by mean and maximum, then decodes by depth. It tests transferable set-to-function learning from a few complete curves (Garnelo et al., 2018). {{cnp_diagnosis}}

**TabPFN joint.** TabPFN is a Transformer that learns a prediction procedure across synthetic tabular tasks (Hollmann et al., 2025). Here its guarded CPT residual prediction is averaged equally with a triangulation estimate. A support-derived complete-curve decision sets the low-cell count. This is a hybrid: TabPFN supplies half of the amplitude base. {{tabpfn_training_evidence}}

**TabPFN graph.** This hybrid averages the residual branch with a graph-Huber estimate and labels a cell low if either branch predicts it. Robust spatial smoothing supports coherent profiles. The union rule favours detection, with an expected increase in false positives and low-value thickness.

{{model_selection_rule}}

<!-- PAGE -->

## 5 Methodology

### Matched investigation replay

Each site-seed allocation fixes an initial set S0, candidate pool C and hidden audit set A. Spatial maximin initialisation reveals at least {{initial_count}} locations in complete groups. The full replay plan S0 ∪ C defines Nfull. A is excluded from every support set and selectable budget. The endpoint is the complete pool.

Bfull completes this plan. Bseq adds maximin batches, choosing groups furthest from revealed CPTs. Padapt adds model-informed batches. Bseq and Padapt share the frozen prediction model, stopping rule and allocation. Their difference isolates adaptive ordering; comparison with Bfull includes the benefit of early stopping.

<!-- FIGURE map 6.15 -->

### Adaptive location selection

Candidate signals are mean GP standard deviation U, low-event model disagreement D and distance to revealed CPTs S. Whole groups receive mean member U and D and minimum support distance. Signals are ranked within the pool. {{acquisition_rule}}

{{batch_rule}} The expanded context is then used for prediction. {{development_boundary}}

<!-- PAGE -->

### Online stopping and independent quality assessment

Stopping uses only revealed-data signals. At all full-plan coordinates and the 24 depths, consecutive predictions yield the 90th-percentile change in log resistance and cumulative thickness. A leave-near-group-out diagnostic re-predicts up to five fixed initial groups using the remaining revealed support. GP uncertainty, coverage of populated spatial quadrants and a minimum support count provide further checks. {{stopping_rule}}

{{quality_rule}} An independent evaluator computes log-MAE, thickness MAE, precision, recall, F1 and false-positive rate on A, then compares stopping with Bfull. Failed early stops remain failures. Without an online trigger, the strategy completes the plan.

### Economic assessment

The owner's direct saving equals complete-plan cost minus AI investigation cost. Time includes production, review, batch waiting and remobilisation. Project advancement is ΔTproject = α ΔTinv; α = 1 is the main scenario, with 0.75 and 0.5 sensitivity cases (U.S. Government Accountability Office, 2015, 2020).

Net benefit is Bnet = Binv + ΔTproject(cday + bday) − CAI. Here cday is avoidable project cost, bday incremental owner operating cash flow and CAI incremental deployment and review cost. Components do not overlap; user time savings remain separate. Failed early stops are repaired to the full plan, including return-to-site cost and delay.

## 6 Implementation

Separate modules handle modelling, acquisition, stopping, audit and economics. Configuration files freeze site roles, seeds, model versions and thresholds. Exported trajectories preserve selected positions, support counts, online signals and stop reasons; audit labels and scores remain outside the decision engine.

{{implementation_details}}

The exported manifest, configuration, model provenance, trajectories, audit and economic tables support reproduction. {{implementation_checks}}

<!-- PAGE -->

## 7 Results and Evaluation

### Prediction and model contribution

{{model_result_summary}}

<!-- TABLE models -->

<!-- FIGURE models 6.15 -->

{{model_interpretation}}

{{ablation_summary}}

<!-- PAGE -->

### Investigation workload and quality at stopping

{{decision_summary}}

<!-- FIGURE quality 6.15 -->

{{quality_interpretation}}

<!-- FIGURE savings 6.15 -->

{{site_interpretation}}

<!-- PAGE -->

### Economic value and sensitivity

New Lock Terneuzen, fully operational from 1 August 2025 (Rijkswaterstaat, 2025), supplies historical data for a counterfactual supplementary-CPT scenario. Replacement concerns the validated shallow task; statutory and deep-design tests remain common.

<!-- TABLE economics_inputs -->

{{economic_summary}}

<!-- FIGURE economics 6.15 -->

{{economic_interpretation}}

{{sensitivity_summary}}

<!-- PAGE -->

## 8 Discussion and Insights

{{discussion_prediction}}

{{discussion_decision}}

Nearby CPTs can disagree on thin layers and resistance extremes. Low average profile error can conceal missed low intervals, while stable predictions can reflect smoothing rather than sufficient information. Independent post-stop checks keep the decision tied to investigation quality.

{{discussion_economics}}

{{discussion_limits}}

A prospective pilot should specify mandatory locations, accepted quality, actual prices and programme dependencies. Stored trajectories support this by retaining chosen positions, stop reasons, audit outcomes and their effects on cost and time.

## 9 Conclusion

{{conclusion}}

<!-- PAGE -->

## 10 References

{{references}}
