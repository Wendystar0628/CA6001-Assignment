# Adaptive CPT investigation and economic assessment

This experiment extends the archived v0.9.1 prediction study with an explicit investigation decision. Its main comparison contains GP, XGBoost residual, CNP, TabPFN joint and TabPFN–graph. The earlier results remain available in `results/current/` and `results/final_test/`.

## Data and roles

The prepared dataset contains 503 source profiles, 495 distinct planned positions and 492 usable representative positions from 31 sites. Each usable curve has 24 depth cells over 2–8 m. The 11,808 cells are repeated depth measurements, not independent sites.

The eight sites with at least 20 usable positions are divided into four development sites and four evaluation sites. Seeds 17, 29 and 43 are fixed in `configs/adaptive_v1.yaml`. Each evaluation site uses its existing outer-fold source weights, excluding that site from both gradient training and checkpoint selection. These sites have historical development or unblinding exposure; the experiment is a limited-data re-evaluation rather than a new blind test.

| Role | Site | Source fold |
|---|---|---:|
| Development | Lower Hutt Town Hall; Tauranga Farmers Building | 0 |
| Development | Christchurch Metro Sports Facility; Marshland | 2 |
| Evaluation | Christchurch Stadium; Wellington Airport | 0 |
| Evaluation | New Lock Terneuzen | 1 |
| Evaluation | Hollywood | 2 |

For every site and seed, coordinates define a frozen initial set S0, candidate pool C and hidden audit set A. Approximately 20% of positions are assigned to A, using whole near-neighbour groups. The initial set is spatially dispersed and includes at least five actual positions. A nominal batch adds at least two actual positions; a whole group may exceed either target. All methods share the same partition and initial set. `N_full = |S0 ∪ C|`, excluding audit CPTs. Full-plan endpoints are evaluated explicitly.

The fixed monitoring grid G is the full planned-location geometry crossed with the 24 prescribed depths. Revealed rows use their public measurements; unrevealed rows use predictions. This grid supports monitoring at fixed planned positions, not a claim of measured accuracy throughout continuous space.

The 0.25 m representation is retained for comparability. Its thickness error against piecewise-linear integration of the original valid measurements is reported separately. The 0.1 m calculation is a representation diagnostic, not a separately trained model result.

## E1 and E2: model selection and contribution

All five models and four matched TabPFN ablations receive the same maximin observation states. Checkpoints are the first whole-group states reaching 5, 7, 9, 11 and 13 positions, followed by the full-plan endpoint. A group-induced overshoot is recorded as the actual number of observed positions; duplicate states are counted once.

Selection uses the lowest development-site-equal mean log-MAE, averaging registered states within each seed, seeds within each site, and then sites equally. Thickness, precision, recall, F1 and false-positive rate are reported alongside it. This registered-state mean differs from the archived 5–13 AUBC and is not relabelled as that statistic.

The four ablations are the raw trained residual path, its sign-guarded counterpart, the same-generation unfinetuned residual path, and unfinetuned joint path. Installed TabPFN v2 weights are reused; no model generation is upgraded. CNP and TabPFN gradient-update diagnostics accompany the results.

## E3: acquisition, monitoring and audit

- **B_full:** reveal all planned S0 ∪ C positions and predict A in a separate reference run.
- **B_seq:** reveal maximin batches and apply the same online stopping rule as the proposed method.
- **P_adapt:** rank candidate groups by GP standard deviation, low-qc disagreement with an auxiliary AI model, and distance to public observations. The first candidate weights are (0.5, 0.3, 0.2); the sole simpler alternative is (0.7, 0, 0.3).

Group uncertainty and disagreement are the member means; spacing is the minimum member distance to revealed positions. Ranks are recomputed within the remaining group pool, with equal values assigned 0.5. Within a batch, uncertainty and disagreement stay fixed; only distance is updated. All selected curves are revealed together after the complete batch is chosen.

Stopping requires at least nine observed positions, coverage of all populated plan quadrants, and two consecutive rounds passing four thresholds: G log-prediction change P90, G thickness change P90, revealed-support leave-group-out log-MAE, and G GP-standard-deviation P90. Support diagnostics use at most five groups fixed from the initial observations. GP uncertainty and disagreement are monitoring and ranking signals, not calibrated failure probabilities. The historical shared risk head is not used to stop.

The development full-plan log-MAE and thickness-MAE 75th percentiles plus margins 0.05 and 0.10 m define the absolute research targets. Relative full-plan tolerances are 0.05 log-MAE and 0.25 m thickness-MAE. Both absolute and relative requirements must pass. These are empirical study targets, not geotechnical acceptance standards. Low-qc recall ≥0.5 and FPR ≤0.1 are separately audited diagnostics; passing reconstruction targets does not imply that identification target has been met. Recall remains undefined when A has no low-qc cells.

Development trajectories are first continued to the full endpoint. Three predeclared common quantiles (0.50, 0.75, 0.90) of signals at acceptable development states give three threshold sets. Acquisition and stopping are selected jointly by development-site-equal verified saved fraction minus false-early-stop rate. The same frozen rule is then applied to both B_seq and P_adapt. The selected rule, source-weight fingerprints, prepared data and split fingerprints are saved before evaluation.

Online prediction files contain no audit measurements. Only after a trajectory returns does the evaluator read A. An online stop that fails audit remains a false early stop; the evaluator cannot move that stop to a later, more favourable state. Failure and full-plan completion remain in every summary. Seeds represent trajectory variation within a site, not independent construction projects.

## E4: economics

The main table uses an owner-financial counterfactual, EUR at 2025 prices, and α=1. It includes investigation costs, avoidable project-time costs, earlier opening net benefits and incremental deployment/review costs. α=0.5/0.75/1, individual input ranges and an α-by-daily-net-benefit grid show sensitivity. User shipping-time benefits are reported separately and are not added to owner cash flow.

New Lock supplies the engineering background. Unit prices, productivity and owner daily values without a recovered contract are explicitly identified as scenario assumptions; the local quotation anchors and official project sources are listed in `ECONOMIC_CASE_SOURCES.md`. Using the same monetary assumptions for other sites is a standardised scenario transfer, not local cost evidence. The full 20 m scenario penetration depth is separate from the 6 m prediction window; only hypothetical supplementary CPTs whose task is replaceable are eligible for whole-hole savings.

Quantities and stopping decisions come from the saved trajectories. Failed early stops are completed to the full plan in the repair scenario, including remobilisation and delay; they receive no reliable saved-CPT credit. Negative net benefits are retained.

## Reproduction and outputs

Model execution time is not a comparative performance outcome in this experiment: the computer is shared with other CPU/GPU workloads. Performance conclusions use prediction and investigation quality. Fieldwork, review and waiting days in the economic scenarios describe site operations and are distinct from model execution time.

Run `python -m cpt_budget adaptive --stage all` in the existing project environment for a fresh experiment directory. Individual stages are `prepare`, `pilot`, `compare-development`, `calibrate`, `freeze`, `compare-evaluation`, `evaluate`, `publish` and `economics`. The `freeze` stage selects and freezes using already completed development trajectories without repeating model inference. A completed evaluation is not reused for a new selection; use a new configured output/work directory for another experiment.

`results/adaptive_v1/` contains the frozen split and configuration, E1/E2 comparisons, online trajectories, separate quality audit, final decisions, resource records, economic input provenance, economic results and sensitivity. Large per-cell predictions stay under ignored `runs/adaptive_v1/`. Four focused guarantees are checked: hidden-label isolation, whole-group/count correctness, same-input comparison, and economic recomputation.
