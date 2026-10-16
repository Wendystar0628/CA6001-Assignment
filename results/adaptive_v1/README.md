# Adaptive investigation evidence

This directory contains the new five-model, whole-group investigation experiment. It does not overwrite the v0.9.1 results in `results/current/` or the previously unblinded `results/final_test/` archive.

## Data and source models

- `splits.json` / `data_manifest.csv`: 24 frozen site–seed S0/C/A partitions. Full-plan counts exclude A; complete near-neighbour groups are never split between roles.
- `profile_manifest.csv` / `site_manifest.csv`: source-profile, representative-position and site roles. The prepared scope is 503 profiles, 495 positions, 492 usable representatives, 31 sites and 11,808 depth cells.
- `source_training_manifest.csv` / `model_diagnostics_sources.json`: actual training and checkpoint-selection exposure by source fold.
- `representation_profiles.csv` / `representation_summary.csv`: original-curve thickness integration and 0.25 m representation error. The 0.1 m version is a diagnostic only.
- `newlock_source_mapping.csv`: the 98 original New Lock CPT files, including the 80 retained profiles and the 18 window-coverage exclusions.
- `model_diagnostics_training.json` / `model_diagnostics_cnp_distribution.json` / `model_diagnostics_probe.csv`: gradient-update evidence, CNP output checks and representative real inference checks.

## Prediction and frozen selection

- `model_comparison.csv`: five AI models, identical registered observation states. The principal fields are `role`, `site_id`, `seed`, `model_id`, actual `budget_n`, `full_endpoint`, `log_mae`, `thickness_mae_m`, `precision`, `recall`, `f1`, `fpr` and `log_error_p90`.
- `ablation.csv`: trained raw/guarded residual, same-generation unfinetuned residual and unfinetuned joint controls under the same inputs.
- `development_model_ranking.csv`: registered-state mean within seed, then within site, then equal-site development ranking. This selects the primary model without evaluation outcomes.
- `stop_calibration.csv`: the six predeclared acquisition/threshold candidates, including verified saving and false early stops.
- `frozen_config.json`: selected graph model, UDS weights, common stopping rule, research quality targets and source/configuration/data fingerprints, frozen before evaluation.

Development has 72 registered states, 72 distinct known sets and 12 audit sets. Evaluation has 63 registered states, 62 distinct known sets and 11 audit sets. The four evaluation sites are the replicate units; three seeds per site describe trajectory variation. Registered-state means include complete-plan endpoints and must not be labelled as the archived 5–13 AUBC.

## Online decisions and separate audit

- `trajectories.csv`: online-only signals, observed IDs, next selected IDs, actual counts and saved stop reasons. `revealed_ids` is the batch arriving at a round; `selected_ids` is the next batch. Terminal rows select nothing further.
- `selections.csv` / `locations.csv`: measured order, geometry, partition and selection scores for the maps.
- `quality_audit.csv`: independent hidden-audit quality for saved prediction states. Absolute and relative reconstruction requirements are separate; low-qc identification is a further diagnostic.
- `signal_diagnostics.csv`: descriptive association between the GP monitoring signal and actual audit error. These audit diagnostics are not fed back into the controller.
- `decisions.csv`: one final record per site, seed and policy, including `n_full`, `n_stop`, `audit_pass`, `false_early_stop`, `is_early_stop` and saved counts.

`audit_pass` refers to empirical profile-and-cumulative-thickness research targets, not an engineering safety standard. `low_qc_audit_pass` is separate; recall is undefined where the audit set has no low-qc cells. A failed audit cannot move an already saved online stop.

## Economic scenarios and presentation

- `economic_inputs.csv`: units, currency/year, sources, assumptions, ranges and overlap notes.
- `economic_results.csv`: α=1 owner-financial scenario, including actual stop counts, claimed versus reliable saved CPTs, repair costs, field/review/waiting days and signed net benefits.
- `sensitivity.csv`: α, individual parameter ranges and the α-by-daily-net-benefit scenario grid.
- `social_user_benefit.csv`: separately labelled shipping-user effects, excluded from owner net benefit.
- `figures/`: six scientific figures exported to PNG, SVG and PDF. The project HTML and experiment Notebook read these actual tables.

New Lock provides the case background. Other sites use the same monetary assumptions as a standardised scenario transfer. Amounts are not historical contract savings; whole-CPT substitution applies only to the explicitly assumed supplementary task. Failed early stops are completed to the full plan with remobilisation and delay. Model execution times are not comparative outcomes; economic fieldwork and review days are separate operational assumptions.

Run commands and the complete methodology are in [`docs/RUN_GUIDE.md`](../../docs/RUN_GUIDE.md) and [`docs/ADAPTIVE_PROTOCOL.md`](../../docs/ADAPTIVE_PROTOCOL.md). Large per-cell prediction records remain locally under ignored `runs/adaptive_v1/`.
