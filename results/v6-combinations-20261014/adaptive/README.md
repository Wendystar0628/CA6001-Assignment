# Observed-only adaptive profile stacking

This second exploration round implements a deployable selection mechanism rather than selecting a fixed mixture from validation scores. Its parameters were agreed in agent coordination and written to `method_protocol.json` before executing these four candidates. No new-combination validation scores were read by this implementation agent.

## Candidates and boundary

The four members are IDW, source-trained FPCA plus target-conditioned KRR, bounded shape registration, and a new three-neighbor inverse-distance-weighted median at each depth. All use the same currently revealed complete CPTs and query coordinates. The last member minimizes local weighted absolute error and does not average away support order statistics.

- `support_stack_l1`: choose a member-weight simplex using revealed-support leave-one-location-out absolute log error.
- `support_stack_structural`: choose the simplex using absolute log error plus 0.5 times soft thickness absolute error in metres plus 0.5 times an event Brier surrogate with false-negative weight 2.
- `support_residual_stack_l1`: fixed 0.5 share of the frozen v0.4 best residual prediction plus 0.5 share of the first stack.
- `support_residual_stack_structural`: the same fixed residual share plus the structural stack.

Each leave-one-out split excludes every depth of the held location, any shared `near_group_id`, and colocated support coordinates. Every member is refitted on the remaining support. The source FPCA basis remains the original fold-specific v0.5 basis and excludes the target validation site. A small independent support of one or two profiles is used as available; a zero-independent-support split has the predeclared fixed log-qc-zero fallback. None of these exceptional cases occurred in this 60-state run.

The weights lie on the nonnegative sum-to-one simplex with 0.25 spacing: 35 possibilities. Both objectives add the unchanged regularizer `0.05 * sum((weight - 0.25)^2)`. The structural proxy is `p = sigmoid(-predicted_log_qc / 0.25)` at the 1 MPa threshold; soft thickness is `0.25 m * sum(p)` and is compared to the held observed profile's hard low-resistance thickness. This is a support-only training surrogate, **not** a native calibrated event probability. Evaluation must still report the common actual `h_point_mae_m` and common risk calibration.

The residual member is deliberately absent from the support LOO grid because its source-trained model would require an additional full inference loop. Therefore the two residual variants use a **fixed 0.5 residual weight**, not a jointly LOO-optimized four/five-member ensemble. This mismatch is a limitation, not hidden hyperparameter selection.

## Execution

```powershell
.venv/Scripts/python.exe -m cpt_budget.v6.adaptive
.venv/Scripts/python.exe -m pytest tests/test_v6_adaptive.py -q
```

Completed 60 states, 540 whole-profile LOO splits, 120 weight selections and four outputs of 8,640 rows each. Runtime was 52.36 seconds on CPU. Five focused tests passed, including whole-profile label withholding, shared-group/colocated-label withholding, query-label poisoning, row order, weighted-median L1 optimality, and selection of a truly predictive synthetic member.

Mean support-selected weights, shown only to describe the mechanism (not validation efficacy):

| Support selection objective | IDW | FPCA KRR | Shape registration | Weighted median |
|---|---:|---:|---:|---:|
| Absolute log error | 0.2333 | 0.3458 | 0.2833 | 0.1375 |
| Structural composite | 0.1500 | 0.3917 | 0.2667 | 0.1917 |

`support_selections.csv` records all weights and LOO losses; `contexts/` records support IDs, exclusions, source-basis sites and timing. Candidate subdirectories contain complete predictions and protocols. Final efficacy, common calibration, and promotion decisions belong to the parent v0.6 comparison, not these fitting diagnostics.
