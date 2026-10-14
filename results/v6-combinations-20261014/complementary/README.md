# Fourth-round complementary point mixtures

This exploration follows the earlier rounds' multiobjective conflict. The parent froze three complementary point mechanisms, each paired with two risk/event heads. The implementation did not read fresh combination validation scores and did not sweep any mixture weight.

| Point mechanism | Frozen construction |
|---|---|
| `adaptive_graph` | 0.5 support-selected structural residual stack + 0.5 graph residual low-union |
| `fpca_graph` | 0.5 FPCA residual w050 + 0.5 graph residual low-union |
| `median_four` | Cellwise median of v0.4 best residual, FPCA KRR, graph Huber and shape registration |

For four values, median means the arithmetic average of the two middle ordered values. This is a robust amplitude combination rather than selection of whichever member happens to fit an evaluation label.

Every point mechanism has two outputs:

- `_gp`: retain the point amplitude and attach the existing GP's **native** `p_low_1` as a separate event head. This probability is not claimed to be the CDF of the new point estimator.
- `_event`: apply the fixed event Gibbs decoder with GP native probabilities, tau 0.5, Potts cost 0.25, cardinality cost 0.25 and sign margin 0.05. The count prior is the 3NN IDW2 weighted median of currently revealed low-resistance profile counts. Report the decoded point amplitudes and the exact Gibbs event marginals.

The event decoder's `H_point` remains the actual final amplitude's low-resistance thickness. Its support count prior is a soft energy penalty, not a substituted evaluation output. Gibbs marginals are model probabilities, not an assertion of calibrated geological uncertainty.

Inputs are aligned by fold, site, seed, budget, location and depth. The only columns read from new candidate artifacts are keys and `pred_log_qc`; the only extra column read from the frozen GP artifact is `p_low_1`. OOF-calibrated `risk_probability` and query truth are not read by the mechanism. Truth is attached only in the common export/evaluation boundary.

All **6 candidates completed 60 states with 8,640 cells per model**, in **17.00 seconds**. `decoding_diagnostics.csv` records support count priors, decoded/expected counts and changed signs; `contexts/` records observed IDs and timing. Final efficacy and gate decisions belong to the parent comparison.

```powershell
.venv/Scripts/python.exe -m cpt_budget.v6.complementary
.venv/Scripts/python.exe -m pytest tests/test_v6_complementary.py -q
```

Two focused tests passed, checking the frozen numerical combinations, robustness to one extreme member, query-label/OOF-risk poisoning, row-order alignment and the equality between reported event count and the actual final amplitude's signs. The shared Gibbs decoder's own correctness tests are maintained separately.
