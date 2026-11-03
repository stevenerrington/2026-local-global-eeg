---
title: Validation
subtitle: Each method checked against simulated data with a known answer
---

All checks are in section 4 of the [analysis notebook](../notebooks/results_analyses_executed.ipynb). Section 5 of the notebook recomputes every number reported in the Results from the saved derivatives: **62/62 reproduced** at the reported precision.

:::{list-table}
:header-rows: 1

* - Check
  - Method
  - Test
  - Result
* - V1
  - Leaky Beta–Bernoulli learner
  - Compare with the closed-form posterior; check reset and forgetting
  - Exact match; reset and half-life behave as specified
* - V2
  - Random-effects BMS (PXP, BOR)
  - 200 simulated groups with a 60% majority model; 200 null groups
  - Majority model wins 95%; under the null mean BOR = 0.85, max PXP > 0.9 in 0%
* - V3
  - Nested cross-validation
  - Simulate M2 data on a real sequence with a known half-life/reset, at 1×, 3× and 10× the empirical effect
  - Model family recovered; half-life/reset recovered 0/10, 4/10, 8/10. Candidate regressors correlate *r* = 0.91–0.998, so parameters are **not identifiable** at the real SNR
* - V4
  - Temporal cluster test
  - 200 null groups with ERP-like autocorrelation
  - Family-wise false-positive rate 6.5% (nominal 5%; within Monte Carlo error)
* - V5
  - 2 × 2 rm-ANOVA
  - Each *F* against the squared contrast *t*
  - Identical
* - V6
  - Added-variable slope
  - Inert regressor correlated (*r* ≈ .2) with the true one
  - Raw slope spurious (*t* = −4.2); added-variable slope correctly null (*p* = .40)
* - V7
  - eLORETA localisation
  - Simulated source in left Heschl's gyrus + STG
  - Peak in left STG; bilateral parahippocampal/entorhinal estimates at 83–95% of peak (spread)
:::

Additional checks in the notebook:

- **Refit:** all six models refitted from scratch for one participant reproduce the saved cross-validated scores exactly.
- **Tie-breaking in surprise bins:** 5–13% of trials have tied surprise values. Random instead of order-based tie-breaking changes every binned *t* by ≤ 0.01.
- **Spread check** for the left insula/IFG source: see [](results-source.md).

:::{note} What validation changed
V3 showed that half-life and reset cannot be estimated at this signal-to-noise ratio. Claims about learner timescales were removed from the Results, and the block-reset result is reported as supportive rather than decisive. V7 showed that the medial temporal involvement in the local source cluster is consistent with spread.
:::
