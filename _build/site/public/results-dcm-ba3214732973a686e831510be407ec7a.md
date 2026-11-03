---
title: 4. Effective connectivity (DCM)
subtitle: Local surprise drives forward auditory connections; global deviance propagates forward from STG to parietal cortex
short_title: DCM
---

In the MMN window, local surprise strengthened the forward A1 → STG connection and top-down input to STG, in both model codings. Global deviance's apparent effect on A1 → STG over the full response did not survive restriction to the MMN window.

```{figure} ../figures/Figure_DCM_windows.png
:label: figdcm
:width: 100%
**DCM, full response vs MMN window.** (A–D) Network changes for local (top) and global (bottom) deviance, fitted over 0–600 ms (left) and 0–250 ms (right). Coloured arrows are robust: credible with the same sign in both codings. Orange = stronger, blue = weaker than for standards; grey = credible in one coding only. (E) Log gain for every connection credible in at least one analysis; outlines mark credible estimates. (F) Variance explained per participant.
```

:::{important} Pending update
The 0–600 ms panels currently use the sign probability of the family-level average. Run `scripts/dcm/dcm_07_search_full.m` to give them the same per-connection test as 0–250 ms, then rerun `scripts/28_figure_dcm_windows.py`.
:::

## Model and controls

- DCM for ERP (SPM12), eight sources: bilateral A1, STG, IFG and IPL. Driving input at A1; forward A1 → STG → IFG/IPL; reciprocal backward connections; lateral IFG ↔ IPL. Equivalent-current-dipole spatial model.
- Condition effects could modulate every forward and backward connection, coded two ways:
  - **Surprise:** each participant's condition-mean learner surprise (M2 local, M3 global), relative to LS/GS and scaled to a maximum of 1. The local term carries context-dependent local surprise (LD/GS 0.59, LD/GD 1.0). The global term is close to the global labels (0, −0.06, 0.96, 0.88). The two are weakly correlated (mean *r* = 0.22).
  - **Interaction:** 0/1 labels plus a local × global term, so "global" is the global effect on local standards only.
- Group inference: parametric empirical Bayes on the modulatory parameters, then per-connection Bayesian model reduction {cite:p}`Friston2016`. Credible = posterior probability of being present > .95.
- **Window control.** With a single input at A1, the late P3b of global deviants must pass through A1 → STG when the model is fitted over 0–600 ms. Both codings were therefore also fitted over 0–250 ms, the conventional MMN window for DCM for ERP {cite:p}`Garrido2007`.
- Fit: median variance explained 89% at 0–600 ms (range 80–96%) and 94–95% at 0–250 ms (range 90–98%) ([](#figdcm)F).

## Local surprise ([](#figdcm)A, B)

- 0–250 ms, robust (log gain, mean of the two codings): forward A1 → STG +1.27 left, +0.80 right. These are the largest effects in the model.
- Backward input to STG was also robust: IFG → STG +0.51 left, +0.44 right; IPL → STG +0.55 left. Right STG → IPL weakened (−0.30).
- At 0–600 ms the forward A1 → STG increase was again robust (+0.50 left, +0.45 right), together with left STG → IFG (+0.71). That connection was not credible at 0–250 ms, so it probably reflects later propagation.

## Global deviance ([](#figdcm)C, D)

- **0–600 ms:** global deviance strengthened A1 → STG in both codings (+0.51 left, +0.30 right), with changes in backward IFG → STG and forward STG → IFG/IPL.
- **0–250 ms:** the A1 → STG effect was no longer robust; its sign depended on the coding (left: surprise −0.53, interaction +0.23). The only robust global effect was forward left STG → IPL (+0.38).
- Backward connections depended on the coding. Surprise coding: the forward-only family won (posterior 0.98). Interaction coding: backward IPL → STG (+0.54) and IFG → STG (+0.27) were credible.

## What the data do and do not support

- **Supported:** local surprise is carried by the forward auditory pathway and modulates top-down input to STG, independent of window and coding.
- **Supported:** the full-window global effect on A1 → STG is largely a consequence of routing the late response through the single A1 input. This agrees with eLORETA, which found no global effect in auditory cortex ([](results-source.md)).
- **Not resolved:** whether global deviance modulates backward connections early, and whether a small early global input to auditory cortex exists. The codings estimate different quantities. Interaction coding assigns the small LS/GD effect (−0.27 µV, [](results-erps.md)) to global; surprise coding first absorbs context-dependent local surprise.
- **Not informative:** the forward/backward family comparison at 0–600 ms, where "forward and backward" won for every effect.
- **Limitations:** template head, dipole locations fixed a priori, a single driving input; late global effects are not modelled with their own input.

:::{seealso}
Code (MATLAB/SPM12): `scripts/dcm/dcm_01_prepare.m` → `dcm_02_fit.m` → `dcm_03_peb.m` → `dcm_04_robustness.m`; surprise and interaction codings `scripts/25_dcm_surprise_export.py` → `dcm_05_surprise.m` (0–600 ms), `dcm_06_early.m` (0–250 ms), `dcm_07_search_full.m`; figure `scripts/28_figure_dcm_windows.py`.
:::
