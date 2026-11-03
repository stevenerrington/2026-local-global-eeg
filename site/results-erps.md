---
title: 1. Early MMN and late P3b
subtitle: Local and global deviance produce an early frontocentral MMN and a late parietal P3b (Figure 1)
short_title: ERPs (Fig. 1)
---

The local contrast (LD − LS) gave an MMN at 80–140 ms and the global contrast (GD − GS) a sustained P3b at 300–700 ms, each in 15/15 participants.

```{figure} ../figures/Figure1.png
:label: fig1
:width: 100%
**Figure 1.** (A–B) Design: local (5th sound repeats, LS, or changes, LD) × global (the block's frequent quintlet, GS, or its rare deviant, GD). (C) Topographies of each contrast over time; dots mark spatio-temporal cluster membership. (D–E) ROI waveforms and differences, with cluster bars. (F) Per-participant window means.
```

## Design and controls

- Each quintlet was labelled by two orthogonal factors: local and global ([](#fig1)A–B).
- The global contrast is physically matched. The same quintlet is the standard in one block and the deviant in another, so GD − GS cannot reflect acoustic differences.
- Windows and ROIs were fixed a priori: frontocentral (Fz, FC1, FC2, F1, F2, Cz) at 80–140 ms; parietal (Pz, P1, P2, CPz, POz, CP1, CP2) at 300–700 ms. Inference uses these windows.
- Whole-scalp and whole-epoch effects: cluster-based permutation {cite:p}`Maris2007` (cluster-forming *p* < .05, 2,000 permutations).
- "x/15" = participants whose effect has the group sign. Tests are two-sided, df = 14.

## Local effect (MMN)

- Frontocentral negativity peaking at 106 ms ([](#fig1)C top, D).
- 80–140 ms: −1.22 µV (95% CI ± 0.28), *t*(14) = −9.4, *p* = 2 × 10⁻⁷; 15/15 ([](#fig1)F).
- ROI time course: negative cluster 68–136 ms (*p* = .019). It was followed by an unhypothesised positive cluster at 160–296 ms (*p* = .007; P3a-like) and a negative cluster at 344–468 ms (*p* = .033).
- Spatio-temporal clusters: an opposite-polarity pair at 30–140 ms (negative *p* = .017, 33 channels; positive *p* = .027, 30 channels). These were followed by 138–586 ms (positive, *p* = .0005) and 150–360 ms (negative, *p* = .0055).
- **Reference control.** With a linked-mastoid reference (TP9/TP10) the MMN at Fz was −3.17 µV, *t*(14) = −9.4. With the average reference it inverted at the mastoids (+1.77 µV, *t*(14) = 9.0; 15/15), the signature of a supratemporal generator.

## Global effect (P3b)

- Sustained parietal positivity ([](#fig1)C bottom, E).
- 300–700 ms: +2.35 µV (95% CI ± 0.94), *t*(14) = 5.4, *p* = 9.7 × 10⁻⁵, peak 432 ms; 15/15.
- ROI time course: positive cluster 168–700 ms (*p* = .001). Spatio-temporal clusters: positive *p* = .0005 and negative *p* = .001, both significant from about 75 ms to 700 ms.

## Local × global interaction

:::{list-table} Cell means (µV) and 2 × 2 repeated-measures ANOVA
:label: tbl-anova
:header-rows: 1

* - Window
  - LS/GS
  - LD/GS
  - LS/GD
  - LD/GD
  - Local
  - Global
  - Local × global
* - Frontocentral 80–140 ms
  - −0.15
  - −1.20
  - −0.42
  - −2.42
  - *F* = 79.2, *p* < .001, η²G = .57
  - *F* = 31.5, *p* < .001, η²G = .24
  - *F* = 12.9, *p* = .003, η²G = .11
* - Parietal 300–700 ms
  - −0.07
  - +0.08
  - +2.54
  - +2.15
  - *F* = 0.8, *p* = .40
  - *F* = 28.8, *p* < .001, η²G = .48
  - *F* = 3.6, *p* = .079
:::

- **Early window: global status modulates the MMN.** The MMN was about twice as large when the local deviant was also a global deviant (LD/GD vs LD/GS: −1.22 µV, *t*(14) = −5.0, *p* = .0002, 15/15). A small global effect was present without local deviance (LS/GD vs LS/GS: −0.27 µV, *t*(14) = −2.5, *p* = .026, 12/15).
- These early global differences follow the local learner's context-dependent surprise: 4.1 bits for LD/GD vs 2.4 for LD/GS, because the deviant sound is rare in those blocks ([](results-models.md)). They are not interpreted as rule-level effects.
- **Late window: the P3b is a pure global effect.** It was present with and without a local change (LS: +2.61 µV, *t*(14) = 4.9; LD: +2.07 µV, *t*(14) = 5.7). There was no local effect and no reliable interaction.

:::{warning} Onset caveat
Global spatio-temporal clusters reached significance from about 75 ms. Cluster tests do not establish onsets {cite:p}`Sassenhagen2019`, and the early part reflects the local context effect above, so the a priori window is used for inference.
:::

:::{seealso}
Code: notebook section 1 ([](../notebooks/results_analyses_executed.ipynb)); `scripts/14_figure1.py`, `scripts/27_results_checks.py`, `scripts/10_mastoid_reference.py`.
:::
