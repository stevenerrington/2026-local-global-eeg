---
title: 3. Source localisation
subtitle: The local effect originates in left auditory cortex; the global effect has no reliable focal source (Figure 3)
short_title: Sources (Fig. 3)
---

The 80–140 ms local effect localised to left Heschl's gyrus and superior temporal gyrus (cluster *p* = .0005), with an MMN-like source time course peaking at 102 ms. No focal generator was found for the global effect.

```{figure} ../figures/Figure3.png
:label: fig3
:width: 55%
**Figure 3.** (A) eLORETA *t*-map of the signed local effect (LD − LS, 80–140 ms), cluster-corrected; auditory cortex outlined. (B) Auditory-cortex (Heschl's gyrus + STG) source time course, left and right.
```

## Method and controls

- eLORETA {cite:p}`PascualMarqui2007` on a template head: MNE "sample" anatomy, three-layer BEM, standard 10-05 electrode positions, per-participant noise covariance from the pre-sequence silence.
- Statistics used the **signed** source difference (deviant − standard). It has zero expectation under the null and, unlike source strength, is not biased by unequal trial counts.
- Group test: cluster-based permutation on the window mean (cluster-forming *p* < .01, 2,000 permutations).

## Local effect

- Significant left-lateralised cluster: *p* = .0005 (further clusters *p* = .001, .0075), 1,153 sources, peak |*t*| = 10.2 ([](#fig3)A).
- Most consistently involved: left Heschl's gyrus (73% of sources significant, mean |*t*| = 5.1) and left STG (80%, mean |*t*| = 5.0).
- Auditory-cortex time course ([](#fig3)B): left peak 102 ms, *t*(14) = −7.5, *p* < .001, 15/15; right peak 108 ms, *t*(14) = −2.7, *p* = .017, 10/15. Source polarity is arbitrary; the MMN is plotted as negative.

## Controls for spread

- **No separable frontal generator.** Left insula/IFG was also in the cluster (*t*(14) = 4.2, peak 96 ms). Its peak did not follow auditory cortex (−6 ms, jackknife *t*(14) = −1.0, *p* = .34). Its amplitude relative to auditory cortex (41%) was within what a simulated source confined to auditory cortex produces through spread (33%, range 16–58%).
- **Medial temporal regions are spread.** Parahippocampal and entorhinal sources were > 80% significant bilaterally, while right auditory cortex was largely outside the cluster (13%). A simulated left Heschl's gyrus + STG source peaked in left STG, but eLORETA also assigned 83–95% of that peak amplitude to bilateral parahippocampal and entorhinal cortex (validation V7). These regions are not interpreted.

## Global effect (negative result)

- The 300–700 ms GD − GS difference formed only a small, scattered cluster: *p* = .0035, 278 sources, no region more than about 25% involved, peak |*t*| ≈ 2.7.
- Source-strength contrasts were positive almost everywhere for both effects, including the global effect at 80–140 ms, and matching trial counts did not change this. Source strength is not spatially specific here and is not interpreted.
- With a template head and 62 channels, no focal generator is claimed for the global effect.

:::{seealso}
Code: notebook section 3 ([](../notebooks/results_analyses_executed.ipynb)); `scripts/11_source_eloreta.py`, `scripts/19_figure3.py`, `scripts/20_frontal_generator_check.py`, `scripts/18_source_balanced.py`.
:::
