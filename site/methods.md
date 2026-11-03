---
title: Methods
subtitle: Dataset, preprocessing, learner models, inference, source localisation and DCM
---

## Dataset

Open dataset ds005429 {cite:p}`Rutiku2024` (CC0, [OpenNeuro](https://openneuro.org/datasets/ds005429)). The recordings and paradigm:

- **Participants and recording:** 15 healthy adults. BrainAmp, 62 scalp EEG channels plus VEOG/HEOG, forehead reference, 2500 Hz (sub-13: 5000 Hz).
- **Stimuli:** quintlets of 50 ms chords (A = 350/700/1400 Hz, B = 500/1000/2000 Hz), inter-quintlet interval 800–1100 ms.
- **Blocks:** 8 blocks. In each block one quintlet type (AAAAA, AAAAB, BBBBB or BBBBA) is the global standard (80%) and the other the global deviant (20%). At least 10 standards open each block, and at least 2 standards precede each deviant. Participants counted global deviants. About 1,120 quintlets per participant.
- **Labels:** triggers carry no sound identity. Labels come from the stimulation sequence files in `sourcedata/`. Each run was aligned to its sequence by matching inter-quintlet intervals to the Psychtoolbox timestamps: all 31 runs aligned with ≤ 5 ms error.
- **Timing:** the actual SOA was about 157 ms for most participants (122 ms for sub-13 and sub-14, 152 ms for sub-15), estimated per participant.

## Preprocessing

:::{list-table}
:header-rows: 1
* - Step
  - Setting
* - Resample
  - 500 Hz
* - Filter
  - 0.1–40 Hz (30 Hz low-pass at the ERP stage)
* - Bad channels
  - automatic (windowed variance / correlation, robust *z*), interpolated
* - Artefacts
  - Picard ICA; EOG components identified from VEOG/HEOG
* - Reference
  - average (linked mastoids as a control)
* - Epochs
  - −800 to 700 ms around the 5th sound; baseline −100 to 0 ms
* - Trial rejection
  - peak-to-peak 150 µV
:::

## Regions of interest and windows

Fixed a priori from the literature, not from these data:

- **Local effect (MMN):** frontocentral ROI (Fz, FC1, FC2, F1, F2, Cz), 80–140 ms.
- **Global effect (P3b):** parietal ROI (Pz, P1, P2, CPz, POz, CP1, CP2), 300–700 ms.

(learners)=
## Learner models

Each model predicts single-trial ROI amplitude $y_t$ from one regressor $x_t$, with a session-drift covariate:

```{math}
:label: eq-glm
y_t = \beta_0 + \beta_1\,\mathrm{drift}_t + \beta_2\,z(x_t) + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \sigma^2)
```

### Leaky Beta–Bernoulli learner (M2, M3)

The learner observes a binary outcome stream $o_t \in \{0, 1\}$ (1 = change) and keeps leaky counts with half-life $h$:

```{math}
:label: eq-learner
\lambda = 2^{-1/h}, \qquad n_1 \leftarrow \lambda n_1 + o_t, \qquad n_0 \leftarrow \lambda n_0 + (1 - o_t)
```

Before each outcome it predicts $\mu_t = (n_1 + 1)/(n_1 + n_0 + 2)$ (uniform prior), and the regressor is the Shannon surprise

```{math}
:label: eq-surprise
S_t = -\log_2 p(o_t) = -\log_2\!\left[\mu_t^{\,o_t}(1 - \mu_t)^{1 - o_t}\right]
```

With $h = \infty$ this is the exact Bayesian posterior predictive (checked in validation V1). Optionally the counts reset to zero at each block start.

- **M2 local:** outcomes are all sound-to-sound transitions (four per quintlet; 1 = the sound changed). The regressor is the surprise of the 4 → 5 transition. Half-life ∈ {10, 50, 200, ∞} transitions × {reset, no reset}.
- **M3 global:** one outcome per quintlet (1 = the 5th sound differs from the first four). Half-life ∈ {2, 5, 10, 20, ∞} quintlets × {reset, no reset}.

```{literalinclude} ../scripts/25_dcm_surprise_export.py
:caption: The learner, as used in every script
:language: python
:lines: 32-43
```

### Other models

- **M0 categories:** LD, GD and LD × GD indicator regressors (the 2 × 2 design, no learning).
- **M1 adaptation:** stimulus-specific adaptation. Each sound adds 1 to its own trace, and traces decay as $e^{-\Delta t/\tau}$ with $\tau$ ∈ {0.25, 0.5, 1, 2, 4, 8, 16, 32} s. The regressor is the trace of the 5th sound just before it plays.
- **M4 / M5:** M2 + M3 and M1 + M3 (preregistered combinations).

### Inference

- **Nested leave-one-block-out cross-validation.** For each held-out block, the half-life/reset (or τ) is chosen by an inner leave-one-block-out over the seven training blocks. The model is then fitted by OLS and scored by the Gaussian log-likelihood of the held-out block. Scores are summed over blocks.
- **Group model selection:** random-effects BMS on the held-out log-likelihoods {cite:p}`Stephan2009`, with PXP and BOR {cite:p}`Rigoux2014`.
- **Model recovery:** 20 simulated datasets per generating model, using each participant's fitted parameters, betas and noise, passed through the full pipeline.

## Statistics

- Paired / one-sample *t*-tests across participants (two-sided, df = 14); 2 × 2 repeated-measures ANOVA (pingouin).
- Cluster-based permutation, temporal and spatio-temporal {cite:p}`Maris2007`: cluster-forming *p* < .05, 2,000 permutations, sign-flip. Cluster extents are not interpreted as onsets {cite:p}`Sassenhagen2019`.
- Surprise–amplitude relations: per-participant slopes on surprise thirds (binned) and single-trial regressions. A within-condition version removes 2 × 2 cell means; an added-variable version tests one learner beyond the other.

## Source localisation

- eLORETA {cite:p}`PascualMarqui2007` (MNE-Python) on the MNE "sample" template: three-layer BEM, oct-6 source space, standard 10-05 positions. Loose orientation 0.2, depth weighting 0.8, λ² = 1/9.
- Noise covariance per participant from the pre-sequence silence.
- Group statistic: signed source difference, window mean, cluster permutation (cluster-forming *p* < .01, 2,000 permutations). Regions from the Desikan–Killiany atlas.

## Dynamic causal modelling

- DCM for ERP in SPM12. Eight sources (MNI coordinates):

  :::{list-table}
  :header-rows: 1
  * - Source
    - Left
    - Right
  * - A1
    - (−42, −22, 7)
    - (46, −14, 8)
  * - STG
    - (−61, −32, 8)
    - (59, −25, 8)
  * - IFG
    - (−46, 20, 8)
    - (46, 20, 8)
  * - IPL
    - (−48, −50, 42)
    - (48, −50, 42)
  :::
- Driving input at A1 (onset 60 ms, 16 ms duration).
- Connections: forward A1 → STG → IFG/IPL; reciprocal backward; lateral IFG ↔ IPL. ECD spatial model, 8 modes, Hanning window.
- Condition effects (B matrices) on all forward and backward connections, with surprise or interaction coding ([](results-dcm.md)). Windows 0–600 ms and 0–250 ms.
- Group: PEB on B, family-wise BMR (none / forward / backward / both per effect), and per-connection BMR {cite:p}`Friston2016`.
