# Header 1 — analysis plan (written before fitting the real data)

Written 2026-10-27, before `06_model_comparison.py` was run on real EEG. The recovery simulation
was running at the time; one real-data fit of sub-01 (global component) had been run as a
smoke test.

**Question.** Do the local and global responses in the local–global paradigm reflect different
computations, running on different timescales ("clocks")?

**Data.** ds005429, 15 participants, ses-LGeffect, preprocessed epochs from `02_preprocess.py`.
All epochs kept by preprocessing, including habituation trials.

**Dependent variables (a priori, from Rutiku et al. 2024).**
- Local: mean amplitude, frontocentral (Fz, FC1, FC2, F1, F2, Cz), 80–140 ms after the 5th sound.
- Global: mean amplitude, parietal (Pz, P1, P2, CPz, POz, CP1, CP2), 300–700 ms after the 5th sound.

**Models.** All include an intercept and a linear drift over the session.
M0 categories (LD, GD, LDxGD) · M1 adaptation (tau 0.25–32 s) · M2 local transition learner
(half-life 10/50/200/inf transitions × block reset yes/no) · M3 global rule learner (half-life
2/5/10/20/inf quintlets × reset yes/no) · M4 = M2 + M3 · M5 = M1 + M3.

**Fitting and scoring.** Leave-one-block-out outer CV, parameters selected by inner
leave-one-block-out CV on the training blocks. Score = summed out-of-sample Gaussian
log-likelihood per subject.

**Group inference.** Random-effects BMS (Stephan et al. 2009) with protected exceedance
probabilities (Rigoux et al. 2014), separately for each component.

**Decision rules.**
1. A model "wins" a component only if its PXP > 0.90 **and** the recovery simulation shows it
   is recovered as the winner in ≥ 80% of simulated datasets when it generated the data. If the
   winner is not recoverable, report the set of models it is confused with, not a single winner.
2. "Different clocks" is supported if the selected half-lives (in seconds) of the best learner
   for each component differ across subjects (Wilcoxon signed-rank, two-sided, α = .05), or if
   the selected block-reset setting differs between components in the majority of subjects.
3. Any result outside these rules is reported as exploratory.

**Known caveats.** The block-reset idea for the global learner came from earlier exploratory
analyses of the same data; here reset is a free parameter selected inside CV rather than fixed.
Participants actively counted global deviants.
