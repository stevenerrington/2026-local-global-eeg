# Local–global EEG

Human scalp-EEG analysis of the auditory local–global paradigm (Bekinschtein et al., 2009),
starting from the open dataset **ds005429** (Rutiku, Fiscone, Massimini & Sarasso, 2024, *EJN* 59:842–859).

## Dataset (ds005429, CC0)

- 15 healthy adults, BrainAmp, 62 scalp EEG + VEOG/HEOG, forehead reference, 2500 Hz
  (sub-13: 5000 Hz), hardware 0.1–250 Hz.
- Three sessions per participant: `ses-MMN` (Optimum-1), `ses-P3b` (learning oddball),
  `ses-LGeffect` (local–global, task `LocalGlobal`, 2 runs; sub-05 has 3).
- LG design: quintlets of 50 ms chords (A = 350/700/1400 Hz, B = 500/1000/2000 Hz),
  nominal SOA 150 ms (see below), inter-quintlet interval 800–1100 ms. 8 blocks; in each block one quintlet
  type (AAAAA, AAAAB, BBBBB or BBBBA) is the global standard (80%), the other the global
  deviant (20%). ≥10 standards open each block; ≥2 standards precede each deviant.
  Participants counted global deviants. ~1120 quintlets/participant.
- `*_events.tsv` are FieldTrip `trl` rows (one per quintlet, t = 0 at the **5th sound**,
  `offset` = −1 s), with dropped triggers already interpolated. Timing only — no labels.
- Every `.vmrk` trigger is `Response/R128` (no sound identity). **Labels come from
  `sourcedata/<sub>/ses-LGeffect/GlobLoc_stimulation_SEQUENCE*.mat`** (`SEQUENCE{block}`,
  n×5, 1 = A, 2 = B). Each run is aligned to the sequence by matching inter-quintlet intervals
  against the PTB sound timestamps (`GlobLoc_critical_events*.mat`): all 31 runs align with
  ≤5 ms error and an unambiguous margin (≥295 ms to the next-best offset).
- sub-05: 17 sequence quintlets were not recorded (#0 and #410–425, around the extra break).
- **Actual SOA ≠ 150 ms**: ~157 ms (most), 122 ms (sub-13, sub-14), 152 ms (sub-15).
  Estimated per subject from the PTB timestamps.

## Layout

```
data/bids/                 raw BIDS download (git-ignored)
derivatives/triggers/      trigger/structure reports from 01
derivatives/preproc/       per-subject epochs, ICA, trial tables, HTML reports from 02
scripts/                   numbered, self-contained pipeline scripts
notebooks/                 analysis notebooks
figures/
```

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 0 | `scripts/00_download.sh` | Downloads the LG session + sourcedata + metadata (~10 GB) |
| 1 | `scripts/01_inspect_triggers.py` | Per subject: blocks, block standards, SOA, marker counts, and the alignment of each run to the stimulation sequence (offset, error, margin). Writes `derivatives/triggers/alignment.tsv`. |
| 2 | `scripts/02_preprocess.py` | Resample → filter → bad channels → ICA (Picard; EOG components via VEOG/HEOG) → interpolate → average ref → epochs around the 5th sound with full LG metadata |
| 3 | `scripts/03_ideal_observer.py` | Leaky Beta–Bernoulli observers over the full presented sequence: LOCAL (position-agnostic transitions) and GLOBAL (5th-position change). Surprise, prediction error, prediction change, Bayesian surprise, precision; half-life grid; resets at block starts (no-reset variant written for comparison). → `derivatives/observer/` |
| 4 | `scripts/04_single_trial_glm.py` | Per subject: ROI timescale grid + CV model comparison; mass-univariate GLM `LD + GD + LDxGD + surprise (within-condition, split by standard/deviant)`. `--group`: spatio-temporal cluster permutation on every beta + figures. → `derivatives/glm/`, `figures/` |

Epochs use hierarchical event names, so `epochs["LD"]` = local deviants, `epochs["GD"]` =
global deviants, `epochs["LS/GD"]` = local-standard global deviants, etc. Metadata columns:
`run, seq_idx, block, block_standard, trial_in_block, habituation, since_gd, quintlet, local, global, condition, t_fifth`
(`since_gd` = standards since the last global deviant in the block, for sequential effects).

## Setup

```bash
conda env create -f environment.yml
conda activate lg-eeg
bash scripts/00_download.sh
python scripts/01_inspect_triggers.py
python scripts/02_preprocess.py            # all subjects
python scripts/02_preprocess.py sub-01     # one subject
```

## Defaults vs. the original paper

| | Rutiku et al. 2024 | Here (editable in each script's CONFIG) |
|---|---|---|
| Resample | 1000 Hz | 500 Hz |
| Filter | 0.1 Hz HW high-pass, 30 Hz low-pass | 0.1–40 Hz (low-pass to 30 Hz at ERP stage) |
| Bad channels / trials | manual + NoiseTools outliers (2 SD) | automatic (windowed variance / correlation, robust z); peak-to-peak 150 µV + optional autoreject |
| ICA | EEGLAB, manual | Picard, EOG-correlated components (optional ICLabel) |
| Reference | average | average |
| Epoch / baseline | −800 to 700 ms re 5th sound, −100–0 ms | same |
| Local effect | 0–400 ms, Fz peak ~105 ms | — |
| Global effect | 0–700 ms; Pz 300–500 & 500–700 ms | — |
