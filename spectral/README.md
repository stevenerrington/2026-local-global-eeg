# Spectral / time–frequency analyses (separate from the main pipeline)

Self-contained analyses that **read** the main pipeline's preprocessed epochs and **never write**
outside this folder.

- Input (read-only): `../derivatives/preproc/<sub>/<sub>_ses-LGeffect_task-LocalGlobal_epo.fif`
- Outputs: `spectral/derivatives/`, `spectral/figures/`
- Scripts: `spectral/scripts/NN_*.py`. Each one is self-contained and has its own CONFIG block.

## Inputs (as produced by `scripts/02_preprocess.py`)

- 500 Hz, 0.1–40 Hz, average reference, ICA-cleaned
- Epochs from −0.8 to 0.7 s around the 5th sound, baseline −0.1–0 s, peak-to-peak 150 µV rejection
- Metadata includes `local`, `global`, `condition`, `block`, `trial_in_block`, `habituation` and `since_gd`
  (hierarchical event names, so `epochs["LS/GD"]` etc. work)

## Constraints worth knowing before choosing methods

- **Short epochs (1.5 s).** Low frequencies (delta and theta) will be dominated by edge effects with
  wavelets. Keep the number of cycles small at low frequencies, crop the edges after the transform, or
  re-epoch with longer windows (see below).
- **40 Hz low-pass.** Gamma above ~35 Hz isn't available from these epochs.
- **Quintlet rhythm.** The SOA is ~157 ms (~6.4 Hz; 122 ms ≈ 8.2 Hz for sub-13/14, 152 ms for sub-15).
  The −0.8 to 0 s window contains the steady-state response to sounds 1–4, so the theta/alpha range
  is entangled with stimulus-locked power. Separate evoked and induced power (subtract the ERP)
  and note the per-subject SOA.
- **Pre-stimulus baseline overlaps sounds 1–4.** −0.1–0 s is not a silent baseline for TF.

`01_reepoch.py` fixes the first two (and leaves baselining until after the transform) by rebuilding the epochs from raw BIDS (see below).
Later spectral scripts should read `spectral/derivatives/epochs/`, not `../derivatives/preproc/`.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 0 | `scripts/00_check_inputs.py` | Loads each subject's epochs (read-only); reports sfreq, window, filter, channels, bad channels and trials per condition → `spectral/derivatives/inputs_check.tsv` |
| 1 | `scripts/01_reepoch.py` | Rebuilds the epochs from raw BIDS, independently of `../derivatives/preproc`: 0.1–100 Hz + 50/100 Hz notch, fresh ICA (1–100 Hz, Picard) with EOG + ICLabel removal (eye, muscle, heart, line, channel noise, p ≥ 0.8), −2.0 to 1.5 s around the 5th sound, no baseline, 150 µV peak-to-peak rejection only within −0.8 to 0.7 s. Labels and bad-channel detection are copied from `scripts/02_preprocess.py`. → `spectral/derivatives/epochs/<sub>/` (~0.5 GB per subject, single precision) |
| 2 | `scripts/02_tfr.py` | Morlet TFRs (3–90 Hz) per 2×2 condition: total, induced (total − ERP power), ITC (equal n). sub-03 excluded (muscle). `--group`: balanced dB contrasts (local, global, interaction), freq × time cluster tests in the frontocentral/parietal ROIs (0–0.7 s), band topographies → `spectral/figures/tfr_*.png`, `derivatives/tfr/group_clusters.tsv` |
| 3 | `scripts/03_block_context.py` | Block-context effect before the 5th sound (homogeneous xxxxx vs alternating xxxxY blocks; sounds 1–4 identical). `--groupA`: channel × freq × time cluster test on 02's TFRs, pre-5th-sound only, wavelet leakage excluded per frequency (`--nperm=`). Per subject: single-trial log power (3–40 Hz) at the ROIs and per channel/band (all quintlets, incl. the opening standards). `--groupB`: effect vs quintlet-in-block, per-subject OLS (block × log trial-in-block, block × interval since previous 5th sound, previous global deviant, block index) → `figures/block_*.png`, `derivatives/block_context/` |
| 4 | `scripts/04_pac.py` | Phase–amplitude coupling 0–0.7 s after the 5th sound: induced signal, Morlet phase 3–12 Hz × amplitude 30–90 Hz, Tort MI (18 bins), equal n per condition (3 draws), z vs 200 trial-shuffled surrogates; checkpoints per condition. `--group`: overall + local/global/interaction comodulograms with cluster tests per ROI, theta × gamma topographies → `figures/pac_*.png`, `derivatives/pac/` |

`01_reepoch.py` needs `mne-icalabel` and `onnxruntime` (`pip install mne-icalabel onnxruntime`).
