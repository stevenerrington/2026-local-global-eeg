"""
00_check_inputs.py — sanity check of the preprocessed epochs used by the spectral analyses.

Reads (never writes) ../derivatives/preproc/<sub>/*_epo.fif from the main pipeline and reports,
per subject: sampling rate, epoch window, filter band, channel count, bad channels, total epochs
and epochs per 2x2 condition. Also prints the lowest frequency resolvable for a few wavelet
cycle counts given the epoch length.

Output: spectral/derivatives/inputs_check.tsv

Usage:  python spectral/scripts/00_check_inputs.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import mne

# ----------------------------------------------------------------------------- CONFIG
HERE = Path(__file__).resolve().parents[1]          # spectral/
PROJECT = HERE.parent                               # 2026-local-global-eeg/
PRE_DIR = PROJECT / "derivatives" / "preproc"       # read-only input
OUT_DIR = HERE / "derivatives"
STEM = "ses-LGeffect_task-LocalGlobal"
CONDITIONS = ["LS/GS", "LD/GS", "LS/GD", "LD/GD"]
N_CYCLES = [2, 3, 5, 7]
# ------------------------------------------------------------------------------------

assert OUT_DIR.resolve().is_relative_to(HERE.resolve()), "outputs must stay inside spectral/"


def summarise(subject):
    f = PRE_DIR / subject / f"{subject}_{STEM}_epo.fif"
    ep = mne.read_epochs(f, preload=False, verbose="error")
    row = dict(subject=subject, sfreq=ep.info["sfreq"], tmin=ep.tmin, tmax=ep.tmax,
               highpass=ep.info["highpass"], lowpass=ep.info["lowpass"],
               n_eeg=len(mne.pick_types(ep.info, eeg=True)), bads=",".join(ep.info["bads"]),
               n_epochs=len(ep))
    for c in CONDITIONS:
        try:
            row[c] = len(ep[c])
        except KeyError:
            row[c] = 0
    return row


def main():
    subjects = sorted(p.name for p in PRE_DIR.glob("sub-*") if p.is_dir())
    df = pd.DataFrame([summarise(s) for s in subjects])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "inputs_check.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df.to_string(index=False))

    dur = float((df["tmax"] - df["tmin"]).min())
    print(f"\nShortest epoch: {dur:.2f} s")
    for nc in N_CYCLES:
        # a Morlet wavelet's length is ~nc / f; it must fit in the epoch (full-length, before cropping)
        print(f"  n_cycles={nc}: lowest f whose wavelet fits in the epoch ~ {nc / dur:.1f} Hz "
              f"(~{2 * nc / dur:.1f} Hz to keep half the epoch free of edge effects)")


if __name__ == "__main__":
    main()
