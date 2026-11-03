"""
21_dcm_export.py — export per-subject 2x2 condition ERPs for DCM in SPM12 (MATLAB).

For each subject: average the preprocessed epochs (average reference, 0.1–40 Hz) per condition
(LSGS, LDGS, LSGD, LDGD — in that order, which the MATLAB scripts assume), baseline −100–0 ms,
crop −100 to 700 ms, decimate 500 -> 250 Hz. EEG channels only (EOG excluded).

Output: derivatives/dcm/<sub>_erp.mat with
    data        nchan x ntime x 4 (µV)
    time        1 x ntime (s)
    label       nchan cell array of channel names (standard 10-10)
    condition   1 x 4 cell array
    n_trials    1 x 4
    fsample     scalar (Hz)
Then run scripts/dcm/dcm_01_prepare.m, dcm_02_fit.m, dcm_03_peb.m in MATLAB with SPM12 on the path.

Usage:  python scripts/21_dcm_export.py [sub-01 ...]
"""
from pathlib import Path
import sys

import numpy as np
from scipy.io import savemat

# ----------------------------------------------------------------------------- CONFIG
ROOT = Path(__file__).resolve().parents[1]
PRE_DIR = ROOT / "derivatives" / "preproc"
OUT_DIR = ROOT / "derivatives" / "dcm"
STEM = "ses-LGeffect_task-LocalGlobal"
CONDS = ["LSGS", "LDGS", "LSGD", "LDGD"]
BASELINE = (-0.1, 0.0)
CROP = (-0.1, 0.7)
DECIM = 2
# ------------------------------------------------------------------------------------


def main(subjects=None):
    import mne
    mne.set_log_level("ERROR")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sub in subjects or sorted(p.name for p in PRE_DIR.glob("sub-*") if p.is_dir()):
        ep = mne.read_epochs(PRE_DIR / sub / f"{sub}_{STEM}_epo.fif", preload=True)
        ep.pick("eeg")
        meta = ep.metadata.reset_index(drop=True)
        evs = []
        for c in CONDS:
            ev = ep[np.where(meta["condition"].eq(c).to_numpy())[0]].average()
            ev.apply_baseline(BASELINE).crop(*CROP)
            evs.append(ev)
        data = np.stack([e.data[:, ::DECIM] for e in evs], axis=-1) * 1e6          # chan x time x cond, µV
        times = evs[0].times[::DECIM]
        savemat(OUT_DIR / f"{sub}_erp.mat", dict(
            data=data, time=times[None, :], label=np.array(evs[0].ch_names, dtype=object)[:, None],
            condition=np.array(CONDS, dtype=object)[None, :],
            n_trials=np.array([[e.nave for e in evs]], float), fsample=float(ep.info["sfreq"] / DECIM)))
        print(f"{sub}: {data.shape}, trials {[e.nave for e in evs]}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
