"""
01_reepoch.py — rebuild wide-band, long epochs from raw BIDS for the spectral analyses.

Independent of ../derivatives/preproc: reads only raw BIDS + sourcedata, writes only to
spectral/derivatives/epochs/<sub>/. Labelling and bad-channel detection are copied from
scripts/02_preprocess.py, so quintlet labels and metadata are identical to the ERP pipeline.

What differs from the ERP pipeline:
  band      0.1–100 Hz (was 0.1–40), 50 + 100 Hz notch
  ICA       refitted on a 1–100 Hz copy at 500 Hz (Picard, extended); removes EOG-correlated
            components + ICLabel eye / muscle / heart / line / channel-noise components (p >= 0.8)
  epochs    −2.0 to 1.5 s around the 5th sound, no baseline (apply it after the TF transform).
            Peak-to-peak rejection (150 µV) is evaluated only within −0.8 to 0.7 s — the window
            that will be analysed — so the padding (which overlaps the neighbouring quintlets)
            does not drive trial loss.

Per subject:
  1. load runs, downsample to 500 Hz channel by channel
  2. label quintlets from sourcedata (sequence alignment, as in 02)
  3. concatenate, band-pass 0.1–100 Hz, notch 50/100 Hz
  4. bad channels (02's algorithm, on a 1–40 Hz QC copy)
  5. average reference (bads excluded)
  6. ICA on a 1–100 Hz copy + ICLabel
  7. interpolate bads, re-reference
  8. epoch, reject, save (single precision)

Outputs (spectral/derivatives/epochs/<sub>/):
  <stem>_epo.fif, <stem>_ica.fif, <stem>_desc-chanqc.tsv, <stem>_desc-iclabel.tsv,
  <stem>_quintlets.tsv, <stem>_summary.json

Usage:  python spectral/scripts/01_reepoch.py [sub-01 sub-02 ...]   (done subjects are skipped;
        add --overwrite to redo)
"""
from pathlib import Path
import sys
import gc
import json
import time
import warnings
from fractions import Fraction

import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA
from scipy.io import loadmat
from scipy.signal import resample_poly

# ----------------------------------------------------------------------------- CONFIG
HERE = Path(__file__).resolve().parents[1]                  # spectral/
PROJECT = HERE.parent
BIDS_ROOT = PROJECT / "data" / "bids"                       # read-only input
OUT_ROOT = HERE / "derivatives" / "epochs"
SESSION, TASK = "LGeffect", "LocalGlobal"
SOUNDS = {1: "A", 2: "B"}
MAX_ALIGN_ERR = 0.010
N_HABITUATION = 10
EOG_CHANNELS = ["VEOG", "HEOG"]
MONTAGE = "colin27_1005" if "colin27_1005" in mne.channels.get_builtin_montages() else "standard_1005"

SFREQ = 500.0
L_FREQ, H_FREQ = 0.1, 100.0
NOTCH = [50.0, 100.0]

# bad channels (identical to 02)
FLAT_THRESH = 0.5e-6
VAR_Z_THRESH = 4.0
CORR_Z_THRESH = 4.0
CORR_ABS = 0.6
VAR_SHARED_R = 0.95
MAX_BAD_FRAC = 0.15

# ICA
ICA_N_COMPONENTS = 30
ICA_HP = 1.0
ICA_REJECT = dict(eeg=500e-6)
EOG_Z_THRESH = 3.0
ICLABEL_REMOVE = ("eye blink", "muscle artifact", "heart beat", "line noise", "channel noise")
ICLABEL_PROB = 0.8

# epochs
TMIN, TMAX = -2.0, 1.5
BASELINE = None
REJECT_PTP = dict(eeg=150e-6)
REJECT_WIN = (-0.8, 0.7)
SEED = 42
# ------------------------------------------------------------------------------------

assert OUT_ROOT.resolve().is_relative_to(HERE.resolve()), "outputs must stay inside spectral/"
mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", message="Not setting positions of 2 eog channels")
warnings.filterwarnings("ignore", message="No bad channels to interpolate")


# ============================================================ quintlet table (as in 02)
def load_sequence(subject):
    src = BIDS_ROOT / "sourcedata" / subject / f"ses-{SESSION}"
    seq = loadmat(next(src.glob("GlobLoc_stimulation_SEQUENCE*.mat")), squeeze_me=True)["SEQUENCE"]
    tz = loadmat(next(src.glob("GlobLoc_critical_events*.mat")), squeeze_me=True)["tz"]
    tz = tz[tz > 0]
    rows = []
    for b, blk in enumerate(seq, start=1):
        blk = np.atleast_2d(blk).astype(int)
        std = "".join(SOUNDS[x] for x in blk[0])
        for i, q in enumerate(blk):
            rows.append(dict(block=b, trial_in_block=i,
                             quintlet="".join(SOUNDS[x] for x in q), block_standard=std))
    qs = pd.DataFrame(rows)
    if len(tz) != 5 * len(qs):
        raise RuntimeError(f"{subject}: {len(tz)} timestamps for {len(qs)} quintlets")
    qs["t5_ptb"] = tz[4::5]
    soa = float(np.median(np.diff(tz.reshape(-1, 5), axis=1)))
    return qs, soa


def align_run(anchors, t5_ptb):
    da, n = np.diff(anchors), len(anchors)
    errs = np.array([np.max(np.abs(np.diff(t5_ptb[o:o + n]) - da))
                     for o in range(len(t5_ptb) - n + 1)])
    o = int(np.argmin(errs))
    return o, float(errs[o])


def build_quintlet_table(subject, vhdrs, sfreqs):
    qs, soa = load_sequence(subject)
    tables = []
    for vhdr, sf in zip(vhdrs, sfreqs):
        run = int(vhdr.name.split("_run-")[1].split("_")[0])
        ev = pd.read_csv(str(vhdr).replace("_eeg.vhdr", "_events.tsv"), sep="\t")
        anchors = (ev["begsample"].to_numpy() - ev["offset"].to_numpy() - 1) / sf
        off, err = align_run(anchors, qs["t5_ptb"].to_numpy())
        if err > MAX_ALIGN_ERR:
            raise RuntimeError(f"{subject} run-{run}: alignment error {err * 1e3:.1f} ms")
        t = qs.iloc[off:off + len(anchors)].copy()
        t["seq_idx"] = np.arange(off, off + len(anchors))
        t["run"], t["t_fifth"] = run, anchors
        tables.append(t)
    qt = pd.concat(tables, ignore_index=True)
    if qt["seq_idx"].duplicated().any():
        raise RuntimeError(f"{subject}: runs overlap in the sequence")
    qt["habituation"] = qt["trial_in_block"] < N_HABITUATION
    qt["local"] = np.where(qt["quintlet"].str[3] == qt["quintlet"].str[4], "LS", "LD")
    qt["global"] = np.where(qt["quintlet"] == qt["block_standard"], "GS", "GD")
    qt["condition"] = qt["local"] + qt["global"]
    qt["event"] = qt["local"] + "/" + qt["global"] + "/" + qt["quintlet"]
    since = []
    for _, g in qt.groupby("block", sort=False):
        c = np.nan
        for gl in g["global"]:
            since.append(c)
            c = 0 if gl == "GD" else (c + 1 if c == c else np.nan)
    qt["since_gd"] = since
    return qt, soa


# ============================================================ loading
def load_resampled(vhdr):
    """BrainVision run -> 500 Hz, one channel at a time (polyphase anti-aliasing)."""
    src = mne.io.read_raw_brainvision(vhdr, eog=EOG_CHANNELS, preload=False)
    sf = src.info["sfreq"]
    frac = Fraction(SFREQ / sf).limit_denominator(1000)
    data = np.vstack([resample_poly(src.get_data(picks=[i])[0], frac.numerator, frac.denominator)
                      for i in range(len(src.ch_names))])
    info = src.info.copy()
    with info._unlock():
        info["sfreq"] = SFREQ
        info["lowpass"] = min(info["lowpass"], SFREQ / 2)
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_annotations(src.annotations)
    raw.set_montage(MONTAGE, on_missing="warn")
    return raw, sf


# ============================================================ channel QC (as in 02)
def find_bad_channels(raw):
    """02's algorithm. The data here are low-passed at 100 Hz, so the QC copy is low-passed
    at 40 Hz before decimating to 100 Hz (02 could decimate directly)."""
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    X = mne.filter.filter_data(raw.get_data(picks=picks), raw.info["sfreq"], None, 40.0, verbose=False)
    step = int(round(raw.info["sfreq"] / 100.0))
    X = np.ascontiguousarray(X[:, ::step])
    X = mne.filter.filter_data(X, 100.0, 1.0, None, verbose=False)
    X -= np.median(X, axis=0, keepdims=True)
    names = np.array(raw.ch_names)[picks]
    n_win = X.shape[1] // 100
    W = X[:, : n_win * 100].reshape(len(names), n_win, 100)
    del X

    def robust_z(v, ref):
        med = np.median(v[ref])
        mad = 1.4826 * np.median(np.abs(v[ref] - med)) + 1e-12
        return (v - med) / mad

    sd = np.median(W.std(axis=2), axis=1)
    flat = sd < FLAT_THRESH
    z = robust_z(np.log(sd + 1e-20), ~flat)
    corr_w = np.empty((len(names), n_win))
    for w in range(n_win):
        R = np.abs(np.corrcoef(W[:, w, :]))
        np.fill_diagonal(R, np.nan)
        corr_w[:, w] = np.nanpercentile(R, 98, axis=1)
    corr = np.nanmedian(corr_w, axis=1)
    corr_z = robust_z(corr, ~flat)
    corr_bad = (corr_z < -CORR_Z_THRESH) & (corr < CORR_ABS)
    var_bad = (np.abs(z) > VAR_Z_THRESH) & (corr < VAR_SHARED_R)
    bad = flat | var_bad | corr_bad
    max_bad = int(MAX_BAD_FRAC * len(names))
    if bad.sum() > max_bad:
        order = np.argsort(-np.abs(z) * bad)
        bad = np.zeros_like(bad)
        bad[order[:max_bad]] = True
    qc = pd.DataFrame(dict(ch=names, sd_uV=sd * 1e6, logsd_z=z, corr98=corr, corr_z=corr_z,
                           flat=flat, var_bad=var_bad, corr_bad=corr_bad, bad=bad))
    return [str(n) for n in names[bad]], qc


# ============================================================ main
def reepoch_subject(subject, overwrite=False):
    stem = f"{subject}_ses-{SESSION}_task-{TASK}"
    out = OUT_ROOT / subject
    if (out / f"{stem}_epo.fif").exists() and not overwrite:
        print(f"{subject}: done, skipping")
        return
    eeg_dir = BIDS_ROOT / subject / f"ses-{SESSION}" / "eeg"
    vhdrs = sorted(eeg_dir.glob(f"{stem}_run-*_eeg.vhdr"))
    if not vhdrs:
        print(f"{subject}: no LG runs in {eeg_dir}")
        return
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"\n=== {subject}: {len(vhdrs)} run(s)", flush=True)

    # 1–2. load + label
    raws, sfreqs = zip(*(load_resampled(v) for v in vhdrs))
    qt, soa = build_quintlet_table(subject, vhdrs, sfreqs)
    print(f"  SOA {soa * 1e3:.1f} ms; quintlets {len(qt)}", flush=True)

    # 3. concatenate + filter
    events_list = []
    for raw, (run, g) in zip(raws, qt.groupby("run", sort=True)):
        samp = raw.first_samp + np.round(g["t_fifth"].to_numpy() * raw.info["sfreq"]).astype(int)
        events_list.append(np.c_[samp, np.zeros_like(samp), g.index.to_numpy()])
    raw, events = mne.concatenate_raws(list(raws), events_list=events_list)
    del raws
    gc.collect()
    raw.filter(L_FREQ, H_FREQ, picks=["eeg", "eog"], phase="zero", fir_design="firwin")
    raw.notch_filter(NOTCH, picks=["eeg", "eog"], fir_design="firwin")

    # 4–5. bad channels, average reference
    bads, chan_qc = find_bad_channels(raw)
    raw.info["bads"] = bads
    chan_qc.to_csv(out / f"{stem}_desc-chanqc.tsv", sep="\t", index=False)
    raw.set_eeg_reference("average", projection=False)
    print(f"  bads: {bads}  [{time.time() - t0:.0f} s]", flush=True)

    # 6. ICA on a 1–100 Hz copy (ICLabel's expected input: 1–100 Hz, average reference)
    raw_ica = raw.copy().filter(ICA_HP, None, picks=["eeg", "eog"], fir_design="firwin", verbose=False)
    ica = ICA(n_components=ICA_N_COMPONENTS, method="picard",
              fit_params=dict(ortho=False, extended=True), max_iter="auto", random_state=SEED)
    ica.fit(raw_ica, picks="eeg", reject=ICA_REJECT, reject_by_annotation=True)
    eog_idx, _ = ica.find_bads_eog(raw_ica, ch_name=[c for c in EOG_CHANNELS if c in raw_ica.ch_names],
                                   threshold=EOG_Z_THRESH, measure="zscore")
    from mne_icalabel import label_components
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        icl = label_components(raw_ica, ica, method="iclabel")
    del raw_ica
    gc.collect()
    icl_df = pd.DataFrame(dict(component=np.arange(ica.n_components_), label=icl["labels"],
                               prob=icl["y_pred_proba"]))
    icl_df["eog_corr"] = icl_df["component"].isin(eog_idx)
    icl_df["exclude"] = icl_df["eog_corr"] | (icl_df["label"].isin(ICLABEL_REMOVE)
                                              & (icl_df["prob"] >= ICLABEL_PROB))
    icl_df.to_csv(out / f"{stem}_desc-iclabel.tsv", sep="\t", index=False)
    ica.exclude = icl_df.loc[icl_df["exclude"], "component"].tolist()
    ica.apply(raw)
    ica.save(out / f"{stem}_ica.fif", overwrite=True)
    excl = icl_df[icl_df["exclude"]]
    print(f"  ICA excluded {len(excl)}: " + ", ".join(f"{c}:{l}" for c, l in zip(excl.component, excl.label))
          + f"  [{time.time() - t0:.0f} s]", flush=True)

    # 7. interpolate, re-reference
    raw.interpolate_bads(reset_bads=True)
    raw.set_eeg_reference("average", projection=False)

    # 8. epochs
    labels = qt.loc[events[:, 2]]
    event_id = {e: i for i, e in enumerate(sorted(labels["event"].unique()), start=1)}
    events[:, 2] = labels["event"].map(event_id).to_numpy()
    meta_cols = ["run", "seq_idx", "block", "block_standard", "trial_in_block", "habituation",
                 "since_gd", "quintlet", "local", "global", "condition", "t_fifth"]
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=TMIN, tmax=TMAX, baseline=BASELINE,
                        picks=["eeg", "eog"], metadata=labels[meta_cols].reset_index(drop=True),
                        reject=REJECT_PTP, reject_tmin=REJECT_WIN[0], reject_tmax=REJECT_WIN[1],
                        reject_by_annotation=True, preload=True)
    del raw
    gc.collect()
    n_kept = len(epochs)
    epochs.save(out / f"{stem}_epo.fif", overwrite=True, fmt="single")
    qt.to_csv(out / f"{stem}_quintlets.tsv", sep="\t", index=False)
    summary = dict(subject=subject, runs=len(vhdrs), soa_ms=round(soa * 1e3, 1), n_quintlets=int(len(qt)),
                   n_epochs=int(n_kept), bads=bads, ica_exclude=[int(i) for i in ica.exclude],
                   ica_exclude_labels=excl["label"].tolist(),
                   per_condition=epochs.metadata["condition"].value_counts().to_dict(),
                   band=[L_FREQ, H_FREQ], notch=NOTCH, window=[TMIN, TMAX], reject_window=list(REJECT_WIN),
                   minutes=round((time.time() - t0) / 60, 1))
    (out / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  epochs kept {n_kept}/{len(qt)}  [{(time.time() - t0) / 60:.1f} min] -> {out}", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    subs = args or sorted(p.name for p in BIDS_ROOT.glob("sub-*") if p.is_dir())
    for s in subs:
        reepoch_subject(s, overwrite="--overwrite" in sys.argv)
