"""
02_preprocess.py — local–global EEG preprocessing (ds005429, ses-LGeffect).

Per subject:
  1. load all LG runs, set EOG channel types + 10-05 montage
  2. label quintlets from the stimulation sequence in sourcedata/ (the .vmrk triggers are all
     'R128' and carry no sound identity). Each run's events.tsv rows (anchored on the 5th sound)
     are aligned to the sequence by matching inter-quintlet intervals against the PTB sound
     timestamps (GlobLoc_critical_events*.mat) -> block, block standard, quintlet type,
     local (LS/LD) and global (GS/GD) labels
  3. resample (500 Hz), concatenate runs, band-pass 0.1–40 Hz
  4. detect bad channels (flat / robust-z variance / low neighbour correlation)
  5. average reference (bads excluded)
  6. ICA (Picard) on a 1–100 Hz copy; remove EOG-correlated (+ optional ICLabel) comps
  7. interpolate bad channels, re-reference to average
  8. epoch −0.8 to 0.7 s around the 5th sound, baseline −0.1–0 s, peak-to-peak rejection
     (optional autoreject)
  9. save epochs, ICA, quintlet table and an HTML QC report

Event names are hierarchical: "LS/GS/AAAAA", "LD/GD/BBBBA", ... so
epochs["LD"], epochs["GD"], epochs["LS/GD"] all work. Full labels live in epochs.metadata.

Note: the actual SOA is not 150 ms — it is ~157 ms for most participants, 122 ms for
sub-13/14 and 152 ms for sub-15. It is estimated per subject from the PTB timestamps.

Usage:  python scripts/02_preprocess.py [sub-01 sub-02 ...]
"""
from pathlib import Path
import sys
import gc
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from mne.preprocessing import ICA
from scipy.io import loadmat

# ----------------------------------------------------------------------------- CONFIG
ROOT = Path(__file__).resolve().parents[1]
BIDS_ROOT = ROOT / "data" / "bids"
OUT_ROOT = ROOT / "derivatives" / "preproc"
TASK, SESSION = "LocalGlobal", "LGeffect"

SOUNDS = {1: "A", 2: "B"}   # SEQUENCE codes: 1 = 350/700/1400 Hz chord, 2 = 500/1000/2000 Hz
MAX_ALIGN_ERR = 0.010       # s, max |interval mismatch| when aligning events.tsv to the sequence
N_HABITUATION = 10          # first N quintlets per block (always standards) flagged as habituation

EOG_CHANNELS = ["VEOG", "HEOG"]
# standard_1005 was renamed colin27_1005 in MNE 1.13
MONTAGE = "colin27_1005" if "colin27_1005" in mne.channels.get_builtin_montages() else "standard_1005"
SFREQ = 500.0
L_FREQ, H_FREQ = 0.1, 40.0

# bad channels
# (computed on median-referenced data in 1 s windows and summarised by the median across windows,
#  so neither the forehead reference nor sparse blinks make frontopolar sites look bad)
FLAT_THRESH = 0.5e-6     # V, median window std below this = flat
VAR_Z_THRESH = 4.0       # |robust z| of log median-window std
CORR_Z_THRESH = 4.0      # robust z (low side) of median-window 98th-pct |r| with other channels
CORR_ABS = 0.6           # ...and that correlation must also be below this (channels near the
                         # forehead reference, e.g. Fp2/F8, are less correlated but not bad)
VAR_SHARED_R = 0.95      # variance outliers with 98th-pct |r| >= this are shared signal, not bad
MAX_BAD_FRAC = 0.15

# ICA
ICA_N_COMPONENTS = 30
ICA_HP = 1.0             # ICA is fitted on a 1 Hz high-passed, 2x-decimated (250 Hz) copy of the
                         # 0.1–40 Hz data (low memory). ICLabel was trained on 1–100 Hz data, so
                         # treat its labels as approximate with this setup.
ICA_REJECT = dict(eeg=500e-6)
EOG_Z_THRESH = 3.0
USE_ICLABEL = False      # requires mne-icalabel; removes muscle/heart/line/channel-noise comps
ICLABEL_PROB = 0.8

# epochs
TMIN, TMAX = -0.8, 0.7   # re 5th-sound onset
BASELINE = (-0.1, 0.0)
REJECT_PTP = dict(eeg=150e-6)
USE_AUTOREJECT = False   # requires autoreject; replaces REJECT_PTP

SEED = 42
# ------------------------------------------------------------------------------------

mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", message="Not setting positions of 2 eog channels")
warnings.filterwarnings("ignore", message="No bad channels to interpolate")


# ============================================================ quintlet / condition table
def load_sequence(subject):
    """All quintlets in presentation order from sourcedata, with block labels and PTB times."""
    src = BIDS_ROOT / "sourcedata" / subject / f"ses-{SESSION}"
    seq = loadmat(next(src.glob("GlobLoc_stimulation_SEQUENCE*.mat")), squeeze_me=True)["SEQUENCE"]
    tz = loadmat(next(src.glob("GlobLoc_critical_events*.mat")), squeeze_me=True)["tz"]
    tz = tz[tz > 0]                                       # preallocated zeros at the end
    rows = []
    for b, blk in enumerate(seq, start=1):
        blk = np.atleast_2d(blk).astype(int)
        std = "".join(SOUNDS[x] for x in blk[0])          # each block opens with standards
        for i, q in enumerate(blk):
            rows.append(dict(block=b, trial_in_block=i,
                             quintlet="".join(SOUNDS[x] for x in q), block_standard=std))
    qs = pd.DataFrame(rows)
    if len(tz) != 5 * len(qs):
        raise RuntimeError(f"{subject}: {len(tz)} timestamps for {len(qs)} quintlets")
    qs["t5_ptb"] = tz[4::5]
    within = np.diff(tz.reshape(-1, 5), axis=1)
    soa = float(np.median(within))
    return qs, soa


def as_ranges(idx):
    """[1, 2, 3, 7] -> '1–3, 7'"""
    out, start = [], None
    for i, x in enumerate(idx):
        if start is None:
            start = x
        if i == len(idx) - 1 or idx[i + 1] != x + 1:
            out.append(f"{start}–{x}" if x != start else f"{x}")
            start = None
    return ", ".join(out)


def align_run(anchors, t5_ptb):
    """Offset into the sequence that best matches this run's inter-quintlet intervals."""
    da, n = np.diff(anchors), len(anchors)
    errs = np.array([np.max(np.abs(np.diff(t5_ptb[o:o + n]) - da))
                     for o in range(len(t5_ptb) - n + 1)])
    o = int(np.argmin(errs))
    return o, float(errs[o])


def build_quintlet_table(subject, vhdrs, sfreqs):
    """One row per recorded quintlet: run, 5th-sound time (s, run time) + labels."""
    qs, soa = load_sequence(subject)
    tables = []
    for vhdr, sf in zip(vhdrs, sfreqs):
        run = int(vhdr.name.split("_run-")[1].split("_")[0])
        ev = pd.read_csv(str(vhdr).replace("_eeg.vhdr", "_events.tsv"), sep="\t")
        anchors = (ev["begsample"].to_numpy() - ev["offset"].to_numpy() - 1) / sf   # 5th sound
        off, err = align_run(anchors, qs["t5_ptb"].to_numpy())
        if err > MAX_ALIGN_ERR:
            raise RuntimeError(f"{subject} run-{run}: alignment error {err * 1e3:.1f} ms")
        t = qs.iloc[off:off + len(anchors)].copy()
        t["seq_idx"] = np.arange(off, off + len(anchors))
        t["run"], t["t_fifth"] = run, anchors
        tables.append(t)
        print(f"  run-{run}: {len(anchors)} quintlets -> sequence {off}–{off + len(anchors) - 1} "
              f"(max interval error {err * 1e3:.1f} ms)")
    qt = pd.concat(tables, ignore_index=True)
    if qt["seq_idx"].duplicated().any():
        raise RuntimeError(f"{subject}: runs overlap in the sequence")
    missing = sorted(set(range(len(qs))) - set(qt["seq_idx"]))
    if missing:
        print(f"  {len(missing)} sequence quintlets not recorded: {as_ranges(missing)}")

    qt["habituation"] = qt["trial_in_block"] < N_HABITUATION
    qt["local"] = np.where(qt["quintlet"].str[3] == qt["quintlet"].str[4], "LS", "LD")
    qt["global"] = np.where(qt["quintlet"] == qt["block_standard"], "GS", "GD")
    qt["condition"] = qt["local"] + qt["global"]
    qt["event"] = qt["local"] + "/" + qt["global"] + "/" + qt["quintlet"]
    # useful for sequential-effect analyses: quintlets since the last global deviant in block
    since = []
    for _, g in qt.groupby("block", sort=False):
        c = np.nan
        for gl in g["global"]:
            since.append(c)
            c = 0 if gl == "GD" else (c + 1 if c == c else np.nan)
    qt["since_gd"] = since
    return qt, soa, missing


# ============================================================ channel QC
def find_bad_channels(raw):
    """Flat, high/low-variance and poorly correlated channels, from 1 s windows."""
    # raw is already low-passed at H_FREQ (< 50 Hz), so plain decimation to 100 Hz is alias-free
    # and avoids copying the full-rate data
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    step = int(round(raw.info["sfreq"] / 100.0))
    X = np.ascontiguousarray(raw._data[picks, ::step])
    X = mne.filter.filter_data(X, 100.0, 1.0, None, verbose=False)
    # robust (median) common reference first: with the forehead recording reference, channel
    # variance otherwise scales with distance from Fpz/Fp2 and frontopolar sites look "dead"
    X -= np.median(X, axis=0, keepdims=True)
    names = np.array(raw.ch_names)[picks]
    n_win = X.shape[1] // 100
    W = X[:, : n_win * 100].reshape(len(names), n_win, 100)        # ch x win x samples

    def robust_z(v, ref):
        med = np.median(v[ref])
        mad = 1.4826 * np.median(np.abs(v[ref] - med)) + 1e-12
        return (v - med) / mad

    sd = np.median(W.std(axis=2), axis=1)
    flat = sd < FLAT_THRESH
    z = robust_z(np.log(sd + 1e-20), ~flat)
    var_bad = np.zeros_like(flat)            # set below, once correlations are known

    corr_w = np.empty((len(names), n_win))
    for w in range(n_win):
        R = np.abs(np.corrcoef(W[:, w, :]))
        np.fill_diagonal(R, np.nan)
        corr_w[:, w] = np.nanpercentile(R, 98, axis=1)
    corr = np.nanmedian(corr_w, axis=1)
    corr_z = robust_z(corr, ~flat)
    corr_bad = (corr_z < -CORR_Z_THRESH) & (corr < CORR_ABS)
    # high/low variance only counts if the signal is channel-specific: frontopolar sites with lots
    # of (shared) ocular activity have high variance but |r| ~ 0.99 with neighbours -> left to ICA
    var_bad = (np.abs(z) > VAR_Z_THRESH) & (corr < VAR_SHARED_R)
    bad = flat | var_bad | corr_bad
    # cap the number of bads: keep the worst by |z|
    max_bad = int(MAX_BAD_FRAC * len(names))
    if bad.sum() > max_bad:
        order = np.argsort(-np.abs(z) * bad)
        bad = np.zeros_like(bad)
        bad[order[:max_bad]] = True
    qc = pd.DataFrame(dict(ch=names, sd_uV=sd * 1e6, logsd_z=z, corr98=corr, corr_z=corr_z,
                           flat=flat, var_bad=var_bad, corr_bad=corr_bad, bad=bad))
    return [str(n) for n in names[bad]], qc


# ============================================================ figures
def qc_erp_figure(epochs, subject, soa):
    """Local effect at Fz and global effect at Pz (30 Hz low-pass), seaborn on matplotlib."""
    sns.set_theme(context="notebook", style="ticks")
    ep = epochs.copy().filter(None, 30.0, verbose=False)
    ep = ep[~ep.metadata["habituation"].to_numpy()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
    specs = [("Fz", "local", {"LS": "#4C72B0", "LD": "#C44E52"}, "Local effect"),
             ("Pz", "global", {"GS": "#4C72B0", "GD": "#C44E52"}, "Global effect")]
    t = ep.times * 1e3
    for ax, (ch, col, pal, title) in zip(axes, specs):
        if ch not in ep.ch_names:
            continue
        for lab, c in pal.items():
            sel = ep[ep.metadata[col].to_numpy() == lab]
            if len(sel) == 0:
                continue
            d = sel.get_data(picks=ch)[:, 0, :] * 1e6
            m, se = d.mean(0), d.std(0, ddof=1) / np.sqrt(len(d))
            ax.plot(t, m, color=c, lw=1.6, label=f"{lab} (n={len(d)})")
            ax.fill_between(t, m - se, m + se, color=c, alpha=0.2, lw=0)
        ax.axvline(0, color="k", lw=0.8)
        for k in range(1, 5):
            ax.axvline(-k * soa * 1e3, color="0.7", lw=0.6, ls=":")
        ax.axhline(0, color="k", lw=0.5)
        ax.set(title=f"{title} — {ch}", xlabel="Time re 5th sound (ms)", ylabel="µV",
               xlim=(t[0], t[-1]))
        ax.legend(frameon=False, fontsize=9)
    sns.despine(fig)
    fig.suptitle(subject, x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    return fig


# ============================================================ main per-subject pipeline
def preprocess_subject(subject):
    eeg_dir = BIDS_ROOT / subject / f"ses-{SESSION}" / "eeg"
    vhdrs = sorted(eeg_dir.glob(f"{subject}_ses-{SESSION}_task-{TASK}_run-*_eeg.vhdr"))
    if not vhdrs:
        print(f"{subject}: no LG runs found in {eeg_dir}")
        return
    out = OUT_ROOT / subject
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{subject}_ses-{SESSION}_task-{TASK}"
    print(f"\n=== {subject}: {len(vhdrs)} run(s)")

    # ---- 1–2. load runs (downsampled channel by channel), quintlet table
    raws, sfreqs = [], []
    for vhdr in vhdrs:
        raw, sf_orig = load_resampled(vhdr)
        raws.append(raw)
        sfreqs.append(sf_orig)
    qt, soa, missing = build_quintlet_table(subject, vhdrs, sfreqs)
    print(f"  SOA {soa * 1e3:.1f} ms;  quintlets: {len(qt)};  blocks: {qt['block'].nunique()}")
    print("  " + qt.groupby("condition").size().to_string().replace("\n", "\n  "))

    # ---- 3. resample + events, concatenate, filter
    events_list = []
    for raw, (run, g) in zip(raws, qt.groupby("run", sort=True)):
        samp = raw.first_samp + np.round(g["t_fifth"].to_numpy() * raw.info["sfreq"]).astype(int)
        events_list.append(np.c_[samp, np.zeros_like(samp), g.index.to_numpy()])
    raw, events = mne.concatenate_raws(raws, events_list=events_list)
    del raws
    raw.filter(L_FREQ, H_FREQ, picks=["eeg", "eog"], phase="zero", fir_design="firwin")

    # ---- 4. bad channels
    bads, chan_qc = find_bad_channels(raw)
    raw.info["bads"] = bads
    chan_qc.to_csv(out / f"{stem}_desc-chanqc.tsv", sep="\t", index=False)
    print(f"  bad channels: {bads}")

    # ---- 5. average reference (bads excluded)
    raw.set_eeg_reference("average", projection=False)

    # ---- 6. ICA on 1–100 Hz copy
    raw_ica = make_ica_raw(raw)
    ica = ICA(n_components=ICA_N_COMPONENTS, method="picard",
              fit_params=dict(ortho=False, extended=True), max_iter="auto", random_state=SEED)
    ica.fit(raw_ica, picks="eeg", reject=ICA_REJECT, reject_by_annotation=True)
    eog_idx, eog_scores = ica.find_bads_eog(
        raw_ica, ch_name=[c for c in EOG_CHANNELS if c in raw_ica.ch_names],
        threshold=EOG_Z_THRESH, measure="zscore")
    exclude = list(eog_idx)
    ic_labels = None
    if USE_ICLABEL:
        from mne_icalabel import label_components
        ic_labels = label_components(raw_ica, ica, method="iclabel")
        for i, (lab, p) in enumerate(zip(ic_labels["labels"], ic_labels["y_pred_proba"])):
            if lab not in ("brain", "other") and p >= ICLABEL_PROB and i not in exclude:
                exclude.append(i)
    ica.exclude = sorted(exclude)
    print(f"  ICA: excluding {[int(i) for i in ica.exclude]} (EOG: {[int(i) for i in eog_idx]})")
    ica.apply(raw)
    ica.save(out / f"{stem}_ica.fif", overwrite=True)

    # ---- 7. interpolate, re-reference
    raw.interpolate_bads(reset_bads=True)
    raw.set_eeg_reference("average", projection=False)

    # ---- 8. epochs
    labels = qt.loc[events[:, 2]]
    event_id = {e: i for i, e in enumerate(sorted(labels["event"].unique()), start=1)}
    events[:, 2] = labels["event"].map(event_id).to_numpy()
    meta_cols = ["run", "seq_idx", "block", "block_standard", "trial_in_block", "habituation",
                 "since_gd", "quintlet", "local", "global", "condition", "t_fifth"]
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=TMIN, tmax=TMAX,
                        baseline=BASELINE, picks=["eeg", "eog"], metadata=labels[meta_cols].reset_index(drop=True),
                        reject_by_annotation=True, preload=True)
    n_pre = len(epochs)
    if USE_AUTOREJECT:
        from autoreject import AutoReject
        ar = AutoReject(random_state=SEED, n_jobs=-1, picks="eeg")
        epochs, reject_log = ar.fit_transform(epochs, return_log=True)
    else:
        epochs.drop_bad(reject=REJECT_PTP)
    print(f"  epochs kept: {len(epochs)}/{n_pre} "
          f"({100 * len(epochs) / max(n_pre, 1):.1f}%)")
    print("  " + epochs.metadata.groupby("condition").size().to_string().replace("\n", "\n  "))
    epochs.save(out / f"{stem}_epo.fif", overwrite=True)
    qt.to_csv(out / f"{stem}_quintlets.tsv", sep="\t", index=False)

    # ---- 9. report
    summary = dict(subject=subject, runs=len(vhdrs), soa_ms=round(soa * 1e3, 1),
                   n_quintlets=int(len(qt)), n_unrecorded=len(missing), n_blocks=int(qt["block"].nunique()),
                   bads=bads, ica_exclude=[int(i) for i in ica.exclude],
                   n_epochs_pre=int(n_pre), n_epochs=int(len(epochs)),
                   per_condition=epochs.metadata["condition"].value_counts().to_dict())
    (out / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))

    # (built incrementally, freeing each data object once it has been plotted — keeps peak
    # memory low enough for small machines/VMs)
    report = mne.Report(title=f"{subject} — local–global preprocessing")
    report.add_html(f"<pre>{json.dumps(summary, indent=2)}</pre>", title="Summary")
    report.add_html(chan_qc.round(3).to_html(index=False), title="Channel QC")
    report.add_raw(raw, title="Cleaned continuous", psd=True, butterfly=False)
    del raw
    report.add_ica(ica, title="ICA", inst=raw_ica, eog_scores=eog_scores, n_jobs=1)
    del raw_ica
    plt.close("all")
    gc.collect()
    report.add_figure(epochs.plot_drop_log(show=False), title="Epoch drop log")
    report.add_figure(qc_erp_figure(epochs, subject, soa), title="Local (Fz) & global (Pz) ERPs")
    report.save(out / f"{stem}_report.html", overwrite=True, open_browser=False)
    plt.close("all")
    print(f"  saved -> {out}")


def make_ica_raw(raw):
    """1 Hz high-passed copy of the (already 0.1–40 Hz, average-referenced) data at half rate."""
    step = 2                                   # 500 -> 250 Hz; alias-free after the 40 Hz low-pass
    info = raw.info.copy()
    with info._unlock():
        info["sfreq"] = raw.info["sfreq"] / step
    r = mne.io.RawArray(np.ascontiguousarray(raw._data[:, ::step]), info, verbose=False)
    r.set_annotations(raw.annotations)
    r.filter(ICA_HP, None, picks=["eeg", "eog"], fir_design="firwin", verbose=False)
    return r


def load_resampled(vhdr):
    """Read a BrainVision run and downsample to SFREQ one channel at a time.

    Avoids holding the full-rate recording in memory (2500/5000 Hz x 64 ch x ~15 min is
    1.2–2.4 GB in float64). Uses polyphase anti-alias filtering (scipy.signal.resample_poly).
    Returns the downsampled Raw and the original sampling rate.
    """
    from fractions import Fraction
    from scipy.signal import resample_poly
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


if __name__ == "__main__":
    subs = sys.argv[1:] or sorted(p.name for p in BIDS_ROOT.glob("sub-*") if p.is_dir())
    for s in subs:
        preprocess_subject(s)
