"""
02_tfr.py — time–frequency analysis of the local and global effects (first pass).

Reads spectral/derivatives/epochs/<sub>/*_epo.fif (from 01_reepoch.py). Excludes sub-03
(heavy muscle contamination: 23/30 ICs labelled muscle, 78% of trials kept).

Per subject (cached in spectral/derivatives/tfr/<sub>.npz; done subjects are skipped):
  - habituation quintlets (first 10 per block) dropped, as in the ERP pipeline
  - Morlet wavelets, 3–90 Hz (36 log-spaced), n_cycles = clip(0.4·f, 3, 12), computed on the full
    −2.0 to 1.5 s epochs, then cropped to −0.8 to 0.7 s and decimated to 100 Hz (no edge effects:
    the longest wavelet, 3 Hz, is ±0.8 s)
  - per 2x2 condition (LS/GS, LD/GS, LS/GD, LD/GD):
      total   power of single trials, averaged
      induced power after subtracting that condition's ERP from each trial (computed exactly as
              total − power of the ERP)
      itc     inter-trial phase coherence on an equal number of trials per condition (the smallest
              condition's n, random draw), because ITC is biased by trial count
  - raw power is stored (no baseline): the pre-5th-sound period contains sounds 1–4, so there is
    no silent baseline. Effects are expressed as condition contrasts instead.

Group (--group):
  - contrasts per subject, conditions weighted equally (balanced 2x2):
      local        10·log10( mean(LD/GS, LD/GD) / mean(LS/GS, LS/GD) )   dB
      global       10·log10( mean(LS/GD, LD/GD) / mean(LS/GS, LD/GS) )   dB
      interaction  10·log10( (LD/GD · LS/GS) / (LS/GD · LD/GS) )         dB
      (ITC: the same contrasts as differences)
  - a-priori ROIs (as in the ERP scripts): frontocentral and parietal. Freq x time cluster
    permutation test (1-sample, 0–0.7 s) per ROI x contrast x measure
  - band topographies of each contrast in a local (0–0.3 s) and global (0.3–0.7 s) window

Figures -> spectral/figures/tfr_<measure>.png, tfr_topo_<measure>.png
Stats   -> spectral/derivatives/tfr/group_clusters.tsv

Usage:  python spectral/scripts/02_tfr.py [sub-01 ...]     per-subject TFRs
        python spectral/scripts/02_tfr.py --group          group stats + figures
"""
from pathlib import Path
import sys
import gc
import glob
import time

import numpy as np
import pandas as pd
import mne
from mne.time_frequency import tfr_array_morlet
from mne.stats import permutation_cluster_1samp_test
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------------- CONFIG
HERE = Path(__file__).resolve().parents[1]
EPO_DIR = HERE / "derivatives" / "epochs"
OUT_DIR = HERE / "derivatives" / "tfr"
FIG_DIR = HERE / "figures"
STEM = "ses-LGeffect_task-LocalGlobal"
EXCLUDE = ["sub-03"]

CONDITIONS = ["LS/GS", "LD/GS", "LS/GD", "LD/GD"]
FREQS = np.geomspace(3, 90, 36)
N_CYCLES = np.clip(0.4 * FREQS, 3, 12)
DECIM = 5                          # 500 -> 100 Hz
CROP = (-0.8, 0.7)
SEED = 42

ROIS = {   # a priori (Rutiku et al. 2024), identical to the ERP scripts
    "frontocentral": ["Fz", "FC1", "FC2", "F1", "F2", "Cz"],
    "parietal": ["Pz", "P1", "P2", "CPz", "POz", "CP1", "CP2"],
}
ROI_COL = {"frontocentral": "#C44E52", "parietal": "#8172B3"}
CONTRASTS = ["local", "global", "interaction"]
TEST_WIN = (0.0, 0.7)
N_PERM = 5000
CLUSTER_P = 0.05
BANDS = {"theta 3–8": (3, 8), "alpha 8–13": (8, 13), "beta 13–30": (13, 30), "gamma 30–90": (30, 90)}
TOPO_WIN = {"local": (0.0, 0.3), "global": (0.3, 0.7), "interaction": (0.3, 0.7)}
LINE = (47, 53)                    # notched; masked in figures
# ------------------------------------------------------------------------------------

assert OUT_DIR.resolve().is_relative_to(HERE.resolve())
mne.set_log_level("ERROR")


# ============================================================ per subject
def subject_tfr(subject):
    out = OUT_DIR / f"{subject}.npz"
    if out.exists():
        try:
            np.load(out)["itc"]
            print(f"{subject}: done, skipping")
            return
        except Exception:
            print(f"{subject}: cached file unreadable (interrupted run?), recomputing")
    t0 = time.time()
    ep = mne.read_epochs(EPO_DIR / subject / f"{subject}_{STEM}_epo.fif", preload=True)
    ep = ep[~ep.metadata["habituation"].to_numpy()].pick("eeg")
    sf = ep.info["sfreq"]
    times = ep.times[::DECIM]
    keep = (times >= CROP[0] - 1e-9) & (times <= CROP[1] + 1e-9)
    rng = np.random.default_rng(SEED)
    n_min = min(len(ep[c]) for c in CONDITIONS)

    res = {k: [] for k in ("total", "induced", "itc")}
    counts = []
    for c in CONDITIONS:
        X = ep[c].get_data()                                     # trials x ch x time
        counts.append(len(X))
        kw = dict(sfreq=sf, freqs=FREQS, n_cycles=N_CYCLES, decim=DECIM, n_jobs=1, zero_mean=True)
        res["total"].append(tfr_array_morlet(X, output="avg_power", **kw)[..., keep])
        # induced = total − evoked, exactly: mean|W(x_i − m)|² = mean|W x_i|² − |W m|² (W is linear)
        evoked = tfr_array_morlet(X.mean(0, keepdims=True), output="avg_power", **kw)[..., keep]
        res["induced"].append(res["total"][-1] - evoked)
        sub = X[np.sort(rng.choice(len(X), n_min, replace=False))]
        res["itc"].append(tfr_array_morlet(sub, output="itc", **kw)[..., keep])
        del X, sub
        gc.collect()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")                      # write then rename: never leaves a partial file
    np.savez_compressed(tmp, **{k: np.array(v, dtype=np.float32) for k, v in res.items()},
                        counts=np.array(counts), n_itc=n_min, freqs=FREQS, times=times[keep],
                        ch_names=np.array(ep.ch_names), conditions=np.array(CONDITIONS))
    tmp.replace(out)
    if not (OUT_DIR / "info.fif").exists():
        mne.io.write_info(OUT_DIR / "info.fif", ep.info)
    print(f"{subject}: n={counts}, itc n={n_min}  [{time.time() - t0:.0f} s]", flush=True)


# ============================================================ group
def contrasts(arr, measure):
    """arr: cond x ch x f x t (order = CONDITIONS) -> dict of contrast arrays (ch x f x t)."""
    ls_gs, ld_gs, ls_gd, ld_gd = arr
    if measure == "itc":
        return {"local": (ld_gs + ld_gd - ls_gs - ls_gd) / 2,
                "global": (ls_gd + ld_gd - ls_gs - ld_gs) / 2,
                "interaction": (ld_gd - ls_gd) - (ld_gs - ls_gs)}
    db = lambda a, b: 10 * np.log10(a / b)
    return {"local": db((ld_gs + ld_gd) / 2, (ls_gs + ls_gd) / 2),
            "global": db((ls_gd + ld_gd) / 2, (ls_gs + ld_gs) / 2),
            "interaction": db(ld_gd * ls_gs, ls_gd * ld_gs)}


def group():
    files = sorted(f for f in OUT_DIR.glob("sub-*.npz") if f.stem not in EXCLUDE and ".tmp" not in f.name)
    subs = [f.stem for f in files]
    d0 = np.load(files[0])
    freqs, times, ch = d0["freqs"], d0["times"], list(d0["ch_names"])
    info = mne.io.read_info(OUT_DIR / "info.fif")
    print(f"group: {len(subs)} subjects ({', '.join(subs)})")

    C = {m: {c: [] for c in CONTRASTS} for m in ("total", "induced", "itc")}
    for f in files:
        d = np.load(f)
        assert list(d["ch_names"]) == ch
        for m in C:
            for c, v in contrasts(d[m].astype(float), m).items():
                C[m][c].append(v)
    C = {m: {c: np.array(v) for c, v in cs.items()} for m, cs in C.items()}   # subj x ch x f x t

    roi_idx = {r: [ch.index(x) for x in chs if x in ch] for r, chs in ROIS.items()}
    tw = (times >= TEST_WIN[0]) & (times <= TEST_WIN[1])
    rows = []
    sns.set_theme(context="paper", style="ticks")
    for m in C:
        unit = "ΔITC" if m == "itc" else "dB"
        fig, axes = plt.subplots(len(ROIS), len(CONTRASTS), figsize=(11, 6.2), sharex=True, sharey=True,
                                 constrained_layout=True)
        for i, (r, idx) in enumerate(roi_idx.items()):
            for j, c in enumerate(CONTRASTS):
                X = C[m][c][:, idx].mean(1)                          # subj x f x t
                mean = X.mean(0)
                T, clus, pv, _ = permutation_cluster_1samp_test(X[:, :, tw], n_permutations=N_PERM, tail=0,
                                                                 seed=SEED, out_type="mask", n_jobs=1)
                sig = np.zeros_like(mean, dtype=bool)
                for cl, p in zip(clus, pv):
                    full = np.zeros_like(mean, dtype=bool)
                    full[:, tw] = cl
                    fi, ti = np.where(full)
                    rows.append(dict(measure=m, roi=r, contrast=c, p=p, sign=np.sign(T[cl].mean()),
                                     f_lo=freqs[fi].min(), f_hi=freqs[fi].max(),
                                     t_lo=times[ti].min(), t_hi=times[ti].max(),
                                     mean_effect=mean[full].mean()))
                    if p < CLUSTER_P:
                        sig |= full
                ax = axes[i, j]
                lim = np.nanpercentile(np.abs(mean), 99)
                plot = mean.copy()
                plot[(freqs >= LINE[0]) & (freqs <= LINE[1])] = np.nan
                pc = ax.pcolormesh(times, freqs, plot, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
                if sig.any():
                    ax.contour(times, freqs, sig.astype(float), levels=[0.5], colors="k", linewidths=1.0)
                ax.set_yscale("log")
                ax.set_yticks([4, 8, 13, 30, 60, 90], labels=["4", "8", "13", "30", "60", "90"])
                ax.axvline(0, color="k", lw=0.8)
                ax.axvspan(TEST_WIN[0], TEST_WIN[1], ymin=0, ymax=0.015, color="0.3")
                fig.colorbar(pc, ax=ax, label=unit, shrink=0.85)
                if i == 0:
                    ax.set_title(c.capitalize())
                if j == 0:
                    ax.set_ylabel(f"{r}\nFrequency (Hz)", color=ROI_COL[r])
                if i == len(ROIS) - 1:
                    ax.set_xlabel("Time re 5th sound (s)")
        fig.suptitle(f"{m.capitalize()} — {'ITC difference' if m == 'itc' else 'power contrast (dB)'}; "
                     f"N = {len(subs)}; outline: cluster p < {CLUSTER_P}", x=0.01, ha="left")
        FIG_DIR.mkdir(exist_ok=True)
        fig.savefig(FIG_DIR / f"tfr_{m}.png", dpi=200)
        plt.close(fig)

        # band topographies
        fig, axes = plt.subplots(len(CONTRASTS), len(BANDS), figsize=(9, 7), constrained_layout=True)
        for i, c in enumerate(CONTRASTS):
            w = (times >= TOPO_WIN[c][0]) & (times <= TOPO_WIN[c][1])
            for j, (b, (lo, hi)) in enumerate(BANDS.items()):
                fb = (freqs >= lo) & (freqs < hi) & ~((freqs >= LINE[0]) & (freqs <= LINE[1]))
                v = C[m][c][:, :, fb][:, :, :, w].mean((2, 3))      # subj x ch
                t = v.mean(0) / (v.std(0, ddof=1) / np.sqrt(len(v)))
                im, _ = mne.viz.plot_topomap(t, info, axes=axes[i, j], cmap="RdBu_r", vlim=(-5, 5), show=False,
                                             contours=0, sensors=False)
                if i == 0:
                    axes[i, j].set_title(b)
                if j == 0:
                    axes[i, j].set_ylabel(f"{c}\n{TOPO_WIN[c][0]:.1f}–{TOPO_WIN[c][1]:.1f} s")
        fig.colorbar(im, ax=axes, label="t (N−1 df)", shrink=0.5)
        fig.suptitle(f"{m.capitalize()} — band topographies (uncorrected t)", x=0.01, ha="left")
        fig.savefig(FIG_DIR / f"tfr_topo_{m}.png", dpi=200)
        plt.close(fig)

    df = pd.DataFrame(rows).sort_values(["measure", "roi", "contrast", "p"])
    df.to_csv(OUT_DIR / "group_clusters.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3g}".format):
        print(df[df["p"] < 0.1].to_string(index=False))


if __name__ == "__main__":
    if "--group" in sys.argv:
        group()
    else:
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        subs = args or sorted(p.name for p in EPO_DIR.glob("sub-*") if p.is_dir())
        for s in subs:
            if s not in EXCLUDE:
                subject_tfr(s)
