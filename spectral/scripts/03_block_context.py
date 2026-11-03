"""
03_block_context.py — does oscillatory power BEFORE the 5th sound depend on block context, and is
that context rebuilt after each block change?

Logic. Sounds 1–4 are physically identical in every quintlet (xxxx), so any difference before the
5th sound reflects context only. Blocks come in two types:
    homogeneous   standard = xxxxx (AAAAA / BBBBB blocks): a change at position 5 is unexpected
    alternating   standard = xxxxY (AAAAB / BBBBA blocks): a change at position 5 is expected
The 2x2 'interaction' in 02_tfr.py is exactly this contrast:
    block = 10·log10( (LS/GS · LD/GD) / (LS/GD · LD/GS) ) = homogeneous − alternating  (dB)

Confound to keep in mind: in alternating blocks most preceding quintlets ended in a local deviant,
so a pre-5th-sound difference could be carry-over of the previous quintlet's response rather than a
prediction. Two handles here:
  (i)  dynamics over trial_in_block: carry-over is fully present from trial 1 (the previous quintlet
       is already of the new type); a learned prediction should build up over several quintlets
  (ii) the jittered interval since the previous 5th sound (~1.47–1.8 s): carry-over should shrink
       with longer intervals, a prediction should not

Part A (from 02's condition TFRs; habituation excluded):
    block contrast at the ROIs over the whole epoch + a channel x freq x time cluster test restricted
    to the pre-5th-sound period. Leakage of the 5th-sound response back in time through the wavelet
    is excluded per frequency: only t <= −2.5·σt(f) is tested (σt = n_cycles / 2πf).
Part B (single trials, ALL quintlets including the first 10 of each block):
    per subject (cached): single-trial log power, 3–40 Hz, at the two ROIs (time-resolved) and per
    channel in a pre-5th-sound window per band.
    group: block difference as a function of trial_in_block (binned); per-subject OLS
      logP ~ block * log1p(trial_in_block) + block * interval_c + prev_global + block_index
    (statsmodels), coefficients tested across subjects (pingouin one-sample t).

Outputs -> spectral/figures/block_*.png, spectral/derivatives/block_context/
Usage:  python spectral/scripts/03_block_context.py [sub-01 ...]     single-trial power (Part B cache)
        python spectral/scripts/03_block_context.py --groupA         Part A
        python spectral/scripts/03_block_context.py --groupB         Part B
"""
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import mne
from mne.time_frequency import tfr_array_morlet
from mne.stats import permutation_cluster_1samp_test, combine_adjacency
import statsmodels.formula.api as smf
import pingouin as pg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------------- CONFIG
HERE = Path(__file__).resolve().parents[1]
EPO_DIR = HERE / "derivatives" / "epochs"
TFR_DIR = HERE / "derivatives" / "tfr"                 # from 02_tfr.py (Part A)
OUT_DIR = HERE / "derivatives" / "block_context"
FIG_DIR = HERE / "figures"
STEM = "ses-LGeffect_task-LocalGlobal"
EXCLUDE = ["sub-03"]
SEED = 42

ROIS = {   # a priori, as in the ERP scripts
    "frontocentral": ["Fz", "FC1", "FC2", "F1", "F2", "Cz"],
    "parietal": ["Pz", "P1", "P2", "CPz", "POz", "CP1", "CP2"],
}
ROI_COL = {"frontocentral": "#C44E52", "parietal": "#8172B3"}
BLOCK_COL = {"homogeneous": "#4C72B0", "alternating": "#DD8452"}

# Part A
F_MAX_A = 40.0
LEAK_SD = 2.5
N_PERM_A = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--nperm=")), 2000))
CLUSTER_P = 0.05

# Part B single-trial power
FREQS_B = np.geomspace(3, 40, 24)
N_CYCLES_B = np.clip(0.4 * FREQS_B, 3, 12)     # same rule as 02
DECIM_B = 10                                    # -> 50 Hz
CROP_B = (-0.8, 0.7)
CH_CHUNK = 8
BANDS = {   # pre-5th-sound windows chosen to stay clear of 5th-sound leakage (≥ 2.5 σt at band floor)
    "theta": ((4, 7), (-0.7, -0.4)),
    "alpha": ((8, 13), (-0.6, -0.15)),
    "beta": ((14, 30), (-0.6, -0.1)),
}
TIB_BINS = [0, 1, 2, 4, 7, 10, 20, 40, 200]          # trial_in_block bin edges (left-closed)
TIB_LABELS = ["0", "1", "2–3", "4–6", "7–9", "10–19", "20–39", "40+"]
# ------------------------------------------------------------------------------------

assert OUT_DIR.resolve().is_relative_to(HERE.resolve())
mne.set_log_level("ERROR")


def block_type(std):
    return np.where(std.str[3] == std.str[4], "homogeneous", "alternating")


def leak_mask(freqs, times, n_cycles):
    """True where power at (f, t) is free of post-5th-sound leakage: t <= −LEAK_SD·σt(f)."""
    sd = n_cycles / (2 * np.pi * freqs)
    return times[None, :] <= -LEAK_SD * sd[:, None]


# ============================================================ Part A
def group_a():
    files = sorted(f for f in TFR_DIR.glob("sub-*.npz") if f.stem not in EXCLUDE and ".tmp" not in f.name)
    d0 = np.load(files[0])
    freqs, times, ch = d0["freqs"], d0["times"], list(d0["ch_names"])
    assert list(d0["conditions"]) == ["LS/GS", "LD/GS", "LS/GD", "LD/GD"]
    fsel = freqs <= F_MAX_A
    freqs = freqs[fsel]
    n_cyc = np.clip(0.4 * freqs, 3, 12)                     # as in 02
    B = []
    for f in files:
        P = np.load(f)["total"].astype(float)[:, :, fsel]   # cond x ch x f x t
        ls_gs, ld_gs, ls_gd, ld_gd = P
        B.append(10 * np.log10((ls_gs * ld_gd) / (ls_gd * ld_gs)))
    B = np.array(B)                                         # subj x ch x f x t  (homogeneous − alternating)
    n = len(B)
    print(f"Part A: N = {n}")

    info = mne.io.read_info(TFR_DIR / "info.fif")
    adj_ch, names = mne.channels.find_ch_adjacency(info, "eeg")
    assert list(names) == ch
    valid = leak_mask(freqs, times, n_cyc)                  # f x t
    tsel = valid.any(0)
    X = B[:, :, :, tsel]
    vmask = valid[:, tsel]
    adjacency = combine_adjacency(adj_ch, vmask.shape[0], vmask.shape[1])

    def stat_fun(x):
        x = x.reshape(len(x), len(ch), *vmask.shape)
        sd = x.std(0, ddof=1)
        t = np.where(sd > 0, x.mean(0) / (sd / np.sqrt(len(x)) + 1e-30), 0.0)
        t[:, ~vmask] = 0.0
        return t.ravel()

    t0 = time.time()
    T, clus, pv, _ = permutation_cluster_1samp_test(X, n_permutations=N_PERM_A, adjacency=adjacency, tail=0,
                                                    stat_fun=stat_fun, seed=SEED, out_type="mask", n_jobs=1,
                                                    buffer_size=None)
    print(f"  cluster test: {len(clus)} clusters [{time.time() - t0:.0f} s]")
    rows = []
    for k, (cl, p) in enumerate(zip(clus, pv)):
        ci, fi, ti = np.where(cl)
        rows.append(dict(cluster=k, p=p, sign=int(np.sign(T[cl].mean())), n_ch=len(np.unique(ci)),
                         f_lo=freqs[fi].min(), f_hi=freqs[fi].max(),
                         t_lo=times[tsel][ti].min(), t_hi=times[tsel][ti].max(),
                         mean_db=X.mean(0)[cl].mean(),
                         channels=",".join(np.array(ch)[np.unique(ci)])))
    cdf = pd.DataFrame(rows).sort_values("p") if rows else pd.DataFrame()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cdf.to_csv(OUT_DIR / "partA_clusters.tsv", sep="\t", index=False)
    if len(cdf):
        with pd.option_context("display.width", 200, "display.max_colwidth", 60):
            print(cdf[cdf.p < 0.2].drop(columns="channels").to_string(index=False))

    # ---- figure: ROI TF maps (whole epoch, leak boundary drawn) + topography of best cluster
    sns.set_theme(context="paper", style="ticks")
    fig = plt.figure(figsize=(12, 4.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.8])
    mean = B.mean(0)
    lim = np.nanpercentile(np.abs([mean[[ch.index(c) for c in chs if c in ch]].mean(0)
                                   for chs in ROIS.values()]), 99)
    sd = n_cyc / (2 * np.pi * freqs)
    for j, (r, chs) in enumerate(ROIS.items()):
        ax = fig.add_subplot(gs[0, j])
        idx = [ch.index(c) for c in chs if c in ch]
        m = mean[idx].mean(0)
        pc = ax.pcolormesh(times, freqs, m, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
        ax.plot(-LEAK_SD * sd, freqs, color="k", ls="--", lw=0.9)
        sig = np.zeros_like(m, bool)
        for cl, p in zip(clus, pv):
            if p < CLUSTER_P:
                full = np.zeros((len(ch), len(freqs), len(times)), bool)
                full[:, :, tsel] = cl
                sig |= full[idx].any(0)
        if sig.any():
            ax.contour(times, freqs, sig.astype(float), levels=[0.5], colors="k", linewidths=1.0)
        ax.set_yscale("log")
        ax.set_yticks([4, 8, 13, 20, 30, 40], labels=["4", "8", "13", "20", "30", "40"])
        ax.axvline(0, color="k", lw=0.8)
        ax.set(xlabel="Time re 5th sound (s)", title=r)
        ax.title.set_color(ROI_COL[r])
        if j == 0:
            ax.set_ylabel("Frequency (Hz)")
        fig.colorbar(pc, ax=ax, label="dB (homogeneous − alternating)", shrink=0.85)
    ax = fig.add_subplot(gs[0, 2])
    fa = (freqs >= 8) & (freqs <= 13)
    va = valid[fa].all(0) & (times >= -0.6)
    v = B[:, :, fa][:, :, :, va].mean((2, 3))
    t = v.mean(0) / (v.std(0, ddof=1) / np.sqrt(n))
    mask = None
    if len(cdf) and (cdf.p < CLUSTER_P).any():
        best = clus[int(cdf.iloc[0].cluster)]
        mask = best.any((1, 2))
    mne.viz.plot_topomap(t, info, axes=ax, cmap="RdBu_r", vlim=(-5, 5), show=False, contours=0, mask=mask,
                         mask_params=dict(marker="o", markerfacecolor="w", markeredgecolor="k", markersize=4))
    ax.set_title("alpha 8–13 Hz, −0.6 s to leak bound\n(t; dots = best cluster channels)")
    fig.suptitle(f"Block context before the 5th sound — N = {n}; dashed: leakage bound (−{LEAK_SD}σt); "
                 f"outline: cluster p < {CLUSTER_P} (pre-5th-sound only)", x=0.01, ha="left")
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "block_A_context_tfr.png", dpi=200)
    plt.close(fig)


# ============================================================ Part B (per subject)
def subject_single_trial(subject):
    out = OUT_DIR / f"{subject}_single_trial.npz"
    if out.exists():
        try:
            np.load(out)["band"]
            print(f"{subject}: done, skipping")
            return
        except Exception:
            print(f"{subject}: cached file unreadable, recomputing")
    t0 = time.time()
    ep = mne.read_epochs(EPO_DIR / subject / f"{subject}_{STEM}_epo.fif", preload=True).pick("eeg")
    md = ep.metadata.reset_index(drop=True).copy()
    sf = ep.info["sfreq"]
    times = ep.times[::DECIM_B]
    keep = (times >= CROP_B[0] - 1e-9) & (times <= CROP_B[1] + 1e-9)
    times = times[keep]
    ch = ep.ch_names
    roi_idx = {r: [ch.index(c) for c in chs if c in ch] for r, chs in ROIS.items()}
    band_f = {b: (FREQS_B >= lo) & (FREQS_B <= hi) for b, ((lo, hi), _) in BANDS.items()}
    band_t = {b: (times >= w[0]) & (times <= w[1]) for b, (_, w) in BANDS.items()}

    X = ep.get_data()
    n_tr = len(X)
    roi_sum = {r: np.zeros((n_tr, len(FREQS_B), keep.sum())) for r in ROIS}
    band = np.zeros((n_tr, len(ch), len(BANDS)), dtype=np.float32)
    for c0 in range(0, len(ch), CH_CHUNK):
        cs = list(range(c0, min(c0 + CH_CHUNK, len(ch))))
        P = tfr_array_morlet(X[:, cs], sfreq=sf, freqs=FREQS_B, n_cycles=N_CYCLES_B, decim=DECIM_B,
                             output="power", n_jobs=1, zero_mean=True)[..., keep]     # tr x ch x f x t
        L = np.log10(P)
        for k, b in enumerate(BANDS):
            band[:, cs, k] = L[:, :, band_f[b]][:, :, :, band_t[b]].mean((2, 3))
        for r, idx in roi_idx.items():
            for i, c in enumerate(cs):
                if c in idx:
                    roi_sum[r] += L[:, i]
        del P, L
    roi = np.stack([roi_sum[r] / len(roi_idx[r]) for r in ROIS], 1).astype(np.float32)   # tr x roi x f x t

    # trial metadata: block type, interval since the previous 5th sound (same run & block, consecutive)
    md["block_type"] = block_type(md["block_standard"])
    q = pd.read_csv(EPO_DIR / subject / f"{subject}_{STEM}_quintlets.tsv", sep="\t").sort_values("seq_idx")
    q["prev_global"] = q["global"].shift()
    q["interval"] = q["t_fifth"].diff()
    same = (q["run"] == q["run"].shift()) & (q["block"] == q["block"].shift()) & (q["seq_idx"].diff() == 1)
    q.loc[~same, ["prev_global", "interval"]] = np.nan
    md = md.merge(q[["seq_idx", "prev_global", "interval"]], on="seq_idx", how="left")
    md["subject"] = subject

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, roi=roi, band=band, times=times, freqs=FREQS_B, ch_names=np.array(ch),
                        bands=np.array(list(BANDS)), rois=np.array(list(ROIS)))
    tmp.replace(out)
    md.to_csv(OUT_DIR / f"{subject}_single_trial_meta.tsv", sep="\t", index=False)
    print(f"{subject}: {n_tr} trials  [{time.time() - t0:.0f} s]", flush=True)


# ============================================================ Part B (group)
def group_b():
    subs = sorted(f.name.split("_")[0] for f in OUT_DIR.glob("sub-*_single_trial.npz")
                  if f.name.split("_")[0] not in EXCLUDE)
    print(f"Part B: N = {len(subs)}")
    d0 = np.load(OUT_DIR / f"{subs[0]}_single_trial.npz")
    ch, bands, rois = list(d0["ch_names"]), list(d0["bands"]), list(d0["rois"])
    times, freqs = d0["times"], d0["freqs"]
    roi_ch = {r: [ch.index(c) for c in chs if c in ch] for r, chs in ROIS.items()}

    curve_rows, coef_rows = [], []
    tf_early, tf_late = [], []
    for s in subs:
        d = np.load(OUT_DIR / f"{s}_single_trial.npz")
        md = pd.read_csv(OUT_DIR / f"{s}_single_trial_meta.tsv", sep="\t")
        md["tib_bin"] = pd.cut(md["trial_in_block"], TIB_BINS, right=False, labels=TIB_LABELS)
        md["hom"] = (md["block_type"] == "homogeneous").astype(float)
        md["log_tib"] = np.log1p(md["trial_in_block"])
        md["interval_c"] = md["interval"] - md["interval"].median()
        md["prev_gd"] = (md["prev_global"] == "GD").astype(float)
        for r, idx in roi_ch.items():
            for k, b in enumerate(bands):
                y = d["band"][:, idx, k].mean(1).astype(float)
                y = 10 * (y - y.mean())                                    # dB re subject mean
                df = md.assign(y=y)
                # curve: homogeneous − alternating per trial_in_block bin
                g = df.groupby(["tib_bin", "block_type"], observed=True)["y"].mean().unstack()
                for lab in TIB_LABELS:
                    if lab in g.index and g.loc[lab].notna().all():
                        curve_rows.append(dict(subject=s, roi=r, band=b, tib_bin=lab,
                                               diff=g.loc[lab, "homogeneous"] - g.loc[lab, "alternating"]))
                # per-subject OLS (trials with a valid previous quintlet in the same block)
                fit = smf.ols("y ~ hom * log_tib + hom * interval_c + prev_gd + block",
                              data=df.dropna(subset=["interval_c"])).fit()
                for term in ["hom", "hom:log_tib", "interval_c", "hom:interval_c", "prev_gd"]:
                    coef_rows.append(dict(subject=s, roi=r, band=b, term=term, beta=fit.params[term]))
        # time-resolved ROI TF: block difference early (trials 1–9) vs late (≥ 20) in block
        R = 10 * (d["roi"].astype(float) - d["roi"].astype(float).mean(0, keepdims=True))   # dB re mean
        hom = (md["block_type"] == "homogeneous").to_numpy()
        early = md["trial_in_block"].between(1, 9).to_numpy()
        late = (md["trial_in_block"] >= 20).to_numpy()
        tf_early.append(R[early & hom].mean(0) - R[early & ~hom].mean(0))
        tf_late.append(R[late & hom].mean(0) - R[late & ~hom].mean(0))

    curve = pd.DataFrame(curve_rows)
    coefs = pd.DataFrame(coef_rows)
    curve.to_csv(OUT_DIR / "partB_curve.tsv", sep="\t", index=False)
    coefs.to_csv(OUT_DIR / "partB_coefs.tsv", sep="\t", index=False)
    stats = []
    for (r, b, term), g in coefs.groupby(["roi", "band", "term"]):
        tt = pg.ttest(g["beta"], 0)
        stats.append(dict(roi=r, band=b, term=term, mean_beta=g["beta"].mean(), T=tt["T"].iloc[0],
                          p=tt.filter(regex="^p[-_]val$").iloc[0, 0], d=tt.filter(regex="^cohen[-_]d$").iloc[0, 0]))  # pingouin <0.6 / >=0.6 names
    stats = pd.DataFrame(stats)
    stats.to_csv(OUT_DIR / "partB_stats.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3g}".format):
        print(stats.to_string(index=False))

    # ---- figure 1: curves
    sns.set_theme(context="paper", style="ticks")
    fig, axes = plt.subplots(len(ROIS), len(bands), figsize=(11, 5.5), sharex=True, constrained_layout=True)
    for i, r in enumerate(ROIS):
        for j, b in enumerate(bands):
            ax = axes[i, j]
            c = curve[(curve.roi == r) & (curve.band == b)]
            sns.pointplot(data=c, x="tib_bin", y="diff", order=TIB_LABELS, errorbar="se", color=ROI_COL[r],
                          markersize=4, linewidth=1.4, ax=ax)
            ax.axhline(0, color="k", lw=0.6)
            ax.axvline(4.5, color="0.6", lw=0.6, ls=":")          # end of habituation (10 standards)
            ax.set(xlabel="Quintlet in block" if i == len(ROIS) - 1 else "",
                   ylabel=f"{r}\nhomogeneous − alternating (dB)" if j == 0 else "")
            ax.tick_params(axis="x", rotation=45)
            if i == 0:
                w = BANDS[b]
                ax.set_title(f"{b} {w[0][0]}–{w[0][1]} Hz, {w[1][0]} to {w[1][1]} s")
    fig.suptitle(f"Pre-5th-sound block-context effect across the block (mean ± SEM, N = {len(subs)}); "
                 "dotted: end of the 10 opening standards", x=0.01, ha="left")
    fig.savefig(FIG_DIR / "block_B_dynamics.png", dpi=200)
    plt.close(fig)

    # ---- figure 2: early vs late TF difference at the ROIs
    E, Lt = np.array(tf_early), np.array(tf_late)                 # subj x roi x f x t
    fig, axes = plt.subplots(len(ROIS), 2, figsize=(9, 6), sharex=True, sharey=True, constrained_layout=True)
    lim = np.nanpercentile(np.abs(np.r_[E.mean(0).ravel(), Lt.mean(0).ravel()]), 99)
    sd = N_CYCLES_B / (2 * np.pi * freqs)
    for i, r in enumerate(ROIS):
        for j, (lab, A) in enumerate([("quintlets 1–9", E), ("quintlets 20+", Lt)]):
            ax = axes[i, j]
            pc = ax.pcolormesh(times, freqs, A.mean(0)[i], cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
            ax.plot(-LEAK_SD * sd, freqs, color="k", ls="--", lw=0.9)
            ax.axvline(0, color="k", lw=0.8)
            ax.set_yscale("log")
            ax.set_yticks([4, 8, 13, 20, 30, 40], labels=["4", "8", "13", "20", "30", "40"])
            if i == 0:
                ax.set_title(lab)
            if j == 0:
                ax.set_ylabel(f"{r}\nFrequency (Hz)", color=ROI_COL[r])
            if i == len(ROIS) - 1:
                ax.set_xlabel("Time re 5th sound (s)")
    fig.colorbar(pc, ax=axes, label="dB (homogeneous − alternating)", shrink=0.6)
    fig.suptitle(f"Block-context effect early vs late in the block (N = {len(subs)}; all quintlets)", x=0.01, ha="left")
    fig.savefig(FIG_DIR / "block_B_early_late_tfr.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    if "--groupA" in sys.argv:
        group_a()
    elif "--groupB" in sys.argv:
        group_b()
    else:
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        subs = args or sorted(p.name for p in EPO_DIR.glob("sub-*") if p.is_dir())
        for s in subs:
            if s not in EXCLUDE:
                subject_single_trial(s)
