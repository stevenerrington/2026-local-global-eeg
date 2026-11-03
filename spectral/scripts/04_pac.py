"""
04_pac.py — phase–amplitude coupling (PAC) after the 5th sound.

Question: is gamma amplitude coupled to the phase of slower activity after the 5th sound, and does
that coupling change for local and/or global deviants?

Per subject (cached in spectral/derivatives/pac/<sub>.npz):
  - epochs from 01_reepoch.py (−2.0 to 1.5 s, 0.1–100 Hz); habituation quintlets dropped; all 62 channels
  - INDUCED signal: each condition's ERP is subtracted from its trials, so coupling is not driven by
    the sharp, stimulus-locked waveform of the evoked response (the main source of spurious PAC)
  - checkpoints each condition (<sub>_part<k>.npz), so an interrupted run resumes
  - equal trial numbers: N_DRAW random draws of n_min trials (smallest condition) per condition;
    PAC is computed per draw and averaged (MI depends on the amount of data)
  - phase: complex Morlet, 3–12 Hz (1 Hz steps, 3 cycles)
    amplitude: complex Morlet, 30–90 Hz (5 Hz steps), n_cycles = max(3, f/6), i.e. a spectral SD of
    ~6 Hz, so the envelope carries modulations up to ~12 Hz (≥ the fastest phase frequency)
    computed on the full epoch, then cut to the analysis window (0–0.7 s re 5th sound): no filter
    edge effects (the 3 Hz phase wavelet's ±2.5 SD support is ±0.4 s)
  - modulation index (Tort et al. 2010; 18 phase bins), samples pooled over trials in the window
  - surrogates: amplitude from one trial paired with phase from another (random trial permutation,
    N_SURR times) within the same condition. This keeps each signal's spectrum and any residual
    stimulus-locked structure but destroys within-trial coupling. Output: MI z-scored against the
    surrogates (and raw MI).

Group (--group):
  - overall coupling: comodulogram of z (mean over conditions) per ROI, cluster test vs 0
  - condition contrasts on z (balanced 2x2): local = LD − LS, global = GD − GS, interaction
    (LD/GD − LS/GD) − (LD/GS − LS/GS); phase x amplitude cluster permutation test per ROI
  - topographies for theta phase (4–8 Hz) x gamma amplitude (30–80 Hz)

Outputs -> spectral/figures/pac_*.png, spectral/derivatives/pac/
Usage:  python spectral/scripts/04_pac.py [sub-01 ...]
        python spectral/scripts/04_pac.py --group
"""
from pathlib import Path
import sys
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
OUT_DIR = HERE / "derivatives" / "pac"
FIG_DIR = HERE / "figures"
STEM = "ses-LGeffect_task-LocalGlobal"
EXCLUDE = ["sub-03"]
SEED = 42

CONDITIONS = ["LS/GS", "LD/GS", "LS/GD", "LD/GD"]
F_PHASE = np.arange(3, 13, 1.0)
NC_PHASE = 3.0
F_AMP = np.arange(30, 91, 5.0)
NC_AMP = np.maximum(3.0, F_AMP / 6.0)
DECIM = 2                       # 500 -> 250 Hz (after the wavelet transform)
WIN = (0.0, 0.7)                # s re 5th sound
N_BINS = 18
N_SURR = 200
N_DRAW = 3

ROIS = {   # a priori, as in the ERP scripts
    "frontocentral": ["Fz", "FC1", "FC2", "F1", "F2", "Cz"],
    "parietal": ["Pz", "P1", "P2", "CPz", "POz", "CP1", "CP2"],
}
ROI_COL = {"frontocentral": "#C44E52", "parietal": "#8172B3"}
CONTRASTS = ["local", "global", "interaction"]
TOPO_PHASE = (4, 8)
TOPO_AMP = (30, 80)
N_PERM = 5000
CLUSTER_P = 0.05
# ------------------------------------------------------------------------------------

assert OUT_DIR.resolve().is_relative_to(HERE.resolve())
mne.set_log_level("ERROR")


def tort_mi(mean_amp):
    """mean_amp: (..., n_bins) -> Tort modulation index."""
    p = mean_amp / mean_amp.sum(-1, keepdims=True)
    return (np.log(N_BINS) + np.sum(p * np.log(p + 1e-30), -1)) / np.log(N_BINS)


def pac_condition(phase, amp, rng):
    """phase: trials x f_phase x samples (radians); amp: trials x f_amp x samples.
    Returns MI (f_phase x f_amp) and surrogate mean/sd from trial-permuted pairings."""
    n, n_fp, n_s = phase.shape
    n_fa = amp.shape[1]
    bins = np.minimum(((phase + np.pi) / (2 * np.pi) * N_BINS).astype(int), N_BINS - 1)
    perms = np.array([rng.permutation(n) for _ in range(N_SURR)])
    mi = np.empty((n_fp, n_fa))
    s_mean = np.empty((n_fp, n_fa))
    s_sd = np.empty((n_fp, n_fa))
    A = amp.transpose(1, 0, 2).reshape(n_fa * n, n_s).astype(np.float32)        # (fa·j) x s
    for p in range(n_fp):
        onehot = np.zeros((n, n_s, N_BINS), dtype=np.float32)
        np.put_along_axis(onehot, bins[:, p, :, None], 1.0, axis=2)
        counts = onehot.sum(1)                                                  # i x k
        # S[fa, j, i, k] = sum_s amp[j, fa, s] · 1[phase_i(s) in bin k]
        S = (A @ onehot.transpose(1, 0, 2).reshape(n_s, n * N_BINS)).reshape(n_fa, n, n, N_BINS)
        tot_counts = counts.sum(0)                                              # k (same for any pairing)
        real = S[:, np.arange(n), np.arange(n), :].sum(1) / tot_counts          # fa x k
        mi[p] = tort_mi(real)
        sur = S[:, perms, np.arange(n)[None, :], :].sum(2) / tot_counts         # fa x surr x k
        smi = tort_mi(sur)
        s_mean[p], s_sd[p] = smi.mean(1), smi.std(1, ddof=1)
    return mi, s_mean, s_sd


# ============================================================ per subject
def subject_pac(subject):
    out = OUT_DIR / f"{subject}.npz"
    if out.exists():
        try:
            np.load(out)["z"]
            print(f"{subject}: done, skipping")
            return
        except Exception:
            print(f"{subject}: cached file unreadable, recomputing")
    t0 = time.time()
    ep = mne.read_epochs(EPO_DIR / subject / f"{subject}_{STEM}_epo.fif", preload=True)
    ep = ep[~ep.metadata["habituation"].to_numpy()].pick("eeg")
    sf, ch = ep.info["sfreq"], ep.ch_names
    times = ep.times[::DECIM]
    win = (times >= WIN[0] - 1e-9) & (times <= WIN[1] + 1e-9)
    info = ep.info
    n_min = min(len(ep[c]) for c in CONDITIONS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not (OUT_DIR / "info.fif").exists():
        mne.io.write_info(OUT_DIR / "info.fif", info)

    shape = (len(CONDITIONS), len(ch), len(F_PHASE), len(F_AMP))
    MI, SM, SS = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    for ci, c in enumerate(CONDITIONS):
        part = OUT_DIR / f"{subject}_part{ci}.npz"           # per-condition checkpoint (resumable)
        if part.exists():
            try:
                d = np.load(part)
                MI[ci], SM[ci], SS[ci] = d["mi"], d["sm"], d["ss"]
                print(f"  {subject} {c}: from checkpoint", flush=True)
                continue
            except Exception:
                pass
        rng = np.random.default_rng([SEED, ci])               # deterministic per condition
        X = ep[c].get_data()
        X = X - X.mean(0, keepdims=True)                      # induced
        draws = [np.sort(rng.choice(len(X), n_min, replace=False)) for _ in range(N_DRAW)]
        used = np.unique(np.concatenate(draws))
        X, draws = X[used], [np.searchsorted(used, d) for d in draws]
        for k in range(len(ch)):
            x = X[:, [k]]
            ph = np.angle(tfr_array_morlet(x, sf, F_PHASE, n_cycles=NC_PHASE, output="complex",
                                           decim=DECIM, zero_mean=True)[:, 0][..., win])
            am = np.abs(tfr_array_morlet(x, sf, F_AMP, n_cycles=NC_AMP, output="complex",
                                         decim=DECIM, zero_mean=True)[:, 0][..., win])
            for d in draws:
                mi, sm, ss = pac_condition(ph[d], am[d], rng)
                MI[ci, k] += mi / len(draws)
                SM[ci, k] += sm / len(draws)
                SS[ci, k] += ss / len(draws)
        tmp = part.with_suffix(".tmp.npz")
        np.savez(tmp, mi=MI[ci], sm=SM[ci], ss=SS[ci])
        tmp.replace(part)
        print(f"  {subject} {c} [{time.time() - t0:.0f} s]", flush=True)
    Z = (MI - SM) / SS
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, z=Z.astype(np.float32), mi=MI.astype(np.float32), n_min=n_min,
                        f_phase=F_PHASE, f_amp=F_AMP, ch_names=np.array(ch), conditions=np.array(CONDITIONS))
    tmp.replace(out)
    print(f"{subject}: n per condition = {n_min} x {N_DRAW} draws  [{time.time() - t0:.0f} s]", flush=True)


# ============================================================ group
def contrasts(z):
    ls_gs, ld_gs, ls_gd, ld_gd = z
    return {"local": (ld_gs + ld_gd - ls_gs - ls_gd) / 2,
            "global": (ls_gd + ld_gd - ls_gs - ld_gs) / 2,
            "interaction": (ld_gd - ls_gd) - (ld_gs - ls_gs)}


def comod_panel(ax, M, sig, title, lim, label, fig):
    pc = ax.pcolormesh(F_PHASE, F_AMP, M.T, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
    if sig is not None and sig.any():
        ax.contour(F_PHASE, F_AMP, sig.T.astype(float), levels=[0.5], colors="k", linewidths=1.0)
    ax.set_title(title)
    fig.colorbar(pc, ax=ax, label=label, shrink=0.85)


def cluster(X):
    T, clus, pv, _ = permutation_cluster_1samp_test(X, n_permutations=N_PERM, tail=0, seed=SEED,
                                                     out_type="mask", n_jobs=1)
    sig = np.zeros(X.shape[1:], bool)
    rows = []
    for cl, p in zip(clus, pv):
        fi, ai = np.where(cl)
        rows.append(dict(p=p, sign=int(np.sign(T[cl].mean())), fp_lo=F_PHASE[fi].min(), fp_hi=F_PHASE[fi].max(),
                         fa_lo=F_AMP[ai].min(), fa_hi=F_AMP[ai].max(), mean=X.mean(0)[cl].mean()))
        if p < CLUSTER_P:
            sig |= cl
    return sig, rows


def group():
    files = sorted(f for f in OUT_DIR.glob("sub-*.npz") if f.stem not in EXCLUDE and ".tmp" not in f.name and "_part" not in f.name)
    subs = [f.stem for f in files]
    d0 = np.load(files[0])
    ch = list(d0["ch_names"])
    Z = np.array([np.load(f)["z"].astype(float) for f in files])        # subj x cond x ch x fp x fa
    n = len(Z)
    print(f"group: N = {n}")
    roi_idx = {r: [ch.index(c) for c in chs if c in ch] for r, chs in ROIS.items()}
    rows = []
    sns.set_theme(context="paper", style="ticks")

    # ---- overall coupling + contrasts, per ROI
    fig, axes = plt.subplots(len(ROIS), 1 + len(CONTRASTS), figsize=(14, 6.4), sharex=True, sharey=True,
                             constrained_layout=True)
    for i, (r, idx) in enumerate(roi_idx.items()):
        zr = Z[:, :, idx].mean(2)                                         # subj x cond x fp x fa
        overall = zr.mean(1)
        sig, rr = cluster(overall)
        rows += [dict(roi=r, test="overall", **x) for x in rr]
        comod_panel(axes[i, 0], overall.mean(0), sig, "Overall coupling" if i == 0 else "",
                    max(np.abs(overall.mean(0)).max(), 1e-6), "MI (z vs surrogates)", fig)
        C = {k: np.array([contrasts(zr[s])[k] for s in range(n)]) for k in CONTRASTS}
        lim = max(np.percentile(np.abs(np.r_[[C[k].mean(0) for k in CONTRASTS]]), 99), 1e-6)
        for j, k in enumerate(CONTRASTS, start=1):
            sig, rr = cluster(C[k])
            rows += [dict(roi=r, test=k, **x) for x in rr]
            comod_panel(axes[i, j], C[k].mean(0), sig, k.capitalize() if i == 0 else "", lim, "Δ z", fig)
        axes[i, 0].set_ylabel(f"{r}\nAmplitude frequency (Hz)", color=ROI_COL[r])
    for ax in axes[-1]:
        ax.set_xlabel("Phase frequency (Hz)")
    fig.suptitle(f"Phase–amplitude coupling, induced signal, {WIN[0]}–{WIN[1]} s after the 5th sound "
                 f"(N = {n}; outline: cluster p < {CLUSTER_P})", x=0.01, ha="left")
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "pac_comodulograms.png", dpi=200)
    plt.close(fig)

    # ---- topographies, theta phase x gamma amplitude
    info = mne.io.read_info(OUT_DIR / "info.fif")
    fp = (F_PHASE >= TOPO_PHASE[0]) & (F_PHASE <= TOPO_PHASE[1])
    fa = (F_AMP >= TOPO_AMP[0]) & (F_AMP <= TOPO_AMP[1])
    zb = Z[:, :, :, fp][:, :, :, :, fa].mean((3, 4))                     # subj x cond x ch
    maps = {"overall": zb.mean(1)}
    for k in CONTRASTS:
        maps[k] = np.array([contrasts(zb[s][:, :, None, None])[k][:, 0, 0] for s in range(n)])
    fig, axes = plt.subplots(1, len(maps), figsize=(12, 3.4), constrained_layout=True)
    for ax, (k, v) in zip(axes, maps.items()):
        t = v.mean(0) / (v.std(0, ddof=1) / np.sqrt(n))
        im, _ = mne.viz.plot_topomap(t, info, axes=ax, cmap="RdBu_r", vlim=(-5, 5), show=False,
                                     contours=0, sensors=False)
        ax.set_title(k.capitalize())
    fig.colorbar(im, ax=axes, label="t (N−1 df)", shrink=0.8)
    fig.suptitle(f"Theta phase ({TOPO_PHASE[0]}–{TOPO_PHASE[1]} Hz) x gamma amplitude "
                 f"({TOPO_AMP[0]}–{TOPO_AMP[1]} Hz), uncorrected t", x=0.01, ha="left")
    fig.savefig(FIG_DIR / "pac_topographies.png", dpi=200)
    plt.close(fig)

    df = pd.DataFrame(rows).sort_values(["roi", "test", "p"])
    df.to_csv(OUT_DIR / "group_clusters.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3g}".format):
        print(df[df.p < 0.2].to_string(index=False))


if __name__ == "__main__":
    if "--group" in sys.argv:
        group()
    else:
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        subs = args or sorted(p.name for p in EPO_DIR.glob("sub-*") if p.is_dir())
        for s in subs:
            if s not in EXCLUDE:
                subject_pac(s)
