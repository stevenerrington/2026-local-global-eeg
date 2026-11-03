"""
09_topo_timecourse.py — topography time series of the local and global effects.

Rows:    Local effect (LD − LS) and Global effect (GD − GS), group mean difference (µV)
Columns: consecutive time bins from T_START to T_END (BIN ms wide), re 5th-sound onset
Marked electrodes: part of a significant spatio-temporal cluster (one-sample cluster permutation
over subjects on the difference, T_START–T_END, two-sided, cluster-forming p < .01, 5000 perms).

Input: per-subject condition averages cached by 07_header1_erps.py
       (derivatives/header1_erps/<sub>.npz; 30 Hz low-passed, baseline −100–0 ms).
Output: figures/topo_timecourse_local_global.png, derivatives/header1_erps/topo_clusters.tsv

Note: epochs currently end at 700 ms (02_preprocess.py TMAX); to extend, raise TMAX and re-run
02 and 07.

Usage:  python scripts/09_topo_timecourse.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from scipy import stats

# ----------------------------------------------------------------------------- CONFIG
ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "derivatives" / "header1_erps"
FIG_DIR = ROOT / "figures"
T_START, T_END, BIN = 0.0, 0.8, 0.05     # s; T_END is clipped to the last available sample
EFFECTS = [("Local effect  (LD − LS)", "LD", "LS"), ("Global effect  (GD − GS)", "GD", "GS")]
# a-priori analysis sites/windows (as in 06/07), highlighted per row: rings = ROI electrodes,
# star = electrode with the largest group difference inside the window (sign as expected),
# coloured bar under the columns = analysis window
ROIS = {"LD": dict(chs=["Fz", "FC1", "FC2", "F1", "F2", "Cz"], win=(0.08, 0.14), sign=-1,
                   color="#E07B00", label="frontocentral ROI (MMN)"),
        "GD": dict(chs=["Pz", "P1", "P2", "CPz", "POz", "CP1", "CP2"], win=(0.30, 0.70), sign=+1,
                   color="#1B9E77", label="parietal ROI (P3b)")}
CLUSTER_P = 0.01
N_PERM = 5000
SEED = 42
# ------------------------------------------------------------------------------------

mne.set_log_level("ERROR")


def main():
    files = sorted(CACHE_DIR.glob("sub-*.npz"))
    info = mne.io.read_info(CACHE_DIR / "info.fif")
    z0 = np.load(files[0])
    times = z0["times"]
    t_end = min(T_END, times[-1] + 1e-9)
    edges = np.arange(T_START, t_end - 1e-9, BIN)
    edges = np.append(edges, min(edges[-1] + BIN, t_end))
    n = len(files)
    adjacency, _ = mne.channels.find_ch_adjacency(info, "eeg")
    tsel = (times >= T_START) & (times <= t_end)

    rows_out = []
    diffs, masks = [], []
    for name, dev, std in EFFECTS:
        D = np.stack([np.load(f)[f"topo__{dev}"] - np.load(f)[f"topo__{std}"] for f in files])  # subj x ch x time
        X = D[:, :, tsel].transpose(0, 2, 1)                                                   # subj x time x ch
        thr = stats.t.ppf(1 - CLUSTER_P / 2, n - 1)
        T, clusters, pv, _ = mne.stats.spatio_temporal_cluster_1samp_test(
            X, threshold=thr, n_permutations=N_PERM, adjacency=adjacency, tail=0, seed=SEED,
            out_type="mask", n_jobs=1, verbose=False)
        sig = np.zeros_like(T, bool)
        for c, p in zip(clusters, pv):
            if p < 0.05:
                sig |= c
                tt = np.where(c.any(1))[0]
                rows_out.append(dict(effect=name, sign="+" if T[c].mean() > 0 else "−", p=p,
                                     start_ms=1e3 * times[tsel][tt[0]], end_ms=1e3 * times[tsel][tt[-1]],
                                     n_channels=int(c.any(0).sum())))
        diffs.append(D.mean(0)[:, tsel])
        masks.append(sig.T)                                                                    # ch x time
    pd.DataFrame(rows_out).to_csv(CACHE_DIR / "topo_clusters.tsv", sep="\t", index=False)
    print(pd.DataFrame(rows_out).round(4).to_string(index=False))

    tt = times[tsel]
    nb = len(edges) - 1
    sns.set_theme(context="notebook", style="white")
    fig, axes = plt.subplots(len(EFFECTS), nb + 1, figsize=(1.32 * nb + 1.1, 3.3 * len(EFFECTS)),
                             gridspec_kw=dict(width_ratios=[1] * nb + [0.08]))
    from mne.channels.layout import _find_topomap_coords
    pos = _find_topomap_coords(info, picks=mne.pick_types(info, eeg=True))
    chs = [info["ch_names"][i] for i in mne.pick_types(info, eeg=True)]
    for r, ((name, dev, _), M, S) in enumerate(zip(EFFECTS, diffs, masks)):
        lim = np.ceil(np.abs(M).max() * 10) / 10
        roi = ROIS[dev]
        ri = [chs.index(c) for c in roi["chs"] if c in chs]
        w = (tt >= roi["win"][0]) & (tt <= roi["win"][1])
        peak = int(np.argmax(roi["sign"] * M[:, w].mean(1)))
        for k in range(nb):
            ax = axes[r, k]
            b = (tt >= edges[k]) & (tt < edges[k + 1] + (1e-9 if k == nb - 1 else 0))
            im, _ = mne.viz.plot_topomap(M[:, b].mean(1), info, axes=ax, show=False, cmap="RdBu_r",
                                         vlim=(-lim, lim), contours=0, sensors=False,
                                         mask=S[:, b].any(1),
                                         mask_params=dict(marker="o", markerfacecolor="k",
                                                          markeredgecolor="k", markersize=2.2))
            ax.scatter(pos[ri, 0], pos[ri, 1], s=38, facecolor="none", edgecolor=roi["color"], lw=1.4, zorder=5)
            in_win = edges[k] < roi["win"][1] - 1e-9 and edges[k + 1] > roi["win"][0] + 1e-9
            if in_win:
                ax.scatter(*pos[peak], s=95, marker="*", color=roi["color"], edgecolor="k", lw=0.5, zorder=6)
                ax.plot([0.08, 0.92], [-0.06, -0.06], transform=ax.transAxes, color=roi["color"], lw=4,
                        clip_on=False, solid_capstyle="butt")
            if r == 0:
                ax.set_title(f"{int(round(edges[k] * 1e3))}–{int(round(edges[k + 1] * 1e3))}", fontsize=9)
        axes[r, 0].text(-0.35, 0.5, name, transform=axes[r, 0].transAxes, rotation=90,
                        ha="center", va="center", fontsize=10, weight="bold")
        axes[r, 0].text(0.0, -0.22, f"○ {roi['label']}: {', '.join(roi['chs'])}   ★ peak: {chs[peak]}   "
                        f"▬ analysis window {int(roi['win'][0] * 1e3)}–{int(roi['win'][1] * 1e3)} ms",
                        transform=axes[r, 0].transAxes, fontsize=8.5, color=roi["color"], ha="left", va="top")
        cb = fig.colorbar(im, cax=axes[r, -1])
        cb.set_label("µV", fontsize=9)
        cb.ax.tick_params(labelsize=8)
    fig.suptitle(f"Group difference topographies, {int(T_START * 1e3)}–{int(round(t_end * 1e3))} ms after the "
                 f"5th sound (n = {n}; {int(round(BIN * 1e3))}-ms bins; black dots = significant spatio-temporal cluster, p < .05)",
                 fontsize=10.5, y=1.0)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "topo_timecourse_local_global.png", dpi=170, bbox_inches="tight")
    print(f"Figure -> {FIG_DIR / 'topo_timecourse_local_global.png'}")


if __name__ == "__main__":
    main()
