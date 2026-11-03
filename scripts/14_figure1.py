"""
14_figure1.py — Figure 1: task, contrasts, topographies, MMN and P3b.

  A  Task: quintlet structure and timing; example blocks for the two rules within a sound pair
  B  The 2 x 2 design and the two contrasts (local: LD − LS; global: GD − GS) and what they index
  C  Group difference topographies 0–700 ms (50-ms bins) for both contrasts; rings = ROI electrodes,
     dots = significant spatio-temporal cluster
  D  MMN: frontocentral ROI, LS vs LD and difference; a-priori window shaded; bar = significant
     temporal cluster of the difference
  E  P3b: parietal ROI, GS vs GD and difference; same conventions

Input: per-subject condition averages cached by 07_header1_erps.py (derivatives/header1_erps/<sub>.npz;
average reference, 30 Hz low-pass, baseline −100–0 ms) and info.fif.
Output: figures/Figure1.png and figures/Figure1.pdf; derivatives/header1_erps/figure1_stats.json

Usage:  python scripts/14_figure1.py            # full version  -> Figure1.png/.pdf
        python scripts/14_figure1.py --lean     # 100-ms maps, no cluster dots, methods text moved to legend
                                                 #          -> Figure1_lean.png/.pdf
        python scripts/14_figure1.py --erp      # panels D & E only, editable text -> Figure1_DE.pdf/.svg/.png
"""
from pathlib import Path
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
import seaborn as sns
import mne
from scipy import stats

# ----------------------------------------------------------------------------- CONFIG
ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "derivatives" / "header1_erps"
FIG_DIR = ROOT / "figures"
TONE = {"A": "#4C72B0", "B": "#DD8452"}
C_LOC, C_GLOB, C_STD = "#C44E52", "#8172B3", "0.35"
ROI = {"local": dict(chs=["Fz", "FC1", "FC2", "F1", "F2", "Cz"], win=(0.08, 0.14), dev="LD", std="LS",
                     ring="#E07B00"),
       "global": dict(chs=["Pz", "P1", "P2", "CPz", "POz", "CP1", "CP2"], win=(0.30, 0.70), dev="GD", std="GS",
                      ring="#1B9E77")}
LEAN = "--lean" in sys.argv
ERP_ONLY = "--erp" in sys.argv          # panels D & E alone, as a vector file with editable text
# Editable text in vector output: TrueType (Type 42) fonts in PDF, real <text> elements in SVG.
TEXT_RC = {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
           "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]}
_b = np.round(np.arange(0.0, 0.701, 0.05), 3)
# lean: four windows matched to the effects (MMN, early positivity, early/late P3b) instead of uniform bins
TOPO_WINS = ([(0.08, 0.14), (0.16, 0.30), (0.30, 0.50), (0.50, 0.70)] if LEAN else list(zip(_b[:-1], _b[1:])))
N_PERM = 2000
SEED = 42
# ------------------------------------------------------------------------------------

mne.set_log_level("ERROR")


# ============================================================ drawing helpers
def quintlet(ax, x, y, q, w=0.012, gap=0.004, h=0.07, edge=None, lw=1.6):
    """Draw a quintlet (string of 'A'/'B') as five bars; B bars taller. Returns right edge."""
    x0 = x
    for c in q:
        hh = h if c == "A" else h * 1.35
        ax.add_patch(Rectangle((x, y), w, hh, fc=TONE[c], ec="none"))
        x += w + gap
    if edge is not None:
        ax.add_patch(FancyBboxPatch((x0 - 0.004, y - 0.012), x - x0 + 0.004, h * 1.35 + 0.024,
                                    boxstyle="round,pad=0.002,rounding_size=0.006", fc="none", ec=edge, lw=lw))
    return x


def panel_label(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=16, weight="bold", va="bottom", ha="left")


def draw_task(ax):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # --- one quintlet, with timing
    ax.text(0.0, 0.93, "One trial = a quintlet of 50-ms chords", fontsize=10.5, weight="bold", va="top")
    x0, y0, w, step = 0.03, 0.60, 0.035, 0.075
    for k, c in enumerate("AAAAB"):
        ax.add_patch(Rectangle((x0 + k * step, y0), w, 0.11 if c == "A" else 0.15, fc=TONE[c], ec="k" if k == 4 else "none",
                               lw=1.5))
    ax.annotate("", xy=(x0 + step, y0 - 0.03), xytext=(x0, y0 - 0.03),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color="0.3"))
    ax.text(x0 + step / 2, y0 - 0.045, "SOA ≈ 157 ms", ha="center", va="top", fontsize=8)
    ax.text(x0 + 4 * step + w / 2, y0 + 0.16, "5th sound (t = 0)", ha="center", va="bottom", fontsize=8.5)
    ax.annotate("", xy=(x0 + 4 * step + w + 0.13, y0 + 0.05), xytext=(x0 + 4 * step + w + 0.01, y0 + 0.05),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color="0.3"))
    ax.text(x0 + 4 * step + w + 0.07, y0 + 0.07, "800–1100 ms", ha="center", va="bottom", fontsize=8)
    ax.text(0.55, 0.92, "■", color=TONE["A"], fontsize=11, va="top")
    ax.text(0.58, 0.92, "chord A" if LEAN else "chord A  350/700/1400 Hz", fontsize=8.5, va="top")
    ax.text(0.55, 0.85, "■", color=TONE["B"], fontsize=11, va="top")
    ax.text(0.58, 0.85, "chord B" if LEAN else "chord B  500/1000/2000 Hz", fontsize=8.5, va="top")
    ax.text(0.55, 0.76, "Task: silently count global deviants" if LEAN else
            "Task: silently count global deviants\n(totals reported after every 4 blocks)",
            fontsize=8.5, va="top", style="italic")
    # --- two example blocks (same sound pair, opposite rules)
    ax.text(0.0, 0.52, "Blocks: the standard quintlet (80%) sets the rule; the other is the deviant (20%)",
            fontsize=10.5, weight="bold", va="top")
    rows = [("Repeat-rule block\n(standard AAAAA)", ["AAAAA"] * 4 + ["AAAAB"] + ["AAAAA"] * 3 + ["AAAAB"] + ["AAAAA"] * 2, "AAAAB"),
            ("Change-rule block\n(standard AAAAB)", ["AAAAB"] * 4 + ["AAAAA"] + ["AAAAB"] * 3 + ["AAAAA"] + ["AAAAB"] * 2, "AAAAA")]
    for r, (lab, seq, dev) in enumerate(rows):
        y = 0.30 - r * 0.22
        ax.text(0.0, y + 0.045, lab, fontsize=8.5, va="center")
        x = 0.20
        for q in seq:
            x = quintlet(ax, x, y, q, w=0.009, gap=0.0025, h=0.06, edge=C_GLOB if q == dev else None) + 0.010
        ax.text(x, y + 0.04, "…", fontsize=12, va="center")
    if LEAN:
        return
    ax.text(0.20, -0.02, "8 blocks × 120–164 quintlets (AAAAA, AAAAB, BBBBB, BBBBA standards, 2 blocks each)\n"
            "≥ 10 standards open every block; 5-s silence between blocks;  purple box = global deviant",
            fontsize=7.5, va="bottom", color="0.3")


def draw_contrasts(ax):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    cx = [0.33, 0.62]
    cy = [0.72, 0.46]
    ax.text(cx[0], 0.94, "Local standard (LS)\n5th = 4th", ha="center", va="top", fontsize=9, weight="bold")
    ax.text(cx[1], 0.94, "Local deviant (LD)\n5th ≠ 4th", ha="center", va="top", fontsize=9, weight="bold",
            color=C_LOC)
    ax.text(0.10, cy[0] + 0.03, "Global\nstandard\n(GS)", ha="center", va="center", fontsize=9, weight="bold")
    ax.text(0.10, cy[1] + 0.03, "Global\ndeviant\n(GD)", ha="center", va="center", fontsize=9, weight="bold",
            color=C_GLOB)
    cells = {(0, 0): ("AAAAA", "in AAAAA block"), (0, 1): ("AAAAB", "in AAAAB block"),
             (1, 0): ("AAAAA", "in AAAAB block"), (1, 1): ("AAAAB", "in AAAAA block")}
    for (r, c), (q, ctx) in cells.items():
        ax.add_patch(FancyBboxPatch((cx[c] - 0.12, cy[r] - 0.09), 0.24, 0.23,
                                    boxstyle="round,pad=0.005,rounding_size=0.015", fc="0.97",
                                    ec=C_GLOB if r == 1 else "0.7", lw=1.4 if r == 1 else 0.8))
        quintlet(ax, cx[c] - 0.085, cy[r] + 0.01, q, w=0.024, gap=0.009, h=0.07,
                 edge=C_LOC if c == 1 else None, lw=1.2)
        ax.text(cx[c], cy[r] - 0.055, f"{['LS', 'LD'][c]}/{['GS', 'GD'][r]}  ·  {ctx}", ha="center",
                va="center", fontsize=7.5)
    # contrast explanations
    ax.text(0.02, 0.335, "Local effect = LD − LS", color=C_LOC, fontsize=10, weight="bold", va="top")
    ax.text(0.02, 0.29, "a change at the 5th sound (tone-to-tone surprise)\n→ early frontocentral MMN, auditory cortex",
            fontsize=8.5, va="top")
    ax.add_patch(FancyArrowPatch((0.905, cy[0] - 0.02), (0.905, cy[1] + 0.10), arrowstyle="-|>",
                                 mutation_scale=12, color=C_GLOB, lw=1.5))
    ax.text(0.02, 0.205, "Global effect = GD − GS", color=C_GLOB, fontsize=10, weight="bold", va="top")
    ax.text(0.02, 0.16, "the quintlet breaks the block's rule\n→ late parietal P3b", fontsize=8.5, va="top")
    if LEAN:
        return
    ax.text(0.02, -0.04, "Physically identical quintlets swap roles across blocks (e.g. AAAAB is GS in one\n"
            "block and GD in another), so the global effect is not a stimulus difference.\nSame design for the B pair.",
            ha="left", va="bottom", fontsize=7.5, color="0.3")


# ============================================================ data
def load():
    files = sorted(CACHE_DIR.glob("sub-*.npz"))
    z0 = np.load(files[0])
    info = mne.io.read_info(CACHE_DIR / "info.fif")
    D = {c: np.stack([np.load(f)[f"topo__{c}"] for f in files]) for c in ["LS", "LD", "GS", "GD"]}
    return z0["times"], list(z0["ch_names"]), info, D, len(files)


def roi_wave(D, c, chs, roi_chs):
    return D[c][:, [chs.index(x) for x in roi_chs]].mean(1)          # subj x time


def cluster_1d(diff, times, tmin=0.0, tmax=0.7):
    sel = (times >= tmin) & (times <= tmax)
    n = len(diff)
    T, clusters, pv, _ = mne.stats.permutation_cluster_1samp_test(
        diff[:, sel], threshold=stats.t.ppf(0.975, n - 1), n_permutations=N_PERM, tail=0, seed=SEED,
        out_type="indices", verbose=False)
    sig = np.zeros(len(times), bool)
    cl = []
    idx = np.where(sel)[0]
    for c, p in zip(clusters, pv):
        if p < 0.05:
            sig[idx[c[0]]] = True
            cl.append(dict(start_ms=float(times[idx[c[0][0]]] * 1e3), end_ms=float(times[idx[c[0][-1]]] * 1e3),
                           p=float(p), sign="+" if T[c[0]].mean() > 0 else "−"))
    return sig, cl


def erp_panel(ax, times, D, chs, key, n, label_std, label_dev, col_dev, title):
    r = ROI[key]
    s, d = roi_wave(D, r["std"], chs, r["chs"]), roi_wave(D, r["dev"], chs, r["chs"])
    diff = d - s
    t = times * 1e3
    ax.axvspan(r["win"][0] * 1e3, r["win"][1] * 1e3, color="0.92", zorder=0)
    for x, col, lab, ls, lw in [(s, C_STD, label_std, "-", 1.8), (d, col_dev, label_dev, "-", 1.8),
                                (diff, "k", f"{r['dev']} − {r['std']}", "--", 1.3)]:
        m, se = x.mean(0), x.std(0, ddof=1) / np.sqrt(n)
        ax.plot(t, m, color=col, lw=lw, ls=ls, label=lab)
        ax.fill_between(t, m - se, m + se, color=col, alpha=0.18, lw=0)
    sig, cl = cluster_1d(diff, times)
    ax.plot(np.where(sig, t, np.nan), np.full(len(t), 0.03), color="k", lw=4, solid_capstyle="butt",
            transform=ax.get_xaxis_transform())
    for k in range(1, 5):
        ax.axvline(-157 * k, color="0.8", lw=0.6, ls=":")
    ax.axvline(0, color="k", lw=0.9)
    ax.axhline(0, color="k", lw=0.5)
    w = (times >= r["win"][0]) & (times <= r["win"][1])
    mw = diff[:, w].mean(1)
    tt = stats.ttest_1samp(mw, 0)
    pk = times[np.argmax(np.abs(diff.mean(0)) * ((times >= r["win"][0] - 0.05) & (times <= r["win"][1])))] * 1e3
    ax.set(xlim=(-200, 700), xlabel="Time re 5th sound (ms)", ylabel="µV")
    ax.set_title(f"{title}\n{', '.join(r['chs'])}", fontsize=10)
    ax.text(0.99, 0.97, f"{int(r['win'][0] * 1e3)}–{int(r['win'][1] * 1e3)} ms: {mw.mean():+.2f} µV\n"
            f"t({n - 1}) = {tt.statistic:.1f}, p = {tt.pvalue:.1g}\n{int((np.sign(mw) == np.sign(mw.mean())).sum())}/{n} subjects",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    return dict(window_ms=[int(r["win"][0] * 1e3), int(r["win"][1] * 1e3)], mean_diff_uV=float(mw.mean()),
                t=float(tt.statistic), p=float(tt.pvalue), peak_ms=float(pk),
                n_same_sign=int((np.sign(mw) == np.sign(mw.mean())).sum()), temporal_clusters=cl)


def topo_rows(fig, gs_cell, times, chs, info, D, n):
    adjacency, _ = mne.channels.find_ch_adjacency(info, "eeg")
    from mne.channels.layout import _find_topomap_coords
    pos = _find_topomap_coords(info, picks=mne.pick_types(info, eeg=True))
    nb = len(TOPO_WINS)
    sub = gs_cell.subgridspec(2, nb + 1, width_ratios=[1] * nb + [0.07], hspace=0.08, wspace=0.05)
    out = {}
    for r, key in enumerate(["local", "global"]):
        roi = ROI[key]
        diff = D[roi["dev"]] - D[roi["std"]]
        sel = (times >= 0) & (times <= 0.7)
        T, clusters, pv, _ = mne.stats.spatio_temporal_cluster_1samp_test(
            diff[:, :, sel].transpose(0, 2, 1), threshold=stats.t.ppf(0.975, n - 1), n_permutations=N_PERM,
            adjacency=adjacency, tail=0, seed=SEED, out_type="mask", verbose=False)
        mask = np.zeros_like(T, bool)
        for c, p in zip(clusters, pv):
            if p < 0.05:
                mask |= c
        out[key] = dict(cluster_p=[float(p) for p in sorted(pv)[:3]])
        M = diff.mean(0)[:, sel]
        ts = times[sel]
        lim = np.ceil(np.abs(M).max() * 10) / 10
        ri = [chs.index(c) for c in roi["chs"]]
        for k in range(nb):
            ax = fig.add_subplot(sub[r, k])
            b = (ts >= TOPO_WINS[k][0]) & (ts < TOPO_WINS[k][1] + (1e-9 if k == nb - 1 else 0))
            im, _ = mne.viz.plot_topomap(M[:, b].mean(1), info, axes=ax, show=False, cmap="RdBu_r", vlim=(-lim, lim),
                                         contours=0, sensors=False, mask=None if LEAN else mask[b].any(0),
                                         mask_params=dict(marker="o", markerfacecolor="k", markeredgecolor="k",
                                                          markersize=1.8))
            ax.scatter(pos[ri, 0], pos[ri, 1], s=40 if LEAN else 26, facecolor="none", edgecolor=roi["ring"], lw=1.2, zorder=5)
            if TOPO_WINS[k][0] < roi["win"][1] - 1e-9 and TOPO_WINS[k][1] > roi["win"][0] + 1e-9:
                ax.plot([0.1, 0.9], [-0.04, -0.04], transform=ax.transAxes, color=roi["ring"], lw=3.5, clip_on=False,
                        solid_capstyle="butt")
            if r == 0:
                ax.set_title(f"{round(TOPO_WINS[k][0] * 1e3)}–{round(TOPO_WINS[k][1] * 1e3)}" + (" ms" if LEAN else ""), fontsize=9 if LEAN else 8)
            if k == 0:
                ax.text(-0.25, 0.5, ["Local\nLD − LS", "Global\nGD − GS"][r], transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=9.5, weight="bold", color=[C_LOC, C_GLOB][r])
        cb = fig.colorbar(im, cax=fig.add_subplot(sub[r, -1]))
        cb.set_label("µV", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    return out


# ============================================================ print-size ERP block (--erp)
PRINT_MM = (165.467, 86.011)          # target box, width x height


def erp_print_block(times, D, chs, n):
    """D/E/F at final print size: conditions (top) and difference (bottom) for local and global,
    sharing x; right column = per-participant window means. Editable text, 7-pt type."""
    mm = 1 / 25.4
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "legend.fontsize": 6, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                         "ytick.major.size": 2.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
                         "axes.labelpad": 2})
    fig = plt.figure(figsize=(PRINT_MM[0] * mm, PRINT_MM[1] * mm))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.5], height_ratios=[1.15, 1],
                          left=0.065, right=0.99, bottom=0.12, top=0.90, wspace=0.36, hspace=0.12)
    t = times * 1e3
    out = {}

    def trace(ax, X, col, lab, ls="-", lw=1.0):
        m, se = X.mean(0), X.std(0, ddof=1) / np.sqrt(n)
        ax.plot(t, m, color=col, lw=lw, ls=ls, label=lab)
        ax.fill_between(t, m - se, m + se, color=col, alpha=0.2, lw=0)

    def furniture(ax, win):
        ax.axvspan(win[0] * 1e3, win[1] * 1e3, color="0.92", zorder=0, lw=0)
        ax.axvline(0, color="k", lw=0.6)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_xlim(-200, 700)
        ax.set_xticks([-200, 0, 200, 400, 600])

    for c, (key, col, lab, names, title) in enumerate([
            ("local", C_LOC, "D", ("LS", "LD"), "Local: MMN (frontocentral)"),
            ("global", C_GLOB, "E", ("GS", "GD"), "Global: P3b (parietal)")]):
        r = ROI[key]
        s_, d_ = roi_wave(D, r["std"], chs, r["chs"]), roi_wave(D, r["dev"], chs, r["chs"])
        diff = d_ - s_
        axc = fig.add_subplot(gs[0, c])
        axd = fig.add_subplot(gs[1, c], sharex=axc)
        for ax in (axc, axd):
            furniture(ax, r["win"])
        trace(axc, s_, C_STD, f"standard ({names[0]})")
        trace(axc, d_, col, f"deviant ({names[1]})")
        trace(axd, diff, "k", f"{names[1]} − {names[0]}")
        sig, cl = cluster_1d(diff, times)
        axd.plot(np.where(sig, t, np.nan), np.full(len(t), 0.04), color=col, lw=2.2, solid_capstyle="butt",
                 transform=axd.get_xaxis_transform())
        axc.set_title(title, fontsize=7, pad=3)
        axc.legend(frameon=False, loc="upper left", handlelength=1.4, borderaxespad=0.2, labelspacing=0.25)
        axd.legend(frameon=False, loc="upper left", handlelength=1.4, borderaxespad=0.2)
        plt.setp(axc.get_xticklabels(), visible=False)
        axc.set_ylabel("µV")
        axd.set_ylabel("Δ µV")
        axd.set_xlabel("Time from 5th sound (ms)")
        sns.despine(ax=axc)
        sns.despine(ax=axd)
        axc.text(-0.2, 1.08, lab, transform=axc.transAxes, fontsize=9, weight="bold", va="bottom")
        w = (times >= r["win"][0]) & (times <= r["win"][1])
        out[key] = dict(per_subject=diff[:, w].mean(1), win=r["win"], col=col, names=names)
    fig.align_ylabels()

    # ---- F: per-participant effects
    axf = fig.add_subplot(gs[:, 2])
    rng = np.random.default_rng(1)
    for k, key in enumerate(["local", "global"]):
        o = out[key]
        y = o["per_subject"]
        tt = stats.ttest_1samp(y, 0)
        ci = stats.t.ppf(0.975, n - 1) * y.std(ddof=1) / np.sqrt(n)
        x = k + rng.uniform(-0.12, 0.12, n)
        axf.scatter(x, y, s=9, color=o["col"], alpha=0.75, lw=0, zorder=3)
        axf.errorbar(k + 0.3, y.mean(), yerr=ci, fmt="o", color="k", ms=3.2, lw=0.9, capsize=0, zorder=4)
        same = int((np.sign(y) == np.sign(y.mean())).sum())
        o.update(t=float(tt.statistic), p=float(tt.pvalue), mean=float(y.mean()), ci=float(ci), n_same=same)
    axf.set_xticks([0, 1])
    axf.set_xticklabels([f"MMN\n{int(ROI['local']['win'][0]*1e3)}–{int(ROI['local']['win'][1]*1e3)}",
                         f"P3b\n{int(ROI['global']['win'][0]*1e3)}–{int(ROI['global']['win'][1]*1e3)}"])
    for k, key in enumerate(["local", "global"]):
        o = out[key]
        axf.get_xticklabels()[k].set_color(o["col"])
        axf.text(k + 0.1, 1.0, f"t = {o['t']:.1f}".replace("-", "−") + f"\n{o['n_same']}/{n}", transform=axf.get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=6, linespacing=1.1)
    axf.axhline(0, color="k", lw=0.4)
    axf.set_xlim(-0.4, 1.55)
    axf.set_ylabel("Deviant − standard (µV)")
    axf.set_xlabel("Window (ms)")
    sns.despine(ax=axf)
    axf.text(-0.45, 1.08, "F", transform=axf.transAxes, fontsize=9, weight="bold", va="bottom")

    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "Figure1_DE.pdf")
    fig.savefig(FIG_DIR / "Figure1_DE.svg")
    svg = FIG_DIR / "Figure1_DE.svg"          # SVG names fonts: point it at Arial (Liberation Sans = same metrics)
    svg.write_text(svg.read_text().replace("Liberation Sans", "Arial"))
    fig.savefig(FIG_DIR / "Figure1_DE.png", dpi=400)
    w_in, h_in = fig.get_size_inches()
    print(f"Figure -> {FIG_DIR / 'Figure1_DE'}.pdf/.svg/.png  ({w_in * 25.4:.1f} x {h_in * 25.4:.1f} mm)")
    for key in ("local", "global"):
        o = out[key]
        print(key, f"mean {o['mean']:+.2f} µV, 95% CI ±{o['ci']:.2f}, t = {o['t']:.2f}, p = {o['p']:.2g}, {o['n_same']}/{n}")


def main():
    times, chs, info, D, n = load()
    sns.set_theme(context="notebook", style="ticks", rc=TEXT_RC)
    if ERP_ONLY:
        erp_print_block(times, D, chs, n)
        return
    fig = plt.figure(figsize=(16, 17))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.15, 0.8 if LEAN else 0.95, 1.05], hspace=0.28)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.45, 1], wspace=0.08)
    axA = fig.add_subplot(top[0, 0])
    draw_task(axA)
    panel_label(axA, "A", x=-0.01, y=0.97)
    axB = fig.add_subplot(top[0, 1])
    draw_contrasts(axB)
    panel_label(axB, "B", x=-0.02, y=0.97)

    topo_stats = topo_rows(fig, outer[1], times, chs, info, D, n)
    fig.text(0.115, outer[1].get_position(fig).y1 + (0.02 if LEAN else 0.005), "C", fontsize=16, weight="bold", va="bottom")
    fig.text(0.14, outer[1].get_position(fig).y1 + (0.022 if LEAN else 0.007),
             ("Difference topographies after the 5th sound (rings = ROI; bar = analysis window)" if LEAN else
              "Difference topographies, 0–700 ms after the 5th sound (50-ms bins; rings = ROI; coloured bar = "
              "analysis window; dots = significant spatio-temporal cluster)"), fontsize=9.5, va="bottom")

    bot = outer[2].subgridspec(1, 2, wspace=0.22)
    axD = fig.add_subplot(bot[0, 0])
    sD = erp_panel(axD, times, D, chs, "local", n, "local standard (LS)", "local deviant (LD)", C_LOC,
                   "Local effect — MMN (frontocentral ROI)")
    panel_label(axD, "D", x=-0.1, y=1.06)
    axE = fig.add_subplot(bot[0, 1])
    sE = erp_panel(axE, times, D, chs, "global", n, "global standard (GS)", "global deviant (GD)", C_GLOB,
                   "Global effect — P3b (parietal ROI)")
    panel_label(axE, "E", x=-0.1, y=1.06)
    for ax in (axD, axE):
        sns.despine(ax=ax)
    FIG_DIR.mkdir(exist_ok=True)
    stem = "Figure1_lean" if LEAN else "Figure1"
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    if LEAN:
        print(f"Figure -> {FIG_DIR / stem}.png")
        return
    st = dict(n_subjects=n, MMN=sD, P3b=sE, topography_cluster_p=topo_stats)
    (CACHE_DIR / "figure1_stats.json").write_text(json.dumps(st, indent=2))
    print(json.dumps(st, indent=2))
    print(f"Figure -> {FIG_DIR / 'Figure1.png'}")


if __name__ == "__main__":
    main()
