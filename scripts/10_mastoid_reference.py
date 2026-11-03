"""
10_mastoid_reference.py — the local effect (MMN) under the classic linked-mastoid reference.

The pipeline uses an average reference. Classic MMN papers use a nose or linked-mastoid
reference, under which the MMN is a frontocentral negativity and the mastoids carry its polarity
inversion. The cap has no M1/M2; TP9/TP10 (over the mastoids) are used as the linked-mastoid
reference. Re-referencing is linear, so it is applied to the cached per-subject averages from
07_header1_erps.py (30 Hz low-passed, baseline −100–0 ms) — identical to re-referencing trials.

Figure (figures/mmn_reference_comparison.png):
  row 1  ERPs at Fz (LS, LD, LD − LS) under average vs linked-mastoid reference, and the
         LD − LS difference at TP9/TP10 under the average reference (polarity inversion)
  row 2  LD − LS topographies 0–400 ms in 50-ms bins under each reference
Stats (derivatives/header1_erps/mmn_reference_stats.tsv): Fz LD − LS mean 80–140 ms (t-test vs 0)
and peak latency (most negative point 50–200 ms) per reference.

Usage:  python scripts/10_mastoid_reference.py
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
MASTOIDS = ["TP9", "TP10"]
SITE = "Fz"
WIN = (0.08, 0.14)            # a-priori MMN window
PEAK_SEARCH = (0.05, 0.20)
TOPO_BINS = np.arange(0.0, 0.401, 0.05)
COL = {"LS": "#4C72B0", "LD": "#C44E52", "diff": "0.2"}
# ------------------------------------------------------------------------------------

mne.set_log_level("ERROR")


def main():
    files = sorted(CACHE_DIR.glob("sub-*.npz"))
    info = mne.io.read_info(CACHE_DIR / "info.fif")
    z0 = np.load(files[0])
    times, chs = z0["times"], list(z0["ch_names"])
    n = len(files)
    mi = [chs.index(c) for c in MASTOIDS]
    fz = chs.index(SITE)

    data = {}
    for ref in ["average", "linked mastoids"]:
        for c in ["LS", "LD"]:
            X = np.stack([np.load(f)[f"topo__{c}"] for f in files])        # subj x ch x time (avg ref)
            if ref == "linked mastoids":
                X = X - X[:, mi, :].mean(1, keepdims=True)
            data[(ref, c)] = X

    # ---- stats at Fz
    rows = []
    w = (times >= WIN[0]) & (times <= WIN[1])
    ps = (times >= PEAK_SEARCH[0]) & (times <= PEAK_SEARCH[1])
    for ref in ["average", "linked mastoids"]:
        d = data[(ref, "LD")][:, fz] - data[(ref, "LS")][:, fz]
        m = d[:, w].mean(1)
        r = stats.ttest_1samp(m, 0)
        lat = times[ps][np.argmin(d[:, ps], axis=1)] * 1e3
        ga = d.mean(0)
        rows.append(dict(reference=ref, site=SITE, mean_80_140_uV=m.mean(), sd=m.std(ddof=1), t=r.statistic,
                         p=r.pvalue, dz=m.mean() / m.std(ddof=1),
                         n_negative=int((m < 0).sum()),
                         group_peak_ms=times[ps][np.argmin(ga[ps])] * 1e3, group_peak_uV=ga[ps].min(),
                         subj_peak_latency_ms_mean=lat.mean(), subj_peak_latency_ms_sd=lat.std(ddof=1)))
    dm = data[("average", "LD")][:, mi].mean(1) - data[("average", "LS")][:, mi].mean(1)
    mm = dm[:, w].mean(1)
    r = stats.ttest_1samp(mm, 0)
    rows.append(dict(reference="average", site="TP9/TP10", mean_80_140_uV=mm.mean(), sd=mm.std(ddof=1),
                     t=r.statistic, p=r.pvalue, dz=mm.mean() / mm.std(ddof=1), n_negative=int((mm < 0).sum())))
    st = pd.DataFrame(rows)
    st.to_csv(CACHE_DIR / "mmn_reference_stats.tsv", sep="\t", index=False)
    print(st.round(3).to_string(index=False))

    # ---- figure
    sns.set_theme(context="notebook", style="ticks")
    t = times * 1e3
    nb = len(TOPO_BINS) - 1
    fig = plt.figure(figsize=(15, 9.2))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.6], hspace=0.32)
    top = outer[0].subgridspec(1, 3, wspace=0.28)
    gs = outer[1].subgridspec(2, nb, hspace=0.25)

    def panel(ax, traces, title, stat_row=None):
        for lab, X, col, ls, lw in traces:
            m, se = X.mean(0), X.std(0, ddof=1) / np.sqrt(n)
            ax.plot(t, m, color=col, ls=ls, lw=lw, label=lab)
            ax.fill_between(t, m - se, m + se, color=col, alpha=0.18, lw=0)
        ax.axvspan(WIN[0] * 1e3, WIN[1] * 1e3, color="0.92", zorder=0)
        ax.axvline(0, color="k", lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set(xlim=(-100, 500), xlabel="Time re 5th sound (ms)", ylabel="µV")
        if stat_row is not None:
            title += (f"\nLD − LS {int(WIN[0] * 1e3)}–{int(WIN[1] * 1e3)} ms: {stat_row.mean_80_140_uV:.2f} µV"
                      f"\nt({n - 1}) = {stat_row.t:.1f}, p = {stat_row.p:.1g}; peak {stat_row.group_peak_ms:.0f} ms")
        ax.set_title(title, fontsize=9.5)
        ax.legend(frameon=False, fontsize=8, loc="upper right")

    axes_top = [fig.add_subplot(top[0, k]) for k in range(3)]
    for ax, ref in zip(axes_top[:2], ["average", "linked mastoids"]):
        ls_, ld_ = data[(ref, "LS")][:, fz], data[(ref, "LD")][:, fz]
        panel(ax, [("LS", ls_, COL["LS"], "-", 1.6), ("LD", ld_, COL["LD"], "-", 1.6),
                   ("LD − LS", ld_ - ls_, COL["diff"], "--", 1.3)],
              f"{SITE}, {ref} reference", st[(st.reference == ref) & (st.site == SITE)].iloc[0])
    mrow = st[st.site == "TP9/TP10"].iloc[0]
    panel(axes_top[2], [("LD − LS at TP9/TP10", dm, "#8172B3", "-", 1.6),
                        ("LD − LS at Fz", data[("average", "LD")][:, fz] - data[("average", "LS")][:, fz],
                         COL["diff"], "--", 1.3)],
          "Polarity inversion, average reference\n"
          f"TP9/TP10 {int(WIN[0] * 1e3)}–{int(WIN[1] * 1e3)} ms: {mrow.mean_80_140_uV:+.2f} µV\n"
          f"t({n - 1}) = {mrow.t:.1f}, p = {mrow.p:.1g}")
    sns.despine(fig)

    for r, ref in enumerate(["average", "linked mastoids"]):
        D = (data[(ref, "LD")] - data[(ref, "LS")]).mean(0)
        lim = np.ceil(np.abs(D[:, (times >= 0) & (times <= 0.4)]).max() * 10) / 10
        for k in range(nb):
            ax = fig.add_subplot(gs[r, k])
            b = (times >= TOPO_BINS[k]) & (times < TOPO_BINS[k + 1])
            im, _ = mne.viz.plot_topomap(D[:, b].mean(1), info, axes=ax, show=False, cmap="RdBu_r",
                                         vlim=(-lim, lim), contours=0, sensors=False,
                                         mask=np.isin(np.arange(len(chs)), [fz] + mi),
                                         mask_params=dict(marker="o", markerfacecolor="none",
                                                          markeredgecolor="k", markersize=5))
            if r == 0:
                ax.set_title(f"{int(TOPO_BINS[k] * 1e3)}–{int(TOPO_BINS[k + 1] * 1e3)} ms", fontsize=9)
            if k == 0:
                ax.text(-0.3, 0.5, f"LD − LS\n{ref}", transform=ax.transAxes, ha="center", va="center",
                        rotation=90, fontsize=9, weight="bold")
        cax = ax.inset_axes([1.08, 0.1, 0.06, 0.8])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("µV", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(f"Local effect (MMN) under average vs linked-mastoid (TP9/TP10) reference (n = {n}); "
                 f"circled: {SITE}, TP9, TP10; shaded = a-priori window", fontsize=10.5, y=0.995)
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "mmn_reference_comparison.png", dpi=160, bbox_inches="tight")
    print(f"Figure -> {FIG_DIR / 'mmn_reference_comparison.png'}")


if __name__ == "__main__":
    main()
