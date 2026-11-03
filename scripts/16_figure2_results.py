"""
16_figure2_results.py — Figure 2 results panels at manuscript size (173.695 x 80 mm).

Rows = response windows (early frontocentral 80–140 ms; late parietal 300–700 ms).
Columns:
  C  head-to-head evidence: held-out log-likelihood of the window's winning learner minus each rival
     (M0 categories, M1 adaptation, the other learner), one dot per participant; n/15 = participants > 0
  D  group verdict: protected exceedance probability (random-effects BMS) over the preregistered six
     models; M4/M5 faded (not separable from their components in model recovery)
  E  what the winner explains: window amplitude per 2x2 condition (bars = data, mean ± SE) with the
     local and global learners' predictions (each learner's surprise mapped to single-trial amplitude
     by within-subject OLS; from 07_header1_erps.py), and the agreement r across the 4 conditions
  F  example ERP: that window's deviants split into per-participant tertiles of the winning learner's
     surprise (LD by local surprise at the frontocentral ROI; GD by global surprise at the parietal ROI),
     with the standard (dashed); slope = per-participant linear trend of window amplitude across tertiles

Inputs: derivatives/model_comparison/{cv_loglik.tsv, summary.json}
        derivatives/header1_erps/sub-*_cond4.tsv, sub-*_windows.tsv, sub-*.npz (from 07_header1_erps.py)
Output: figures/Figure2_results.pdf / .svg / .png (editable text; SVG font named Arial)

Usage:  python scripts/16_figure2_results.py
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ----------------------------------------------------------------------------- CONFIG
ROOT = Path(__file__).resolve().parents[1]
MC_DIR = ROOT / "derivatives" / "model_comparison"
ERP_DIR = ROOT / "derivatives" / "header1_erps"
FIG_DIR = ROOT / "figures"
SIZE_MM = (173.695, 80.0)
MODELS = ["M0 categories", "M1 adaptation", "M2 local", "M3 global", "M4 local+global", "M5 adapt+global"]
SHORT = {m: m.split()[0] for m in MODELS}
MCOL = {"M0 categories": "0.35", "M1 adaptation": "#55A868", "M2 local": "#C44E52", "M3 global": "#8172B3",
        "M4 local+global": "#937860", "M5 adapt+global": "#64B5CD"}
ROWS = [  # model-comparison component, ERP key, row label, winning model, binned-ERP spec (deviants, surprise)
    ("local_80-140ms", "local", "Early\n80–140 ms", "M2 local", ("LD", "S_loc", "LS")),
    ("global_300-700ms", "global", "Late\n300–700 ms", "M3 global", ("GD", "S_glob", "GS"))]
WIN = {"local": (0.08, 0.14), "global": (0.30, 0.70)}
BIN_PAL = {"local": ["#F2B8B9", "#D9666A", "#8E1F24"], "global": ["#CFCBE8", "#9890CB", "#4E4596"]}
BIN_KIND = "raw"          # tertiles of the learner's surprise on that window's deviant trials
CONDS = ["LSGS", "LDGS", "LSGD", "LDGD"]
COND_LAB = ["LS\nGS", "LD\nGS", "LS\nGD", "LD\nGD"]
LEARNERS = [("local learner", "M2 local", "D"), ("global learner", "M3 global", "o")]
TEXT_RC = {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
           "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
           "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
           "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
           "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
           "axes.labelpad": 2}
SEED = 1
# ------------------------------------------------------------------------------------


def main():
    ll = pd.read_csv(MC_DIR / "cv_loglik.tsv", sep="\t")
    summ = json.loads((MC_DIR / "summary.json").read_text())
    c4 = pd.concat([pd.read_csv(f, sep="\t") for f in sorted(ERP_DIR.glob("sub-*_cond4.tsv"))])
    n = ll["subject"].nunique()
    rng = np.random.default_rng(SEED)

    sns.set_theme(style="ticks", rc=TEXT_RC)
    mm = 1 / 25.4
    fig = plt.figure(figsize=(SIZE_MM[0] * mm, SIZE_MM[1] * mm))
    gs = fig.add_gridspec(2, 4, width_ratios=[0.62, 0.62, 0.95, 1.05], left=0.105, right=0.99, bottom=0.1, top=0.9,
                          wspace=0.5, hspace=0.42)
    stats_out = {}
    for r, (comp, ek, rowlab, winner, (dev, svar, std)) in enumerate(ROWS):
        # ---------------- C: head-to-head — winner minus each alternative, per participant
        ax = fig.add_subplot(gs[r, 0])
        d = ll[ll["component"] == comp].pivot(index="subject", columns="model", values="cv_ll")
        rivals = ["M0 categories", "M1 adaptation", "M3 global" if winner == "M2 local" else "M2 local"]
        stats_out[comp] = {}
        for j, m in enumerate(rivals):
            y = (d[winner] - d[m]).to_numpy()
            ax.scatter(j + rng.uniform(-0.13, 0.13, n), y, s=6, lw=0, color=MCOL[winner], alpha=0.8, zorder=3)
            ax.errorbar(j + 0.32, y.mean(), yerr=y.std(ddof=1) / np.sqrt(n), fmt="o", ms=2.8, lw=0.8, color="k",
                        zorder=4)
            k = int((y > 0).sum())
            ax.text(j, 1.0, f"{k}/{n}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.5,
                    color="0.3")
            stats_out[comp][f"{SHORT[winner]}_minus_{SHORT[m]}"] = dict(mean=float(y.mean()), n_positive=k,
                                                                         total=float(y.sum()))
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(range(len(rivals)), [f"vs {SHORT[m]}" for m in rivals])
        for tl, m in zip(ax.get_xticklabels(), rivals):
            tl.set_color(MCOL[m])
        ax.set_xlim(-0.5, len(rivals) - 0.35)
        ax.set_ylabel(f"Δ log-lik. ({SHORT[winner]} − rival)", color=MCOL[winner])
        sns.despine(ax=ax)
        ax.text(-0.62, 0.5, rowlab, transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=7,
                weight="bold")
        if r == 0:
            ax.set_title("Winner vs rivals", pad=9)

        # ---------------- D: PXP
        ax = fig.add_subplot(gs[r, 1])
        pxp = [summ[comp]["pxp"][m] for m in MODELS]
        faint = ("M4 local+global", "M5 adapt+global")     # not recoverable at this n: shown, de-emphasised
        ax.bar(range(6), pxp, color=[MCOL[m] for m in MODELS], width=0.7,
               alpha=None)
        for patch, m in zip(ax.patches, MODELS):
            patch.set_alpha(0.35 if m in faint else 1.0)
        ax.set_xticks(range(6), [SHORT[m] for m in MODELS])
        for tl, m in zip(ax.get_xticklabels(), MODELS):
            tl.set_color(MCOL[m])
            tl.set_alpha(0.45 if m in faint else 1.0)
        win = int(np.argmax(pxp))
        ax.text(win, pxp[win] + 0.03, f"{pxp[win]:.2f}", ha="center", va="bottom", fontsize=6)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel("PXP")
        sns.despine(ax=ax)
        if r == 0:
            ax.set_title("Group verdict", pad=9)
        stats_out[comp].update(pxp=dict(zip(MODELS, pxp)), bor=summ[comp]["bor"])

        # ---------------- E: data vs learner predictions per condition
        ax = fig.add_subplot(gs[r, 2])
        q = c4[c4["component"] == ek]
        dat = q[q["source"] == "data"].pivot(index="subject", columns="condition", values="amp")[CONDS]
        xs = np.arange(4)
        ax.bar(xs, dat.mean(), yerr=dat.std(ddof=1) / np.sqrt(len(dat)), color="0.82", width=0.7,
               error_kw=dict(lw=0.7, capsize=0), label="data")
        for k, (lname, m, mk) in enumerate(LEARNERS):
            pr = q[q["source"] == lname].pivot(index="subject", columns="condition", values="amp")[CONDS]
            off = -0.13 if k == 0 else 0.13
            rv = float(np.corrcoef(pr.mean(), dat.mean())[0, 1])
            ax.plot(xs + off, pr.mean(), marker=mk, ls="none", ms=3.6, mfc="white", mec=MCOL[m], mew=1.0,
                    label=f"{SHORT[m]}  r = {rv:.2f}")
            stats_out[comp][f"r_{lname}"] = rv
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(xs, COND_LAB)
        ax.set_ylabel("Amplitude (µV)")
        sns.despine(ax=ax)
        ax.legend(frameon=False, loc="lower left" if r == 0 else "upper left", handlelength=1.0,
                  borderaxespad=0.1, handletextpad=0.3, labelspacing=0.3)
        if r == 0:
            ax.set_title("Data vs predictions", pad=9)

        # ---------------- F: example ERP — deviants split by the winning learner's surprise
        ax = fig.add_subplot(gs[r, 3])
        files = sorted(ERP_DIR.glob("sub-*.npz"))
        t = np.load(files[0])["times"] * 1e3
        ref = np.stack([np.load(f)[f"cond__{std}"] for f in files])
        ax.axvspan(WIN[ek][0] * 1e3, WIN[ek][1] * 1e3, color="0.93", lw=0, zorder=0)
        ax.plot(t, ref.mean(0), color="0.55", lw=0.8, ls="--", label=f"{std} (standard)")
        for k, lab in enumerate(["low", "mid", "high"]):
            X = np.stack([np.load(f)[f"bin__{dev}__{svar}__{BIN_KIND}__{ek}__{k}"] for f in files])
            m_, se = X.mean(0), X.std(0, ddof=1) / np.sqrt(len(X))
            ax.plot(t, m_, color=BIN_PAL[ek][k], lw=1.0, label=lab)
            ax.fill_between(t, m_ - se, m_ + se, color=BIN_PAL[ek][k], alpha=0.18, lw=0)
        ax.axvline(0, color="k", lw=0.5)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_xlim(-100, 700)
        ax.set_xticks([0, 200, 400, 600])
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (µV)")
        sns.despine(ax=ax)
        ax.legend(frameon=False, loc="lower right", handlelength=1.0, ncol=2,
                  borderaxespad=0.1, handletextpad=0.3, labelspacing=0.2, columnspacing=0.6, fontsize=5.5,
                  title=f"{dev} by {SHORT[winner]} surprise", title_fontsize=5.5)
        # per-participant slope of window amplitude across surprise tertiles
        w = pd.concat([pd.read_csv(f, sep="\t") for f in sorted(ERP_DIR.glob("sub-*_windows.tsv"))])
        for kind in ("raw", "within"):
            q = w[(w["deviants"] == dev) & (w["var"] == svar) & (w["kind"] == kind) & (w["component"] == ek)]
            piv = q.pivot(index="subject", columns="bin", values="amp")
            sl = piv.apply(lambda row: np.polyfit(np.arange(3), row.to_numpy(), 1)[0], axis=1)
            tt = stats.ttest_1samp(sl, 0)
            stats_out[comp][f"surprise_slope_{kind}"] = dict(mean_uV_per_tertile=float(sl.mean()), t=float(tt.statistic),
                                                             p=float(tt.pvalue), n_same_sign=int((np.sign(sl) == np.sign(sl.mean())).sum()))
        sr = stats_out[comp][f"surprise_slope_{BIN_KIND}"]
        ax.text(0.03, 1.0, f"slope t({n - 1}) = {sr['t']:.1f}".replace("-", "−"),
                transform=ax.transAxes, ha="left", va="top", fontsize=5.5)
        if r == 0:
            ax.set_title("ERP by model surprise", pad=9)

    for lab, ax in zip("CDEF", fig.axes[:4]):
        fig.text(ax.get_position().x0 - 0.05, 0.985, lab, fontsize=9, weight="bold", va="top")

    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "Figure2_results.pdf")
    fig.savefig(FIG_DIR / "Figure2_results.svg")
    svg = FIG_DIR / "Figure2_results.svg"            # SVG names fonts: Liberation Sans -> Arial (same metrics)
    svg.write_text(svg.read_text().replace("Liberation Sans", "Arial"))
    fig.savefig(FIG_DIR / "Figure2_results.png", dpi=400)
    (MC_DIR / "figure2_results_stats.json").write_text(json.dumps(stats_out, indent=2))
    print(json.dumps(stats_out, indent=2))
    print(f"Figure -> {FIG_DIR / 'Figure2_results'}.pdf/.svg/.png "
          f"({fig.get_size_inches()[0] * 25.4:.1f} x {fig.get_size_inches()[1] * 25.4:.1f} mm)")


if __name__ == "__main__":
    main()
