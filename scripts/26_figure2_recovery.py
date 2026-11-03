"""
26_figure2_recovery.py — Figure 2B: model recovery (50 x 50 mm).

Confusion matrices from the recovery simulation in 06_model_comparison.py (--recovery): data generated
from each model with each participant's own fitted parameters, betas and noise, then passed through
the full pipeline (nested leave-one-block-out CV -> random-effects BMS). Cell = % of simulations
(20 per generating model) in which each model had the highest protected exceedance probability.
Left: early window (80–140 ms); right: late window (300–700 ms). M4/M5 faded as in panel D.

Input : derivatives/model_comparison/recovery.tsv
Output: figures/Figure2_recovery.pdf / .svg / .png

Usage:  python scripts/26_figure2_recovery.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------------- CONFIG
ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "derivatives" / "model_comparison" / "recovery.tsv"
FIG_DIR = ROOT / "figures"
SIZE_MM = (50.0, 50.0)
MODELS = ["M0 categories", "M1 adaptation", "M2 local", "M3 global", "M4 local+global", "M5 adapt+global"]
SHORT = [m.split()[0] for m in MODELS]
MCOL = {"M0": "0.35", "M1": "#55A868", "M2": "#C44E52", "M3": "#8172B3", "M4": "#937860", "M5": "#64B5CD"}
FAINT = {"M4", "M5"}
WINDOWS = [("local_80-140ms", "Early"), ("global_300-700ms", "Late")]
TEXT_RC = {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
           "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
           "font.size": 6, "axes.titlesize": 6.5, "axes.labelsize": 6, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
           "axes.linewidth": 0.5, "xtick.major.pad": 1, "ytick.major.pad": 1, "axes.labelpad": 1.5}
# ------------------------------------------------------------------------------------


def colour_ticks(labels):
    for tl in labels:
        m = tl.get_text()
        tl.set_color(MCOL[m])
        tl.set_alpha(0.45 if m in FAINT else 1.0)


def main():
    d = pd.read_csv(REC, sep="\t")
    sns.set_theme(style="white", rc=TEXT_RC)
    mm = 1 / 25.4
    fig, axes = plt.subplots(1, 2, figsize=(SIZE_MM[0] * mm, SIZE_MM[1] * mm),
                             gridspec_kw=dict(left=0.14, right=0.99, bottom=0.2, top=0.86, wspace=0.1))
    for k, (ax, (comp, lab)) in enumerate(zip(axes, WINDOWS)):
        g = d[d["component"] == comp]
        M = pd.crosstab(g["generating"], g["winner_pxp"], normalize="index").reindex(
            index=MODELS, columns=MODELS, fill_value=0).to_numpy() * 100
        ax.imshow(M, cmap="Greys", vmin=0, vmax=100, aspect="auto")
        for i in range(6):
            for j in range(6):
                if M[i, j] >= 5:
                    ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=4.6,
                            color="white" if M[i, j] > 55 else "0.15")
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=MCOL[SHORT[i]], lw=0.8))
        ax.set_xticks(range(6), SHORT, rotation=90)
        ax.set_yticks(range(6), SHORT if k == 0 else [])
        colour_ticks(ax.get_xticklabels())
        if k == 0:
            colour_ticks(ax.get_yticklabels())
            ax.set_ylabel("Generating model")
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(lab, pad=2)
    fig.text(0.56, 0.035, "Recovered model (% of simulations)", ha="center", fontsize=6)
    fig.text(0.565, 0.985, "Model recovery", ha="center", va="top", fontsize=6.5)

    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "Figure2_recovery.pdf")
    fig.savefig(FIG_DIR / "Figure2_recovery.svg")
    svg = FIG_DIR / "Figure2_recovery.svg"           # SVG names fonts: Liberation Sans -> Arial (same metrics)
    svg.write_text(svg.read_text().replace("Liberation Sans", "Arial"))
    fig.savefig(FIG_DIR / "Figure2_recovery.png", dpi=600)
    print(f"Figure -> {FIG_DIR / 'Figure2_recovery'}.pdf/.svg/.png")


if __name__ == "__main__":
    main()
