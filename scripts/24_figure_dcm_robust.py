"""
24_figure_dcm_robust.py — DCM figure built on robustness (173.695 x 78 mm).

Five analyses of the same data: main (0–600 ms, free-orientation dipoles, window mean removed) and
four variants from dcm_04_robustness.m — win400 (0–400 ms), iplsup (IPL node moved to angular gyrus),
h0 (window mean not removed), img (cortical patches with anatomically fixed orientation).
A connection change counts as supported in an analysis if its group posterior (PEB + BMR, Bayesian model
average) has Pp > .95. Support = number of analyses (of 5) in which it is credible with the majority sign.

  A/B  Local / global deviance network: coloured arrows = supported in ≥ 3 of 5 analyses (orange =
       stronger, blue = weaker than for standards), width = number of supporting analyses; grey = < 3.
  C    Every connection supported in ≥ 3 analyses for either effect: estimate per analysis
       (log gain; colour) with credible estimates outlined, and the support count.
  D    Model fit per analysis: variance explained per participant (in the DCM's spatial modes).

Inputs : derivatives/dcm/robustness/robustness_summary_corrected.csv (all five analyses, Pp recomputed
         from each BMA), derivatives/dcm/fits/DCM_*_full.mat, derivatives/dcm/robustness/<variant>/DCM_*.mat
Outputs: figures/Figure_DCM_robust.pdf / .svg / .png; derivatives/dcm/robustness/support_table.csv

Usage:  python scripts/24_figure_dcm_robust.py
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from scipy.io import loadmat

# ----------------------------------------------------------------------------- CONFIG
ROOT = Path(__file__).resolve().parents[1]
DCM_DIR = ROOT / "derivatives" / "dcm"
FIG_DIR = ROOT / "figures"
SIZE_MM = (173.695, 78.0)
ANALYSES = ["main", "win400", "iplsup", "h0", "img"]
ALAB = {"main": "Main", "win400": "0–400 ms", "iplsup": "IPL moved", "h0": "No mean\nremoval",
        "img": "Fixed-orient.\npatches"}
FIT_GLOB = {"main": "fits/DCM_sub-*_full.mat", **{v: f"robustness/{v}/DCM_sub-*.mat" for v in ANALYSES[1:]}}
PP_THR, MIN_SUPPORT = 0.95, 3
SRC = ["lA1", "rA1", "lSTG", "rSTG", "lIFG", "rIFG", "lIPL", "rIPL"]
LABEL = {s: s[1:] for s in SRC}
POS = {"lA1": (-1.0, 0.0), "lSTG": (-1.0, 1.0), "lIFG": (-1.55, 2.05), "lIPL": (-0.45, 2.05),
       "rA1": (1.0, 0.0), "rSTG": (1.0, 1.0), "rIFG": (1.55, 2.05), "rIPL": (0.45, 2.05)}
FWD = [("lA1", "lSTG"), ("rA1", "rSTG"), ("lSTG", "lIFG"), ("rSTG", "rIFG"), ("lSTG", "lIPL"), ("rSTG", "rIPL")]
LAT = [("lIFG", "lIPL"), ("rIFG", "rIPL")]
C_UP, C_DOWN, C_NS = "#D55E00", "#0072B2", "0.85"
C_EFF = {"local": "#C44E52", "global": "#8172B3"}
TEXT_RC = {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
           "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
           "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
           "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
           "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
           "axes.labelpad": 2}
# ------------------------------------------------------------------------------------


def dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x, float)


def support_table(d):
    """Per effect x connection: estimates per analysis, credible flags, majority sign, support count."""
    rows = []
    for (eff, conn), g in d.groupby(["effect", "connection"]):
        g = g.set_index("variant").reindex(ANALYSES)
        cred = g["Pp"] > PP_THR
        signs = np.sign(g["Ep"][cred])
        sign = int(np.sign(signs.sum())) if cred.any() and signs.sum() != 0 else 0
        k = int(((np.sign(g["Ep"]) == sign) & cred).sum()) if sign else 0
        rows.append(dict(effect=eff, connection=conn, sign=sign, support=k,
                         **{f"Ep_{a}": g.loc[a, "Ep"] for a in ANALYSES},
                         **{f"cred_{a}": bool(cred[a]) for a in ANALYSES}))
    return pd.DataFrame(rows)


def r2_per_analysis():
    out = {}
    for a, pat in FIT_GLOB.items():
        vals = []
        for f in sorted(DCM_DIR.glob(pat)):
            D = loadmat(f, squeeze_me=True, struct_as_record=False)["DCM"]
            H = [dense(h) for h in np.atleast_1d(D.H)]
            R = [dense(x) for x in np.atleast_1d(D.R)]
            vals.append(1 - sum((x ** 2).sum() for x in R) / sum(((h + x) ** 2).sum() for h, x in zip(H, R)))
        out[a] = np.array(vals)
    return out


def draw_network(ax, sup, eff):
    ax.set_xlim(-2.05, 2.05)
    ax.set_ylim(-0.55, 2.45)
    ax.set_aspect("equal")
    ax.set_axis_off()
    for a, b in LAT:
        (x0, y0), (x1, y1) = POS[a], POS[b]
        ax.plot([x0, x1], [y0, y1], ls=(0, (2, 2)), color="0.75", lw=0.6, zorder=1)
    for side in ("l", "r"):
        x, y = POS[f"{side}A1"]
        ax.add_patch(FancyArrowPatch((x, y - 0.5), (x, y - 0.2), arrowstyle="-|>", mutation_scale=6,
                                     color="0.45", lw=0.7))
    s = sup[sup["effect"] == eff].set_index("connection")
    for lo, hi in FWD:
        for p0, p1 in ((lo, hi), (hi, lo)):                                   # forward, then backward
            row = s.loc[f"{p0}->{p1}"]
            ok = row["support"] >= MIN_SUPPORT
            col = (C_UP if row["sign"] > 0 else C_DOWN) if ok else C_NS
            lw = 0.9 * row["support"] - 1.2 if ok else 0.5
            ax.add_patch(FancyArrowPatch(POS[p0], POS[p1], connectionstyle="arc3,rad=0.28", arrowstyle="-|>",
                                         mutation_scale=5 + 2 * lw, lw=lw, color=col, shrinkA=8, shrinkB=8,
                                         zorder=3 if ok else 2))
    for name, (x, y) in POS.items():
        ax.add_patch(FancyBboxPatch((x - 0.22, y - 0.13), 0.44, 0.26, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc="white", ec="0.3", lw=0.7, zorder=4))
        ax.text(x, y, LABEL[name], ha="center", va="center", fontsize=6.5, zorder=5)
    ax.text(-1.0, 2.42, "Left", ha="center", va="top", fontsize=6, color="0.4")
    ax.text(1.0, 2.42, "Right", ha="center", va="top", fontsize=6, color="0.4")


def pretty(conn):
    a, b = conn.split("->")
    return f"{a[0].upper()} {a[1:]} → {b[1:]}"


def main():
    d = pd.read_csv(DCM_DIR / "robustness" / "robustness_summary_corrected.csv")
    sup = support_table(d)
    sup.to_csv(DCM_DIR / "robustness" / "support_table.csv", index=False)
    r2 = r2_per_analysis()

    sns.set_theme(style="ticks", rc=TEXT_RC)
    mm = 1 / 25.4
    fig = plt.figure(figsize=(SIZE_MM[0] * mm, SIZE_MM[1] * mm))
    # ---------------- A/B networks
    for k, (eff, lab) in enumerate((("local", "A"), ("global", "B"))):
        ax = fig.add_axes([0.0 + k * 0.255, 0.13, 0.25, 0.80])
        draw_network(ax, sup, eff)
        ax.set_title(f"{eff.capitalize()} deviance", color=C_EFF[eff], weight="bold", pad=0)
        fig.text(0.005 + k * 0.255, 0.985, lab, fontsize=9, weight="bold", va="top")
    # legend
    x = 0.02
    for col, txt in ((C_UP, "stronger"), (C_DOWN, "weaker")):
        fig.add_artist(plt.Line2D([x, x + 0.022], [0.06, 0.06], color=col, lw=1.6, transform=fig.transFigure))
        fig.text(x + 0.026, 0.06, txt, va="center", fontsize=6)
        x += 0.085
    for kk in (3, 4, 5):
        fig.add_artist(plt.Line2D([x, x + 0.022], [0.06, 0.06], color="0.3", lw=0.9 * kk - 1.2,
                                  transform=fig.transFigure))
        fig.text(x + 0.026, 0.06, f"{kk}/5", va="center", fontsize=6)
        x += 0.06
    fig.add_artist(plt.Line2D([x, x + 0.022], [0.06, 0.06], color=C_NS, lw=0.6, transform=fig.transFigure))
    fig.text(x + 0.026, 0.06, "< 3/5", va="center", fontsize=6)
    fig.text(0.02, 0.015, "Analyses supporting the change (Pp > .95, same sign)", fontsize=5.5, color="0.4")

    # ---------------- C: estimate matrix for supported connections
    keep = sup[sup["support"] >= MIN_SUPPORT].copy()
    keep["order"] = keep["effect"].map({"local": 0, "global": 1})
    keep = keep.sort_values(["order", "support"], ascending=[True, False]).reset_index(drop=True)
    axC = fig.add_axes([0.60, 0.16, 0.20, 0.74])
    E = keep[[f"Ep_{a}" for a in ANALYSES]].to_numpy(float)
    lim = np.ceil(np.nanmax(np.abs(E)) * 10) / 10
    im = axC.imshow(E, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(len(keep)):
        for j, a in enumerate(ANALYSES):
            if keep.loc[i, f"cred_{a}"]:
                axC.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="k", lw=0.7))
        axC.text(len(ANALYSES) - 0.35, i, f"{keep.loc[i, 'support']}/5", va="center", fontsize=5.5)
    axC.set_yticks(range(len(keep)), [pretty(c) for c in keep["connection"]], fontsize=5.5)
    for tl, eff in zip(axC.get_yticklabels(), keep["effect"]):
        tl.set_color(C_EFF[eff])
    axC.set_xticks(range(len(ANALYSES)), [ALAB[a] for a in ANALYSES], rotation=90, fontsize=5.5)
    axC.tick_params(length=0)
    nloc = int((keep["effect"] == "local").sum())
    axC.axhline(nloc - 0.5, color="k", lw=0.8)
    for sp in axC.spines.values():
        sp.set_visible(False)
    cax = fig.add_axes([0.60, 0.935, 0.11, 0.02])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=5, length=1.5, pad=1)
    cb.set_label("log gain", fontsize=5.5, labelpad=1)
    cb.ax.xaxis.set_label_position("top")
    fig.text(0.49, 0.985, "C", fontsize=9, weight="bold", va="top")

    # ---------------- D: fit quality
    axD = fig.add_axes([0.915, 0.30, 0.08, 0.60])
    for j, a in enumerate(ANALYSES):
        v = r2[a]
        axD.scatter(np.full(len(v), j) + np.random.default_rng(j).uniform(-0.15, 0.15, len(v)), v, s=3,
                    color="0.55", lw=0)
        axD.plot([j - 0.3, j + 0.3], [np.median(v)] * 2, color="k", lw=1)
    axD.set_xticks(range(len(ANALYSES)), [ALAB[a].replace("\n", " ") for a in ANALYSES], rotation=90, fontsize=5)
    axD.set_ylim(0.7, 1.0)
    axD.set_ylabel("Variance explained")
    sns.despine(ax=axD)
    fig.text(0.85, 0.985, "D", fontsize=9, weight="bold", va="top")

    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "Figure_DCM_robust.pdf")
    fig.savefig(FIG_DIR / "Figure_DCM_robust.svg")
    svg = FIG_DIR / "Figure_DCM_robust.svg"
    svg.write_text(svg.read_text().replace("Liberation Sans", "Arial"))
    fig.savefig(FIG_DIR / "Figure_DCM_robust.png", dpi=400)
    print(keep[["effect", "connection", "sign", "support"]].to_string(index=False))
    print(json.dumps({a: dict(median=float(np.median(v)), min=float(v.min()), n=len(v)) for a, v in r2.items()}, indent=1))


if __name__ == "__main__":
    main()
