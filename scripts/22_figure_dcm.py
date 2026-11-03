"""
22_figure_dcm.py — DCM figure (173.695 x 68 mm): how local and global deviance change the network.

  A  Local deviance: group (PEB, Bayesian model average) modulation of each forward / backward
     connection; coloured = posterior probability > .95 (orange = stronger, blue = weaker than for
     standards), width ∝ |effect| (log scale: +0.5 ≈ 1.6x); grey = not credible. Dashed = fixed
     lateral IFG–IPL connections; arrows below A1 = auditory input.
  B  Global deviance: same.
  C  Family comparison (which connection types each effect modulates): posterior probability.
  D  Left IPL -> STG backward connection per participant: modulation by local vs global deviance
     (individual DCM estimates; lines join a participant), the connection whose modulation reverses.
  Fit quality (variance explained per participant) is reported in derivatives/dcm/figure_dcm_stats.json.

Inputs : derivatives/dcm/peb/{BMR.mat, family_local.mat, family_global.mat}, derivatives/dcm/fits/DCM_*_full.mat
Outputs: figures/Figure_DCM.pdf / .svg / .png; derivatives/dcm/figure_dcm_stats.json

Usage:  python scripts/22_figure_dcm.py
"""
from pathlib import Path
import json
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import seaborn as sns
from scipy.io import loadmat
from scipy import stats

# ----------------------------------------------------------------------------- CONFIG
ROOT = Path(__file__).resolve().parents[1]
DCM_DIR = ROOT / "derivatives" / "dcm"
FIG_DIR = ROOT / "figures"
SIZE_MM = (173.695, 68.0)
SRC = ["lA1", "rA1", "lSTG", "rSTG", "lIFG", "rIFG", "lIPL", "rIPL"]          # order used in dcm_02_fit.m
LABEL = {"lA1": "A1", "rA1": "A1", "lSTG": "STG", "rSTG": "STG", "lIFG": "IFG", "rIFG": "IFG",
         "lIPL": "IPL", "rIPL": "IPL"}
POS = {"lA1": (-1.0, 0.0), "lSTG": (-1.0, 1.0), "lIFG": (-1.55, 2.05), "lIPL": (-0.45, 2.05),
       "rA1": (1.0, 0.0), "rSTG": (1.0, 1.0), "rIFG": (1.55, 2.05), "rIPL": (0.45, 2.05)}
FWD = [(3, 1), (4, 2), (5, 3), (6, 4), (7, 3), (8, 4)]                        # (to, from), 1-based
LAT = [(5, 7), (6, 8)]
PP_THR = 0.95
C_UP, C_DOWN, C_NS = "#D55E00", "#0072B2", "0.82"
C_EFF = {"local": "#C44E52", "global": "#8172B3"}
FAM_NAMES = ["none", "fwd", "bwd", "fwd+bwd"]
KEY = (3, 7)                                                                  # lIPL -> lSTG (to, from)
TEXT_RC = {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
           "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
           "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
           "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
           "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
           "axes.labelpad": 2}
# ------------------------------------------------------------------------------------


def dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def load_bma():
    """Group BMA: dict (effect, to, from) -> (Ep, Pp); Pp = P(|effect| > 0) under the Gaussian posterior."""
    B = loadmat(DCM_DIR / "peb" / "BMR.mat", squeeze_me=True, struct_as_record=False)["BMA"]
    Ep, var = dense(B.Ep).ravel(), dense(B.Cp).ravel()
    var = np.diag(dense(B.Cp)) if dense(B.Cp).ndim == 2 and dense(B.Cp).shape[0] == dense(B.Cp).shape[1] > 1 else var
    sd = np.sqrt(np.clip(var, 0, None))
    out = {}
    for i, name in enumerate(np.atleast_1d(B.Pnames)):
        k, r, c = map(int, re.findall(r"\d+", str(name)))
        pp = stats.norm.cdf(abs(Ep[i]) / sd[i]) if sd[i] > 0 else 0.5
        out[("local" if k == 1 else "global", r, c)] = (float(Ep[i]), float(pp))
    return out


def load_subjects():
    """Per-participant B estimates and variance explained from the individual full DCMs."""
    B, r2 = {}, {}
    for f in sorted((DCM_DIR / "fits").glob("DCM_sub-*_full.mat")):
        sub = f.stem.split("_")[1]
        D = loadmat(f, squeeze_me=True, struct_as_record=False)["DCM"]
        Bk = [np.asarray(b) for b in np.atleast_1d(D.Ep.B)]
        B[sub] = {"local": Bk[0], "global": Bk[1]}
        H = [np.asarray(h) for h in np.atleast_1d(D.H)]
        R = [np.asarray(x) for x in np.atleast_1d(D.R)]
        r2[sub] = float(1 - sum((x ** 2).sum() for x in R) / sum(((h + x) ** 2).sum() for h, x in zip(H, R)))
    return B, r2


def draw_network(ax, bma, eff):
    ax.set_xlim(-2.05, 2.05)
    ax.set_ylim(-0.55, 2.45)
    ax.set_aspect("equal")
    ax.set_axis_off()
    for a, b in LAT:                                                          # fixed lateral connections
        (x0, y0), (x1, y1) = POS[SRC[a - 1]], POS[SRC[b - 1]]
        ax.plot([x0, x1], [y0, y1], ls=(0, (2, 2)), color="0.7", lw=0.6, zorder=1)
    for side in ("l", "r"):                                                   # auditory input
        x, y = POS[f"{side}A1"]
        ax.add_patch(FancyArrowPatch((x, y - 0.5), (x, y - 0.2), arrowstyle="-|>", mutation_scale=6,
                                     color="0.45", lw=0.7))
    for to, frm in FWD:
        for direction, (r, c) in (("fwd", (to, frm)), ("bwd", (frm, to))):
            ep, pp = bma[(eff, r, c)]
            sig = pp > PP_THR
            col = (C_UP if ep > 0 else C_DOWN) if sig else C_NS
            lw = 0.5 + 3.2 * abs(ep) if sig else 0.5
            p0, p1 = POS[SRC[c - 1]], POS[SRC[r - 1]]
            rad = 0.28 if direction == "fwd" else 0.28
            ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                         mutation_scale=5 + 2 * lw, lw=lw, color=col, shrinkA=8, shrinkB=8,
                                         zorder=3 if sig else 2))
    for name, (x, y) in POS.items():
        ax.add_patch(FancyBboxPatch((x - 0.22, y - 0.13), 0.44, 0.26, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc="white", ec="0.3", lw=0.7, zorder=4))
        ax.text(x, y, LABEL[name], ha="center", va="center", fontsize=6.5, zorder=5)
    ax.text(-1.0, 2.42, "Left", ha="center", va="top", fontsize=6, color="0.4")
    ax.text(1.0, 2.42, "Right", ha="center", va="top", fontsize=6, color="0.4")


def main():
    bma = load_bma()
    fam = {e: np.atleast_1d(loadmat(DCM_DIR / "peb" / f"family_{e}.mat", squeeze_me=True,
                                    struct_as_record=False)[f"fam{e[0].upper()}"].family.post).astype(float)
           for e in ("local", "global")}
    Bs, r2 = load_subjects()
    subs = sorted(Bs)
    n = len(subs)

    sns.set_theme(style="ticks", rc=TEXT_RC)
    mm = 1 / 25.4
    fig = plt.figure(figsize=(SIZE_MM[0] * mm, SIZE_MM[1] * mm))
    axA = fig.add_axes([0.0, 0.1, 0.30, 0.84])
    axB = fig.add_axes([0.31, 0.1, 0.30, 0.84])
    for ax, eff, lab in ((axA, "local", "A"), (axB, "global", "B")):
        draw_network(ax, bma, eff)
        ax.set_title(f"{eff.capitalize()} deviance", color=C_EFF[eff], weight="bold", pad=2)
        fig.text(ax.get_position().x0 + 0.005, 0.985, lab, fontsize=9, weight="bold", va="top")
    # legend for the networks
    lx = 0.05
    for col, txt in ((C_UP, "stronger"), (C_DOWN, "weaker"), (C_NS, "n.s.")):
        fig.add_artist(plt.Line2D([lx, lx + 0.025], [0.045, 0.045], color=col, lw=1.6, transform=fig.transFigure))
        fig.text(lx + 0.03, 0.045, txt, va="center", fontsize=6)
        lx += 0.1
    fig.text(lx + 0.01, 0.045, f"(Pp > {PP_THR:.2f}; line width scales with effect size)", va="center", fontsize=6, color="0.4")

    # ---------------- C: family posteriors
    axC = fig.add_axes([0.69, 0.62, 0.29, 0.28])
    w = 0.38
    x = np.arange(4)
    for k, eff in enumerate(("local", "global")):
        axC.bar(x + (k - 0.5) * w, fam[eff], width=w, color=C_EFF[eff], label=eff)
    axC.set_xticks(x, FAM_NAMES)
    axC.set_ylim(0, 1.05)
    axC.set_yticks([0, 0.5, 1])
    axC.set_ylabel("Family posterior")
    axC.legend(frameon=False, loc="upper left", handlelength=1, borderaxespad=0.1, fontsize=6)
    sns.despine(ax=axC)
    fig.text(0.625, 0.985, "C", fontsize=9, weight="bold", va="top")
    axC.set_title("Connection types modulated", pad=3)

    # ---------------- D: key connection per participant
    axD = fig.add_axes([0.69, 0.13, 0.29, 0.33])
    to, frm = KEY
    loc = np.array([Bs[s]["local"][to - 1, frm - 1] for s in subs])
    glo = np.array([Bs[s]["global"][to - 1, frm - 1] for s in subs])
    for a, b in zip(loc, glo):
        axD.plot([0, 1], [a, b], color="0.75", lw=0.5, zorder=1)
    axD.scatter(np.zeros(n), loc, s=8, color=C_EFF["local"], zorder=2, lw=0)
    axD.scatter(np.ones(n), glo, s=8, color=C_EFF["global"], zorder=2, lw=0)
    axD.axhline(0, color="k", lw=0.4)
    tt = stats.ttest_rel(loc, glo)
    axD.set_xticks([0, 1], ["local", "global"])
    axD.set_xlim(-0.4, 1.4)
    axD.set_ylabel("Modulation (log gain)")
    axD.set_title(f"Left IPL → STG (backward)", pad=3)
    ptxt = "p < .001" if tt.pvalue < 0.001 else f"p = {tt.pvalue:.3f}".replace("0.", ".")
    axD.text(0.5, 1.0, f"t({n - 1}) = {tt.statistic:.1f}, {ptxt}\n{int((loc > glo).sum())}/{n} local > global",
             transform=axD.transAxes, ha="center", va="top", fontsize=5.5)
    sns.despine(ax=axD)
    fig.text(0.625, 0.52, "D", fontsize=9, weight="bold", va="top")

    out = dict(r2_per_subject=r2, r2_median=float(np.median(list(r2.values()))),
               r2_range=[float(min(r2.values())), float(max(r2.values()))],
               family_posterior={e: dict(zip(FAM_NAMES, map(float, fam[e]))) for e in fam},
               bma={f"{e}|{SRC[c - 1]}->{SRC[r - 1]}": dict(Ep=v[0], Pp=v[1]) for (e, r, c), v in bma.items()},
               key_connection=dict(name="lIPL->lSTG", local_mean=float(loc.mean()), global_mean=float(glo.mean()),
                                   n_local_pos=int((loc > 0).sum()), n_global_neg=int((glo < 0).sum()),
                                   paired_t=float(tt.statistic), p=float(tt.pvalue)))
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "Figure_DCM.pdf")
    fig.savefig(FIG_DIR / "Figure_DCM.svg")
    svg = FIG_DIR / "Figure_DCM.svg"                 # SVG names fonts: Liberation Sans -> Arial (same metrics)
    svg.write_text(svg.read_text().replace("Liberation Sans", "Arial"))
    fig.savefig(FIG_DIR / "Figure_DCM.png", dpi=400)
    (DCM_DIR / "figure_dcm_stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "bma"}, indent=2))


if __name__ == "__main__":
    main()
