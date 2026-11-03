"""
15_figure2_schematic.py — Figure 2 schematic at manuscript size (173.695 x 65 mm).

  A  Concept matrix: the three candidate mechanisms (adaptation, local transition learning,
     global rule learning) plus fixed category labels as pictograms; each model (row) is the
     set of components it contains (filled dots)
  B  The regressors on real trials around a rule switch (one subject), sharing one x-axis:
     M0 category labels, M1 adaptation trace, M2 local surprise, M3 global surprise.
     (M4/M5 are sums of these, so they are not drawn separately.)

Illustrative parameters = group-typical values from 06_model_comparison.py
(adaptation tau = 8 s; local learner: no forgetting within block, reset; global learner:
half-life 10 quintlets, reset). The task itself is in Figure 1, so it is not repeated here.

Needs only sourcedata (no EEG). Editable text: TrueType in PDF, <text> in SVG (font named Arial).
Output: figures/Figure2_schematic.pdf / .svg / .png

Usage:  python scripts/15_figure2_schematic.py [sub-01]
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import seaborn as sns
from scipy.io import loadmat

# ----------------------------------------------------------------------------- CONFIG
ROOT = Path(__file__).resolve().parents[1]
BIDS_ROOT = ROOT / "data" / "bids"
FIG_DIR = ROOT / "figures"
SOUNDS = {1: "A", 2: "B"}
SUBJECT = "sub-01"
SIZE_MM = (173.695, 65.0)
SWITCH_BLOCK = 2                 # show the end of block 1 and the start of block 2
N_BEFORE, N_AFTER = 20, 40       # quintlets either side of the switch
TAU, H_LOC, RESET_LOC, H_GLOB, RESET_GLOB = 8.0, np.inf, True, 10, True
MCOL = {"categories": "0.35", "adaptation": "#55A868", "local": "#C44E52", "global": "#8172B3",
        "local+global": "#937860", "adapt+global": "#64B5CD"}
TEXT_RC = {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
           "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
           "font.size": 7, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
           "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
           "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
           "axes.labelpad": 2}
TONE = {"A": "#4C72B0", "B": "#DD8452"}
CARDS = [  # name, colour key, pictogram(s), short gist
    ("M0  Categories", "categories", ["categories"], "fixed labels, no learning"),
    ("M1  Adaptation", "adaptation", ["adaptation"], "sensory fatigue"),
    ("M2  Local", "local", ["local"], "sound-to-sound prediction"),
    ("M3  Global", "global", ["global"], "rule prediction"),
    ("M4  Local + global", "local+global", ["local", "global"], "both predictions"),
    ("M5  Adapt. + global", "adapt+global", ["adaptation", "global"], "fatigue + rule"),
]
# ------------------------------------------------------------------------------------


def load(subject):
    src = BIDS_ROOT / "sourcedata" / subject / "ses-LGeffect"
    seq = loadmat(next(src.glob("GlobLoc_stimulation_SEQUENCE*.mat")), squeeze_me=True)["SEQUENCE"]
    tz = loadmat(next(src.glob("GlobLoc_critical_events*.mat")), squeeze_me=True)["tz"]
    tz = tz[tz > 0]
    rows = []
    for b, blk in enumerate(seq, start=1):
        blk = np.atleast_2d(blk).astype(int)
        std = "".join(SOUNDS[x] for x in blk[0])
        for i, q in enumerate(blk):
            rows.append(dict(block=b, trial_in_block=i, quintlet="".join(SOUNDS[x] for x in q), std=std))
    qs = pd.DataFrame(rows)
    qs["local"] = np.where(qs["quintlet"].str[3] == qs["quintlet"].str[4], "LS", "LD")
    qs["global"] = np.where(qs["quintlet"] == qs["std"], "GS", "GD")
    return qs, tz


def leaky(outcomes, h, resets):
    """Leaky Beta-Bernoulli learner; returns surprise (bits) and p(change) before each outcome."""
    lam = 1.0 if np.isinf(h) else 2.0 ** (-1.0 / h)
    n1 = n0 = 0.0
    S, MU = np.empty(len(outcomes)), np.empty(len(outcomes))
    for t, y in enumerate(outcomes):
        if resets is not None and resets[t]:
            n1 = n0 = 0.0
        mu = (n1 + 1) / (n1 + n0 + 2)
        MU[t], S[t] = mu, -np.log2(mu if y else 1 - mu)
        n1, n0 = lam * n1 + y, lam * n0 + (1 - y)
    return S, MU


def adaptation(times, ids, tau):
    """Stimulus-specific adaptation trace of each sound just before it is played."""
    a = {"A": 0.0, "B": 0.0}
    last, out = times[0], np.empty(len(times))
    for i, (t, k) in enumerate(zip(times, ids)):
        dec = np.exp(-(t - last) / tau)
        a["A"] *= dec
        a["B"] *= dec
        out[i] = a[k]
        a[k] += 1
        last = t
    return out


def regressors(qs, tz):
    first, fifth = qs["quintlet"].str[0].to_numpy(), qs["quintlet"].str[4].to_numpy()
    start = (qs["trial_in_block"] == 0).to_numpy()
    lo = np.zeros((len(qs), 4), int)          # 4 transitions per quintlet; only 4->5 can change
    lo[:, 3] = fifth != first
    lo = lo.ravel()
    lr = np.zeros(len(lo), bool)
    lr[::4] = start
    S_loc, _ = leaky(lo, H_LOC, lr if RESET_LOC else None)
    S_glob, _ = leaky((fifth != first).astype(int), H_GLOB, start if RESET_GLOB else None)
    ids = np.array([c for q in qs["quintlet"] for c in q])
    qs = qs.copy()
    qs["S_loc"], qs["S_glob"] = S_loc[3::4], S_glob
    qs["adapt"] = adaptation(tz, ids, TAU)[4::5]
    return qs


def _quint(ax, x, y, q, w, gap, h):
    for c in q:
        ax.add_patch(Rectangle((x, y), w, h if c == "A" else h * 1.4, fc=TONE[c], ec="none"))
        x += w + gap
    return x


def pictogram(ax, kind, x0, y0, w, h):
    """Conceptual icon for each model component, drawn inside the box (x0, y0, w, h) in inches."""
    if kind == "categories":                          # 2 x 2 table of fixed levels
        s = min(w, h) * 0.38
        cx, cy = x0 + w / 2 + 0.04, y0 + h / 2 - 0.03
        for i, lv in enumerate([0.95, 0.75, 0.6, 0.35]):
            r, c = divmod(i, 2)
            ax.add_patch(Rectangle((cx - s + c * s, cy - r * s), s * 0.94, s * 0.94, fc=str(lv), ec="none"))
        if w < 0.8:
            return
        ax.text(cx - s / 2, cy + s + 0.03, "LS", ha="center", fontsize=5, color="0.4")
        ax.text(cx + s / 2, cy + s + 0.03, "LD", ha="center", fontsize=5, color="0.4")
        ax.text(cx - s - 0.03, cy + s / 2, "GS", ha="right", va="center", fontsize=5, color="0.4")
        ax.text(cx - s - 0.03, cy - s / 2, "GD", ha="right", va="center", fontsize=5, color="0.4")
    elif kind == "adaptation":                        # trace jumps at each sound, decays between
        t = np.linspace(0, 1, 300)
        ev = [0.08, 0.16, 0.24, 0.32, 0.62, 0.70]
        tr = np.zeros_like(t)
        for e in ev:
            tr += (t >= e) * np.exp(-(t - e) / 0.18)
        tr /= tr.max()
        ax.plot(x0 + 0.04 + t * (w - 0.08), y0 + 0.08 + tr * (h - 0.2), color=MCOL["adaptation"], lw=0.9)
        for e in ev:
            ax.plot([x0 + 0.04 + e * (w - 0.08)] * 2, [y0 + 0.02, y0 + 0.07], color="0.3", lw=0.7)
        ax.plot([x0 + 0.04, x0 + w - 0.04], [y0 + 0.045, y0 + 0.045], color="0.6", lw=0.4)
    elif kind == "local":                             # five sounds; arcs between neighbours
        tw, gap = w * 0.1, w * 0.07
        xs0 = x0 + (w - (5 * tw + 4 * gap)) / 2
        yb = y0 + h * 0.18
        _quint(ax, xs0, yb, "AAAAB", tw, gap, h * 0.28)
        for k in range(4):
            xa, xb = xs0 + k * (tw + gap) + tw / 2, xs0 + (k + 1) * (tw + gap) + tw / 2
            last = k == 3
            ax.add_patch(matplotlib.patches.Arc(((xa + xb) / 2, yb + h * 0.45), xb - xa, h * 0.28, theta1=0,
                                                theta2=180, color=MCOL["local"] if last else "0.55",
                                                lw=1.1 if last else 0.6))
        ax.text(xs0 + 3.5 * (tw + gap) + tw / 2 - gap / 2, yb + h * 0.66, "?", ha="center", fontsize=7,
                color=MCOL["local"], weight="bold")
    elif kind == "global":                            # whole quintlets; the rule is learned across them
        tw, gap, qg = w * 0.028, w * 0.012, w * 0.06
        qw = 5 * tw + 4 * gap
        xs0 = x0 + (w - (3 * qw + 2 * qg)) / 2
        yb = y0 + h * 0.18
        for k, q in enumerate(["AAAAA", "AAAAA", "AAAAB"]):
            _quint(ax, xs0 + k * (qw + qg), yb, q, tw, gap, h * 0.28)
        xe = xs0 + 2 * (qw + qg)
        ax.plot([xs0, xs0, xe - qg * 0.5, xe - qg * 0.5], [yb + h * 0.5, yb + h * 0.56, yb + h * 0.56, yb + h * 0.5],
                color=MCOL["global"], lw=0.8)
        ax.text((xs0 + xe) / 2 - qg / 2, yb + h * 0.6, "rule", ha="center", fontsize=5.5, color=MCOL["global"])
        ax.text(xe + qw / 2, yb + h * 0.48, "?", ha="center", fontsize=7, color=MCOL["global"], weight="bold")


COMPONENTS = [("categories", "categories"), ("adaptation", "adaptation"), ("local", "local"),
              ("global", "global")]           # (pictogram, colour key) for the matrix columns
MODEL_SETS = [("M0", "categories", {"categories"}), ("M1", "adaptation", {"adaptation"}),
              ("M2", "local", {"local"}), ("M3", "global", {"global"})]
# M4 (local + global) and M5 (adaptation + global) were also fitted (preregistered) but are not
# separable from their components in model recovery; they are named in the legend, not drawn here.


def draw_matrix(ax, W, H):
    """Panel A in inch coordinates: component pictograms as columns, models as rows, dots = included."""
    lab_w, icon_h, top = 0.3, 0.72, H - 0.12
    cw = (W - lab_w) / len(COMPONENTS)
    rh = (top - icon_h - 0.1) / len(MODEL_SETS)
    for j, (pic, ck) in enumerate(COMPONENTS):
        x0 = lab_w + j * cw
        ax.add_patch(FancyBboxPatch((x0 + 0.03, top - icon_h), cw - 0.06, icon_h,
                                    boxstyle="round,pad=0,rounding_size=0.04", fc="0.97", ec="none"))
        pictogram(ax, pic, x0 + 0.05, top - icon_h + 0.04, cw - 0.1, icon_h - 0.08)
    for i, (name, mk, comps) in enumerate(MODEL_SETS):
        yc = top - icon_h - 0.1 - (i + 0.5) * rh
        if i % 2 == 0:
            ax.add_patch(Rectangle((0, yc - rh / 2), W, rh, fc="0.965", ec="none", zorder=0))
        ax.text(lab_w - 0.06, yc, name, ha="right", va="center", fontsize=7, weight="bold", color=MCOL[mk])
        on = [j for j, (pic, _) in enumerate(COMPONENTS) if pic in comps]
        if len(on) == 2:
            ax.plot([lab_w + (on[0] + 0.5) * cw, lab_w + (on[1] + 0.5) * cw], [yc, yc], color=MCOL[mk], lw=1.2,
                    zorder=1)
        for j, (pic, ck) in enumerate(COMPONENTS):
            xc = lab_w + (j + 0.5) * cw
            if pic in comps:
                ax.add_patch(matplotlib.patches.Circle((xc, yc), 0.055, fc=MCOL[mk], ec="none", zorder=2))
            else:
                ax.add_patch(matplotlib.patches.Circle((xc, yc), 0.028, fc="0.85", ec="none", zorder=2))


def main(subject):
    qs, tz = load(subject)
    qs = regressors(qs, tz)
    sw = qs.index[(qs["block"] == SWITCH_BLOCK) & (qs["trial_in_block"] == 0)][0]
    sns.set_theme(style="ticks", rc=TEXT_RC)
    W, H = SIZE_MM[0] / 25.4, SIZE_MM[1] / 25.4
    fig = plt.figure(figsize=(W, H))

    # ---------------- A: model cards (inch coordinates)
    wa = 2.55
    axA = fig.add_axes([0.0, 0.0, wa / W, 1.0])
    axA.set_axis_off()
    axA.set_xlim(-0.02, wa)
    axA.set_ylim(0, H)
    draw_matrix(axA, wa - 0.06, H)
    fig.text(0.0, 0.995, "A", fontsize=9, weight="bold", va="top")

    # ---------------- B: regressors on real trials around a rule switch
    seg = qs.loc[sw - N_BEFORE: sw + N_AFTER - 1]
    xs = np.arange(len(seg)) - N_BEFORE
    gd = seg["global"].eq("GD").to_numpy()
    ld = seg["local"].eq("LD").to_numpy()
    rows = [("categories", "M0", None), ("adaptation", "M1", "adapt"),
            ("local", "M2", "S_loc"), ("global", "M3", "S_glob")]
    x_left, x_right, y_bot, y_top = (wa + 0.42) / W, 0.995, 0.17, 0.90
    h = (y_top - y_bot) / len(rows)
    axes = []
    for r, (key, lab, col) in enumerate(rows):
        a = fig.add_axes([x_left, y_top - (r + 1) * h + 0.012, x_right - x_left, h - 0.024],
                         sharex=axes[0] if axes else None)
        axes.append(a)
        if key == "categories":
            a.scatter(xs[ld], np.full(ld.sum(), 1), marker="|", s=18, lw=0.9, color=MCOL["local"])
            a.scatter(xs[gd], np.full(gd.sum(), 0), marker="|", s=18, lw=0.9, color=MCOL["global"])
            a.set_yticks([0, 1], ["GD", "LD"])
            a.set_ylim(-0.7, 1.7)
        else:
            y = seg[col].to_numpy()
            a.plot(xs, y, color=MCOL[key], lw=0.8, marker="o", ms=1.6, mew=0)
            a.scatter(xs[gd], y[gd], s=9, facecolor="none", edgecolor="k", lw=0.5, zorder=3)
            a.set_yticks([])
            a.spines["left"].set_visible(False)
        a.axvline(-0.5, color="k", lw=0.6, ls="--")
        a.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=6.5, color=MCOL[key], weight="bold")
        a.set_xlim(xs[0] - 1, xs[-1] + 1)
        if r < len(rows) - 1:
            plt.setp(a.get_xticklabels(), visible=False)
        sns.despine(ax=a)
    axes[-1].set_xticks([-20, -10, 0, 10, 20, 30])
    axes[-1].set_xlabel("Trials from rule switch")
    axes[0].text(-1.0, 1.9, f"rule: {qs.loc[sw - 1, 'std']}", ha="right", va="bottom", fontsize=5.8,
                 transform=axes[0].transData)
    axes[0].text(0.5, 1.9, f"rule: {qs.loc[sw, 'std']}", ha="left", va="bottom", fontsize=5.8)
    fig.text(wa / W + 0.005, 0.995, "B", fontsize=9, weight="bold", va="top")

    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "Figure2_schematic.pdf")
    fig.savefig(FIG_DIR / "Figure2_schematic.svg")
    svg = FIG_DIR / "Figure2_schematic.svg"          # SVG names fonts: Liberation Sans -> Arial (same metrics)
    svg.write_text(svg.read_text().replace("Liberation Sans", "Arial"))
    fig.savefig(FIG_DIR / "Figure2_schematic.png", dpi=400)
    print(f"Figure -> {FIG_DIR / 'Figure2_schematic'}.pdf/.svg/.png "
          f"({fig.get_size_inches()[0] * 25.4:.1f} x {fig.get_size_inches()[1] * 25.4:.1f} mm)")
    print("block standards:", qs.loc[sw - 1, "std"], "->", qs.loc[sw, "std"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else SUBJECT)
