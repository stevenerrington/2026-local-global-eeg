"""
08_model_schematic.py — explanatory figure for Header 1: the task, the six models, what each model's
regressor represents trial by trial, and the model-comparison pipeline.

Panel A  stimulus stream around a rule switch (real sequence from one subject)
Panel B  the six models: what each tracks, its regressor at the 5th sound, its free parameters
Panel C  the regressors themselves on the same real trials (illustrative parameters = group-typical
         values from 06: adaptation tau = 8 s; local learner: no forgetting within block, reset;
         global learner: half-life 10 quintlets, reset)
Panel D  comparison pipeline: nested leave-one-block-out CV -> CV log-likelihood -> random-effects
         BMS (PXP), with the model-recovery check

Needs only sourcedata (no EEG). Output: figures/header1_model_schematic.png

Usage:  python scripts/08_model_schematic.py [sub-01]
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import seaborn as sns
from scipy.io import loadmat

# ----------------------------------------------------------------------------- CONFIG
ROOT = Path(__file__).resolve().parents[1]
BIDS_ROOT = ROOT / "data" / "bids"
FIG_DIR = ROOT / "figures"
SOUNDS = {1: "A", 2: "B"}
SUBJECT = "sub-01"
SWITCH_BLOCK = 2            # show the end of block SWITCH_BLOCK-1 and the start of SWITCH_BLOCK
N_BEFORE, N_AFTER = 25, 45  # quintlets shown either side of the switch (panel C)
TAU, H_LOC, RESET_LOC, H_GLOB, RESET_GLOB = 8.0, np.inf, True, 10, True
TONE = {"A": "#4C72B0", "B": "#DD8452"}
MCOL = {"categories": "0.35", "adaptation": "#55A868", "local": "#C44E52", "global": "#8172B3",
        "local+global": "#937860", "adapt+global": "#64B5CD"}
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
    lo = np.zeros((len(qs), 4), int)
    lo[:, 3] = fifth != first
    lo = lo.ravel()
    lr = np.zeros(len(lo), bool)
    lr[::4] = start
    S_loc, mu_loc = leaky(lo, H_LOC, lr if RESET_LOC else None)
    S_glob, mu_glob = leaky((fifth != first).astype(int), H_GLOB, start if RESET_GLOB else None)
    ids = np.array([c for q in qs["quintlet"] for c in q])
    ad = adaptation(tz, ids, TAU)
    qs = qs.copy()
    qs["S_loc"], qs["p_change_loc"] = S_loc[3::4], mu_loc[3::4]
    qs["S_glob"], qs["p_change_glob"] = S_glob, mu_glob
    qs["adapt"] = ad[4::5]
    return qs


def box(ax, x, y, w, h, text, fc, ec="k", fs=8, weight="normal", ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc=fc, ec=ec, lw=1))
    ax.text(x + (w / 2 if ha == "center" else 0.012), y + h / 2, text, ha=ha, va="center",
            fontsize=fs, weight=weight, wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, lw=1.2,
                                 color="0.25"))


def main(subject):
    qs, tz = load(subject)
    qs = regressors(qs, tz)
    sw = qs.index[(qs["block"] == SWITCH_BLOCK) & (qs["trial_in_block"] == 0)][0]
    sns.set_theme(context="notebook", style="ticks")
    fig = plt.figure(figsize=(14, 18))
    gs = fig.add_gridspec(4, 1, height_ratios=[0.75, 2.3, 1.9, 1.05], hspace=0.32)

    # ---------------- A: stimulus stream
    ax = fig.add_subplot(gs[0])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0, 1.02, "A   Task: quintlets of chords; the block rule sets which quintlet is the global standard",
            fontsize=11, weight="bold", va="bottom")
    show = list(range(sw - 4, sw)) + list(range(sw, sw + 2))
    x = 0.01
    tw, gap_t, gap_q = 0.0135, 0.0035, 0.02
    for k, i in enumerate(show):
        if i == sw:
            ax.add_patch(Rectangle((x - 0.004, 0.18), 0.075, 0.62, fc="0.93", ec="none"))
            ax.text(x + 0.033, 0.49, "5-s silence\nnew block\n(rule flips)", ha="center", va="center", fontsize=8)
            x += 0.085
        q = qs.loc[i]
        x0 = x
        for j, c in enumerate(q["quintlet"]):
            hh = 0.36 if c == "A" else 0.5
            ax.add_patch(Rectangle((x, 0.3), tw, hh, fc=TONE[c], ec="k" if j == 4 else "none",
                                   lw=1.4 if j == 4 else 0))
            x += tw + gap_t
        ax.text((x0 + x) / 2, 0.2, f"{q['local']}/{q['global']}", ha="center", va="top", fontsize=8.5,
                color="#C44E52" if q["global"] == "GD" else "0.2", weight="bold" if q["global"] == "GD" else "normal")
        x += gap_q
    ax.text(0.01, 0.92, f"Block {SWITCH_BLOCK - 1}: standard {qs.loc[sw - 1, 'std']}", fontsize=9)
    ax.text(0.47, 0.92, f"Block {SWITCH_BLOCK}: standard {qs.loc[sw, 'std']}", fontsize=9)
    ax.text(0.73, 0.82, "■ A  350/700/1400 Hz", color=TONE["A"], fontsize=8.5)
    ax.text(0.73, 0.70, "■ B  500/1000/2000 Hz", color=TONE["B"], fontsize=8.5)
    ax.text(0.73, 0.56, "50-ms chords, SOA ≈ 157 ms; 800–1100 ms between quintlets\n"
            "outlined = 5th sound (t = 0 for all analyses)\n"
            "LS / LD: 5th sound repeats / changes (local level)\n"
            "GS / GD: quintlet is the block standard / deviant (global level)\n"
            "≥ 10 standards open each block; deviants 20%",
            fontsize=7.5, va="top")

    # ---------------- B: model cards
    ax = fig.add_subplot(gs[1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0, 1.0, "B   Six models of the single-trial response to the 5th sound "
            "(all also include an intercept and a linear drift)", fontsize=11, weight="bold", va="bottom")
    cards = [
        ("M0  categories", "categories",
         "No learning. The 2x2 description of\nthe design.",
         "LD (±½), GD (±½), LD×GD", "none"),
        ("M1  adaptation", "adaptation",
         "Stimulus-specific adaptation: each\nchord leaves a trace that decays in\nreal time. No inference.",
         "A(t) = Σ exp(−Δt / τ) over past\npresentations of the 5th chord",
         "τ ∈ {0.25 … 32 s}"),
        ("M2  local learner", "local",
         "Bayesian learner of tone-to-tone\ntransitions (repeat vs change), any\nposition in the quintlet.",
         "S_loc = −log₂ p(4→5 transition)\np(change) from leaky Beta counts",
         "half-life {10, 50, 200, ∞}\ntransitions × block reset"),
        ("M3  global learner", "global",
         "Bayesian learner of the quintlet rule:\n'does the 5th sound differ from the\nfirst four?' One update per quintlet.",
         "S_glob = −log₂ p(5th change | rule)\np(change) from leaky Beta counts",
         "half-life {2, 5, 10, 20, ∞}\nquintlets × block reset"),
        ("M4  local + global", "local+global",
         "Both learners in parallel; their\ncontributions are additive.",
         "S_loc + S_glob", "both sets"),
        ("M5  adaptation + global", "adapt+global",
         "Adaptation for the local level,\ninference for the global level.",
         "A(t) + S_glob", "τ and global set"),
    ]
    cw, ch = 0.315, 0.45
    for k, (name, key, what, reg, par) in enumerate(cards):
        r, c = divmod(k, 3)
        x0, y0 = 0.005 + c * (cw + 0.02), 0.5 - r * (ch + 0.04)
        ax.add_patch(FancyBboxPatch((x0, y0), cw, ch, boxstyle="round,pad=0.006,rounding_size=0.015",
                                    fc="white", ec=MCOL[key], lw=2))
        ax.add_patch(Rectangle((x0, y0 + ch - 0.075), cw, 0.075, fc=MCOL[key], ec="none", alpha=0.9))
        ax.text(x0 + 0.01, y0 + ch - 0.037, name, color="white", weight="bold", fontsize=10, va="center")
        ax.text(x0 + 0.01, y0 + ch - 0.095, "Represents", fontsize=7.5, color="0.4", va="top")
        ax.text(x0 + 0.01, y0 + ch - 0.12, what, fontsize=8.2, va="top")
        ax.text(x0 + 0.01, y0 + 0.225, "Regressor at the 5th sound", fontsize=7.5, color="0.4", va="top")
        ax.text(x0 + 0.01, y0 + 0.195, reg, fontsize=8.2, va="top", family="DejaVu Sans")
        ax.text(x0 + 0.01, y0 + 0.085, "Free parameters (chosen inside CV)", fontsize=7.5, color="0.4", va="top")
        ax.text(x0 + 0.01, y0 + 0.058, par.replace("\n", " "), fontsize=8, va="top")

    # ---------------- C: regressor traces on real trials
    sub_gs = gs[2].subgridspec(4, 1, hspace=0.12)
    seg = qs.loc[sw - N_BEFORE: sw + N_AFTER - 1].copy()
    xs = np.arange(len(seg)) - N_BEFORE
    gd = seg["global"].eq("GD").to_numpy()
    ld = seg["local"].eq("LD").to_numpy()
    rows = [("categories", "Categories\n(M0)", None),
            ("adaptation", "Adaptation trace\nof 5th chord (M1)", "adapt"),
            ("local", "Local surprise\nS_loc, bits (M2)", "S_loc"),
            ("global", "Global surprise\nS_glob, bits (M3)", "S_glob")]
    for r, (key, lab, col) in enumerate(rows):
        a = fig.add_subplot(sub_gs[r])
        if r == 0:
            a.set_title("C   What each regressor represents, trial by trial (real sequence, "
                        f"{subject}, around a rule switch)", fontsize=11, weight="bold", loc="left")
        if key == "categories":
            a.scatter(xs[ld], np.full(ld.sum(), 1), marker="|", s=120, color="#C44E52")
            a.scatter(xs[gd], np.full(gd.sum(), 0), marker="|", s=120, color="#8172B3")
            a.set_yticks([0, 1], ["GD", "LD"])
            a.set_ylim(-0.6, 1.6)
        else:
            a.plot(xs, seg[col], color=MCOL[key], lw=1.4, marker="o", ms=3)
            a.scatter(xs[gd], seg[col].to_numpy()[gd], s=28, facecolor="none", edgecolor="k", zorder=3, lw=0.9)
        a.axvline(-0.5, color="k", lw=1, ls="--")
        a.set_ylabel(lab, fontsize=8.5, rotation=0, ha="right", va="center", labelpad=10)
        a.set_xlim(xs[0] - 1, xs[-1] + 1)
        if r < 3:
            a.set_xticklabels([])
        else:
            a.set_xlabel("Quintlet relative to block start (dashed line = rule switch; open circles = global deviants)")
        sns.despine(ax=a)
    a0 = fig.axes[-4]
    a0.text(1.0, 1.25, "local surprise is high for every 5th-sound change, scaled by how often changes have\n"
            "occurred in this block; global surprise is high for whichever quintlet breaks the current\n"
            "rule, and is relearned after the switch (reset); adaptation follows physical repetition only",
            transform=a0.transAxes, ha="right", va="bottom", fontsize=8, color="0.25")

    # ---------------- D: pipeline
    ax = fig.add_subplot(gs[3])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0, 1.0, "D   Model comparison (per response component: frontocentral 80–140 ms; parietal 300–700 ms)",
            fontsize=11, weight="bold", va="bottom")
    steps = [
        ("Single-trial EEG\nmean amplitude in the\na-priori ROI × window", "#E8EEF7"),
        ("Design matrix per model\nintercept + drift +\nmodel regressor(s), z-scored", "#E8EEF7"),
        ("Nested leave-one-block-out CV\nouter: predict held-out block\ninner: choose τ / half-life / reset\non the 7 training blocks only", "#FFF2CC"),
        ("Score per subject × model\nsummed out-of-sample\nGaussian log-likelihood", "#E8EEF7"),
        ("Random-effects BMS\n(Stephan 2009; Rigoux 2014)\n→ protected exceedance\nprobability (PXP)", "#E2F0D9"),
    ]
    bw, bh, y0 = 0.17, 0.5, 0.38
    for k, (txt, fc) in enumerate(steps):
        x0 = 0.005 + k * (bw + 0.03)
        box(ax, x0, y0, bw, bh, txt, fc, fs=8.3)
        if k < len(steps) - 1:
            arrow(ax, x0 + bw + 0.003, y0 + bh / 2, x0 + bw + 0.027, y0 + bh / 2)
    box(ax, 0.25, 0.0, 0.62, 0.25,
        "Recovery check (run first): simulate each subject's data from each model with that subject's own\n"
        "fitted parameters, betas and noise; re-run the whole pipeline; confusion matrix of winners.\n"
        "Rule: a winner needs PXP > 0.90 and ≥ 80% recovery when it generated the data.", "#FBE5D6", fs=7.8)
    arrow(ax, 0.6, 0.26, 0.6, 0.37)

    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "header1_model_schematic.png", dpi=150, bbox_inches="tight")
    print(f"Figure -> {FIG_DIR / 'header1_model_schematic.png'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else SUBJECT)
