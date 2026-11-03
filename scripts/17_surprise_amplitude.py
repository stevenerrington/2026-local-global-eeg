"""
17_surprise_amplitude.py — model surprise vs single-trial amplitude (binned scatter), 173.695 x 80 mm.

Rows = windows (early frontocentral 80–140 ms; late parietal 300–700 ms).
Columns:
  1  the window's own learner (early: local surprise, M2; late: global surprise, M3), all trials
  2  the other learner's surprise after removing the own learner's (added-variable plot: amplitude
     and other surprise both residualised on own surprise) — what the other learner adds, if anything
  3  own learner, within condition: surprise and amplitude both centred within each 2x2 condition,
     so only trial-by-trial learning remains (what the category labels cannot express)

Each participant's trials are split into deciles of surprise (rank-based); faint points = one
participant's deciles, black = group mean ± SE per decile. Amplitude is centred per participant.
Line = mean of the per-participant single-trial OLS fits; slope (µV per bit) tested against zero
across participants. Surprise uses each participant's own selected parameters from 06
(M2 parameters chosen on the early window, M3 on the late window), as in 07.

Inputs: derivatives/model_comparison/{trials/sub-*.tsv, clocks.tsv}; sourcedata sequences
Output: figures/Figure2_surprise_scatter.pdf / .svg / .png; derivatives/model_comparison/surprise_slopes.json

Usage:  python scripts/17_surprise_amplitude.py
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
from scipy.io import loadmat

# ----------------------------------------------------------------------------- CONFIG
ROOT = Path(__file__).resolve().parents[1]
BIDS_ROOT = ROOT / "data" / "bids"
MC_DIR = ROOT / "derivatives" / "model_comparison"
FIG_DIR = ROOT / "figures"
SOUNDS = {1: "A", 2: "B"}
SIZE_MM = (173.695, 80.0)
N_BINS = 10
MCOL = {"S_loc": "#C44E52", "S_glob": "#8172B3"}
NAME = {"S_loc": "M2 local surprise (bits)", "S_glob": "M3 global surprise (bits)"}
ROWS = [("local_80-140ms", "Early\n80–140 ms", "S_loc", "S_glob"),
        ("global_300-700ms", "Late\n300–700 ms", "S_glob", "S_loc")]
TEXT_RC = {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
           "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
           "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
           "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
           "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
           "axes.labelpad": 2}
# ------------------------------------------------------------------------------------


def load_sequence(subject):
    src = BIDS_ROOT / "sourcedata" / subject / "ses-LGeffect"
    seq = loadmat(next(src.glob("GlobLoc_stimulation_SEQUENCE*.mat")), squeeze_me=True)["SEQUENCE"]
    rows = []
    for b, blk in enumerate(seq, start=1):
        for i, q in enumerate(np.atleast_2d(blk).astype(int)):
            rows.append(dict(block=b, trial_in_block=i, quintlet="".join(SOUNDS[x] for x in q)))
    qs = pd.DataFrame(rows)
    qs.insert(0, "seq_idx", np.arange(len(qs)))
    return qs


def leaky_surprise(outcomes, half_life, resets):
    """Leaky Beta-Bernoulli learner: surprise (bits) of each outcome before updating."""
    lam = 1.0 if np.isinf(half_life) else 2.0 ** (-1.0 / half_life)
    n1 = n0 = 0.0
    S = np.empty(len(outcomes))
    for t, y in enumerate(outcomes):
        if resets is not None and resets[t]:
            n1 = n0 = 0.0
        mu = (n1 + 1) / (n1 + n0 + 2)
        S[t] = -np.log2(mu if y else 1 - mu)
        n1, n0 = lam * n1 + y, lam * n0 + (1 - y)
    return S


def learner_surprise(subject, clocks):
    """S_loc and S_glob per quintlet with the participant's own selected parameters."""
    qs = load_sequence(subject)
    c = clocks[clocks["subject"] == subject]
    p_loc = c[(c["component"] == "local_80-140ms") & (c["model"] == "M2 local")].iloc[0]
    p_glo = c[(c["component"] == "global_300-700ms") & (c["model"] == "M3 global")].iloc[0]
    first, fifth = qs["quintlet"].str[0].to_numpy(), qs["quintlet"].str[4].to_numpy()
    start = (qs["trial_in_block"] == 0).to_numpy()
    loc_out = np.zeros((len(qs), 4), int)              # 4 transitions per quintlet; only 4->5 can change
    loc_out[:, 3] = fifth != first
    loc_out = loc_out.ravel()
    loc_reset = np.zeros(len(loc_out), bool)
    loc_reset[::4] = start
    qs["S_loc"] = leaky_surprise(loc_out, float(p_loc["param"]), loc_reset if bool(p_loc["reset"]) else None)[3::4]
    qs["S_glob"] = leaky_surprise((fifth != first).astype(int), float(p_glo["param"]),
                                  start if bool(p_glo["reset"]) else None)
    return qs[["seq_idx", "S_loc", "S_glob"]]


def _resid(v, z):
    Z = np.column_stack([np.ones(len(z)), z])
    return v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]


def prepare(d, comp, var, within, control=None):
    """Per-participant x (surprise) and y (amplitude): centred per participant, centred within
    condition (within=True), or residualised on another surprise (control=column name)."""
    x, y = d[var].astype(float), d[comp].astype(float)
    if control is not None:
        z = d[control].to_numpy(float)
        return _resid(x.to_numpy(), z), _resid(y.to_numpy(), z)
    if within:
        x = x - d.groupby("condition")[var].transform("mean")
        y = y - d.groupby("condition")[comp].transform("mean")
    else:
        y = y - y.mean()
    return x.to_numpy(), y.to_numpy()


def main():
    clocks = pd.read_csv(MC_DIR / "clocks.tsv", sep="\t")
    subs = sorted(p.stem for p in (MC_DIR / "trials").glob("sub-*.tsv"))
    data = {s: pd.read_csv(MC_DIR / "trials" / f"{s}.tsv", sep="\t").merge(
        learner_surprise(s, clocks), on="seq_idx", how="left", validate="one_to_one") for s in subs}
    n = len(subs)

    sns.set_theme(style="ticks", rc=TEXT_RC)
    mm = 1 / 25.4
    fig, axes = plt.subplots(2, 3, figsize=(SIZE_MM[0] * mm, SIZE_MM[1] * mm),
                             gridspec_kw=dict(left=0.11, right=0.99, bottom=0.11, top=0.9, wspace=0.42, hspace=0.55))
    out = {}
    cols = [("own", "Own learner"), ("other", "Other learner, beyond own"), ("within", "Own learner, within condition")]
    for r, (comp, rowlab, own, other) in enumerate(ROWS):
        for c, (kind, title) in enumerate(cols):
            ax = axes[r, c]
            var = own if kind != "other" else other
            bx, by, slopes, icpt = [], [], [], []
            for s in subs:
                x, y = prepare(data[s], comp, var, kind == "within", control=own if kind == "other" else None)
                b = pd.qcut(pd.Series(x).rank(method="first"), N_BINS, labels=False).to_numpy()
                mx = np.array([x[b == k].mean() for k in range(N_BINS)])
                my = np.array([y[b == k].mean() for k in range(N_BINS)])
                bx.append(mx)
                by.append(my)
                ax.plot(mx, my, "o", ms=1.4, mew=0, color=MCOL[var], alpha=0.3, zorder=1)
                sl, ic = np.polyfit(x, y, 1)
                slopes.append(sl)
                icpt.append(ic)
            bx, by, slopes = np.array(bx), np.array(by), np.array(slopes)
            ax.errorbar(bx.mean(0), by.mean(0), yerr=by.std(0, ddof=1) / np.sqrt(n), fmt="o", ms=2.6, lw=0.7,
                        color="k", zorder=3)
            xx = np.linspace(np.percentile(bx, 1), np.percentile(bx, 99), 50)
            ax.plot(xx, np.mean(icpt) + slopes.mean() * xx, color=MCOL[var], lw=1.1, zorder=2)
            ax.axhline(0, color="k", lw=0.4)
            tt = stats.ttest_1samp(slopes, 0)
            same = int((np.sign(slopes) == np.sign(slopes.mean())).sum())
            ptxt = "p < .001" if tt.pvalue < 0.001 else f"p = {tt.pvalue:.3f}".replace("0.", ".")
            ax.text(0.98, 0.97 if r == 1 else 0.03,
                    f"{slopes.mean():+.2f} µV/bit\nt({n - 1}) = {tt.statistic:.1f}, {ptxt}\n{same}/{n}".replace("-", "−"),
                    transform=ax.transAxes, ha="right", va="top" if r == 1 else "bottom", fontsize=5.5,
                    linespacing=1.15)
            xl = {"own": NAME[var], "other": NAME[var].replace("(bits)", "(residual)"),
                  "within": NAME[var].replace("(bits)", "(centred)")}[kind]
            ax.set_xlabel(xl, color=MCOL[var])
            if c == 0:
                ax.set_ylabel("Amplitude (µV)")
                ax.text(-0.42, 0.5, rowlab, transform=ax.transAxes, rotation=90, ha="center", va="center",
                        fontsize=7, weight="bold")
            if r == 0:
                ax.set_title(title, pad=4)
            sns.despine(ax=ax)
            out[f"{comp}|{kind}|{var}"] = dict(slope_uV_per_bit=float(slopes.mean()), t=float(tt.statistic),
                                               p=float(tt.pvalue), n_same_sign=same)
    for k, ax in enumerate(axes[0]):
        fig.text(ax.get_position().x0 - (0.085 if k == 0 else 0.05), 0.985, "ABC"[k], fontsize=9, weight="bold",
                 va="top")

    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "Figure2_surprise_scatter.pdf")
    fig.savefig(FIG_DIR / "Figure2_surprise_scatter.svg")
    svg = FIG_DIR / "Figure2_surprise_scatter.svg"   # SVG names fonts: Liberation Sans -> Arial (same metrics)
    svg.write_text(svg.read_text().replace("Liberation Sans", "Arial"))
    fig.savefig(FIG_DIR / "Figure2_surprise_scatter.png", dpi=400)
    (MC_DIR / "surprise_slopes.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"Figure -> {FIG_DIR / 'Figure2_surprise_scatter'}.pdf/.svg/.png")


if __name__ == "__main__":
    main()
