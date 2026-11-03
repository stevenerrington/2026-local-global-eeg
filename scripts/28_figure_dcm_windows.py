"""
28_figure_dcm_windows.py — DCM figure: full response (0–600 ms) vs MMN window (0–250 ms), 173.695 x 80 mm.

Question: does global deviance really modulate the auditory forward connection (A1 -> STG), or is that
forced by the model? The only driving input enters at A1, so over 0–600 ms the late P3b of global
deviants has to pass through A1 -> STG. Fitting 0–250 ms removes the P3b.

Each window was fitted with two codings of the between-condition effects (dcm_05_surprise.m, 0–600 ms;
dcm_06_early.m, 0–250 ms):
  surprise     local / global = each participant's condition-mean learner surprise (M2 / M3)
  interaction  0/1 labels + local x global term (global = global effect on local standards only)

Credible = posterior probability of being present > .95 after per-connection Bayesian model reduction
(Pp_search). If a CSV has no Pp_search column (dcm_05 output before dcm_07_search_full.m was run), the
sign probability of the family-level BMA (Pp_sign) is used instead and a warning is printed.
Robust = credible with the same sign in BOTH codings.

  A–D  Networks: rows local / global deviance, columns 0–600 / 0–250 ms. Coloured arrows = robust
       (orange stronger, blue weaker than for standards; width ~ mean |log gain| over codings);
       thin grey = credible in only one coding or with opposite signs; absent = not credible.
  E    Every connection credible in at least one coding x window: log gain per coding x window,
       credible estimates outlined.
  F    Fit: variance explained per participant (skipped if the DCM .mat files are not present).

Inputs : derivatives/dcm/surprise_coded/<coding>/connections.csv (0–600 ms),
         derivatives/dcm/early_window/<coding>/connections.csv (0–250 ms), and their DCM_sub-*.mat
Outputs: figures/Figure_DCM_windows.pdf / .svg / .png; derivatives/dcm/early_window/window_comparison.csv

Usage:  python scripts/28_figure_dcm_windows.py
"""
from pathlib import Path

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
SIZE_MM = (173.695, 80.0)
WINDOWS = {"full": ("surprise_coded", "0–600 ms"), "early": ("early_window", "0–250 ms")}
CODINGS = {"surprise": "Surprise", "interaction": "Interaction"}
EFFECTS = ["local", "global"]
PP_THR = 0.95
SRC = ["lA1", "rA1", "lSTG", "rSTG", "lIFG", "rIFG", "lIPL", "rIPL"]
POS = {"lA1": (-1.0, 0.0), "lSTG": (-1.0, 1.0), "lIFG": (-1.55, 2.05), "lIPL": (-0.45, 2.05),
       "rA1": (1.0, 0.0), "rSTG": (1.0, 1.0), "rIFG": (1.55, 2.05), "rIPL": (0.45, 2.05)}
FWD = [("lA1", "lSTG"), ("rA1", "rSTG"), ("lSTG", "lIFG"), ("rSTG", "rIFG"), ("lSTG", "lIPL"), ("rSTG", "rIPL")]
LAT = [("lIFG", "lIPL"), ("rIFG", "rIPL")]
C_UP, C_DOWN, C_NS = "#D55E00", "#0072B2", "0.72"
C_EFF = {"local": "#C44E52", "global": "#8172B3"}
TEXT_RC = {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
           "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
           "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
           "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
           "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
           "axes.labelpad": 2}
# ------------------------------------------------------------------------------------


def load_all():
    """One long table: window, coding, effect, connection, Ep, credible."""
    out = []
    for win, (sub, _) in WINDOWS.items():
        for cod in CODINGS:
            d = pd.read_csv(DCM_DIR / sub / cod / "connections.csv")
            if "Pp_search" in d:
                ep, pp = d["Ep_search"], d["Pp_search"]
            else:
                print(f"WARNING: {sub}/{cod} has no per-connection BMR; using Pp_sign "
                      f"(run scripts/dcm/dcm_07_search_full.m)")
                ep, pp = d["Ep"], d["Pp"]
            out.append(pd.DataFrame(dict(window=win, coding=cod, effect=d["effect"], connection=d["connection"],
                                         Ep=ep, credible=pp > PP_THR)))
    d = pd.concat(out, ignore_index=True)
    return d[d["effect"].isin(EFFECTS)]


def robustness(d):
    """Per window x effect x connection: robust (credible, same sign, both codings) and mean log gain."""
    rows = []
    for (win, eff, conn), g in d.groupby(["window", "effect", "connection"]):
        g = g.set_index("coding").reindex(list(CODINGS))
        cred = g["credible"].astype(bool)
        same = np.sign(g["Ep"]).nunique() == 1
        rows.append(dict(window=win, effect=eff, connection=conn, n_credible=int(cred.sum()),
                         robust=bool(cred.all() and same), sign=int(np.sign(g["Ep"].mean())),
                         Ep_mean=float(g["Ep"].mean()),
                         **{f"Ep_{c}": float(g.loc[c, "Ep"]) for c in CODINGS},
                         **{f"cred_{c}": bool(cred[c]) for c in CODINGS}))
    return pd.DataFrame(rows)


def r2_by_window():
    out = {}
    for win, (sub, _) in WINDOWS.items():
        for cod in CODINGS:
            vals = []
            for f in sorted((DCM_DIR / sub / cod).glob("DCM_sub-*.mat")):
                D = loadmat(f, squeeze_me=True, struct_as_record=False, variable_names=["DCM"])["DCM"]
                H = np.concatenate([np.ravel(h) for h in np.atleast_1d(D.H)])
                R = np.concatenate([np.ravel(x) for x in np.atleast_1d(D.R)])
                vals.append(1 - np.sum(R ** 2) / np.sum((H + R) ** 2))
            if vals:
                out[(win, cod)] = np.array(vals)
    return out


def draw_network(ax, rob, win, eff):
    ax.set_xlim(-2.05, 2.05)
    ax.set_ylim(-0.55, 2.4)
    ax.set_aspect("equal")
    ax.set_axis_off()
    for a, b in LAT:
        (x0, y0), (x1, y1) = POS[a], POS[b]
        ax.plot([x0, x1], [y0, y1], ls=(0, (2, 2)), color="0.8", lw=0.5, zorder=1)
    for side in ("l", "r"):
        x, y = POS[f"{side}A1"]
        ax.add_patch(FancyArrowPatch((x, y - 0.5), (x, y - 0.17), arrowstyle="-|>", mutation_scale=6,
                                     color="0.45", lw=0.7))
    s = rob[(rob["window"] == win) & (rob["effect"] == eff)].set_index("connection")
    for lo, hi in FWD:
        for p0, p1 in ((lo, hi), (hi, lo)):                                   # forward, then backward
            row = s.loc[f"{p0}->{p1}"]
            if row["robust"]:
                col = C_UP if row["sign"] > 0 else C_DOWN
                lw = 0.6 + 1.6 * min(abs(row["Ep_mean"]), 1.2)
                z = 3
            elif row["n_credible"] > 0:
                col, lw, z = C_NS, 0.5, 2
            else:
                continue
            ax.add_patch(FancyArrowPatch(POS[p0], POS[p1], connectionstyle="arc3,rad=0.28", arrowstyle="-|>",
                                         mutation_scale=4 + 2 * lw, lw=lw, color=col, shrinkA=8, shrinkB=8,
                                         zorder=z))
    for name, (x, y) in POS.items():
        ax.add_patch(FancyBboxPatch((x - 0.22, y - 0.13), 0.44, 0.26, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc="white", ec="0.3", lw=0.6, zorder=4))
        ax.text(x, y, name[1:], ha="center", va="center", fontsize=5.8, zorder=5)
    ax.text(-1.45, -0.5, "L", ha="center", va="bottom", fontsize=5.5, color="0.45")
    ax.text(1.45, -0.5, "R", ha="center", va="bottom", fontsize=5.5, color="0.45")


def pretty(conn):
    a, b = conn.split("->")
    return f"{a[0].upper()} {a[1:]} → {b[1:]}"


def main():
    d = load_all()
    rob = robustness(d)
    rob.to_csv(DCM_DIR / "early_window" / "window_comparison.csv", index=False)
    for win in WINDOWS:
        for eff in EFFECTS:
            r = rob[(rob.window == win) & (rob.effect == eff) & rob.robust]
            print(f"{WINDOWS[win][1]:9s} {eff:6s} robust: " +
                  ", ".join(f"{c} {e:+.2f}" for c, e in zip(r.connection, r.Ep_mean)))
    r2 = r2_by_window()

    sns.set_theme(style="ticks", rc=TEXT_RC)
    mm = 1 / 25.4
    fig = plt.figure(figsize=(SIZE_MM[0] * mm, SIZE_MM[1] * mm))

    # ---------------- A–D: networks (rows = effect, columns = window)
    W, H = 0.205, 0.42
    letters = iter("ABCD")
    for i, eff in enumerate(EFFECTS):
        for j, win in enumerate(WINDOWS):
            x0, y0 = 0.035 + j * (W + 0.035), 0.50 - i * (H + 0.02)
            ax = fig.add_axes([x0, y0, W, H])
            draw_network(ax, rob, win, eff)
            fig.text(x0 - 0.012, y0 + H + 0.01, next(letters), fontsize=9, weight="bold", va="top")
            if i == 0:
                ax.set_title(WINDOWS[win][1], pad=1)
        fig.text(0.012, 0.50 - i * (H + 0.02) + H / 2, f"{eff.capitalize()} deviance", rotation=90,
                 ha="center", va="center", color=C_EFF[eff], weight="bold")
    # legend
    x = 0.04
    for col, txt in ((C_UP, "stronger"), (C_DOWN, "weaker")):
        fig.add_artist(plt.Line2D([x, x + 0.02], [0.03, 0.03], color=col, lw=1.6, transform=fig.transFigure))
        fig.text(x + 0.024, 0.03, txt, va="center", fontsize=5.8)
        x += 0.075
    fig.add_artist(plt.Line2D([x, x + 0.02], [0.03, 0.03], color=C_NS, lw=0.6, transform=fig.transFigure))
    fig.text(x + 0.024, 0.03, "one coding only", va="center", fontsize=5.8)

    # ---------------- E: estimate matrix
    cols = [(w, c) for w in WINDOWS for c in CODINGS]
    keep = rob.groupby(["effect", "connection"]).filter(lambda g: g["n_credible"].sum() > 0)
    wide = []
    for (eff, conn), g in keep.groupby(["effect", "connection"]):
        g = g.set_index("window")
        wide.append(dict(effect=eff, connection=conn,
                         **{f"{w}|{c}": g.loc[w, f"Ep_{c}"] for w, c in cols},
                         **{f"cred {w}|{c}": g.loc[w, f"cred_{c}"] for w, c in cols},
                         order=(EFFECTS.index(eff), SRC.index(conn.split("->")[0]),
                                SRC.index(conn.split("->")[1]))))
    wide = pd.DataFrame(wide).sort_values("order").reset_index(drop=True)
    axE = fig.add_axes([0.60, 0.17, 0.17, 0.73])
    E = wide[[f"{w}|{c}" for w, c in cols]].to_numpy(float)
    lim = np.ceil(np.nanmax(np.abs(E)) * 10) / 10
    im = axE.imshow(E, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(len(wide)):
        for j, (w, c) in enumerate(cols):
            if wide.loc[i, f"cred {w}|{c}"]:
                axE.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="k", lw=0.6))
    axE.set_yticks(range(len(wide)), [pretty(c) for c in wide["connection"]], fontsize=5.2)
    for tl, eff in zip(axE.get_yticklabels(), wide["effect"]):
        tl.set_color(C_EFF[eff])
    axE.set_xticks(range(len(cols)), [CODINGS[c] for _, c in cols], rotation=90, fontsize=5.5)
    for j, w in enumerate(WINDOWS):
        axE.text(1.5 * j + 0.5 * j + 0.5, -1.0, WINDOWS[w][1], ha="center", va="bottom", fontsize=5.8)
    axE.axvline(1.5, color="k", lw=0.8)
    axE.axhline(int((wide["effect"] == "local").sum()) - 0.5, color="k", lw=0.8)
    axE.tick_params(length=0)
    for sp in axE.spines.values():
        sp.set_visible(False)
    cax = fig.add_axes([0.785, 0.70, 0.01, 0.2])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=5, length=1.5, pad=1)
    cb.set_label("log gain", fontsize=5.5, labelpad=1)
    fig.text(0.475, 0.985, "E", fontsize=9, weight="bold", va="top")

    # ---------------- F: fit
    if r2:
        axF = fig.add_axes([0.905, 0.30, 0.09, 0.60])
        for j, key in enumerate(cols):
            if key not in r2:
                continue
            v = r2[key]
            axF.scatter(np.full(len(v), j) + np.random.default_rng(j).uniform(-0.15, 0.15, len(v)), v, s=3,
                        color="0.55", lw=0)
            axF.plot([j - 0.3, j + 0.3], [np.median(v)] * 2, color="k", lw=1)
        axF.set_xticks(range(len(cols)), [f"{WINDOWS[w][1].replace(' ms', '')} {CODINGS[c][:4]}." for w, c in cols],
                       rotation=90, fontsize=5)
        axF.set_ylim(0.75, 1.0)
        axF.set_ylabel("Variance explained")
        sns.despine(ax=axF)
        fig.text(0.85, 0.985, "F", fontsize=9, weight="bold", va="top")

    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(FIG_DIR / f"Figure_DCM_windows.{ext}")
    svg = FIG_DIR / "Figure_DCM_windows.svg"             # SVG names fonts: Liberation Sans -> Arial (same metrics)
    svg.write_text(svg.read_text().replace("Liberation Sans", "Arial"))
    fig.savefig(FIG_DIR / "Figure_DCM_windows.png", dpi=600)
    print(f"Figure -> {FIG_DIR / 'Figure_DCM_windows'}.pdf/.svg/.png")


if __name__ == "__main__":
    main()
