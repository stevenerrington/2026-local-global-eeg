"""
07_header1_erps.py — ERP figures for Header 1 (different computations for local and global).

Uses each subject's own selected parameters from 06_model_comparison.py
(derivatives/model_comparison/clocks.tsv): the local transition learner (M2) fitted to the
local component, the global rule learner (M3) fitted to the global component.

Figures (figures/header1_erp_*.png):
  A  standard ERPs: frontocentral LS vs LD and parietal GS vs GD (a-priori windows shaded),
     with group difference topographies in those windows
  B  local deviants binned (tertiles, per subject) by LOCAL-learner surprise — frontocentral ERPs;
     left: all LD trials; right: within-condition surprise (condition mean removed), i.e.
     learning dynamics the 2x2 labels cannot express
  C  global deviants binned by GLOBAL-learner surprise — parietal ERPs, same layout
  D  2x2 condition ERPs at both ROIs, with each learner's predicted window amplitude per
     condition (surprise -> amplitude mapped by single-trial OLS within subject)

Per-subject averages are cached in derivatives/header1_erps/<sub>.npz (subjects already done are
skipped), so the figures can be re-styled without re-reading the epochs (--plot).

Usage:  python scripts/07_header1_erps.py          (compute + plot)
        python scripts/07_header1_erps.py --plot   (plot from cache)
"""
from pathlib import Path
import sys

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
PRE_DIR = ROOT / "derivatives" / "preproc"
MC_DIR = ROOT / "derivatives" / "model_comparison"
OUT_DIR = ROOT / "derivatives" / "header1_erps"
FIG_DIR = ROOT / "figures"
STEM = "ses-LGeffect_task-LocalGlobal"
SOUNDS = {1: "A", 2: "B"}

ROIS = {   # a priori (Rutiku et al. 2024), identical to 06
    "local": (["Fz", "FC1", "FC2", "F1", "F2", "Cz"], (0.08, 0.14)),
    "global": (["Pz", "P1", "P2", "CPz", "POz", "CP1", "CP2"], (0.30, 0.70)),
}
LOWPASS = 30.0          # display filter (Hz)
N_BINS = 3
COL = {"LS": "#4C72B0", "LD": "#C44E52", "GS": "#4C72B0", "GD": "#8172B3"}
BIN_PAL = {"local": ["#F4C7A1", "#E8925A", "#B5541B"], "global": ["#C7C3E3", "#8C84C4", "#4E4596"]}
# ------------------------------------------------------------------------------------


# ============================================================ learner regressors (as in 06)
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
    """Local (M2 @ local component) and global (M3 @ global component) surprise per quintlet."""
    qs = load_sequence(subject)
    first, fifth = qs["quintlet"].str[0].to_numpy(), qs["quintlet"].str[4].to_numpy()
    start = (qs["trial_in_block"] == 0).to_numpy()
    cl = clocks[clocks["subject"] == subject]
    p_loc = cl[(cl["component"].str.startswith("local")) & (cl["model"] == "M2 local")].iloc[0]
    p_glo = cl[(cl["component"].str.startswith("global")) & (cl["model"] == "M3 global")].iloc[0]
    loc_out = np.zeros((len(qs), 4), int)
    loc_out[:, 3] = fifth != first
    loc_out = loc_out.ravel()
    loc_reset = np.zeros(len(loc_out), bool)
    loc_reset[::4] = start
    qs["S_loc"] = leaky_surprise(loc_out, float(p_loc["param"]),
                                 loc_reset if bool(p_loc["reset"]) else None)[3::4]
    qs["S_glob"] = leaky_surprise((fifth != first).astype(int), float(p_glo["param"]),
                                  start if bool(p_glo["reset"]) else None)
    return qs[["seq_idx", "S_loc", "S_glob"]]


def tertiles(x):
    return pd.qcut(pd.Series(x).rank(method="first"), N_BINS, labels=False).to_numpy()


def within(meta, col):
    return (meta[col] - meta.groupby("condition")[col].transform("mean")).to_numpy()


# ============================================================ compute per-subject averages
def compute(subjects=None):
    """Per-subject averages -> derivatives/header1_erps/<sub>.npz (skips subjects already done)."""
    import mne
    mne.set_log_level("ERROR")
    clocks = pd.read_csv(MC_DIR / "clocks.tsv", sep="\t")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sub in subjects or sorted(clocks["subject"].unique()):
        f = OUT_DIR / f"{sub}.npz"
        if f.exists() and "cond4__LSGS__local" in np.load(f).files:
            continue
        ep = mne.read_epochs(PRE_DIR / sub / f"{sub}_{STEM}_epo.fif", preload=True)
        ep.pick("eeg").filter(None, LOWPASS, verbose=False)
        times = ep.times
        if not (OUT_DIR / "info.fif").exists():
            mne.io.write_info(OUT_DIR / "info.fif", ep.info)
        meta = ep.metadata.reset_index(drop=True).merge(learner_surprise(sub, clocks), on="seq_idx",
                                                        how="left", validate="one_to_one")
        X = ep.get_data() * 1e6                                   # trials x ch x time
        roi = {k: X[:, [ep.ch_names.index(c) for c in chs]].mean(1) for k, (chs, _) in ROIS.items()}
        cache, win_rows = {}, []
        for c in ["LS", "LD"]:
            sel = meta["local"].eq(c).to_numpy()
            cache[f"cond|{c}"] = roi["local"][sel].mean(0)
            cache[f"topo|{c}"] = X[sel].mean(0)
        for c in ["GS", "GD"]:
            sel = meta["global"].eq(c).to_numpy()
            cache[f"cond|{c}"] = roi["global"][sel].mean(0)
            cache[f"topo|{c}"] = X[sel].mean(0)
        # 2x2 condition ERPs at both ROIs + learner predictions of the window amplitude:
        # each learner's surprise is mapped to single-trial window amplitude by OLS on all trials
        # (intercept + slope), then averaged per condition — what that learner "expects" the ERP to do
        pred_rows = []
        for comp in ROIS:
            t0, t1 = ROIS[comp][1]
            tm = (times >= t0) & (times <= t1)
            y = roi[comp][:, tm].mean(1)
            preds = {}
            for lname, var in [("local learner", "S_loc"), ("global learner", "S_glob")]:
                Xd = np.column_stack([np.ones(len(y)), meta[var].to_numpy()])
                beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
                preds[lname] = Xd @ beta
            for cond in ["LSGS", "LDGS", "LSGD", "LDGD"]:
                sel = meta["condition"].eq(cond).to_numpy()
                cache[f"cond4|{cond}|{comp}"] = roi[comp][sel].mean(0)
                pred_rows.append(dict(subject=sub, component=comp, condition=cond, source="data",
                                      amp=y[sel].mean()))
                for lname, pv in preds.items():
                    pred_rows.append(dict(subject=sub, component=comp, condition=cond, source=lname,
                                          amp=pv[sel].mean()))
        pd.DataFrame(pred_rows).to_csv(OUT_DIR / f"{sub}_cond4.tsv", sep="\t", index=False)

        # binned ERPs: deviants of each level, by that level's and the other level's surprise
        for dev, col in [("LD", "local"), ("GD", "global")]:
            sel = meta[col].eq(dev).to_numpy()
            m = meta[sel].reset_index(drop=True)
            for var in ["S_loc", "S_glob"]:
                for kind, x in [("raw", m[var].to_numpy()), ("within", within(m, var))]:
                    b = tertiles(x)
                    for k in range(N_BINS):
                        for comp in ROIS:
                            wv = roi[comp][sel][b == k].mean(0)
                            cache[f"bin|{dev}|{var}|{kind}|{comp}|{k}"] = wv
                            t0, t1 = ROIS[comp][1]
                            tm = (times >= t0) & (times <= t1)
                            win_rows.append(dict(subject=sub, deviants=dev, var=var, kind=kind,
                                                 component=comp, bin=k, amp=wv[tm].mean(),
                                                 surprise=x[b == k].mean(), n=int((b == k).sum())))
        np.savez_compressed(OUT_DIR / f"{sub}.npz", times=times, ch_names=np.array(ep.ch_names),
                            **{k.replace("|", "__"): v for k, v in cache.items()})
        pd.DataFrame(win_rows).to_csv(OUT_DIR / f"{sub}_windows.tsv", sep="\t", index=False)
        print(f"{sub}: done", flush=True)


# ============================================================ plotting
def load_cache():
    files = sorted(OUT_DIR.glob("sub-*.npz"))
    d, subs = {}, []
    for f in files:
        sub = f.stem
        z = np.load(f)
        times, chs = z["times"], list(z["ch_names"])
        d.update({f"{sub}|" + k.replace("__", "|"): z[k] for k in z.files if "__" in k})
        subs.append(sub)
    return times, chs, d, subs


def ga(d, subs, key):
    a = np.stack([d[f"{s}|{key}"] for s in subs])
    return a.mean(0), a.std(0, ddof=1) / np.sqrt(len(subs)), a


def wave(ax, t, m, se, color, label, lw=1.8, ls="-"):
    ax.plot(t, m, color=color, lw=lw, ls=ls, label=label)
    ax.fill_between(t, m - se, m + se, color=color, alpha=0.18, lw=0)


def decorate(ax, comp, t):
    t0, t1 = ROIS[comp][1]
    ax.axvspan(t0 * 1e3, t1 * 1e3, color="0.92", zorder=0)
    ax.axvline(0, color="k", lw=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(-100, 700)
    ax.set_xlabel("Time re 5th sound (ms)")
    ax.set_ylabel("µV")


def trend_stats(w, deviants, var, kind, comp):
    d = w[(w.deviants == deviants) & (w["var"] == var) & (w.kind == kind) & (w.component == comp)]
    piv = d.pivot(index="subject", columns="bin", values="amp")
    slopes = piv.apply(lambda r: np.polyfit(np.arange(N_BINS), r.to_numpy(), 1)[0], axis=1)
    r = stats.ttest_1samp(slopes, 0)
    return piv, slopes.mean(), r.statistic, r.pvalue


def plot():
    import mne
    times, chs, d, subs = load_cache()
    info = mne.io.read_info(OUT_DIR / "info.fif")
    w = pd.concat([pd.read_csv(f, sep="\t") for f in sorted(OUT_DIR.glob("sub-*_windows.tsv"))])
    t = times * 1e3
    sns.set_theme(context="notebook", style="ticks")
    FIG_DIR.mkdir(exist_ok=True)
    n = len(subs)

    # ---------------- A: standard ERPs + topographies
    fig = plt.figure(figsize=(11, 6.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1])
    for col, (comp, (std, dev)) in enumerate([("local", ("LS", "LD")), ("global", ("GS", "GD"))]):
        ax = fig.add_subplot(gs[0, 2 * col:2 * col + 2])
        for c in (std, dev):
            m, se, _ = ga(d, subs, f"cond|{c}")
            wave(ax, t, m, se, COL[c], c)
        m, se, _ = ga(d, subs, f"cond|{dev}")
        ms, _, _ = ga(d, subs, f"cond|{std}")
        a = np.stack([d[f"{s}|cond|{dev}"] - d[f"{s}|cond|{std}"] for s in subs])
        wave(ax, t, a.mean(0), a.std(0, ddof=1) / np.sqrt(n), "0.3", f"{dev} − {std}", lw=1.2, ls="--")
        decorate(ax, comp, t)
        roi_txt = ", ".join(ROIS[comp][0])
        ax.set_title(f"{'Local' if comp == 'local' else 'Global'} effect — {roi_txt}", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        # topographies: difference in window, and at a second window for context
        wins = [ROIS[comp][1], (0.15, 0.25) if comp == "local" else (0.08, 0.14)]
        for k, (t0, t1) in enumerate(wins):
            axt = fig.add_subplot(gs[1, 2 * col + k])
            tm = (times >= t0) & (times <= t1)
            diff = np.stack([d[f"{s}|topo|{dev}"] - d[f"{s}|topo|{std}"] for s in subs])[:, :, tm].mean(-1)
            tval = diff.mean(0) / (diff.std(0, ddof=1) / np.sqrt(n))
            mne.viz.plot_topomap(diff.mean(0), info, axes=axt, show=False, cmap="RdBu_r",
                                 vlim=(-np.abs(diff.mean(0)).max(), np.abs(diff.mean(0)).max()),
                                 contours=0, mask=np.abs(tval) > stats.t.ppf(0.975, n - 1),
                                 mask_params=dict(markersize=3))
            axt.set_title(f"{dev} − {std}  {int(t0 * 1e3)}–{int(t1 * 1e3)} ms"
                          + ("  (a priori)" if k == 0 else ""), fontsize=9)
    sns.despine(fig)
    fig.suptitle(f"Local–global ERPs (n = {n}; shaded = a-priori window; dots = |t| > t$_{{crit}}$)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "header1_erp_A_conditions.png", dpi=150)

    # ---------------- B / C: binned ERPs by the matched learner's surprise
    specs = [("B", "local", "LD", "S_loc", "local learner"), ("C", "global", "GD", "S_glob", "global learner")]
    for tag, comp, dev, var, lname in specs:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), gridspec_kw=dict(width_ratios=[1.3, 1.3, 0.8]))
        std = "LS" if comp == "local" else "GS"
        ms, ses, _ = ga(d, subs, f"cond|{std}")
        stats_txt = []
        for ax, kind, title in [(axes[0], "raw", f"All {dev} trials"),
                                (axes[1], "within", "Within condition (condition mean removed)")]:
            wave(ax, t, ms, ses, "0.55", f"{std} (reference)", lw=1.2)
            for k in range(N_BINS):
                m, se, _ = ga(d, subs, f"bin|{dev}|{var}|{kind}|{comp}|{k}")
                wave(ax, t, m, se, BIN_PAL[comp][k], ["low", "mid", "high"][k] + " surprise")
            decorate(ax, comp, t)
            piv, sl, tv, pv = trend_stats(w, dev, var, kind, comp)
            ax.set_title(f"{title}\nslope {sl:+.2f} µV/tertile, t({n - 1}) = {tv:.2f}, p = {pv:.3g}", fontsize=9)
            ax.legend(frameon=False, fontsize=7, loc="upper left")
            stats_txt.append((kind, piv))
        ax = axes[2]
        for kind, piv in stats_txt:
            off = 0 if kind == "raw" else 0.15
            mk = "o" if kind == "raw" else "s"
            for s, r in piv.iterrows():
                ax.plot(np.arange(N_BINS) + off, r.to_numpy(), color="0.85", lw=0.6, zorder=0)
            ax.errorbar(np.arange(N_BINS) + off, piv.mean(), yerr=piv.std(ddof=1) / np.sqrt(n), color="k",
                        marker=mk, ms=5, lw=1.4, capsize=2, label="all" if kind == "raw" else "within")
        ax.set_xticks(range(N_BINS), ["low", "mid", "high"])
        t0, t1 = ROIS[comp][1]
        ax.set(xlabel=f"{lname} surprise tertile", ylabel=f"µV, {int(t0 * 1e3)}–{int(t1 * 1e3)} ms",
               title="Window amplitude", xlim=(-0.3, N_BINS - 0.55))
        ax.legend(frameon=False, fontsize=8)
        sns.despine(fig)
        fig.suptitle(f"{'Local' if comp == 'local' else 'Global'} deviants binned by {lname} surprise "
                     f"({'frontocentral' if comp == 'local' else 'parietal'} ROI)", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"header1_erp_{tag}_{comp}_by_surprise.png", dpi=150)

    # ---------------- D: 2x2 conditions at both ROIs, with each learner's predicted amplitude
    c4 = pd.concat([pd.read_csv(f, sep="\t") for f in sorted(OUT_DIR.glob("sub-*_cond4.tsv"))])
    conds = ["LSGS", "LDGS", "LSGD", "LDGD"]
    cpal = {"LSGS": "#4C72B0", "LDGS": "#C44E52", "LSGD": "#8172B3", "LDGD": "#55A868"}
    clab = {"LSGS": "LS/GS (xxxxx in xxxxx block)", "LDGS": "LD/GS (xxxxY in xxxxY block)",
            "LSGD": "LS/GD (xxxxx in xxxxY block)", "LDGD": "LD/GD (xxxxY in xxxxx block)"}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), gridspec_kw=dict(width_ratios=[1.5, 1]))
    out_stats = {}
    for r, comp in enumerate(["local", "global"]):
        ax = axes[r, 0]
        for cond in conds:
            m, se, _ = ga(d, subs, f"cond4|{cond}|{comp}")
            wave(ax, t, m, se, cpal[cond], clab[cond], lw=1.6)
        decorate(ax, comp, t)
        ax.set_title(f"{'Frontocentral' if comp == 'local' else 'Parietal'} ROI", fontsize=10)
        if r == 0:
            ax.legend(frameon=False, fontsize=7.5, loc="upper left")
        ax = axes[r, 1]
        q = c4[c4["component"] == comp]
        dat = q[q["source"] == "data"].pivot(index="subject", columns="condition", values="amp")[conds]
        xs = np.arange(len(conds))
        ax.bar(xs, dat.mean(), yerr=dat.std(ddof=1) / np.sqrt(n), color=[cpal[c] for c in conds],
               alpha=0.75, capsize=3, label="data")
        for lname, mk, colr in [("local learner", "D", "#B5541B"), ("global learner", "o", "#4E4596")]:
            pr = q[q["source"] == lname].pivot(index="subject", columns="condition", values="amp")[conds]
            ax.plot(xs, pr.mean(), marker=mk, ls="none", ms=8, mfc="white", mec=colr, mew=2,
                    label=f"{lname} prediction")
            # agreement: correlation across the 4 condition means, and mean abs error (µV)
            out_stats[(comp, lname)] = dict(
                r=float(np.corrcoef(pr.mean(), dat.mean())[0, 1]),
                mae=float(np.abs(pr.mean() - dat.mean()).mean()))
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(xs, ["LS/GS", "LD/GS", "LS/GD", "LD/GD"], fontsize=9)
        t0, t1 = ROIS[comp][1]
        ax.set(ylabel=f"µV, {int(t0 * 1e3)}–{int(t1 * 1e3)} ms",
               title="Window amplitude: data vs learner predictions")
        txt = "\n".join(f"{ln}: r = {out_stats[(comp, ln)]['r']:.2f}, MAE = {out_stats[(comp, ln)]['mae']:.2f} µV"
                        for ln in ["local learner", "global learner"])
        ax.text(0.02, 0.02 if comp == "local" else 0.98, txt, transform=ax.transAxes, fontsize=7.5,
                va="bottom" if comp == "local" else "top")
        ax.legend(frameon=False, fontsize=7.5, loc="lower left" if comp == "global" else "center left",
                  bbox_to_anchor=(0.0, 0.25) if comp == "global" else (0.0, 0.35))
    sns.despine(fig)
    fig.suptitle("Each window follows a different learner: the early response ignores global surprise "
                 "(LS/GD), the late response ignores local change (LD/GS)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "header1_erp_D_conditions_vs_learners.png", dpi=150)
    pd.DataFrame([dict(component=k[0], learner=k[1], **v) for k, v in out_stats.items()]).to_csv(
        OUT_DIR / "learner_prediction_agreement.tsv", sep="\t", index=False)
    print(pd.DataFrame([dict(component=k[0], learner=k[1], **v) for k, v in out_stats.items()]).round(3).to_string(index=False))
    print(f"Figures -> {FIG_DIR}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--plot"]:
        plot()
    else:
        compute([a for a in args if a.startswith("sub-")] or None)
        if not [a for a in args if a.startswith("sub-")]:
            plot()
