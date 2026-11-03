"""
04_single_trial_glm.py — single-trial EEG regression on ideal-observer variables.

Requires 02_preprocess.py (epochs) and 03_ideal_observer.py (observer tables).

Per subject (run first, one or more subjects per call):
  A. TIMESCALE ANALYSIS (ROI). For each observer half-life, cross-validated R² of single-trial
     ROI amplitude for  categories + surprise  minus  categories only  (ΔR²_cv). Answers: does
     trial-wise surprise explain variance *beyond* the four condition labels, and at what memory
     timescale?  Local surprise -> frontocentral ROI 80–140 ms; global surprise -> parietal ROI
     300–600 ms.
  B. MASS-UNIVARIATE GLM (all channels x times) with
         intercept + LD + GD + LDxGD + S_loc|LS + S_loc|LD + S_glob|GS + S_glob|GD
     Category regressors are ±0.5 coded. Surprise is centred within each of the four conditions
     (raw surprise is almost a deterministic function of the labels, VIF ~100), so its betas index
     trial-to-trial modulation *within* condition — learning dynamics the labels cannot capture.
     Each surprise effect is estimated separately for standards and deviants of its own level
     (e.g. S_glob|GD = global-surprise slope among global deviants).
     A second, observer-only GLM (intercept + z(S_loc) + z(S_glob), raw surprise) is saved too.
  C. MODEL COMPARISON at the ROIs: 10-fold CV R² of categories-only, observer-only and both.

Group (run with --group after all subjects):
  spatio-temporal cluster permutation tests (one-sample over subjects) on every GLM beta,
  timescale curves, CV model comparison, and figures in figures/.

Usage:
  python scripts/04_single_trial_glm.py sub-01 sub-02 ...
  python scripts/04_single_trial_glm.py --group
"""
from pathlib import Path
import sys
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mne

# ----------------------------------------------------------------------------- CONFIG
ROOT = Path(__file__).resolve().parents[1]
PRE_DIR = ROOT / "derivatives" / "preproc"
OBS_DIR = ROOT / "derivatives" / "observer"
OUT_DIR = ROOT / "derivatives" / "glm"
FIG_DIR = ROOT / "figures"
STEM = "ses-LGeffect_task-LocalGlobal"

TMIN, TMAX = -0.1, 0.7          # analysis window re 5th sound (s)
DECIM = 2                       # 500 -> 250 Hz (epochs are low-passed at 40 Hz)
H_LOCAL_GRID = [5, 10, 20, 50, 100, 200, 500, np.inf]      # must match 03
H_GLOBAL_GRID = [2, 3, 5, 10, 20, 50, 100, np.inf]

ROIS = {   # name: (channels, time window in s, surprise variable tested there)
    "frontocentral_80-140ms": (["Fz", "FC1", "FC2", "F1", "F2", "Cz"], (0.08, 0.14), "loc"),
    "parietal_300-600ms": (["Pz", "P1", "P2", "CPz", "POz", "CP1", "CP2"], (0.30, 0.60), "glob"),
}
N_FOLDS = 10
N_PERM = 2000
CLUSTER_P = 0.01                 # cluster-forming threshold (two-sided, t-distribution)
SEED = 42

GLM_FULL = ["intercept", "LD", "GD", "LDxGD", "S_loc|LS", "S_loc|LD", "S_glob|GS", "S_glob|GD"]
GLM_OBS = ["intercept", "S_loc", "S_glob"]
# ------------------------------------------------------------------------------------

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")


# ============================================================ helpers
def zscore(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std()


def within_condition(meta, col):
    """Surprise centred within each of the 4 conditions (orthogonal to the category regressors),
    then scaled to unit SD: indexes trial-to-trial modulation *within* a condition."""
    x = meta[col] - meta.groupby("condition")[col].transform("mean")
    return x.to_numpy() / x.std()


def design(meta, which, s_loc="S_loc", s_glob="S_glob"):
    """Category regressors ±0.5 coded. In models that include the categories, surprise enters
    within-condition centred (S_loc / S_glob); in observer-only models it enters raw (z-scored)."""
    ld = np.where(meta["local"] == "LD", 0.5, -0.5)
    gd = np.where(meta["global"] == "GD", 0.5, -0.5)
    with_cat = "LD" in which
    sl = within_condition(meta, s_loc) if with_cat else zscore(meta[s_loc])
    sg = within_condition(meta, s_glob) if with_cat else zscore(meta[s_glob])
    is_ld, is_gd = meta["local"].eq("LD").to_numpy(), meta["global"].eq("GD").to_numpy()
    cols = {"intercept": np.ones(len(meta)), "LD": ld, "GD": gd, "LDxGD": ld * gd * 4,
            "S_loc": sl, "S_glob": sg,
            # surprise effect estimated separately within each level of its own factor
            "S_loc|LS": np.where(is_ld, 0, sl), "S_loc|LD": np.where(is_ld, sl, 0),
            "S_glob|GS": np.where(is_gd, 0, sg), "S_glob|GD": np.where(is_gd, sg, 0)}
    return np.column_stack([cols[c] for c in which])


def ols(X, Y):
    """Betas for Y (n_trials x ...)."""
    shp = Y.shape
    B, *_ = np.linalg.lstsq(X, Y.reshape(shp[0], -1), rcond=None)
    return B.reshape((X.shape[1],) + shp[1:])


def cv_r2(X, y, n_folds=N_FOLDS, seed=SEED):
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(y)), n_folds)
    pred = np.empty_like(y)
    for f in folds:
        train = np.setdiff1d(np.arange(len(y)), f)
        b, *_ = np.linalg.lstsq(X[train], y[train], rcond=None)
        pred[f] = X[f] @ b
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def load_subject(sub):
    ep = mne.read_epochs(PRE_DIR / sub / f"{sub}_{STEM}_epo.fif", preload=True)
    ep.pick("eeg").crop(TMIN, TMAX)
    ep.decimate(DECIM)
    obs = pd.read_csv(OBS_DIR / f"{sub}_observer.tsv", sep="\t")
    meta = ep.metadata.reset_index(drop=True)
    obs_cols = [c for c in obs.columns if c not in meta.columns or c == "seq_idx"]
    meta = meta.merge(obs[obs_cols], on="seq_idx", how="left", validate="one_to_one")
    assert meta["S_glob"].notna().all(), "observer table does not cover all epochs"
    # labels must agree between preprocessing and observer
    chk = obs.set_index("seq_idx").loc[meta["seq_idx"], ["local", "global"]].to_numpy()
    assert (chk == meta[["local", "global"]].to_numpy()).all()
    return ep, meta


# ============================================================ per subject
def run_subject(sub):
    ep, meta = load_subject(sub)
    Y = ep.get_data() * 1e6                                   # trials x ch x times, µV
    times, chs = ep.times, ep.ch_names

    # ---- A + C: ROI analyses
    rows = []
    for roi, (roi_chs, (t0, t1), level) in ROIS.items():
        picks = [chs.index(c) for c in roi_chs if c in chs]
        tm = (times >= t0) & (times <= t1)
        y = Y[:, picks][:, :, tm].mean(axis=(1, 2))
        base = cv_r2(design(meta, ["intercept", "LD", "GD", "LDxGD"]), y)
        grid = H_LOCAL_GRID if level == "loc" else H_GLOBAL_GRID
        for h in grid:
            col = f"S_{level}_h{h:g}"
            dev = (meta["local"] if level == "loc" else meta["global"]).isin(["LD", "GD"]).to_numpy()
            w = within_condition(meta, col)
            Xc = np.column_stack([design(meta, ["intercept", "LD", "GD", "LDxGD"]),
                                  np.where(dev, w, 0), np.where(dev, 0, w)])
            Xo = np.column_stack([np.ones(len(y)), zscore(meta[col])])
            rows.append(dict(subject=sub, roi=roi, level=level, half_life=h,
                             r2_categories=base, r2_cat_plus_surprise=cv_r2(Xc, y),
                             r2_surprise_only=cv_r2(Xo, y)))
        # model comparison with default half-lives, both surprise terms
        rows.append(dict(subject=sub, roi=roi, level="default", half_life=np.nan,
                         r2_categories=base,
                         r2_cat_plus_surprise=cv_r2(design(meta, GLM_FULL), y),
                         r2_surprise_only=cv_r2(design(meta, GLM_OBS), y)))
        # same, observer without block resets (03 writes the opposite variant with a suffix)
        alt = "_noreset" if "S_glob_noreset" in meta else "_reset"
        if f"S_glob{alt}" in meta:
            kw = dict(s_loc=f"S_loc{alt}", s_glob=f"S_glob{alt}")
            rows.append(dict(subject=sub, roi=roi, level=f"default{alt}", half_life=np.nan,
                             r2_categories=base,
                             r2_cat_plus_surprise=cv_r2(design(meta, GLM_FULL, **kw), y),
                             r2_surprise_only=cv_r2(design(meta, GLM_OBS, **kw), y)))
    roi_df = pd.DataFrame(rows)
    roi_df["dR2"] = roi_df["r2_cat_plus_surprise"] - roi_df["r2_categories"]

    # ---- B: mass-univariate GLMs
    X_full, X_obs = design(meta, GLM_FULL), design(meta, GLM_OBS)
    vif = np.diag(np.linalg.inv(np.corrcoef(X_full[:, 1:], rowvar=False)))
    B_full, B_obs = ols(X_full, Y), ols(X_obs, Y)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / f"{sub}_glm.npz", B_full=B_full, B_obs=B_obs, times=times,
                        ch_names=np.array(chs), regs_full=np.array(GLM_FULL),
                        regs_obs=np.array(GLM_OBS))
    roi_df.to_csv(OUT_DIR / f"{sub}_roi.tsv", sep="\t", index=False)
    if not (OUT_DIR / "info-ave.fif").exists():          # channel info/positions for group plots
        mne.write_evokeds(OUT_DIR / "info-ave.fif", ep.average(), overwrite=True)

    d = roi_df[roi_df["level"] == "default"].set_index("roi")
    print(f"{sub}: {len(meta)} trials | VIF " + ", ".join(f"{r} {v:.2f}" for r, v in zip(GLM_FULL[1:], vif)))
    for roi, r in d.iterrows():
        print(f"   {roi}: CV R² cat={r.r2_categories:.4f}  obs={r.r2_surprise_only:.4f}  "
              f"both={r.r2_cat_plus_surprise:.4f}")


# ============================================================ group
def cluster_test(X, adjacency):
    """X: subjects x times x channels -> list of (cluster mask, p, sign)."""
    from scipy.stats import t as tdist
    n = X.shape[0]
    thr = tdist.ppf(1 - CLUSTER_P / 2, n - 1)
    T, clusters, pvals, _ = mne.stats.spatio_temporal_cluster_1samp_test(
        X, threshold=thr, n_permutations=N_PERM, adjacency=adjacency, tail=0,
        seed=SEED, out_type="mask", n_jobs=1, verbose=False)
    res = [(c, p, np.sign(T[c].mean())) for c, p in zip(clusters, pvals)]
    return T, sorted(res, key=lambda r: r[1])


def roi_timecourse(B, regs, reg, chs, roi_chs):
    picks = [chs.index(c) for c in roi_chs if c in chs]
    return B[:, regs.index(reg), picks, :].mean(axis=1)            # subjects x times


def run_group():
    FIG_DIR.mkdir(exist_ok=True)
    files = sorted(OUT_DIR.glob("sub-*_glm.npz"))
    subs = [f.name.split("_")[0] for f in files]
    dat = [np.load(f, allow_pickle=True) for f in files]
    times, chs = dat[0]["times"], list(dat[0]["ch_names"])
    Bf = np.stack([d["B_full"] for d in dat])          # subj x reg x ch x time
    Bo = np.stack([d["B_obs"] for d in dat])
    info = mne.read_evokeds(OUT_DIR / "info-ave.fif")[0].info
    adjacency, _ = mne.channels.find_ch_adjacency(info, "eeg")
    print(f"group: {len(subs)} subjects")

    # ---- cluster tests on every beta (time >= 0)
    t_sel = times >= 0
    summary, T_maps, sig_masks = [], {}, {}
    for model, B, regs in [("full", Bf, GLM_FULL), ("obs", Bo, GLM_OBS)]:
        for r, reg in enumerate(regs):
            if reg == "intercept":
                continue
            X = B[:, r][:, :, t_sel].transpose(0, 2, 1)           # subj x time x ch
            T, res = cluster_test(X, adjacency)
            T_maps[(model, reg)] = T
            mask = np.zeros_like(T, bool)
            for c, p, sgn in res:
                if p < 0.05:
                    mask |= c
                    tt = np.where(c.any(1))[0]
                    cc = np.where(c.any(0))[0]
                    peak = np.unravel_index(np.argmax(np.abs(T) * c), T.shape)
                    summary.append(dict(model=model, regressor=reg, sign="+" if sgn > 0 else "-",
                                        p=p, t_start_ms=1e3 * times[t_sel][tt[0]],
                                        t_end_ms=1e3 * times[t_sel][tt[-1]], n_channels=len(cc),
                                        peak_ms=1e3 * times[t_sel][peak[0]],
                                        peak_ch=chs[peak[1]], peak_t=T[peak]))
            sig_masks[(model, reg)] = mask
    summ = pd.DataFrame(summary)
    summ.to_csv(OUT_DIR / "group_clusters.tsv", sep="\t", index=False)
    print("Significant clusters (p < .05):\n" + (summ.round(3).to_string(index=False) if len(summ) else "  none"))

    # ---- ROI tables
    roi = pd.concat([pd.read_csv(OUT_DIR / f"{s}_roi.tsv", sep="\t") for s in subs])
    roi.to_csv(OUT_DIR / "group_roi.tsv", sep="\t", index=False)

    sns.set_theme(context="notebook", style="ticks")
    pal = {"LD": "#C44E52", "GD": "#8172B3", "LDxGD": "0.5", "S_loc": "#DD8452", "S_glob": "#4C72B0",
           "S_loc|LS": "#F2B27E", "S_loc|LD": "#DD8452", "S_glob|GS": "#9CB8DE", "S_glob|GD": "#4C72B0"}

    # ---- Fig 1: timescale curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (rname, (_, _, level)) in zip(axes, ROIS.items()):
        d = roi[(roi["roi"] == rname) & (roi["level"] == level)].copy()
        d["h"] = d["half_life"].replace(np.inf, np.nan)
        grid = H_LOCAL_GRID if level == "loc" else H_GLOBAL_GRID
        xs = np.arange(len(grid))
        d["x"] = d["half_life"].map({h: i for i, h in enumerate(grid)})
        d["within"] = 100 * d["dR2"]
        d["observer"] = 100 * (d["r2_surprise_only"] - d["r2_categories"])
        long = d.melt(id_vars=["subject", "x"], value_vars=["observer", "within"],
                      var_name="comparison", value_name="pct")
        long["comparison"] = long["comparison"].map(
            {"observer": "observer only − categories", "within": "categories + within-cond. surprise − categories"})
        sns.lineplot(data=long, x="x", y="pct", hue="comparison", errorbar="se", marker="o", ax=ax,
                     palette=[pal["S_" + level], "0.35"])
        ax.axhline(0, color="k", lw=0.6)
        ax.legend(frameon=False, fontsize=7, title=None)
        ax.set_xticks(xs, [("∞" if np.isinf(h) else f"{h:g}") for h in grid])
        unit = "transitions" if level == "loc" else "quintlets"
        ax.set(xlabel=f"Observer memory half-life ({unit})", ylabel="ΔR²$_{CV}$ vs 4-category model (%)",
               title=f"{'Local' if level == 'loc' else 'Global'} surprise — {rname.replace('_', ' ')}")
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "observer_timescales.png", dpi=150)

    # ---- Fig 2: beta time courses at ROIs (full GLM), with cluster extent shading
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
    tms = times * 1e3
    for ax, (rname, (roi_chs, _, _)) in zip(axes, ROIS.items()):
        for reg in GLM_FULL[1:]:
            if reg == "LDxGD":
                continue
            tc = roi_timecourse(Bf, GLM_FULL, reg, chs, roi_chs)
            m, se = tc.mean(0), tc.std(0, ddof=1) / np.sqrt(len(tc))
            ax.plot(tms, m, color=pal[reg], lw=1.7, label=reg.replace("_", " "))
            ax.fill_between(tms, m - se, m + se, color=pal[reg], alpha=0.18, lw=0)
        ax.axvline(0, color="k", lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set(title=rname.replace("_", " "), xlabel="Time re 5th sound (ms)", ylabel="β (µV per unit)")
        ax.legend(frameon=False, fontsize=8, ncol=2)
    # significance rasters at the bottom of each axis: times where the regressor's significant
    # cluster includes any ROI channel
    for ax, (rname, (roi_chs, _, _)) in zip(axes, ROIS.items()):
        picks = [chs.index(c) for c in roi_chs if c in chs]
        lo, hi = ax.get_ylim()
        step = (hi - lo) * 0.035
        for k, reg in enumerate([r for r in GLM_FULL[1:] if r != "LDxGD"]):
            sig_t = sig_masks[("full", reg)][:, picks].any(1)
            ax.plot(np.where(sig_t, tms[t_sel], np.nan), np.full(sig_t.size, lo + step * (k + 0.5)),
                    color=pal[reg], lw=3, solid_capstyle="butt")
        ax.set_ylim(lo, hi)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "glm_betas_roi.png", dpi=150)

    # ---- Fig 3: topographies of surprise betas (group t) at peak times
    fig, axes = plt.subplots(4, 4, figsize=(11, 10))
    for row, reg in enumerate(["S_loc|LS", "S_loc|LD", "S_glob|GS", "S_glob|GD"]):
        T = T_maps[("full", reg)]
        mask = sig_masks[("full", reg)]
        for col, (a, b) in enumerate([(0.08, 0.14), (0.15, 0.25), (0.30, 0.45), (0.45, 0.60)]):
            ts = (times[t_sel] >= a) & (times[t_sel] <= b)
            mne.viz.plot_topomap(T[ts].mean(0), info, axes=axes[row, col], show=False,
                                 cmap="RdBu_r", vlim=(-6, 6), contours=0,
                                 mask=mask[ts].any(0), mask_params=dict(markersize=3))
            axes[row, col].set_title(f"{reg.replace('_', ' ')}  {int(a * 1e3)}–{int(b * 1e3)} ms",
                                     fontsize=9)
    fig.suptitle("Group t of within-condition surprise betas (full GLM; dots = significant cluster)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "glm_surprise_topomaps.png", dpi=150)

    # ---- Fig 4: CV model comparison at ROIs
    d = roi[roi["level"].str.startswith("default")].melt(
        id_vars=["subject", "roi", "level"],
        value_vars=["r2_categories", "r2_surprise_only", "r2_cat_plus_surprise"],
        var_name="model", value_name="R2")
    d["model"] = d["model"].map({"r2_categories": "categories", "r2_surprise_only": "observer",
                                 "r2_cat_plus_surprise": "both"})
    d = d[~((d["model"] == "categories") & (d["level"] != "default"))]
    d["model"] = np.where(d["level"] == "default", d["model"], d["model"] + "\n(no reset)")
    order = ["categories", "observer", "both", "observer\n(no reset)", "both\n(no reset)"]
    d["R2_pct"] = 100 * d["R2"]
    order = [o for o in order if o in set(d["model"])]
    g = sns.catplot(data=d, x="model", y="R2_pct", col="roi", kind="point", errorbar="se",
                    order=order, color="0.2", height=3.6, aspect=1.4, sharey=False)
    for ax, (rname, sub_d) in zip(g.axes.flat, d.groupby("roi", sort=False)):
        sns.stripplot(data=sub_d, x="model", y="R2_pct", order=order, ax=ax, color="0.7",
                      size=3, jitter=0.1)
    g.set_axis_labels("", "CV R² (%)")
    g.set_titles("{col_name}")
    g.savefig(FIG_DIR / "glm_model_comparison.png", dpi=150)

    # ---- numeric summary
    from scipy.stats import ttest_1samp, ttest_rel
    out = {}
    for rname, (_, _, level) in ROIS.items():
        dd = roi[(roi["roi"] == rname) & (roi["level"] == level)]
        curve = dd.groupby("half_life")["dR2"].agg(["mean", "sem"])
        best = curve["mean"].idxmax()
        obs_curve = (dd["r2_surprise_only"] - dd["r2_categories"]).groupby(dd["half_life"]).mean()
        bvals = dd[dd["half_life"] == best]["dR2"]
        dflt = roi[(roi["roi"] == rname) & (roi["level"] == "default")]
        out[rname] = dict(
            observer_minus_categories_pct_by_half_life={("inf" if np.isinf(h) else f"{h:g}"): round(100 * v, 3)
                                                         for h, v in obs_curve.items()},
            best_half_life=("inf" if np.isinf(best) else float(best)),
            dR2_at_best_pct=float(100 * bvals.mean()),
            t_dR2_at_best=float(ttest_1samp(bvals, 0).statistic),
            p_dR2_at_best=float(ttest_1samp(bvals, 0).pvalue),
            n_positive=int((bvals > 0).sum()),
            observer_vs_categories_t=float(ttest_rel(dflt["r2_surprise_only"], dflt["r2_categories"]).statistic),
            observer_vs_categories_p=float(ttest_rel(dflt["r2_surprise_only"], dflt["r2_categories"]).pvalue),
            both_vs_categories_t=float(ttest_rel(dflt["r2_cat_plus_surprise"], dflt["r2_categories"]).statistic),
            both_vs_categories_p=float(ttest_rel(dflt["r2_cat_plus_surprise"], dflt["r2_categories"]).pvalue),
            mean_cv_r2_pct={"categories": float(100 * dflt["r2_categories"].mean()),
                            "observer": float(100 * dflt["r2_surprise_only"].mean()),
                            "both": float(100 * dflt["r2_cat_plus_surprise"].mean())},
        )
        alt = roi[(roi["roi"] == rname) & roi["level"].str.startswith("default_")]
        if len(alt):
            a = alt.set_index("subject").loc[dflt["subject"]]
            r = ttest_rel(dflt["r2_surprise_only"].to_numpy(), a["r2_surprise_only"].to_numpy())
            out[rname].update(observer_reset_vs_noreset_t=float(r.statistic),
                              observer_reset_vs_noreset_p=float(r.pvalue),
                              mean_cv_r2_pct_noreset={"observer": float(100 * a["r2_surprise_only"].mean()),
                                                      "both": float(100 * a["r2_cat_plus_surprise"].mean())})
    (OUT_DIR / "group_roi_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    plt.close("all")
    print(f"Figures -> {FIG_DIR}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--group"]:
        run_group()
    else:
        for s in args or sorted(p.name for p in PRE_DIR.glob("sub-*") if p.is_dir()):
            run_subject(s)
