"""
05_reset_validation.py — does the global learner reset at block boundaries?

The block reset in 03 was adopted after model-free checks on the same data, so it needs
confirmation. No independent local–global dataset with block structure is openly available
(the other ds005429 sessions are continuous sequences), so this script runs two tests that do not
reuse the observations that motivated the reset:

TEST 1 — novel prediction (not examined before).
    A learner that carries memory across blocks predicts that an early-block global deviant is
    LESS surprising when its structure (5th sound changes vs repeats) was the previous block's
    STANDARD structure (it has just heard ~120 of them) than when it was not. A resetting learner
    predicts no difference.
    Measure: P3b (parietal ROI, 300–600 ms) on the first N_EARLY global deviants of blocks 2–8,
    split by history ("was previous standard" vs "other"). Paired t-test + JZS Bayes factor.
    The no-reset observer's own predicted surprise difference is reported alongside, to show the
    size of effect a carry-over learner would produce.

TEST 2 — out-of-sample estimate of carry-over.
    Global observer with memory carried across a block boundary scaled by rho
    (n <- rho * n at each block start; rho = 0 full reset, rho = 1 no reset).
    For each rho on a grid: leave-one-block-out CV prediction of single-trial parietal P3b from
        intercept + LD + GD + LDxGD + S_glob|GS + S_glob|GD   (within-condition surprise)
    i.e. regression weights are always fitted on the other 7 blocks. Per-subject CV R² profile
    over rho; group mean profile; per-subject best rho; bootstrap CI over subjects of the
    group-best rho.

Outputs: derivatives/reset_validation/*.tsv, figures/reset_validation.png

Usage:  python scripts/05_reset_validation.py
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from scipy import integrate, stats
from scipy.io import loadmat

# ----------------------------------------------------------------------------- CONFIG
ROOT = Path(__file__).resolve().parents[1]
BIDS_ROOT = ROOT / "data" / "bids"
PRE_DIR = ROOT / "derivatives" / "preproc"
OUT_DIR = ROOT / "derivatives" / "reset_validation"
FIG_DIR = ROOT / "figures"
STEM = "ses-LGeffect_task-LocalGlobal"
SOUNDS = {1: "A", 2: "B"}

ROI_CHS = ["Pz", "P1", "P2", "CPz", "POz", "CP1", "CP2"]
ROI_WIN = (0.30, 0.60)
H_GLOBAL = 10                   # quintlets, as in 03
RHO_GRID = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
N_EARLY = 3                     # first N global deviants of each block (Test 1)
N_BOOT = 5000
SEED = 42
# ------------------------------------------------------------------------------------

mne.set_log_level("ERROR")


def load_sequence(subject):
    src = BIDS_ROOT / "sourcedata" / subject / "ses-LGeffect"
    seq = loadmat(next(src.glob("GlobLoc_stimulation_SEQUENCE*.mat")), squeeze_me=True)["SEQUENCE"]
    rows = []
    for b, blk in enumerate(seq, start=1):
        blk = np.atleast_2d(blk).astype(int)
        std = "".join(SOUNDS[x] for x in blk[0])
        for i, q in enumerate(blk):
            rows.append(dict(block=b, trial_in_block=i, quintlet="".join(SOUNDS[x] for x in q),
                             block_standard=std))
    qs = pd.DataFrame(rows)
    qs.insert(0, "seq_idx", np.arange(len(qs)))
    return qs


def global_surprise(qs, h, rho):
    """Leaky Beta–Bernoulli on 'does the 5th sound differ from the first four', with memory
    scaled by rho at every block start. Returns surprise (bits) of each observed outcome."""
    lam = 1.0 if np.isinf(h) else 2.0 ** (-1.0 / h)
    y = (qs["quintlet"].str[4] != qs["quintlet"].str[0]).to_numpy().astype(int)
    start = (qs["trial_in_block"] == 0).to_numpy()
    n1 = n0 = 0.0
    S = np.empty(len(y))
    for t in range(len(y)):
        if start[t]:
            n1, n0 = rho * n1, rho * n0
        mu = (n1 + 1) / (n1 + n0 + 2)
        S[t] = -np.log2(mu if y[t] else 1 - mu)
        n1, n0 = lam * n1 + y[t], lam * n0 + (1 - y[t])
    return S


def jzs_bf10_paired(d, r=np.sqrt(2) / 2):
    """JZS Bayes factor (Rouder et al. 2009) for a one-sample / paired t-test."""
    d = np.asarray(d, float)
    n = len(d)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    v = n - 1

    def integrand(g):
        return ((1 + n * g * r ** 2) ** -0.5 * (1 + t ** 2 / ((1 + n * g * r ** 2) * v)) ** (-(v + 1) / 2)
                * (2 * np.pi) ** -0.5 * g ** -1.5 * np.exp(-1 / (2 * g)))
    num, _ = integrate.quad(integrand, 0, np.inf)
    den = (1 + t ** 2 / v) ** (-(v + 1) / 2)
    return num / den


def within_condition(x, cond):
    x = pd.Series(x)
    x = x - x.groupby(cond).transform("mean")
    sd = x.std()
    return (x / sd).to_numpy() if sd > 0 else x.to_numpy()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    subs = sorted(p.name for p in PRE_DIR.glob("sub-*") if p.is_dir())
    t1_rows, t2_rows = [], []

    for sub in subs:
        ep = mne.read_epochs(PRE_DIR / sub / f"{sub}_{STEM}_epo.fif", preload=True)
        tm = (ep.times >= ROI_WIN[0]) & (ep.times <= ROI_WIN[1])
        p3 = ep.get_data(picks=ROI_CHS)[:, :, tm].mean(axis=(1, 2)) * 1e6
        meta = ep.metadata.reset_index(drop=True).copy()
        meta["p3"] = p3
        qs = load_sequence(sub)
        for rho in RHO_GRID:
            qs[f"S_rho{rho:g}"] = global_surprise(qs, H_GLOBAL, rho)
        meta = meta.merge(qs[["seq_idx"] + [c for c in qs.columns if c.startswith("S_rho")]],
                          on="seq_idx", how="left", validate="one_to_one")

        # ---- TEST 1: early-block deviants by previous-block history
        # the global learner represents "5th sound changes vs repeats", irrespective of A/B, so
        # history is defined on that structure: does this deviant's structure match the
        # previous block's standard?
        std_change = qs.groupby("block")["block_standard"].first().map(lambda q: q[4] != q[0])
        prev_std_change = std_change.shift(1)
        gd = meta[meta["global"] == "GD"].copy()
        gd["gd_ord"] = gd.groupby("block").cumcount() + 1
        gd = gd[(gd["block"] > 1) & (gd["gd_ord"] <= N_EARLY)]
        dev_change = gd["quintlet"].str[4] != gd["quintlet"].str[0]
        gd["history"] = np.where(dev_change == gd["block"].map(prev_std_change).astype(bool),
                                 "was_prev_standard", "other")
        for h, g in gd.groupby("history"):
            t1_rows.append(dict(subject=sub, history=h, n=len(g), p3=g["p3"].mean(),
                                S_noreset=g["S_rho1"].mean(), S_reset=g["S_rho0"].mean()))

        # ---- TEST 2: leave-one-block-out CV over rho
        y = meta["p3"].to_numpy()
        ld = np.where(meta["local"] == "LD", 0.5, -0.5)
        gdv = np.where(meta["global"] == "GD", 0.5, -0.5)
        is_gd = meta["global"].eq("GD").to_numpy()
        blocks = meta["block"].to_numpy()
        for rho in RHO_GRID:
            w = within_condition(meta[f"S_rho{rho:g}"], meta["condition"])
            X = np.column_stack([np.ones(len(y)), ld, gdv, ld * gdv * 4,
                                 np.where(is_gd, 0, w), np.where(is_gd, w, 0)])
            pred = np.empty_like(y)
            for b in np.unique(blocks):
                te = blocks == b
                beta, *_ = np.linalg.lstsq(X[~te], y[~te], rcond=None)
                pred[te] = X[te] @ beta
            sse = np.sum((y - pred) ** 2)
            t2_rows.append(dict(subject=sub, rho=rho, cv_r2=1 - sse / np.sum((y - y.mean()) ** 2),
                                cv_loglik=-0.5 * len(y) * np.log(sse / len(y))))
        print(f"{sub}: done ({len(meta)} trials, {len(gd)} early-block deviants)")

    t1 = pd.DataFrame(t1_rows)
    t2 = pd.DataFrame(t2_rows)
    t1.to_csv(OUT_DIR / "test1_history.tsv", sep="\t", index=False)
    t2.to_csv(OUT_DIR / "test2_rho_cv.tsv", sep="\t", index=False)

    # ---- TEST 1 stats
    w1 = t1.pivot(index="subject", columns="history", values="p3").dropna()
    d = w1["was_prev_standard"] - w1["other"]
    tt = stats.ttest_rel(w1["was_prev_standard"], w1["other"])
    bf10 = jzs_bf10_paired(d.to_numpy())
    ws = t1.pivot(index="subject", columns="history", values="S_noreset").loc[w1.index]
    wr = t1.pivot(index="subject", columns="history", values="S_reset").loc[w1.index]
    res1 = dict(n_subjects=int(len(w1)),
                trials_per_subject=t1.groupby("history")["n"].mean().round(1).to_dict(),
                p3_was_prev_standard=float(w1["was_prev_standard"].mean()),
                p3_other=float(w1["other"].mean()), diff=float(d.mean()),
                t=float(tt.statistic), p=float(tt.pvalue), BF10=float(bf10), BF01=float(1 / bf10),
                predicted_surprise_diff_noreset_bits=float((ws["was_prev_standard"] - ws["other"]).mean()),
                predicted_surprise_diff_reset_bits=float((wr["was_prev_standard"] - wr["other"]).mean()))

    # ---- TEST 2 stats: per-subject best rho, group profile, bootstrap CI for group-best rho
    t2["dR2_vs_reset"] = t2["cv_r2"] - t2["subject"].map(t2[t2["rho"] == 0].set_index("subject")["cv_r2"])
    prof = t2.pivot(index="subject", columns="rho", values="cv_loglik")
    best = prof.idxmax(axis=1)
    rng = np.random.default_rng(SEED)
    boot = [prof.iloc[rng.integers(0, len(prof), len(prof))].sum().idxmax() for _ in range(N_BOOT)]
    r2 = t2.pivot(index="subject", columns="rho", values="cv_r2")
    tt2 = stats.ttest_rel(r2[0.0], r2[1.0])
    res2 = dict(group_best_rho=float(prof.sum().idxmax()),
                bootstrap_95ci_rho=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                bootstrap_fraction_rho_le_0p1=float(np.mean(np.array(boot) <= 0.1)),
                per_subject_best_rho=best.astype(float).to_dict(),
                n_subjects_best_rho_le_0p1=int((best <= 0.1).sum()),
                mean_cv_r2_pct=(100 * r2.mean()).round(3).to_dict(),
                reset_vs_noreset_cv_r2_t=float(tt2.statistic), reset_vs_noreset_cv_r2_p=float(tt2.pvalue),
                reset_better_in_n_subjects=int((r2[0.0] > r2[1.0]).sum()))
    res = dict(test1_novel_prediction=res1, test2_cv_carryover=res2)
    (OUT_DIR / "summary.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))

    # ---- figure
    sns.set_theme(context="notebook", style="ticks")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw=dict(width_ratios=[1, 1.4]))
    ax = axes[0]
    long = w1.reset_index().melt(id_vars="subject", var_name="history", value_name="p3")
    long["history"] = long["history"].map({"was_prev_standard": "structure was previous\nblock's standard",
                                           "other": "other"})
    order = ["structure was previous\nblock's standard", "other"]
    for _, g in long.groupby("subject"):
        g = g.set_index("history").loc[order]
        ax.plot([0, 1], g["p3"], color="0.8", lw=0.8, zorder=0)
    sns.pointplot(data=long, x="history", y="p3", order=order, errorbar="se", color="#4C72B0", ax=ax)
    ax.set(xlabel="", ylabel="P3b (µV, Pz ROI 300–600 ms)",
           title=f"Early-block deviants (first {N_EARLY})\nBF$_{{01}}$ = {res1['BF01']:.1f}, p = {res1['p']:.2f}")
    ax = axes[1]
    t2["dR2_pct"] = 100 * t2["dR2_vs_reset"]
    for _, g in t2.groupby("subject"):
        ax.plot(range(len(RHO_GRID)), g.sort_values("rho")["dR2_pct"], color="0.82", lw=0.7, zorder=0)
    sns.pointplot(data=t2, x="rho", y="dR2_pct", errorbar="se", color="#4C72B0", ax=ax)
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xlabel="Memory carried across block boundary (ρ)\n0 = full reset, 1 = no reset",
           ylabel="Leave-one-block-out ΔR² vs reset (%)",
           title=f"Out-of-sample carry-over; group best ρ = {res2['group_best_rho']:g} "
                 f"[{res2['bootstrap_95ci_rho'][0]:g}, {res2['bootstrap_95ci_rho'][1]:g}]")
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reset_validation.png", dpi=150)
    print(f"Figure -> {FIG_DIR / 'reset_validation.png'}")


if __name__ == "__main__":
    main()
