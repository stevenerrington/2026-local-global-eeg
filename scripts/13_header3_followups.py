"""
13_header3_followups.py — Header 3 follow-ups to the pre-5th-sound rule decoding (12).

Uses the binned features from 12 (derivatives/rule_decoding/features), single-trial ROI amplitudes
from 06 (derivatives/model_comparison/trials) and the stimulation sequences (sourcedata).

PART 1  What carries the rule signal?
  a. Rule-difference ERP before the 5th sound: change-rule minus repeat-rule blocks (baseline:
     silence before the quintlet), group topographies in 100-ms bins and a spatio-temporal cluster
     test over −650–0 ms; time courses at a central (CNV) and a parietal site group.
  b. Slow vs fast decoding, same block-pair cross-validation as 12, window −650–0 ms:
       slow = per channel trial mean and linear slope over the window (2 x 62 features)
       fast = the binned signal after removing each trial's per-channel linear trend
              (sound-locked responses to sounds 1–4)
PART 2  Rebuilding after a switch
  Decoder (whole window, as in 12) trained only on established trials (trial_in_block >= 15) of
  the training blocks, applied to all trials of the held-out block pair (each block is held out
  in 4 folds; evidence averaged). Rule evidence = decision value signed towards the true rule
  (> 0 = correct). Group curves by position in block, compared with the resetting global
  learner's confidence |p(change) − ½| (half-life 10, reset), and split by whether the previous
  block had the opposite rule ("switch") or the same rule ("stay"): a carry-over learner predicts
  negative evidence early in switch blocks.
PART 3  Does trial-wise rule evidence predict the response to the 5th sound?
  Evidence e = held-out decision value towards "change" (not aligned), centred within block.
  Per subject: amplitude ~ e * outcome (5th change vs repeat), within each 2x2 condition centred;
  the e x outcome interaction is the test (a shared-baseline artefact would affect both outcomes
  equally). MMN: frontocentral 80–140 ms; P3b: parietal 300–700 ms. Group t-test on per-subject
  interaction slopes, plus the simple slopes per outcome.

Outputs: figures/header3_A_signal_source.png, header3_B_rebuild.png, header3_C_prediction_error.png, derivatives/rule_decoding/followups_summary.json

Usage:  python scripts/13_header3_followups.py
"""
from pathlib import Path
import itertools
import json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat

# ----------------------------------------------------------------------------- CONFIG
ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "derivatives" / "rule_decoding" / "features"
TRIALS_DIR = ROOT / "derivatives" / "model_comparison" / "trials"
BIDS_ROOT = ROOT / "data" / "bids"
INFO = ROOT / "derivatives" / "header1_erps" / "info.fif"
OUT_DIR = ROOT / "derivatives" / "rule_decoding"
FIG_DIR = ROOT / "figures"
SOUNDS = {1: "A", 2: "B"}
WIN = (-0.65, 0.0)
C_WW, C_SLOW, C_FAST = 0.01, 0.1, 0.01
ESTABLISHED = 15
POS_BINS = [0, 3, 6, 10, 15, 25, 40, 70, 200]
H_GLOB = 10
CENTRAL = ["Cz", "FC1", "FC2", "C1", "C2", "CPz"]
PARIETAL = ["Pz", "P1", "P2", "POz", "CP1", "CP2"]
SEED = 42
# ------------------------------------------------------------------------------------


def clf(C):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan


def global_learner(subject):
    """p(5th changes) before each quintlet, resetting leaky Beta learner (as in 03/06)."""
    src = BIDS_ROOT / "sourcedata" / subject / "ses-LGeffect"
    seq = loadmat(next(src.glob("GlobLoc_stimulation_SEQUENCE*.mat")), squeeze_me=True)["SEQUENCE"]
    rows = []
    for b, blk in enumerate(seq, start=1):
        for i, q in enumerate(np.atleast_2d(blk).astype(int)):
            rows.append(dict(block=b, trial_in_block=i, change=int(q[4] != q[0])))
    qs = pd.DataFrame(rows)
    qs["seq_idx"] = np.arange(len(qs))
    lam = 2.0 ** (-1.0 / H_GLOB)
    n1 = n0 = 0.0
    mu = np.empty(len(qs))
    for t, (y, tib) in enumerate(zip(qs["change"], qs["trial_in_block"])):
        if tib == 0:
            n1 = n0 = 0.0
        mu[t] = (n1 + 1) / (n1 + n0 + 2)
        n1, n0 = lam * n1 + y, lam * n0 + (1 - y)
    qs["p_change"] = mu
    return qs[["seq_idx", "p_change"]]


def load(sub):
    z = np.load(FEAT_DIR / f"{sub}.npz")
    meta = pd.read_csv(FEAT_DIR / f"{sub}.tsv", sep="\t")
    meta["rule"] = (meta["block_standard"].str[4] != meta["block_standard"].str[0]).astype(int)
    meta["outcome_change"] = (meta["quintlet"].str[4] != meta["quintlet"].str[0]).astype(int)
    meta["condition"] = meta["local"] + meta["global"]
    amp = pd.read_csv(TRIALS_DIR / f"{sub}.tsv", sep="\t")[["seq_idx", "local_80-140ms", "global_300-700ms"]]
    meta = meta.merge(amp, on="seq_idx", how="left").merge(global_learner(sub), on="seq_idx", how="left")
    return z, meta


def folds_for(meta):
    b = meta.groupby("block")["rule"].first()
    return list(itertools.product(b.index[b == 0], b.index[b == 1]))


def run_subject(sub):
    z, meta = load(sub)
    ttr, Xtr = z["ttr"], z["Xtr"].astype(float)
    w = (ttr > WIN[0]) & (ttr < WIN[1])
    Xw = Xtr[:, :, w]                                            # trials x ch x bins (25 ms)
    tw = ttr[w]
    y = meta["rule"].to_numpy()
    blk = meta["block"].to_numpy()
    folds = folds_for(meta)
    out = {}

    # ---- 1a rule-difference ERP (binned)
    out["diff_erp"] = Xtr[y == 1].mean(0) - Xtr[y == 0].mean(0)            # ch x all bins

    # ---- 1b slow vs fast features
    tc = tw - tw.mean()
    slope = (Xw * tc).sum(-1) / (tc ** 2).sum()
    mean = Xw.mean(-1)
    slow = np.concatenate([mean, slope], axis=1)
    fast = (Xw - mean[..., None] - slope[..., None] * tc).reshape(len(Xw), -1)
    full = Xw.reshape(len(Xw), -1)
    # 50-ms binning for the full/fast decoders, as in 12 (fewer features)
    def bin50(A):
        nb = A.shape[-1] // 2
        return A[..., :nb * 2].reshape(*A.shape[:-1], nb, 2).mean(-1)
    full = bin50(Xw).reshape(len(Xw), -1)
    fast = bin50(Xw - mean[..., None] - slope[..., None] * tc).reshape(len(Xw), -1)
    for name, F, C in [("full", full, C_WW), ("slow", slow, C_SLOW), ("fast", fast, C_FAST)]:
        out[f"auc_{name}"] = np.nanmean([auc(y[np.isin(blk, f)],
                                             clf(C).fit(F[~np.isin(blk, f)], y[~np.isin(blk, f)])
                                             .decision_function(F[np.isin(blk, f)])) for f in folds])

    # ---- 2 + 3: held-out evidence from a decoder trained on established trials
    est = meta["trial_in_block"].to_numpy() >= ESTABLISHED
    ev_sum, ev_n = np.zeros(len(y)), np.zeros(len(y))
    for f in folds:
        te = np.isin(blk, f)
        trn = ~te & est
        d = clf(C_WW).fit(full[trn], y[trn]).decision_function(full[te])
        ev_sum[te] += d
        ev_n[te] += 1
    d = ev_sum / ev_n                                                         # towards "change"
    meta["d"] = d
    meta["evidence"] = np.where(y == 1, d, -d)                                 # towards true rule
    out["auc_established_decoder"] = auc(y, d)

    # previous block's rule
    brule = meta.groupby("block")["rule"].first()
    prev = brule.shift(1)
    meta["switch"] = meta["block"].map(lambda b: np.nan if pd.isna(prev.get(b)) else float(prev[b] != brule[b]))
    meta["learner_conf"] = np.where(y == 1, meta["p_change"] - 0.5, 0.5 - meta["p_change"])  # towards true rule
    meta["pos_bin"] = pd.cut(meta["trial_in_block"], POS_BINS, right=False, labels=False)
    out["curve"] = meta.groupby("pos_bin")[["evidence", "learner_conf"]].mean()
    out["curve_switch"] = meta[meta["switch"] == 1].groupby("pos_bin")["evidence"].mean()
    out["curve_stay"] = meta[meta["switch"] == 0].groupby("pos_bin")["evidence"].mean()
    out["early_switch"] = meta[(meta["switch"] == 1) & (meta["trial_in_block"] < 6)]["evidence"].mean()
    out["early_stay"] = meta[(meta["switch"] == 0) & (meta["trial_in_block"] < 6)]["evidence"].mean()
    # trial-wise relation of evidence to the learner's confidence (within subject)
    m = meta.dropna(subset=["evidence", "learner_conf"])
    out["r_evidence_learner"] = stats.pearsonr(m["evidence"], m["learner_conf"])[0]

    # ---- 3 evidence x outcome -> MMN / P3b, within block-centred evidence, condition-centred amplitude
    m = meta.dropna(subset=["local_80-140ms", "global_300-700ms"]).copy()
    m["e"] = m["d"] - m.groupby("block")["d"].transform("mean")
    m["e"] /= m["e"].std()
    for comp in ["local_80-140ms", "global_300-700ms"]:
        a = m[comp] - m.groupby("condition")[comp].transform("mean")
        oc = np.where(m["outcome_change"] == 1, 0.5, -0.5)
        X = np.column_stack([np.ones(len(m)), m["e"], oc, m["e"] * oc])
        beta, *_ = np.linalg.lstsq(X, a.to_numpy(), rcond=None)
        out[f"{comp}_interaction"] = beta[3]
        for lab, sel in [("change", oc > 0), ("repeat", oc < 0)]:
            Xs = np.column_stack([np.ones(sel.sum()), m["e"][sel]])
            out[f"{comp}_slope_{lab}"] = np.linalg.lstsq(Xs, a[sel].to_numpy(), rcond=None)[0][1]
    out["ttr"] = ttr
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import mne
    mne.set_log_level("ERROR")
    subs = sorted(p.stem for p in FEAT_DIR.glob("sub-*.npz"))
    R = {}
    for s in subs:
        R[s] = run_subject(s)
        r = R[s]
        print(f"{s}: AUC full {r['auc_full']:.3f} slow {r['auc_slow']:.3f} fast {r['auc_fast']:.3f} | "
              f"early evidence switch {r['early_switch']:+.2f} stay {r['early_stay']:+.2f} | "
              f"P3b e x outcome {r['global_300-700ms_interaction']:+.2f}", flush=True)
    n = len(subs)
    ttr = R[subs[0]]["ttr"]
    info = mne.io.read_info(INFO)
    chs = info["ch_names"]
    summ = {}

    # ---------- part 1
    for k in ["full", "slow", "fast"]:
        v = np.array([R[s][f"auc_{k}"] for s in subs])
        tt = stats.ttest_1samp(v, 0.5, alternative="greater")
        summ[f"auc_{k}"] = dict(mean=float(v.mean()), sem=float(v.std(ddof=1) / np.sqrt(n)),
                                t=float(tt.statistic), p=float(tt.pvalue), n_above=int((v > 0.5).sum()))
    vs, vf = np.array([R[s]["auc_slow"] for s in subs]), np.array([R[s]["auc_fast"] for s in subs])
    tt = stats.ttest_rel(vs, vf)
    summ["slow_vs_fast"] = dict(t=float(tt.statistic), p=float(tt.pvalue))
    D = np.stack([R[s]["diff_erp"] for s in subs])                      # subj x ch x bins
    pre = ttr < 0
    adjacency, _ = mne.channels.find_ch_adjacency(info, "eeg")
    T, clusters, pv, _ = mne.stats.spatio_temporal_cluster_1samp_test(
        D[:, :, pre & (ttr > -0.65)].transpose(0, 2, 1), threshold=stats.t.ppf(0.975, n - 1),
        n_permutations=5000, adjacency=adjacency, seed=SEED, out_type="mask", verbose=False)
    sigmask = np.zeros_like(T, bool)
    cl = []
    tsel = ttr[pre & (ttr > -0.65)]
    for c, p in zip(clusters, pv):
        if p < 0.05:
            sigmask |= c
            ti = np.where(c.any(1))[0]
            cl.append(dict(p=float(p), sign="+" if T[c].mean() > 0 else "−", start_ms=float(tsel[ti[0]] * 1e3),
                           end_ms=float(tsel[ti[-1]] * 1e3), n_channels=int(c.any(0).sum()),
                           channels=[chs[i] for i in np.where(c.any(0))[0]][:12]))
    summ["diff_erp_clusters"] = cl

    # ---------- part 2
    curves = pd.concat({s: R[s]["curve"] for s in subs}, names=["subject"]).reset_index()
    cs = pd.concat({s: R[s]["curve_switch"] for s in subs}, names=["subject"]).reset_index()
    cy = pd.concat({s: R[s]["curve_stay"] for s in subs}, names=["subject"]).reset_index()
    es, ey = np.array([R[s]["early_switch"] for s in subs]), np.array([R[s]["early_stay"] for s in subs])
    rr = np.array([R[s]["r_evidence_learner"] for s in subs])
    first = curves[curves["pos_bin"] == 0].set_index("subject")["evidence"]
    late = curves[curves["pos_bin"] >= 5].groupby("subject")["evidence"].mean()
    summ["rebuild"] = dict(
        evidence_first_3_trials=float(first.mean()), t_first=float(stats.ttest_1samp(first, 0).statistic),
        p_first=float(stats.ttest_1samp(first, 0).pvalue),
        evidence_late_25plus=float(late.mean()), t_late=float(stats.ttest_1samp(late, 0).statistic),
        p_late=float(stats.ttest_1samp(late, 0).pvalue),
        late_vs_first_t=float(stats.ttest_rel(late, first.loc[late.index]).statistic),
        late_vs_first_p=float(stats.ttest_rel(late, first.loc[late.index]).pvalue),
        early_switch_mean=float(np.nanmean(es)), early_stay_mean=float(np.nanmean(ey)),
        early_switch_t_vs0=float(stats.ttest_1samp(es, 0, nan_policy="omit").statistic),
        early_switch_p_vs0=float(stats.ttest_1samp(es, 0, nan_policy="omit").pvalue),
        early_switch_vs_stay_t=float(stats.ttest_rel(es, ey, nan_policy="omit").statistic),
        early_switch_vs_stay_p=float(stats.ttest_rel(es, ey, nan_policy="omit").pvalue),
        r_trialwise_evidence_learner_mean=float(rr.mean()),
        r_t=float(stats.ttest_1samp(np.arctanh(rr), 0).statistic),
        r_p=float(stats.ttest_1samp(np.arctanh(rr), 0).pvalue))

    # ---------- part 3
    p3 = {}
    for comp in ["local_80-140ms", "global_300-700ms"]:
        inter = np.array([R[s][f"{comp}_interaction"] for s in subs])
        sc = np.array([R[s][f"{comp}_slope_change"] for s in subs])
        sr = np.array([R[s][f"{comp}_slope_repeat"] for s in subs])
        p3[comp] = dict(interaction=float(inter.mean()), t=float(stats.ttest_1samp(inter, 0).statistic),
                        p=float(stats.ttest_1samp(inter, 0).pvalue),
                        slope_change=float(sc.mean()), p_change=float(stats.ttest_1samp(sc, 0).pvalue),
                        slope_repeat=float(sr.mean()), p_repeat=float(stats.ttest_1samp(sr, 0).pvalue))
    summ["prediction_error"] = p3
    (OUT_DIR / "followups_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))

    # ---------- figures (three files)
    sns.set_theme(context="notebook", style="ticks")
    FIG_DIR.mkdir(exist_ok=True)
    gm = D.mean(0)
    edges = np.arange(-0.65, 0.051, 0.1)
    lim = np.abs(gm[:, (ttr > -0.65) & (ttr < 0)]).max()

    # A: what carries the signal
    fig = plt.figure(figsize=(16, 7.2))
    gA = fig.add_gridspec(2, 1, height_ratios=[1, 1.25], hspace=0.35)
    top = gA[0].subgridspec(1, 8, width_ratios=[1] * 7 + [0.06], wspace=0.08)
    for k in range(7):
        ax = fig.add_subplot(top[0, k])
        b = (ttr >= edges[k]) & (ttr < edges[k + 1])
        bs = (tsel >= edges[k]) & (tsel < edges[k + 1])
        im, _ = mne.viz.plot_topomap(gm[:, b].mean(1), info, axes=ax, show=False, cmap="RdBu_r",
                                     vlim=(-lim, lim), contours=0, sensors=False,
                                     mask=sigmask[bs].any(0) if bs.any() else None,
                                     mask_params=dict(marker="o", markerfacecolor="k", markeredgecolor="k",
                                                      markersize=2))
        ax.set_title(f"{int(round(edges[k] * 1e3))} to {int(round(min(edges[k + 1], 0) * 1e3))} ms", fontsize=9)
    fig.colorbar(im, cax=fig.add_subplot(top[0, 7])).set_label("µV", fontsize=9)
    bot = gA[1].subgridspec(1, 2, width_ratios=[1.7, 1], wspace=0.25)
    ax = fig.add_subplot(bot[0, 0])
    for name, grp, col in [("central (CNV sites: " + ", ".join(CENTRAL) + ")", CENTRAL, "#C44E52"),
                           ("parietal (" + ", ".join(PARIETAL) + ")", PARIETAL, "#4C72B0")]:
        idx = [chs.index(c) for c in grp]
        x = D[:, idx].mean(1)
        ax.plot(ttr * 1e3, x.mean(0), color=col, lw=1.8, label=name)
        ax.fill_between(ttr * 1e3, x.mean(0) - x.std(0, ddof=1) / np.sqrt(n),
                        x.mean(0) + x.std(0, ddof=1) / np.sqrt(n), color=col, alpha=0.2, lw=0)
    for k in range(1, 5):
        ax.axvline(-157 * k, color="0.75", lw=0.7, ls=":")
    ax.axvspan(0, 400, color="0.94", zorder=0)
    ax.text(200, ax.get_ylim()[1], "after the 5th sound:\nstimulus differs", ha="center", va="top", fontsize=8,
            color="0.4")
    ax.axvline(0, color="k", lw=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlim=(-650, 400), xlabel="Time re 5th sound (ms)  (dotted = sounds 1–4)", ylabel="µV",
           title="Change-rule minus repeat-rule blocks")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax = fig.add_subplot(bot[0, 1])
    dd = pd.DataFrame({k: [R[s][f"auc_{k}"] for s in subs] for k in ["full", "slow", "fast"]}, index=subs)
    for _, r in dd.iterrows():
        ax.plot(range(3), r.to_numpy(), color="0.82", lw=0.8)
    ax.errorbar(range(3), dd.mean(), yerr=dd.std(ddof=1) / np.sqrt(n), color="k", marker="o", capsize=3)
    ax.axhline(0.5, color="k", lw=0.6)
    ax.set_xticks(range(3), ["full", "slow\n(mean + ramp)", "fast\n(detrended)"])
    sf = summ["slow_vs_fast"]
    ax.set(ylabel="ROC AUC (held-out block pair)",
           title=f"Decoding from slow vs sound-locked features\nslow vs fast: t({n - 1}) = {sf['t']:.2f}, p = {sf['p']:.2g}")
    sns.despine(fig)
    fig.suptitle("Header 3 — what carries the rule signal before the 5th sound (baseline: silence before the "
                 "quintlet; dots = significant spatio-temporal cluster)", fontsize=11)
    fig.savefig(FIG_DIR / "header3_A_signal_source.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # B: rebuilding after a switch
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(width_ratios=[1.7, 1]))
    ax = axes[0]
    labels = [f"{a}–{b - 1}" for a, b in zip(POS_BINS[:-1], POS_BINS[1:])]
    xs = np.arange(len(labels))
    for name, df, col, off, lw in [("decoder, all blocks", curves, "k", 0, 1.8),
                                   ("decoder, switch blocks", cs, "#C44E52", -0.12, 1.1),
                                   ("decoder, stay blocks", cy, "#4C72B0", 0.12, 1.1)]:
        piv = df.pivot(index="subject", columns="pos_bin", values="evidence").reindex(columns=range(len(labels)))
        ax.errorbar(xs + off, piv.mean(), yerr=piv.std(ddof=1) / np.sqrt(piv.notna().sum()), color=col,
                    marker="o", ms=4, capsize=2, lw=lw, label=name)
    ax.axhline(0, color="k", lw=0.6)
    ax2 = ax.twinx()
    lc = curves.pivot(index="subject", columns="pos_bin", values="learner_conf").mean()
    ax2.plot(xs, lc.reindex(range(len(labels))), color="#8172B3", ls="--", lw=1.8, label="resetting learner (right axis)")
    ax2.set_ylabel("Learner confidence in the true rule", color="#8172B3", fontsize=9)
    ax2.tick_params(axis="y", colors="#8172B3")
    ax.set_xticks(xs, labels, fontsize=8.5)
    rb = summ["rebuild"]
    ax.set(xlabel="Quintlet position in block", ylabel="Decoder evidence for the true rule",
           title=f"Rule evidence is rebuilt within each block\nfirst 3 trials {rb['evidence_first_3_trials']:+.2f} "
                 f"(p = {rb['p_first']:.2g}); 25+ trials {rb['evidence_late_25plus']:+.2f} (p = {rb['p_late']:.2g}); "
                 f"late vs first p = {rb['late_vs_first_p']:.2g}")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="lower right")
    ax = axes[1]
    vals = pd.DataFrame({"switch": es, "stay": ey}, index=subs)
    for _, r in vals.iterrows():
        ax.plot([0, 1], r.to_numpy(), color="0.82", lw=0.8)
    ax.errorbar([0, 1], vals.mean(), yerr=vals.std(ddof=1) / np.sqrt(n), color="k", marker="o", capsize=3)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks([0, 1], ["previous block:\nopposite rule", "previous block:\nsame rule"])
    ax.set(ylabel="Decoder evidence, first 6 trials", xlim=(-0.4, 1.4),
           title=f"Carry-over test (negative = old rule)\nswitch vs 0: p = {rb['early_switch_p_vs0']:.2g}; "
                 f"switch vs stay: p = {rb['early_switch_vs_stay_p']:.2g}")
    sns.despine(ax=axes[1])
    sns.despine(ax=axes[0], right=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "header3_B_rebuild.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # C: prediction -> error
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, (comp, ttl) in zip(axes, [("local_80-140ms", "MMN (frontocentral 80–140 ms)"),
                                      ("global_300-700ms", "P3b (parietal 300–700 ms)")]):
        sc = np.array([R[s][f"{comp}_slope_change"] for s in subs])
        sr = np.array([R[s][f"{comp}_slope_repeat"] for s in subs])
        for i in range(n):
            ax.plot([0, 1], [sr[i], sc[i]], color="0.82", lw=0.8)
        ax.errorbar([0, 1], [sr.mean(), sc.mean()], yerr=[sr.std(ddof=1) / np.sqrt(n), sc.std(ddof=1) / np.sqrt(n)],
                    color="k", marker="o", capsize=3)
        ax.axhline(0, color="k", lw=0.6)
        q = p3[comp]
        ax.set_xticks([0, 1], ["5th sound repeats", "5th sound changes"])
        ax.set(xlim=(-0.4, 1.4), ylabel="Amplitude slope (µV per SD of\n'expect change' evidence)",
               title=f"{ttl}\nevidence × outcome: t({n - 1}) = {q['t']:.2f}, p = {q['p']:.2g}")
    sns.despine(fig)
    fig.suptitle("Does pre-5th rule evidence on each trial predict the response to the 5th sound? "
                 "(the test is the interaction)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "header3_C_prediction_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
