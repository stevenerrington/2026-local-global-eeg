"""
12_rule_decoding.py — Header 3 feasibility: is the current block rule present in the EEG *before*
the 5th sound?

Within a sound pair the first four sounds are physically identical across rules (AAAA in both
AAAAA- and AAAAB-standard blocks; BBBB likewise), so information about the rule ("expect the 5th
to repeat" vs "expect it to change") before the 5th sound must reflect expectation.

Controls
  * baseline: epochs are re-baselined to the silent gap before the quintlet (−800 to −650 ms; the
    first sound starts at ≈ −628 ms, or −488 ms at the 122-ms SOA). The preprocessing baseline
    (−100–0 ms) lies inside the decoding window and is undone by this.
  * cross-validation: every test fold = one held-out repeat-rule block + one held-out change-rule
    block (4 x 4 = 16 folds); the classifier never sees trials from the test blocks, so
    block-specific drift cannot be learned. Score = ROC AUC on the held-out pair.
  * null: exact block-level permutation — rule labels re-assigned to blocks within each sound
    pair (2 of 4 blocks per pair -> 6 x 6 = 36 assignments per subject, incl. the true one);
    applied to the whole-window decoder.
  * history-matched test: the previous quintlet usually ends in a change in change-rule blocks
    and in a repeat in repeat-rule blocks; residual responses could therefore carry the rule.
    AUC is also computed only on test trials whose previous quintlet ended the *same* way in both
    held-out blocks (averaged over the "prev repeat" and "prev change" subsets).
  * positive control: decoding local deviant vs local standard (LD vs LS) on the same folds —
    should be at chance before the 5th sound and high after it. (Decoding the sound pair, A vs B,
    was tried first and is at chance: the chords differ only in pitch, which scalp EEG carries
    little information about.)

Decoders: L2 logistic regression on standardised features.
  time-resolved: 62 channels, 25-ms bins from −650 ms to +400 ms
  whole-window: 62 channels x 13 bins of 50 ms over −650–0 ms

Outputs: derivatives/rule_decoding/<sub>.npz; with --group: figures/header3_rule_decoding.png and
derivatives/rule_decoding/group_summary.json

Usage:  python scripts/12_rule_decoding.py --features          (needs epochs; fast)
        python scripts/12_rule_decoding.py [sub-01 ...]     (decoding per subject; skips done subjects)
        python scripts/12_rule_decoding.py --group
"""
from pathlib import Path
import itertools
import json
import sys

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- CONFIG
ROOT = Path(__file__).resolve().parents[1]
PRE_DIR = ROOT / "derivatives" / "preproc"
OBS_DIR = ROOT / "derivatives" / "observer"
OUT_DIR = ROOT / "derivatives" / "rule_decoding"
FIG_DIR = ROOT / "figures"
STEM = "ses-LGeffect_task-LocalGlobal"

BASELINE = (-0.80, -0.65)
TR_START, TR_END, TR_BIN = -0.65, 0.40, 0.025      # time-resolved bins (s)
WW_START, WW_END, WW_BIN = -0.65, 0.0, 0.05        # whole-window decoder bins (s)
C_TR, C_WW = 1.0, 0.01                             # inverse regularisation strengths
SEED = 42
# ------------------------------------------------------------------------------------


def bin_data(X, times, start, end, width):
    edges = np.arange(start, end + 1e-9, width)
    out, centres = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (times >= a) & (times < b)
        out.append(X[:, :, m].mean(-1))
        centres.append((a + b) / 2)
    return np.stack(out, axis=-1), np.array(centres)          # trials x ch x bins


def make_clf(C):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan


def extract_features(sub):
    """Epochs -> binned features + trial table (needs MNE and the epochs); saved for decoding."""
    import mne
    mne.set_log_level("ERROR")
    out = OUT_DIR / "features" / f"{sub}.npz"
    if out.exists() and np.load(out)["ttr"][-1] >= TR_END - TR_BIN:
        return
    ep = mne.read_epochs(PRE_DIR / sub / f"{sub}_{STEM}_epo.fif", preload=True)
    ep.pick("eeg").apply_baseline(BASELINE)
    meta = ep.metadata.reset_index(drop=True)
    obs = pd.read_csv(OBS_DIR / f"{sub}_observer.tsv", sep="\t")[["seq_idx", "block", "quintlet"]].set_index("seq_idx")
    meta["prev_quintlet"] = meta["seq_idx"].map(lambda i: obs["quintlet"].get(i - 1, None))
    meta["prev_block"] = meta["seq_idx"].map(lambda i: obs["block"].get(i - 1, -1))
    X = ep.get_data() * 1e6
    Xtr, ttr = bin_data(X, ep.times, TR_START, TR_END, TR_BIN)
    Xww, _ = bin_data(X, ep.times, WW_START, WW_END, WW_BIN)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, Xtr=Xtr.astype(np.float32), ttr=ttr, Xww=Xww.astype(np.float32),
                        ch_names=np.array(ep.ch_names))
    meta[["seq_idx", "block", "trial_in_block", "block_standard", "quintlet", "local", "global",
          "prev_quintlet", "prev_block"]].to_csv(out.with_suffix(".tsv"), sep="\t", index=False)
    print(f"{sub}: features {Xtr.shape}", flush=True)


def run_subject(sub):
    out = OUT_DIR / f"{sub}.npz"
    if out.exists():
        return
    fz = np.load(OUT_DIR / "features" / f"{sub}.npz")
    meta = pd.read_csv(OUT_DIR / "features" / f"{sub}.tsv", sep="\t")
    Xtr, ttr = fz["Xtr"].astype(float), fz["ttr"]
    Xww = fz["Xww"].astype(float).reshape(len(meta), -1)
    meta["rule_change"] = (meta["block_standard"].str[4] != meta["block_standard"].str[0]).astype(int)
    meta["pair"] = meta["block_standard"].str[0]
    meta["prev_change"] = np.where(meta["prev_block"] == meta["block"],
                                   (meta["prev_quintlet"].astype(str).str[4] != meta["prev_quintlet"].astype(str).str[0])
                                   .astype(float), np.nan)

    blocks = meta.groupby("block")[["rule_change", "pair"]].first()
    rep_b = blocks.index[blocks["rule_change"] == 0].to_numpy()
    chg_b = blocks.index[blocks["rule_change"] == 1].to_numpy()
    folds = list(itertools.product(rep_b, chg_b))
    blk = meta["block"].to_numpy()
    y = meta["rule_change"].to_numpy()
    pc = meta["prev_change"].to_numpy()

    auc_tr = np.full((len(folds), len(ttr)), np.nan)
    auc_tr_hist = np.full((len(folds), len(ttr)), np.nan)
    auc_ww, auc_ww_hist = [], []
    for f, (br, bc) in enumerate(folds):
        te = np.isin(blk, [br, bc])
        tr = ~te
        for k in range(len(ttr)):
            clf = make_clf(C_TR).fit(Xtr[tr, :, k], y[tr])
            s = clf.decision_function(Xtr[te, :, k])
            auc_tr[f, k] = auc(y[te], s)
            auc_tr_hist[f, k] = np.nanmean([auc(y[te][pc[te] == v], s[pc[te] == v]) for v in (0, 1)])
        clf = make_clf(C_WW).fit(Xww[tr], y[tr])
        s = clf.decision_function(Xww[te])
        auc_ww.append(auc(y[te], s))
        auc_ww_hist.append(np.nanmean([auc(y[te][pc[te] == v], s[pc[te] == v]) for v in (0, 1)]))

    # positive control: local deviant vs local standard (differs only after the 5th sound), on the
    # same held-out block pairs
    yl = (meta["local"] == "LD").astype(int).to_numpy()
    auc_pair = np.full((len(folds), len(ttr)), np.nan)
    for f, (br, bc) in enumerate(folds):
        te = np.isin(blk, [br, bc])
        for k in range(len(ttr)):
            clf = make_clf(C_TR).fit(Xtr[~te, :, k], yl[~te])
            auc_pair[f, k] = auc(yl[te], clf.decision_function(Xtr[te, :, k]))

    # exact block-level permutation null for the whole-window decoder (labels re-assigned to
    # blocks within each sound pair; 36 assignments incl. the true one)
    perm_auc = []
    pairs = {p: blocks.index[blocks["pair"] == p].to_numpy() for p in ("A", "B")}
    for ca in itertools.combinations(pairs["A"], 2):
        for cb in itertools.combinations(pairs["B"], 2):
            chg = set(ca) | set(cb)
            yperm = np.isin(blk, list(chg)).astype(int)
            rp = [b for b in blocks.index if b not in chg]
            cp = sorted(chg)
            vals = []
            for br, bc in itertools.product(rp, cp):
                te = np.isin(blk, [br, bc])
                clf = make_clf(C_WW).fit(Xww[~te], yperm[~te])
                vals.append(auc(yperm[te], clf.decision_function(Xww[te])))
            perm_auc.append(np.nanmean(vals))
    perm_auc = np.array(perm_auc)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, times=ttr, auc_tr=np.nanmean(auc_tr, 0), auc_tr_hist=np.nanmean(auc_tr_hist, 0),
                        auc_pair=np.nanmean(auc_pair, 0), auc_ww=np.nanmean(auc_ww),
                        auc_ww_hist=np.nanmean(auc_ww_hist), perm_auc=perm_auc,
                        n_trials=len(y), n_change=int(y.sum()))
    p_perm = (np.sum(perm_auc >= np.nanmean(auc_ww) - 1e-12)) / len(perm_auc)
    print(f"{sub}: whole-window rule AUC {np.nanmean(auc_ww):.3f} (history-matched {np.nanmean(auc_ww_hist):.3f}); "
          f"block-permutation p = {p_perm:.3f}; LD-vs-LS control AUC (80–140 ms) "
          f"{np.nanmean(np.nanmean(auc_pair, 0)[(ttr > 0.08) & (ttr < 0.14)]):.3f}", flush=True)


def run_group():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    import mne
    files = sorted(OUT_DIR.glob("sub-*.npz"))
    subs = [f.stem for f in files]
    Z = [np.load(f) for f in files]
    t = Z[0]["times"]
    n = len(Z)
    A = {k: np.stack([z[k] for z in Z]) for k in ["auc_tr", "auc_tr_hist", "auc_pair"]}
    ww = np.array([float(z["auc_ww"]) for z in Z])
    wwh = np.array([float(z["auc_ww_hist"]) for z in Z])
    perm = [z["perm_auc"] for z in Z]
    p_sub = np.array([np.mean(pz >= w - 1e-12) for pz, w in zip(perm, ww)])

    # group block-permutation test: mean AUC over subjects vs. distribution of means of
    # independently drawn per-subject permutation AUCs (Monte Carlo, 100k draws)
    rng = np.random.default_rng(SEED)
    null_means = np.mean([rng.choice(pz, 100000) for pz in perm], axis=0)
    p_group = (np.sum(null_means >= ww.mean()) + 1) / (len(null_means) + 1)

    # time-resolved: cluster permutation (sign-flip) on AUC − 0.5
    res_tr = {}
    for k in A:
        T, clusters, pv, _ = mne.stats.permutation_cluster_1samp_test(
            A[k] - 0.5, threshold=stats.t.ppf(0.975, n - 1), n_permutations=5000, tail=1, seed=SEED,
            out_type="indices", verbose=False)
        sig = np.zeros(len(t), bool)
        cl = []
        for ci, p in zip(clusters, pv):
            c = np.zeros(len(t), bool)
            c[ci[0]] = True
            if p < 0.05:
                sig |= c
                cl.append(dict(start_ms=float(t[c][0] * 1e3), end_ms=float(t[c][-1] * 1e3), p=float(p)))
        res_tr[k] = dict(sig=sig, clusters=cl)

    summ = dict(
        n_subjects=n,
        whole_window_auc_mean=float(ww.mean()), whole_window_auc_sem=float(ww.std(ddof=1) / np.sqrt(n)),
        whole_window_t_vs_05=float(stats.ttest_1samp(ww, 0.5).statistic),
        whole_window_p_ttest=float(stats.ttest_1samp(ww, 0.5, alternative="greater").pvalue),
        whole_window_group_block_permutation_p=float(p_group),
        n_subjects_auc_above_05=int((ww > 0.5).sum()),
        n_subjects_block_perm_p_le_0p05=int((p_sub <= 0.05).sum()),
        history_matched_auc_mean=float(np.nanmean(wwh)),
        history_matched_t=float(stats.ttest_1samp(wwh, 0.5, nan_policy="omit").statistic),
        history_matched_p=float(stats.ttest_1samp(wwh, 0.5, alternative="greater", nan_policy="omit").pvalue),
        time_resolved_clusters={k: v["clusters"] for k, v in res_tr.items()},
        control_LD_vs_LS_auc_pre5th=float(A["auc_pair"][:, t < 0].mean()),
        control_LD_vs_LS_auc_80_140=float(A["auc_pair"][:, (t > 0.08) & (t < 0.14)].mean()),
        per_subject=dict(zip(subs, [dict(auc=float(a), hist=float(h), perm_p=float(p)) for a, h, p in zip(ww, wwh, p_sub)])),
    )
    (OUT_DIR / "group_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps({k: v for k, v in summ.items() if k != "per_subject"}, indent=2))

    # figure
    sns.set_theme(context="notebook", style="ticks")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw=dict(width_ratios=[1.6, 1.6, 1]))
    tm = t * 1e3
    sound_on = [-157 * k for k in range(4, 0, -1)]
    for ax, keys, ttl in [(axes[0], [("auc_pair", "#55A868", "local deviant vs standard")],
                           "Positive control: local deviance (5th sound)"),
                          (axes[1], [("auc_tr", "#4C72B0", "block rule, all test trials"),
                                     ("auc_tr_hist", "#C44E52", "block rule, history-matched")],
                           "Decoding the block rule before the 5th sound")]:
        for k, col, lab in keys:
            m, se = A[k].mean(0), A[k].std(0, ddof=1) / np.sqrt(n)
            ax.plot(tm, m, color=col, lw=1.8, label=lab)
            ax.fill_between(tm, m - se, m + se, color=col, alpha=0.2, lw=0)
            sig = res_tr[k]["sig"]
            ax.plot(np.where(sig, tm, np.nan), np.full(len(tm), 0.03 if k != "auc_tr_hist" else 0.06),
                    color=col, lw=4, solid_capstyle="butt", transform=ax.get_xaxis_transform())
        for s_on in sound_on:
            ax.axvline(s_on, color="0.75", lw=0.8, ls=":")
        ax.axvline(0, color="k", lw=0.9)
        ax.axhline(0.5, color="k", lw=0.6)
        ax.set(xlabel="Time re 5th sound (ms)", ylabel="ROC AUC (held-out block pair)", title=ttl,
               xlim=(tm[0] - 10, tm[-1] + 10))
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    for a in axes[:2]:
        a.text(0.99, 0.98, "dotted = sounds 1–4 (157-ms SOA)\nbars = significant cluster (p < .05)",
               transform=a.transAxes, ha="right", va="top", fontsize=7.5, color="0.4")
    ax = axes[2]
    xs = np.arange(n)
    order = np.argsort(ww)
    ax.scatter(ww[order], xs, color="#4C72B0", s=28, label="whole-window AUC", zorder=3)
    for i, s in enumerate(order):
        lo, hi = np.percentile(perm[s], [2.5, 97.5])
        ax.plot([lo, hi], [i, i], color="0.75", lw=3, solid_capstyle="round", zorder=1)
    ax.scatter(wwh[order], xs, marker="x", color="#C44E52", s=26, label="history-matched", zorder=3)
    ax.axvline(0.5, color="k", lw=0.6)
    ax.set_yticks(xs, [subs[s] for s in order], fontsize=7)
    ax.set(xlabel="ROC AUC, −650–0 ms decoder",
           title=f"Per subject (−650–0 ms)\nmean {ww.mean():.3f}, block-perm. p {'< .0001' if p_group < 1e-4 else f'= {p_group:.2g}'}")
    ax.text(0.02, 0.02, "grey = block-permutation\n95% range", transform=ax.transAxes, fontsize=7.5, color="0.4")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    sns.despine(fig)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "header3_rule_decoding.png", dpi=150)
    print(f"Figure -> {FIG_DIR / 'header3_rule_decoding.png'}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--group"]:
        run_group()
    elif args == ["--features"]:
        for s in sorted(p.name for p in PRE_DIR.glob("sub-*") if p.is_dir()):
            extract_features(s)
    else:
        for s in args or sorted(p.stem for p in (OUT_DIR / "features").glob("sub-*.npz")):
            run_subject(s)
