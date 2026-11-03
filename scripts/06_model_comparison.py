"""
06_model_comparison.py — Header 1: do local and global prediction errors run on different clocks?

Formal comparison of computational accounts of single-trial EEG at two a-priori response
components (windows/sites from Rutiku et al. 2024):
    LOCAL   frontocentral (Fz, FC1, FC2, F1, F2, Cz),        80–140 ms after the 5th sound
    GLOBAL  parietal      (Pz, P1, P2, CPz, POz, CP1, CP2),  300–700 ms after the 5th sound

MODELS (every model also has an intercept and a linear drift over the session)
    M0 categories     LD, GD, LDxGD (the standard 2x2 description; no learning)
    M1 adaptation     stimulus-specific adaptation of the 5th sound: exponentially decaying
                      trace of past presentations of that sound in real time (tau, seconds);
                      no inference
    M2 local          leaky Bayesian learner of tone-to-tone transitions (repeat/change),
                      position-agnostic; surprise of the 4->5 transition (half-life, reset)
    M3 global         leaky Bayesian learner of the quintlet-level rule ("5th sound changes?");
                      surprise of the 5th sound (half-life, reset)
    M4 local+global   M2 + M3, each with its own parameters
    M5 adapt+global   M1 + M3
Parameters (half-life, block reset yes/no, tau) are chosen on training blocks only:
nested leave-one-block-out CV (outer: predict each block; inner: select parameters by
leave-one-block-out within the remaining blocks). Score = summed out-of-sample Gaussian
log-likelihood per subject (an approximation to log model evidence), then random-effects
Bayesian model selection (Stephan et al. 2009) with protected exceedance probabilities
(Rigoux et al. 2014).

"Clocks": every learner's selected timescale is converted to a half-life in seconds
(local: transitions -> quintlets / 4 x mean quintlet period; global: quintlets x period;
adaptation: tau x ln 2) so the two components can be compared on one axis.

MODES
    --extract    (needs MNE; run where the epochs are) single-trial ROI amplitudes ->
                 derivatives/model_comparison/trials/<sub>.tsv
    --recovery   model-recovery simulation: data generated from each model with each subject's
                 own fitted parameters, betas and noise; full pipeline re-run -> confusion matrix
    (default)    fit all models to the real data, BMS, clocks, figures

Usage:
    python scripts/06_model_comparison.py --extract
    python scripts/06_model_comparison.py --recovery
    python scripts/06_model_comparison.py
"""
from pathlib import Path
import itertools
import json
import sys

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.special import digamma, gammaln, logsumexp

# ----------------------------------------------------------------------------- CONFIG
ROOT = Path(__file__).resolve().parents[1]
BIDS_ROOT = ROOT / "data" / "bids"
PRE_DIR = ROOT / "derivatives" / "preproc"
OUT_DIR = ROOT / "derivatives" / "model_comparison"
FIG_DIR = ROOT / "figures"
STEM = "ses-LGeffect_task-LocalGlobal"
SOUNDS = {1: "A", 2: "B"}

COMPONENTS = {   # name: (channels, window in s)  — a priori, from Rutiku et al. 2024
    "local_80-140ms": (["Fz", "FC1", "FC2", "F1", "F2", "Cz"], (0.08, 0.14)),
    "global_300-700ms": (["Pz", "P1", "P2", "CPz", "POz", "CP1", "CP2"], (0.30, 0.70)),
}
H_LOCAL = [10, 50, 200, np.inf]          # transitions
H_GLOBAL = [2, 5, 10, 20, np.inf]        # quintlets
RESETS = [False, True]
TAUS = [0.25, 0.5, 1, 2, 4, 8, 16, 32]   # s, adaptation time constant
MODELS = ["M0 categories", "M1 adaptation", "M2 local", "M3 global",
          "M4 local+global", "M5 adapt+global"]
N_RECOVERY_REPS = 20
SEED = 42
# ------------------------------------------------------------------------------------


# ============================================================ sequence + regressors
def load_sequence(subject):
    """Per-quintlet table and per-sound onset times (PTB clock, s)."""
    src = BIDS_ROOT / "sourcedata" / subject / "ses-LGeffect"
    seq = loadmat(next(src.glob("GlobLoc_stimulation_SEQUENCE*.mat")), squeeze_me=True)["SEQUENCE"]
    tz = loadmat(next(src.glob("GlobLoc_critical_events*.mat")), squeeze_me=True)["tz"]
    tz = tz[tz > 0]
    rows = []
    for b, blk in enumerate(seq, start=1):
        blk = np.atleast_2d(blk).astype(int)
        std = "".join(SOUNDS[x] for x in blk[0])
        for i, q in enumerate(blk):
            rows.append(dict(block=b, trial_in_block=i, quintlet="".join(SOUNDS[x] for x in q),
                             block_standard=std))
    qs = pd.DataFrame(rows)
    qs.insert(0, "seq_idx", np.arange(len(qs)))
    assert len(tz) == 5 * len(qs)
    sound_id = np.array([c for q in qs["quintlet"] for c in q])
    return qs, tz, sound_id


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


def adaptation_trace(times, ids, tau):
    """Trace of past presentations of the same sound, read out just before each sound."""
    a = {"A": 0.0, "B": 0.0}
    last = times[0]
    out = np.empty(len(times))
    for i, (t, k) in enumerate(zip(times, ids)):
        decay = np.exp(-(t - last) / tau)
        a["A"] *= decay
        a["B"] *= decay
        out[i] = a[k]
        a[k] += 1.0
        last = t
    return out


def regressor_bank(subject):
    """All parameterised regressors, one value per presented quintlet (5th sound)."""
    qs, tz, sid = load_sequence(subject)
    first, fifth = qs["quintlet"].str[0].to_numpy(), qs["quintlet"].str[4].to_numpy()
    start = (qs["trial_in_block"] == 0).to_numpy()
    period = float(np.median(np.diff(tz[4::5])[np.diff(tz[4::5]) < 3]))   # quintlet period, s
    bank = {}
    loc_out = np.zeros((len(qs), 4), int)
    loc_out[:, 3] = (fifth != first)
    loc_out = loc_out.ravel()
    loc_reset = np.zeros(len(loc_out), bool)
    loc_reset[::4] = start
    for h, r in itertools.product(H_LOCAL, RESETS):
        bank[("loc", h, r)] = leaky_surprise(loc_out, h, loc_reset if r else None)[3::4]
    glob_out = (fifth != first).astype(int)
    for h, r in itertools.product(H_GLOBAL, RESETS):
        bank[("glob", h, r)] = leaky_surprise(glob_out, h, start if r else None)
    for tau in TAUS:
        bank[("adapt", tau)] = adaptation_trace(tz, sid, tau)[4::5]
    return qs, bank, period


def model_configs(model, bank):
    """List of (config label, feature list) for a model; features are bank keys."""
    loc = [k for k in bank if k[0] == "loc"]
    glob = [k for k in bank if k[0] == "glob"]
    ad = [k for k in bank if k[0] == "adapt"]
    if model == "M0 categories":
        return [(("cat",), ["LD", "GD", "LDxGD"])]
    if model == "M1 adaptation":
        return [((k,), [k]) for k in ad]
    if model == "M2 local":
        return [((k,), [k]) for k in loc]
    if model == "M3 global":
        return [((k,), [k]) for k in glob]
    if model == "M4 local+global":
        return [((a, b), [a, b]) for a in loc for b in glob]
    if model == "M5 adapt+global":
        return [((a, b), [a, b]) for a in ad for b in glob]
    raise ValueError(model)


def design_matrix(trials, bank, features):
    n = len(trials)
    cols = [np.ones(n), (trials["seq_idx"] - trials["seq_idx"].mean()) / trials["seq_idx"].std()]
    ld = np.where(trials["local"] == "LD", 0.5, -0.5)
    gd = np.where(trials["global"] == "GD", 0.5, -0.5)
    for f in features:
        if f == "LD":
            cols.append(ld)
        elif f == "GD":
            cols.append(gd)
        elif f == "LDxGD":
            cols.append(4 * ld * gd)
        else:
            x = bank[f][trials["seq_idx"].to_numpy()]
            cols.append((x - x.mean()) / (x.std() + 1e-12))
    return np.column_stack(cols)


# ============================================================ CV machinery (sufficient statistics)
def block_stats(X, y, blocks, ub):
    return {b: (X[blocks == b].T @ X[blocks == b], X[blocks == b].T @ y[blocks == b],
                float(y[blocks == b] @ y[blocks == b]), int((blocks == b).sum())) for b in ub}


def fit_eval(stats, train, test):
    XtX = sum(stats[b][0] for b in train)
    Xty = sum(stats[b][1] for b in train)
    yty = sum(stats[b][2] for b in train)
    n = sum(stats[b][3] for b in train)
    p = XtX.shape[0]
    beta = np.linalg.solve(XtX + 1e-9 * np.eye(p), Xty)
    sse_tr = yty - 2 * beta @ Xty + beta @ XtX @ beta
    s2 = max(sse_tr / (n - p), 1e-12)
    ll = 0.0
    for b in test:
        A, c, d, m = stats[b]
        sse = d - 2 * beta @ c + beta @ A @ beta
        ll += -0.5 * m * np.log(2 * np.pi * s2) - sse / (2 * s2)
    return ll, beta, s2


def nested_cv(trials, y, bank, model):
    """Outer LOBO CV log-likelihood with inner LOBO parameter selection."""
    blocks = trials["block"].to_numpy()
    ub = np.unique(blocks)
    configs = model_configs(model, bank)
    st = [block_stats(design_matrix(trials, bank, feats), y, blocks, ub) for _, feats in configs]
    # inner scores: for every config, every pair (held-out outer, held-out inner)
    total, chosen = 0.0, []
    for bo in ub:
        train = [b for b in ub if b != bo]
        if len(configs) == 1:
            best = 0
        else:
            inner = [sum(fit_eval(s, [b for b in train if b != bi], [bi])[0] for bi in train) for s in st]
            best = int(np.argmax(inner))
        total += fit_eval(st[best], train, [bo])[0]
        chosen.append(configs[best][0])
    # full-data parameter estimate (single-level LOBO), for clocks and simulation
    full = [sum(fit_eval(s, [b for b in ub if b != bi], [bi])[0] for bi in ub) for s in st]
    kbest = int(np.argmax(full))
    return total, chosen, configs[kbest]


# ============================================================ random-effects BMS
def rfx_bms(L, n_samples=100000, seed=SEED):
    """Stephan et al. 2009 VB + Rigoux et al. 2014 protected exceedance probabilities.
    L: subjects x models log evidence."""
    n, K = L.shape
    a0 = np.ones(K)
    a = a0.copy()
    for _ in range(1000):
        lu = L + digamma(a) - digamma(a.sum())
        g = np.exp(lu - logsumexp(lu, axis=1, keepdims=True))
        a_new = a0 + g.sum(0)
        if np.max(np.abs(a_new - a)) < 1e-8:
            a = a_new
            break
        a = a_new
    rng = np.random.default_rng(seed)
    r = rng.dirichlet(a, n_samples)
    ep = np.bincount(r.argmax(1), minlength=K) / n_samples
    # free energies: null (equal frequencies) vs RFX
    F0 = np.sum(logsumexp(L, axis=1) - np.log(K))
    Elogr = digamma(a) - digamma(a.sum())
    kl = (gammaln(a.sum()) - gammaln(a).sum() - gammaln(a0.sum()) + gammaln(a0).sum()
          + np.sum((a - a0) * Elogr))
    F1 = np.sum(g * (L + Elogr - np.log(g + 1e-300))) - kl
    bor = 1 / (1 + np.exp(np.clip(F1 - F0, -700, 700)))
    pxp = ep * (1 - bor) + bor / K
    return dict(alpha=a, expected_freq=a / a.sum(), ep=ep, bor=bor, pxp=pxp)


# ============================================================ modes
def extract():
    import mne
    mne.set_log_level("ERROR")
    out = OUT_DIR / "trials"
    out.mkdir(parents=True, exist_ok=True)
    for sub in sorted(p.name for p in PRE_DIR.glob("sub-*") if p.is_dir()):
        ep = mne.read_epochs(PRE_DIR / sub / f"{sub}_{STEM}_epo.fif", preload=True)
        m = ep.metadata.reset_index(drop=True)[["seq_idx", "block", "trial_in_block", "quintlet",
                                                "local", "global", "condition"]].copy()
        for comp, (chs, (t0, t1)) in COMPONENTS.items():
            tm = (ep.times >= t0) & (ep.times <= t1)
            m[comp] = ep.get_data(picks=chs)[:, :, tm].mean(axis=(1, 2)) * 1e6
        m.to_csv(out / f"{sub}.tsv", sep="\t", index=False)
        print(f"{sub}: {len(m)} trials")


def subjects():
    return sorted(p.stem for p in (OUT_DIR / "trials").glob("sub-*.tsv"))


def fit_all(trials_by_sub, ys_by_sub, banks):
    """Returns L[comp] (subjects x models), chosen params, full-data configs."""
    res = {}
    for comp in COMPONENTS:
        L = np.zeros((len(trials_by_sub), len(MODELS)))
        chosen, best = {}, {}
        for i, sub in enumerate(trials_by_sub):
            for j, mdl in enumerate(MODELS):
                L[i, j], ch, bf = nested_cv(trials_by_sub[sub], ys_by_sub[sub][comp], banks[sub], mdl)
                chosen[(sub, mdl)], best[(sub, mdl)] = ch, bf
        res[comp] = dict(L=L, chosen=chosen, best=best)
    return res


def _recovery_task(args):
    comp, gi, rep, gen_spec, trials, banks, seed = args
    rng = np.random.default_rng(seed)
    subs = list(trials)
    L = np.zeros((len(subs), len(MODELS)))
    for i, s in enumerate(subs):
        X, beta, sd, _ = gen_spec[s]
        y = X @ beta + rng.normal(0, sd, len(trials[s]))
        for j, mdl in enumerate(MODELS):
            L[i, j] = nested_cv(trials[s], y, banks[s], mdl)[0]
    b = rfx_bms(L)
    return dict(component=comp, generating=MODELS[gi], rep=rep,
                winner_pxp=MODELS[int(np.argmax(b["pxp"]))], max_pxp=float(b["pxp"].max()),
                **{f"pxp_{m}": v for m, v in zip(MODELS, b["pxp"])},
                frac_subjects_correct=float(np.mean(L.argmax(1) == gi)))


def recovery(n_jobs=None):
    """Simulate from each model with each subject's own full-data parameters/betas/noise."""
    from multiprocessing import Pool
    subs = subjects()
    trials = {s: pd.read_csv(OUT_DIR / "trials" / f"{s}.tsv", sep="\t") for s in subs}
    banks = {s: regressor_bank(s)[1] for s in subs}
    real = fit_all(trials, {s: {c: trials[s][c].to_numpy() for c in COMPONENTS} for s in subs}, banks)
    tasks = []
    for comp in COMPONENTS:
        for gi, gen in enumerate(MODELS):
            gen_spec = {}
            for s in subs:
                cfg, feats = real[comp]["best"][(s, gen)]
                X = design_matrix(trials[s], banks[s], feats)
                y = trials[s][comp].to_numpy()
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                gen_spec[s] = (X, beta, np.std(y - X @ beta, ddof=X.shape[1]), cfg)
            for rep in range(N_RECOVERY_REPS):
                tasks.append((comp, gi, rep, gen_spec, trials, banks,
                              SEED + 1000 * gi + rep + (0 if comp == list(COMPONENTS)[0] else 10 ** 6)))
    with Pool(n_jobs) as pool:
        rows = []
        for k, r in enumerate(pool.imap_unordered(_recovery_task, tasks)):
            rows.append(r)
            if (k + 1) % 20 == 0:
                print(f"  recovery {k + 1}/{len(tasks)}", flush=True)
    df = pd.DataFrame(rows).sort_values(["component", "generating", "rep"])
    df.to_csv(OUT_DIR / "recovery.tsv", sep="\t", index=False)
    for comp in COMPONENTS:
        d = df[df["component"] == comp]
        print(comp)
        print(pd.crosstab(d["generating"], d["winner_pxp"], normalize="index").round(2).to_string())
    plot_recovery(df)


def plot_recovery(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(context="notebook", style="white")
    short = [m.split(" ", 1)[1] for m in MODELS]
    fig, axes = plt.subplots(1, len(COMPONENTS), figsize=(5.2 * len(COMPONENTS), 4.6))
    for ax, comp in zip(np.atleast_1d(axes), COMPONENTS):
        d = df[df["component"] == comp]
        cm = pd.crosstab(d["generating"], d["winner_pxp"], normalize="index").reindex(
            index=MODELS, columns=MODELS, fill_value=0)
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax,
                    xticklabels=short, yticklabels=short)
        ax.set(title=f"Model recovery — {comp}", xlabel="Winning model (max PXP)",
               ylabel="Generating model")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "header1_model_recovery.png", dpi=150)


def half_life_seconds(cfg_key, period):
    kind = cfg_key[0]
    if kind == "loc":
        return np.inf if np.isinf(cfg_key[1]) else cfg_key[1] / 4 * period
    if kind == "glob":
        return np.inf if np.isinf(cfg_key[1]) else cfg_key[1] * period
    if kind == "adapt":
        return cfg_key[1] * np.log(2)
    return np.nan


def main():
    subs = subjects()
    trials = {s: pd.read_csv(OUT_DIR / "trials" / f"{s}.tsv", sep="\t") for s in subs}
    rb = {s: regressor_bank(s) for s in subs}
    banks = {s: rb[s][1] for s in subs}
    periods = {s: rb[s][2] for s in subs}
    res = fit_all(trials, {s: {c: trials[s][c].to_numpy() for c in COMPONENTS} for s in subs}, banks)

    summary, clock_rows, ll_rows = {}, [], []
    for comp in COMPONENTS:
        L = res[comp]["L"]
        b = rfx_bms(L)
        summary[comp] = dict(
            pxp=dict(zip(MODELS, np.round(b["pxp"], 4))),
            expected_freq=dict(zip(MODELS, np.round(b["expected_freq"], 3))),
            bor=round(float(b["bor"]), 4),
            n_subjects_best=dict(pd.Series(np.array(MODELS)[L.argmax(1)]).value_counts()),
            summed_dLL_vs_M0=dict(zip(MODELS, np.round((L - L[:, [0]]).sum(0), 2))))
        for i, s in enumerate(subs):
            for j, m in enumerate(MODELS):
                ll_rows.append(dict(component=comp, subject=s, model=m, cv_ll=L[i, j],
                                    dLL_vs_M0=L[i, j] - L[i, 0]))
            for m in MODELS[1:]:
                cfg = res[comp]["best"][(s, m)][0]
                for k in cfg:
                    clock_rows.append(dict(component=comp, subject=s, model=m, term=k[0],
                                           param=k[1], reset=(k[2] if len(k) > 2 else None),
                                           half_life_s=half_life_seconds(k, periods[s])))
    ll = pd.DataFrame(ll_rows)
    clocks = pd.DataFrame(clock_rows)
    # plan rule 2: half-life of the best learner at each component (local learner @ local,
    # global learner @ global), Wilcoxon signed-rank; reset choice compared too
    from scipy.stats import wilcoxon
    c_loc = clocks[(clocks["component"] == list(COMPONENTS)[0]) & (clocks["model"] == "M2 local")].set_index("subject")
    c_glo = clocks[(clocks["component"] == list(COMPONENTS)[1]) & (clocks["model"] == "M3 global")].set_index("subject")
    a, b = c_loc["half_life_s"].replace(np.inf, 1e6), c_glo["half_life_s"].loc[c_loc.index].replace(np.inf, 1e6)
    w = wilcoxon(a, b)
    summary["clocks"] = dict(
        local_learner_half_life_s_median=float(c_loc["half_life_s"].median()),
        global_learner_half_life_s_median=float(c_glo["half_life_s"].median()),
        n_local_longer=int((a > b).sum()), n_equal=int((a == b).sum()),
        wilcoxon_W=float(w.statistic), wilcoxon_p=float(w.pvalue),
        n_reset_local=int(c_loc["reset"].sum()), n_reset_global=int(c_glo["reset"].sum()))
    ll.to_csv(OUT_DIR / "cv_loglik.tsv", sep="\t", index=False)
    clocks.to_csv(OUT_DIR / "clocks.tsv", sep="\t", index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    plot_real(ll, clocks, summary)


def plot_real(ll, clocks, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(context="notebook", style="ticks")
    short = {m: m.split(" ", 1)[1] for m in MODELS}
    comps = list(COMPONENTS)
    fig, axes = plt.subplots(2, len(comps), figsize=(5.6 * len(comps), 7.6))
    for c, comp in enumerate(comps):
        d = ll[(ll["component"] == comp) & (ll["model"] != MODELS[0])].copy()
        d["model"] = d["model"].map(short)
        ax = axes[0, c]
        sns.stripplot(data=d, x="model", y="dLL_vs_M0", color="0.6", size=4, jitter=0.15, ax=ax)
        sns.pointplot(data=d, x="model", y="dLL_vs_M0", errorbar="se", color="k", ax=ax,
                      linestyle="none")
        ax.axhline(0, color="k", lw=0.6)
        pxp = summary[comp]["pxp"]
        for i, m in enumerate(MODELS[1:]):
            ax.text(i, ax.get_ylim()[1], f"PXP\n{pxp[m]:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set(xlabel="", ylabel="CV log-likelihood vs categories")
        ax.set_title(comp.replace("_", " "), pad=30)
        ax.tick_params(axis="x", rotation=30)
        ax = axes[1, c]
        k = clocks[(clocks["component"] == comp) & clocks["model"].isin(
            ["M1 adaptation", "M2 local", "M3 global"])].copy()
        k["hl"] = np.log10(k["half_life_s"].replace(np.inf, 1e4))
        k["model"] = k["model"].map(short)
        k["reset"] = k["reset"].map({True: "reset", False: "no reset"}).fillna("n/a (adaptation)")
        sns.stripplot(data=k, x="model", y="hl", hue="reset", size=6, jitter=0.15, ax=ax,
                      palette={"reset": "#C44E52", "no reset": "#4C72B0", "n/a (adaptation)": "0.5"})
        ax.set_yticks([-1, 0, 1, 2, 3, 4], ["0.1", "1", "10", "100", "1000", "∞"])
        ax.set(xlabel="", ylabel="Selected memory half-life (s)", title="Clocks (per subject)")
        ax.legend(title=None, frameon=False, fontsize=8)
    sns.despine(fig)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "header1_model_comparison.png", dpi=150)
    print(f"Figure -> {FIG_DIR / 'header1_model_comparison.png'}")


if __name__ == "__main__":
    arg = sys.argv[1:]
    if arg == ["--extract"]:
        extract()
    elif arg == ["--recovery"]:
        recovery()
    else:
        main()
