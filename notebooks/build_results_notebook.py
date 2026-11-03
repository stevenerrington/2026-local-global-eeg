"""Builds notebooks/results_analyses.ipynb (run once; the notebook is the deliverable)."""
import nbformat as nbf
from pathlib import Path

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# =====================================================================================================
md(r"""
# Local–global EEG: every statistic in the Results, recomputed in one place

This notebook reproduces **all numbers reported in the Results draft** (three sections, one per
figure) from the saved per-participant data, with every function written out inline, and then
**validates the statistical machinery on simulated data** where the ground truth is known.

| Section | Figure | What is computed |
|---|---|---|
| 1 | Figure 1 | MMN / P3b window tests, temporal and spatio-temporal clusters, 2 × 2 ANOVA, reference control |
| 2 | Figure 2 | learners, cross-validated model comparison (BMS/PXP), recovery, head-to-head, condition fits, surprise–amplitude relations, timescales, block reset |
| 3 | Figure 3 | eLORETA cluster of the local effect, auditory-cortex time course, frontal/spread checks, global null |
| 4 | — | validation of every method on simulated data |
| 5 | — | table: every Results number vs this notebook's recomputation |

**Inputs** (all under the project root): per-participant averages from `07_header1_erps.py`
(`derivatives/header1_erps/`), single-trial window amplitudes (`derivatives/model_comparison/trials/`),
the stimulus sequences (`data/bids/sourcedata/`), the model-comparison outputs (`derivatives/model_comparison/`),
the reset-validation outputs (`derivatives/reset_validation/`) and the source-analysis inputs
(`derivatives/source/`: template head + noise covariances).

**Expensive steps** (refitting all cross-validated models, the recovery simulation) are loaded from their saved
outputs by default; flags in the next cell switch on full recomputation. Everything else is recomputed here.
One participant's cross-validated model scores *are* refitted from scratch as a reproducibility check (Section 2).

Conventions: n = 15; tests are two-sided across participants (df = 14); "x/15" = participants whose effect has
the group sign; a-priori ROIs/windows from Rutiku et al. (2024).
""")

code(r"""
# ---------------------------------------------------------------- configuration
from pathlib import Path
import json, re, warnings, itertools, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, integrate
from scipy.io import loadmat
from scipy.special import digamma, gammaln, logsumexp
import pingouin as pg
import mne

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(context="notebook", style="ticks")

# project root: this notebook lives in <root>/notebooks/
ROOT = Path.cwd().parent if (Path.cwd().parent / "derivatives").exists() else Path.cwd()
ERP_DIR = ROOT / "derivatives" / "header1_erps"        # per-participant condition averages (07)
MC_DIR = ROOT / "derivatives" / "model_comparison"     # single-trial amplitudes, CV scores, recovery (06)
RV_DIR = ROOT / "derivatives" / "reset_validation"     # block-reset tests (05)
SRC_DIR = ROOT / "derivatives" / "source"              # template head, noise covariances (11)
SUBJECTS_DIR = SRC_DIR / "subjects"
BIDS_ROOT = ROOT / "data" / "bids"

# switches for the expensive steps (False = load the saved outputs)
RECOMPUTE_ALL_CV = False        # refit all 15 x 6 models x 2 windows (minutes)
RECOMPUTE_SOURCE = False        # recompute the eLORETA t-map + cluster test (~1 min); False = local_tmap.npz
RUN_SPREAD_CHECK = True         # frontal-generator latency + leakage simulation (~2 min)

# a-priori regions of interest and windows (Rutiku et al., 2024)
ROI = {"local": dict(chs=["Fz", "FC1", "FC2", "F1", "F2", "Cz"], win=(0.08, 0.14), dev="LD", std="LS"),
       "global": dict(chs=["Pz", "P1", "P2", "CPz", "POz", "CP1", "CP2"], win=(0.30, 0.70), dev="GD", std="GS")}
CONDS = ["LSGS", "LDGS", "LSGD", "LDGD"]
MODELS = ["M0 categories", "M1 adaptation", "M2 local", "M3 global", "M4 local+global", "M5 adapt+global"]
N_PERM, SEED = 2000, 42
SOUNDS = {1: "A", 2: "B"}
print("project root:", ROOT)
""")

# =====================================================================================================
md(r"""
## 0. Shared functions

All functions are defined here, once, and used by every section. Each mirrors the corresponding project script
(named in its docstring) so the two can be compared line by line.
""")

code(r"""
# ---------------------------------------------------------------- data loading
def load_erps():
    '''Per-participant condition averages from 07_header1_erps.py.
    Returns times (s), channel names, and a dict cond -> array (subjects x channels x time) in µV
    (average reference, 30 Hz low-pass, baseline -100..0 ms).'''
    files = sorted(ERP_DIR.glob("sub-*.npz"))
    z0 = np.load(files[0])
    D = {c: np.stack([np.load(f)[f"topo__{c}"] for f in files]) for c in ["LS", "LD", "GS", "GD"]}
    return z0["times"], list(z0["ch_names"]), D, [f.stem for f in files]


def roi_wave(D, cond, chs, roi_chs):
    '''ROI average: subjects x time.'''
    return D[cond][:, [chs.index(c) for c in roi_chs]].mean(1)


def window_test(diff, times, win):
    '''Window mean of a difference wave per participant -> one-sample t-test vs 0.
    Returns mean, 95% CI half-width, t, p, number of participants with the group sign.'''
    w = (times >= win[0]) & (times <= win[1])
    m = diff[:, w].mean(1)
    t = stats.ttest_1samp(m, 0)
    ci = stats.t.ppf(0.975, len(m) - 1) * m.std(ddof=1) / np.sqrt(len(m))
    return dict(mean=m.mean(), ci95=ci, t=t.statistic, p=t.pvalue, same_sign=int((np.sign(m) == np.sign(m.mean())).sum())), m


def cluster_1d(diff, times, tmin=0.0, tmax=0.7, n_perm=N_PERM, seed=SEED):
    '''Temporal cluster permutation test (sign-flip) on a difference wave, as in 14_figure1.py.
    Cluster-forming threshold = two-sided p < .05 for t(n-1).'''
    sel = (times >= tmin) & (times <= tmax)
    n = len(diff)
    T, clusters, pv, _ = mne.stats.permutation_cluster_1samp_test(
        diff[:, sel], threshold=stats.t.ppf(0.975, n - 1), n_permutations=n_perm, tail=0, seed=seed,
        out_type="indices", verbose=False)
    ts = times[sel]
    rows = [dict(start_ms=ts[c[0]].min() * 1e3, end_ms=ts[c[0]].max() * 1e3, p=p,
                 sign="+" if T[c[0]].mean() > 0 else "-") for c, p in zip(clusters, pv) if p < 0.05]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- stimulus sequence + learners
def load_sequence(subject):
    '''Quintlet table (one row per presented quintlet) and sound onset times (PTB clock, s),
    from the sourcedata SEQUENCE / critical-events files (as 06_model_comparison.py).'''
    src = BIDS_ROOT / "sourcedata" / subject / "ses-LGeffect"
    seq = loadmat(next(src.glob("GlobLoc_stimulation_SEQUENCE*.mat")), squeeze_me=True)["SEQUENCE"]
    tz = loadmat(next(src.glob("GlobLoc_critical_events*.mat")), squeeze_me=True)["tz"]
    tz = tz[tz > 0]
    rows = []
    for b, blk in enumerate(seq, start=1):
        blk = np.atleast_2d(blk).astype(int)
        std = "".join(SOUNDS[x] for x in blk[0])          # first quintlet of a block = its standard
        for i, q in enumerate(blk):
            rows.append(dict(block=b, trial_in_block=i, quintlet="".join(SOUNDS[x] for x in q), block_standard=std))
    qs = pd.DataFrame(rows)
    qs.insert(0, "seq_idx", np.arange(len(qs)))
    assert len(tz) == 5 * len(qs), "one onset per sound expected"
    return qs, tz


def leaky_surprise(outcomes, half_life, resets):
    '''Leaky Beta-Bernoulli learner (06/07/17/25).
    outcomes: 0/1 per observation; prior Beta(1,1); counts decay by 2^(-1/half_life) per observation
    (half_life = inf: no forgetting); counts set to 0 where resets is True (block start).
    Returns the surprise -log2 p(observed outcome) computed BEFORE updating on it.'''
    lam = 1.0 if np.isinf(half_life) else 2.0 ** (-1.0 / half_life)
    n1 = n0 = 0.0
    S = np.empty(len(outcomes))
    for t, y in enumerate(outcomes):
        if resets is not None and resets[t]:
            n1 = n0 = 0.0
        mu = (n1 + 1) / (n1 + n0 + 2)                     # posterior mean p(change)
        S[t] = -np.log2(mu if y else 1 - mu)
        n1, n0 = lam * n1 + y, lam * n0 + (1 - y)
    return S


def adaptation_trace(times, ids, tau):
    '''Stimulus-specific adaptation (M1): each sound adds 1 to its own trace, all traces decay as
    exp(-dt/tau); the value read out is the trace of the sound just BEFORE it is played.'''
    a = {"A": 0.0, "B": 0.0}
    last, out = times[0], np.empty(len(times))
    for i, (t, k) in enumerate(zip(times, ids)):
        dec = np.exp(-(t - last) / tau)
        a["A"] *= dec; a["B"] *= dec
        out[i] = a[k]
        a[k] += 1.0
        last = t
    return out


def learner_outcomes(qs):
    '''Outcome streams for the two learners.
    Local (M2): every sound-to-sound transition (4 per quintlet), 1 = change; only 4->5 can change here.
    Global (M3): one outcome per quintlet, 1 = 5th sound differs from the first four.'''
    first, fifth = qs["quintlet"].str[0].to_numpy(), qs["quintlet"].str[4].to_numpy()
    start = (qs["trial_in_block"] == 0).to_numpy()
    loc = np.zeros((len(qs), 4), int); loc[:, 3] = fifth != first
    loc_reset = np.zeros(4 * len(qs), bool); loc_reset[::4] = start
    return loc.ravel(), loc_reset, (fifth != first).astype(int), start


def learner_surprise(subject, clocks):
    '''Per-quintlet local (S_loc) and global (S_glob) surprise with the participant's own selected
    parameters (M2 chosen on the early window, M3 on the late window; clocks.tsv from 06).'''
    qs, _ = load_sequence(subject)
    c = clocks[clocks["subject"] == subject]
    p_loc = c[(c["component"] == "local_80-140ms") & (c["model"] == "M2 local")].iloc[0]
    p_glo = c[(c["component"] == "global_300-700ms") & (c["model"] == "M3 global")].iloc[0]
    lo, lr, go, gr = learner_outcomes(qs)
    qs["S_loc"] = leaky_surprise(lo, float(p_loc["param"]), lr if bool(p_loc["reset"]) else None)[3::4]
    qs["S_glob"] = leaky_surprise(go, float(p_glo["param"]), gr if bool(p_glo["reset"]) else None)
    return qs[["seq_idx", "S_loc", "S_glob"]]


# ---------------------------------------------------------------- random-effects BMS
def rfx_bms(L, n_samples=100000, seed=SEED):
    '''Random-effects Bayesian model selection (Stephan et al., 2009) with protected exceedance
    probabilities (Rigoux et al., 2014). L: subjects x models log evidence (here: held-out log-likelihood).
    Variational update of the Dirichlet posterior over model frequencies; exceedance probabilities by
    sampling; Bayesian omnibus risk (BOR) = posterior probability that all models are equally frequent.'''
    n, K = L.shape
    a0 = np.ones(K); a = a0.copy()
    for _ in range(1000):
        lu = L + digamma(a) - digamma(a.sum())
        g = np.exp(lu - logsumexp(lu, axis=1, keepdims=True))   # posterior model assignment per subject
        a_new = a0 + g.sum(0)
        if np.max(np.abs(a_new - a)) < 1e-8:
            a = a_new; break
        a = a_new
    r = np.random.default_rng(seed).dirichlet(a, n_samples)
    ep = np.bincount(r.argmax(1), minlength=K) / n_samples      # exceedance probability
    F0 = np.sum(logsumexp(L, axis=1) - np.log(K))                # evidence for the null (equal frequencies)
    Elogr = digamma(a) - digamma(a.sum())
    kl = gammaln(a.sum()) - gammaln(a).sum() - gammaln(a0.sum()) + gammaln(a0).sum() + np.sum((a - a0) * Elogr)
    F1 = np.sum(g * (L + Elogr - np.log(g + 1e-300))) - kl       # evidence for the RFX model
    bor = 1 / (1 + np.exp(np.clip(F1 - F0, -700, 700)))
    return dict(alpha=a, expected_freq=a / a.sum(), ep=ep, bor=bor, pxp=ep * (1 - bor) + bor / K)


def jzs_bf10(d, r=np.sqrt(2) / 2):
    '''JZS Bayes factor for a one-sample/paired t-test (Rouder et al., 2009), as 05_reset_validation.py.'''
    d = np.asarray(d, float); n = len(d); v = n - 1
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    f = lambda g: ((1 + n * g * r ** 2) ** -0.5 * (1 + t ** 2 / ((1 + n * g * r ** 2) * v)) ** (-(v + 1) / 2)
                   * (2 * np.pi) ** -0.5 * g ** -1.5 * np.exp(-1 / (2 * g)))
    return integrate.quad(f, 0, np.inf)[0] / (1 + t ** 2 / v) ** (-(v + 1) / 2)


REPORT = []      # (section, quantity, value recomputed here) -> compared with the Results draft in Section 5
def report(section, name, value):
    REPORT.append(dict(section=section, quantity=name, notebook=value))
    return value
""")

# =====================================================================================================
md(r"""
## 1. Figure 1 — local deviance gives an early frontocentral MMN, global deviance a late parietal P3b

**Logic.** Each quintlet has two orthogonal labels: *local* (5th sound repeats, LS, or changes, LD) and *global*
(the quintlet is the block's standard, GS, or its rare deviant, GD). Physically identical quintlets are standards in
one block and deviants in another, so GD − GS is acoustically matched. Inference is on the a-priori ROI × window
means; cluster permutation tests describe where/when effects occur without a-priori windows.
""")

code(r"""
times, chs, D, subs = load_erps()
n = len(subs)
print(f"{n} participants, {len(chs)} channels, {len(times)} samples ({times[0]:.2f} to {times[-1]:.2f} s)")

fig1 = {}
for key, r in ROI.items():
    std = roi_wave(D, r["std"], chs, r["chs"]); dev = roi_wave(D, r["dev"], chs, r["chs"])
    diff = dev - std
    res, per_sub = window_test(diff, times, r["win"])
    # peak of the group-mean difference: most extreme point from 50 ms before the window to its end
    sel = (times >= r["win"][0] - 0.05) & (times <= r["win"][1])
    res["peak_ms"] = times[sel][np.argmax(np.abs(diff.mean(0)[sel]))] * 1e3
    res["clusters"] = cluster_1d(diff, times)
    fig1[key] = res
    print(f"\n{key.upper()}  {r['dev']} - {r['std']}, {int(r['win'][0]*1e3)}-{int(r['win'][1]*1e3)} ms, ROI {r['chs']}")
    print(f"  mean {res['mean']:+.2f} µV (95% CI ± {res['ci95']:.2f}), t({n-1}) = {res['t']:.2f}, p = {res['p']:.2g}, "
          f"{res['same_sign']}/{n} same sign, peak {res['peak_ms']:.0f} ms")
    print("  temporal clusters (0-700 ms):"); print(res["clusters"].round(4).to_string(index=False))

report(1, "MMN mean (µV)", fig1["local"]["mean"]); report(1, "MMN t", fig1["local"]["t"])
report(1, "MMN 95% CI half-width", fig1["local"]["ci95"]); report(1, "MMN peak (ms)", fig1["local"]["peak_ms"])
report(1, "P3b mean (µV)", fig1["global"]["mean"]); report(1, "P3b t", fig1["global"]["t"])
report(1, "P3b 95% CI half-width", fig1["global"]["ci95"]); report(1, "P3b peak (ms)", fig1["global"]["peak_ms"])
""")

md(r"""
### 1.1 Spatio-temporal clusters (Figure 1C)

Same data, all 62 channels × 0–700 ms; channel neighbourhoods from the montage; sign-flip permutation.
Cluster onsets are *descriptive*: a cluster test controls the family-wise error of "an effect exists in this
cluster", not of its exact extent (Sassenhagen & Draschkow, 2019).
""")

code(r"""
info = mne.io.read_info(ERP_DIR / "info.fif")
adjacency, _ = mne.channels.find_ch_adjacency(info, "eeg")
sel = (times >= 0) & (times <= 0.7); ts = times[sel]
st_clusters = {}
for key, (dev, std) in {"local": ("LD", "LS"), "global": ("GD", "GS")}.items():
    X = (D[dev] - D[std])[:, :, sel].transpose(0, 2, 1)          # subjects x time x channels
    T, clusters, pv, _ = mne.stats.spatio_temporal_cluster_1samp_test(
        X, threshold=stats.t.ppf(0.975, n - 1), n_permutations=N_PERM, adjacency=adjacency,
        tail=0, seed=SEED, out_type="mask", verbose=False)
    rows = []
    for c, p in zip(clusters, pv):
        if p < 0.05:
            tt = np.where(c.any(1))[0]
            rows.append(dict(sign="+" if T[c].mean() > 0 else "-", p=p, onset_ms=ts[tt[0]] * 1e3,
                             offset_ms=ts[tt[-1]] * 1e3, n_channels=int(c.any(0).sum()), peak_abs_t=np.abs(T[c]).max()))
    st_clusters[key] = pd.DataFrame(rows).sort_values("onset_ms")
    print(f"\n{key}: significant spatio-temporal clusters"); print(st_clusters[key].round(4).to_string(index=False))
""")

md(r"""
### 1.2 Local × global: 2 × 2 repeated-measures ANOVA at each a-priori window

The four 2 × 2 cells are averaged in each ROI × window per participant (`sub-*_cond4.tsv`, written by 07 from
the same epochs). Simple effects are paired t-tests. Section 4 checks the ANOVA against the equivalent contrast
t-tests (for a 2 × 2 within design, each F(1, n−1) equals the squared t of its contrast).
""")

code(r"""
c4 = pd.concat([pd.read_csv(f, sep="\t") for f in sorted(ERP_DIR.glob("sub-*_cond4.tsv"))])
cells = c4[c4["source"] == "data"].copy()
cells["local"], cells["global"] = cells["condition"].str[:2], cells["condition"].str[2:]
anova = {}
for comp in ["local", "global"]:
    q = cells[cells["component"] == comp]
    aov = pg.rm_anova(data=q, dv="amp", within=["local", "global"], subject="subject", detailed=True)
    piv = q.pivot(index="subject", columns="condition", values="amp")[CONDS]
    anova[comp] = (aov, piv)
    print(f"\n{comp} window — cell means (µV):", piv.mean().round(2).to_dict())
    print(aov[["Source", "F", "ddof1", "ddof2", [c for c in aov.columns if c.startswith("p")][0], "ng2"]].round(4).to_string(index=False))
    for name, a, b in [("LD: GD vs GS", "LDGD", "LDGS"), ("LS: GD vs GS", "LSGD", "LSGS"),
                       ("GS: LD vs LS", "LDGS", "LSGS"), ("GD: LD vs LS", "LDGD", "LSGD")]:
        d = piv[a] - piv[b]; t = stats.ttest_rel(piv[a], piv[b])
        print(f"   {name}: {d.mean():+.2f} µV, t({n-1}) = {t.statistic:.2f}, p = {t.pvalue:.2g}, "
              f"{int((np.sign(d) == np.sign(d.mean())).sum())}/{n}")
        report(1, f"{comp} simple effect {name} t", t.statistic)

pcol = lambda a: [c for c in a.columns if c.startswith("p")][0]
for comp in ["local", "global"]:
    aov = anova[comp][0].set_index("Source")
    for eff in ["local", "global", "local * global"]:
        report(1, f"ANOVA {comp}-window {eff} F", float(aov.loc[eff, "F"]))
""")

md(r"""
### 1.3 Reference control

Re-referencing is linear, so it is applied to the averages: linked mastoids = subtract the mean of TP9/TP10.
A supratemporal MMN generator predicts a larger frontocentral MMN under this reference and a polarity
inversion at the mastoids under the average reference.
""")

code(r"""
mi, fz = [chs.index(c) for c in ["TP9", "TP10"]], chs.index("Fz")
w = (times >= 0.08) & (times <= 0.14)
avg_fz = (D["LD"][:, fz] - D["LS"][:, fz])[:, w].mean(1)
lm = {c: D[c] - D[c][:, mi, :].mean(1, keepdims=True) for c in ["LD", "LS"]}
lm_fz = (lm["LD"][:, fz] - lm["LS"][:, fz])[:, w].mean(1)
mast = ((D["LD"] - D["LS"])[:, mi].mean(1))[:, w].mean(1)
for name, x in [("Fz, average reference", avg_fz), ("Fz, linked mastoids", lm_fz), ("TP9/TP10, average reference", mast)]:
    t = stats.ttest_1samp(x, 0)
    print(f"{name:30s} {x.mean():+.2f} µV, t({n-1}) = {t.statistic:.2f}, p = {t.pvalue:.2g}, {int((np.sign(x)==np.sign(x.mean())).sum())}/{n}")
report(1, "MMN Fz linked mastoids (µV)", lm_fz.mean()); report(1, "Mastoid inversion (µV)", mast.mean())
""")

# =====================================================================================================
md(r"""
## 2. Figure 2 — each response follows a different learner

**Models** of the single-trial window amplitude (all with intercept + linear drift over the session):

| | regressor(s) | free parameters (chosen inside CV) |
|---|---|---|
| M0 categories | LD, GD, LD×GD (±½ coding) | none |
| M1 adaptation | adaptation trace of the 5th sound | τ ∈ {0.25 … 32 s} |
| M2 local learner | surprise of the 4→5 transition | half-life ∈ {10, 50, 200, ∞} transitions × reset |
| M3 global learner | surprise of the quintlet's 5th-sound change | half-life ∈ {2, 5, 10, 20, ∞} quintlets × reset |
| M4 / M5 | M2 + M3 / M1 + M3 | both sets |

**Score** = summed out-of-sample Gaussian log-likelihood from *nested* leave-one-block-out CV: each block is
predicted by a model whose regression weights **and** parameters were chosen on the other seven blocks only
(inner LOBO), so flexibility is not rewarded and no penalty term is needed.
""")

code(r"""
# ---------------------------------------------------------------- the model-comparison machinery (as 06)
H_LOCAL, H_GLOBAL, RESETS = [10, 50, 200, np.inf], [2, 5, 10, 20, np.inf], [False, True]
TAUS = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
COMPONENTS = ["local_80-140ms", "global_300-700ms"]

def regressor_bank(subject):
    '''Every parameterised regressor, one value per presented quintlet (the 5th sound).'''
    qs, tz = load_sequence(subject)
    lo, lr, go, gr = learner_outcomes(qs)
    sid = np.array([c for q in qs["quintlet"] for c in q])
    bank = {}
    for h, r in itertools.product(H_LOCAL, RESETS):
        bank[("loc", h, r)] = leaky_surprise(lo, h, lr if r else None)[3::4]
    for h, r in itertools.product(H_GLOBAL, RESETS):
        bank[("glob", h, r)] = leaky_surprise(go, h, gr if r else None)
    for tau in TAUS:
        bank[("adapt", tau)] = adaptation_trace(tz, sid, tau)[4::5]
    return bank

def model_configs(model, bank):
    loc = [k for k in bank if k[0] == "loc"]; glob = [k for k in bank if k[0] == "glob"]
    ad = [k for k in bank if k[0] == "adapt"]
    return {"M0 categories": [(("cat",), ["LD", "GD", "LDxGD"])],
            "M1 adaptation": [((k,), [k]) for k in ad], "M2 local": [((k,), [k]) for k in loc],
            "M3 global": [((k,), [k]) for k in glob],
            "M4 local+global": [((a, b), [a, b]) for a in loc for b in glob],
            "M5 adapt+global": [((a, b), [a, b]) for a in ad for b in glob]}[model]

def design_matrix(trials, bank, features):
    '''Intercept + standardised session drift + the model's regressors (standardised). Standardising a
    regressor is an affine change absorbed by the intercept/slope, so it does not leak information.'''
    nT = len(trials)
    cols = [np.ones(nT), (trials["seq_idx"] - trials["seq_idx"].mean()) / trials["seq_idx"].std()]
    ld = np.where(trials["local"] == "LD", 0.5, -0.5); gd = np.where(trials["global"] == "GD", 0.5, -0.5)
    for f in features:
        if f == "LD": cols.append(ld)
        elif f == "GD": cols.append(gd)
        elif f == "LDxGD": cols.append(4 * ld * gd)
        else:
            x = bank[f][trials["seq_idx"].to_numpy()]
            cols.append((x - x.mean()) / (x.std() + 1e-12))
    return np.column_stack(cols)

def block_stats(X, y, blocks, ub):
    '''Per-block sufficient statistics (X'X, X'y, y'y, n) so every train/test split is a cheap sum.'''
    return {b: (X[blocks == b].T @ X[blocks == b], X[blocks == b].T @ y[blocks == b],
                float(y[blocks == b] @ y[blocks == b]), int((blocks == b).sum())) for b in ub}

def fit_eval(st, train, test):
    '''OLS on the training blocks; Gaussian log-likelihood of the test blocks with the training noise variance.'''
    XtX = sum(st[b][0] for b in train); Xty = sum(st[b][1] for b in train)
    yty = sum(st[b][2] for b in train); nn = sum(st[b][3] for b in train)
    p = XtX.shape[0]
    beta = np.linalg.solve(XtX + 1e-9 * np.eye(p), Xty)
    s2 = max((yty - 2 * beta @ Xty + beta @ XtX @ beta) / (nn - p), 1e-12)
    ll = sum(-0.5 * st[b][3] * np.log(2 * np.pi * s2) - (st[b][2] - 2 * beta @ st[b][1] + beta @ st[b][0] @ beta) / (2 * s2)
             for b in test)
    return ll, beta, s2

def nested_cv(trials, y, bank, model):
    '''Outer leave-one-block-out prediction; parameters chosen by inner leave-one-block-out within the
    seven training blocks. Returns the summed held-out log-likelihood and the configuration chosen per fold.'''
    blocks = trials["block"].to_numpy(); ub = np.unique(blocks)
    configs = model_configs(model, bank)
    st = [block_stats(design_matrix(trials, bank, feats), y, blocks, ub) for _, feats in configs]
    total, chosen = 0.0, []
    for bo in ub:
        train = [b for b in ub if b != bo]
        best = 0 if len(configs) == 1 else int(np.argmax(
            [sum(fit_eval(s, [b for b in train if b != bi], [bi])[0] for bi in train) for s in st]))
        total += fit_eval(st[best], train, [bo])[0]
        chosen.append(configs[best][0])
    return total, chosen

def load_trials(subject):
    return pd.read_csv(MC_DIR / "trials" / f"{subject}.tsv", sep="\t")
""")

md(r"""
### 2.1 Reproducibility check: refit one participant from scratch

The saved scores (`cv_loglik.tsv`) were produced by `06_model_comparison.py`. Here all six models are refitted
for one participant with the functions above; the two must agree to numerical precision.
""")

code(r"""
cv_saved = pd.read_csv(MC_DIR / "cv_loglik.tsv", sep="\t")
check_sub = "sub-01"
t0 = time.time()
tr, bank = load_trials(check_sub), regressor_bank(check_sub)
rows = []
for comp in COMPONENTS:
    for m in MODELS:
        ll, _ = nested_cv(tr, tr[comp].to_numpy(), bank, m)
        saved = cv_saved[(cv_saved.subject == check_sub) & (cv_saved.component == comp) & (cv_saved.model == m)].cv_ll.iloc[0]
        rows.append(dict(component=comp, model=m, refit=ll, saved=saved, abs_diff=abs(ll - saved)))
chk = pd.DataFrame(rows)
print(chk.round(4).to_string(index=False)); print(f"({time.time()-t0:.0f} s)")
assert chk.abs_diff.max() < 1e-6, "refit does not reproduce the saved scores"
print("OK: refit reproduces the saved cross-validated scores exactly")

if RECOMPUTE_ALL_CV:
    rows = []
    for s in sorted(p.stem for p in (MC_DIR / "trials").glob("sub-*.tsv")):
        tr, bank = load_trials(s), regressor_bank(s)
        for comp in COMPONENTS:
            for m in MODELS:
                rows.append(dict(component=comp, subject=s, model=m, cv_ll=nested_cv(tr, tr[comp].to_numpy(), bank, m)[0]))
    cv_saved = pd.DataFrame(rows)
""")

md(r"""
### 2.2 Group model comparison (Figure 2D) and head-to-head evidence (Figure 2C)
""")

code(r"""
win_name = {"local_80-140ms": "early", "global_300-700ms": "late"}
bms = {}
for comp in COMPONENTS:
    Lm = cv_saved[cv_saved.component == comp].pivot(index="subject", columns="model", values="cv_ll")[MODELS]
    b = rfx_bms(Lm.to_numpy()); bms[comp] = b
    best = pd.Series(np.array(MODELS)[Lm.to_numpy().argmax(1)]).value_counts().to_dict()
    print(f"\n{win_name[comp].upper()} window ({comp})")
    print(pd.DataFrame(dict(PXP=b["pxp"], EP=b["ep"], expected_freq=b["expected_freq"]), index=MODELS).round(3).to_string())
    print(f"  BOR = {b['bor']:.3f};  best model per participant: {best}")
    # sensitivity: preregistered six models vs M0-M3 only
    b4 = rfx_bms(Lm[MODELS[:4]].to_numpy())
    print(f"  M0-M3 only: PXP {dict(zip([m.split()[0] for m in MODELS[:4]], b4['pxp'].round(3)))}, BOR = {b4['bor']:.3f}")
    winner = "M2 local" if comp.startswith("local") else "M3 global"
    report(2, f"PXP {winner} ({win_name[comp]})", b["pxp"][MODELS.index(winner)]); report(2, f"BOR {win_name[comp]}", b["bor"])
    report(2, f"PXP {winner} M0-M3 only ({win_name[comp]})", b4["pxp"][MODELS.index(winner)])
    # head-to-head: winner minus each rival, per participant
    other = "M3 global" if winner == "M2 local" else "M2 local"
    for rival in ["M0 categories", "M1 adaptation", other]:
        d = Lm[winner] - Lm[rival]
        print(f"  {winner.split()[0]} - {rival.split()[0]}: mean {d.mean():+.1f} per participant, {int((d > 0).sum())}/{len(d)} favour {winner.split()[0]}")
        report(2, f"{win_name[comp]} {winner.split()[0]}-{rival.split()[0]} mean dLL", d.mean())
""")

md(r"""
### 2.3 Model recovery (Figure 2B)

From `06_model_comparison.py --recovery`: data simulated from each model with each participant's own fitted
parameters, betas and residual noise, then analysed with the full pipeline; 20 simulations per generating model.
Rows = generating model, columns = model with the highest PXP.
""")

code(r"""
rec = pd.read_csv(MC_DIR / "recovery.tsv", sep="\t")
for comp in COMPONENTS:
    g = rec[rec.component == comp]
    M = pd.crosstab(g.generating, g.winner_pxp, normalize="index").reindex(index=MODELS, columns=MODELS, fill_value=0) * 100
    print(f"\n{win_name[comp]} window: % of simulations won (n = {g.groupby('generating').size().iloc[0]} per model)")
    print(M.round(0).astype(int).rename(columns=lambda c: c.split()[0], index=lambda c: c.split()[0]).to_string())
    for m in ["M2 local", "M3 global"]:
        report(2, f"recovery {m.split()[0]} {win_name[comp]} (%)", M.loc[m, m])
""")

md(r"""
### 2.4 What each learner predicts for the four conditions (Figure 2E)

Within each participant, each learner's surprise is mapped to single-trial window amplitude by OLS (intercept +
slope) and averaged per 2 × 2 cell (written by 07 into `sub-*_cond4.tsv`). The correlation over the four group
means is **descriptive** (four points).
""")

code(r"""
for comp in ["local", "global"]:
    q = c4[c4.component == comp]
    dat = q[q.source == "data"].pivot(index="subject", columns="condition", values="amp")[CONDS].mean()
    for lname in ["local learner", "global learner"]:
        pr = q[q.source == lname].pivot(index="subject", columns="condition", values="amp")[CONDS].mean()
        r_ = np.corrcoef(pr, dat)[0, 1]
        print(f"{comp:6s} window, {lname:14s}: r = {r_:.2f}   (predicted {pr.round(2).to_dict()})")
        report(2, f"cond-mean r {comp} window {lname}", r_)
""")

md(r"""
### 2.5 Amplitude as a function of model surprise (Figure 2F and unbinned tests)

**(a) Binned (Figure 2F).** Deviants (LD for the early window, GD for the late one) split into within-participant
thirds of the winning learner's surprise; slope of window amplitude over thirds, t-tested across participants.
`raw` = surprise as is; `within` = surprise centred within each 2 × 2 condition (only trial-by-trial learning left).

**(b) Unbinned.** Single-trial OLS of window amplitude on surprise per participant; slopes (µV/bit) t-tested.
The *added-variable* version residualises amplitude and the other learner's surprise on the window's own learner,
i.e. asks what the other learner explains **beyond** it.

**(c) Tie-breaking check.** 07 formed thirds with `rank(method="first")`; when surprise values tie (common with a
block reset — the k-th deviant of every block can have identical surprise), ties are broken by trial order, which
could confound surprise with time in session. The binned slopes are recomputed here from the single-trial data
with ties broken at random (100 permutations) to check this.
""")

code(r"""
# (a) binned slopes, as written by 07 (sub-*_windows.tsv)
win = pd.concat([pd.read_csv(f, sep="\t") for f in sorted(ERP_DIR.glob("sub-*_windows.tsv"))])
for dev, var, comp in [("LD", "S_loc", "local"), ("GD", "S_glob", "global")]:
    for kind in ["raw", "within"]:
        q = win[(win.deviants == dev) & (win["var"] == var) & (win.kind == kind) & (win.component == comp)]
        piv = q.pivot(index="subject", columns="bin", values="amp")
        sl = piv.apply(lambda r: np.polyfit(np.arange(3), r.to_numpy(), 1)[0], axis=1)
        t = stats.ttest_1samp(sl, 0)
        print(f"{dev} by {var} ({kind:6s}), {comp} window: slope {sl.mean():+.2f} µV/third, t({len(sl)-1}) = {t.statistic:.2f}, "
              f"p = {t.pvalue:.3g}, {int((np.sign(sl)==np.sign(sl.mean())).sum())}/{len(sl)}")
        report(2, f"binned slope t {comp} {kind}", t.statistic)
""")

code(r"""
# (b) unbinned single-trial slopes, and (c) binned slopes with random tie-breaking
clocks = pd.read_csv(MC_DIR / "clocks.tsv", sep="\t")
subs_mc = sorted(p.stem for p in (MC_DIR / "trials").glob("sub-*.tsv"))
trials = {s: load_trials(s).merge(learner_surprise(s, clocks), on="seq_idx", validate="one_to_one") for s in subs_mc}
WIN_COL = {"local": "local_80-140ms", "global": "global_300-700ms"}

def resid(v, z):
    Z = np.column_stack([np.ones(len(z)), z])
    return v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]

def slopes(comp, var, mode, control=None):
    out = []
    for s, d in trials.items():
        y, x = d[WIN_COL[comp]].to_numpy(float), d[var].to_numpy(float)
        if mode == "within":     # centre both within each 2x2 condition
            x = x - d.groupby("condition")[var].transform("mean").to_numpy()
            y = y - d.groupby("condition")[WIN_COL[comp]].transform("mean").to_numpy()
        if mode == "partial":    # added-variable: residualise on the window's own learner
            z = d[control].to_numpy(float); x, y = resid(x, z), resid(y, z)
        out.append(np.polyfit(x, y, 1)[0])
    out = np.array(out); t = stats.ttest_1samp(out, 0)
    return out.mean(), t.statistic, t.pvalue, int((np.sign(out) == np.sign(out.mean())).sum())

for comp, own, other in [("local", "S_loc", "S_glob"), ("global", "S_glob", "S_loc")]:
    for label, args in [("own learner", (own, "raw")), ("own learner, within condition", (own, "within")),
                        ("other learner beyond own", (other, "partial", own))]:
        m, t, p, k = slopes(comp, *args)
        print(f"{comp:6s} window, {label:32s}: {m:+.2f} µV/bit, t(14) = {t:.2f}, p = {p:.3g}, {k}/15")
        report(2, f"single-trial slope t {comp} {label}", t)

rng = np.random.default_rng(SEED)
def tertile_slopes(comp, var, dev_col, dev, within, tie="first"):
    out = []
    for s, d in trials.items():
        q = d[d[dev_col] == dev].reset_index(drop=True)
        x = q[var].to_numpy(float); y = q[WIN_COL[comp]].to_numpy(float)
        if within:
            x = x - q.groupby("condition")[var].transform("mean").to_numpy()
        key = pd.Series(x + (rng.uniform(0, 1e-9, len(x)) if tie == "random" else 0)).rank(method="first")
        b = pd.qcut(key, 3, labels=False).to_numpy()
        out.append(np.polyfit(np.arange(3), [y[b == k].mean() for k in range(3)], 1)[0])
    return stats.ttest_1samp(out, 0).statistic

for comp, var, dev_col, dev in [("local", "S_loc", "local", "LD"), ("global", "S_glob", "global", "GD")]:
    for within in [False, True]:
        frac_tied = np.mean([1 - d[d[dev_col] == dev][var].nunique() / (d[dev_col] == dev).sum() for d in trials.values()])
        t_first = tertile_slopes(comp, var, dev_col, dev, within, "first")
        t_rand = np.array([tertile_slopes(comp, var, dev_col, dev, within, "random") for _ in range(100)])
        print(f"{comp:6s} {'within' if within else 'raw   '}: tied values {100*frac_tied:4.1f}%  t(order tie-break) = {t_first:+.2f}  "
              f"t(random tie-break) = {t_rand.mean():+.2f} [{np.percentile(t_rand,2.5):+.2f}, {np.percentile(t_rand,97.5):+.2f}]")
""")

md(r"""
### 2.6 Timescales of the two learners (preregistered "clocks" test) and block reset

**Timescales.** Each learner's selected half-life (from the full-data LOBO fit) is converted to seconds
(local: transitions ÷ 4 × quintlet period; global: quintlets × period; ∞ = no forgetting, ranked highest) and
compared between levels with a Wilcoxon signed-rank test. Preregistered criterion: p < .05.

**Reset** (from `05_reset_validation.py`, which needs the epochs): *Test 1* — a learner that carries memory across
blocks predicts a smaller P3b (300–600 ms) to early-block deviants whose structure was the previous block's
standard; tested with a paired t-test and JZS Bayes factor. *Test 2* — the global learner with memory scaled by
ρ at each block start (ρ = 0 full reset, 1 none), leave-one-block-out CV over ρ. Both are recomputed here from the
saved per-participant tables.
""")

code(r"""
c_loc = clocks[(clocks.component == "local_80-140ms") & (clocks.model == "M2 local")].set_index("subject")
c_glo = clocks[(clocks.component == "global_300-700ms") & (clocks.model == "M3 global")].set_index("subject")
a = c_loc["half_life_s"].replace(np.inf, 1e6); b = c_glo["half_life_s"].loc[a.index].replace(np.inf, 1e6)
w_ = stats.wilcoxon(a, b)
print(f"half-life (s): local median {c_loc.half_life_s.median():.2f}, global median {c_glo.half_life_s.median():.2f}; "
      f"Wilcoxon W = {w_.statistic:.1f}, p = {w_.pvalue:.3f}; local longer in {(a > b).sum()}, equal in {(a == b).sum()}")
print(f"reset selected: local learner {int(c_loc.reset.sum())}/15, global learner {int(c_glo.reset.sum())}/15")
report(2, "clocks Wilcoxon p", w_.pvalue)

t1 = pd.read_csv(RV_DIR / "test1_history.tsv", sep="\t")
w1 = t1.pivot(index="subject", columns="history", values="p3").dropna()
d1 = w1["was_prev_standard"] - w1["other"]; tt1 = stats.ttest_rel(w1["was_prev_standard"], w1["other"])
bf10 = jzs_bf10(d1.to_numpy())
print(f"\nReset test 1: P3b difference {d1.mean():+.2f} µV, t({len(d1)-1}) = {tt1.statistic:.2f}, p = {tt1.pvalue:.2f}, BF01 = {1/bf10:.2f}")
report(2, "reset test1 BF01", 1 / bf10)

t2 = pd.read_csv(RV_DIR / "test2_rho_cv.tsv", sep="\t")
prof = t2.pivot(index="subject", columns="rho", values="cv_loglik")
boot = [prof.iloc[rng.integers(0, len(prof), len(prof))].sum().idxmax() for _ in range(5000)]
best = prof.idxmax(axis=1)
r2 = t2.pivot(index="subject", columns="rho", values="cv_r2"); tt2 = stats.ttest_rel(r2[0.0], r2[1.0])
print(f"Reset test 2: group-best rho = {prof.sum().idxmax()}, bootstrap 95% CI [{np.percentile(boot,2.5)}, {np.percentile(boot,97.5)}]; "
      f"per-participant best rho <= 0.1 in {(best <= 0.1).sum()}/15 (rho = 1 in {(best == 1).sum()}/15); "
      f"CV R² reset > no reset in {(r2[0.0] > r2[1.0]).sum()}/15, t = {tt2.statistic:.2f}, p = {tt2.pvalue:.3f}")
report(2, "reset group-best rho", prof.sum().idxmax())
""")

# =====================================================================================================
md(r"""
## 3. Figure 3 — the local effect originates in left auditory cortex

eLORETA on a template head (MNE "sample": FreeSurfer anatomy, 3-layer BEM, oct-6 source space; standard 10-05
positions coregistered by fiducials + ICP), loose orientation 0.2, depth 0.8, λ² = 1/9, per-participant noise
covariance from the silent pre-sequence period. Statistics use the **signed** normal-orientation difference
(zero expectation under the null, unaffected by unequal trial counts). Cluster test on the 80–140 ms window mean
(cluster-forming p < .01, 2,000 sign flips, source-space adjacency).
""")

code(r"""
WIN_SRC, LAMBDA2, LOOSE, DEPTH = (0.08, 0.14), 1 / 9, 0.2, 0.8
AC = ["transversetemporal", "superiortemporal"]

def src_info():
    inf = mne.io.read_info(ERP_DIR / "info.fif")
    inf = mne.pick_info(inf, mne.pick_types(inf, eeg=True))
    with inf._unlock():
        inf["projs"] = []
    raw0 = mne.io.RawArray(np.zeros((len(inf["ch_names"]), 1)), inf)
    raw0.set_eeg_reference("average", projection=True)
    return raw0.info

def make_forward(inf):
    '''Coregister standard electrode positions to the template head (fiducials, then ICP) and build the
    BEM forward model (as 11/19).'''
    coreg = mne.coreg.Coregistration(inf, "sample", subjects_dir=SUBJECTS_DIR, fiducials="auto")
    coreg.fit_fiducials(verbose=False); coreg.fit_icp(n_iterations=40, nasion_weight=2.0, verbose=False)
    coreg.omit_head_shape_points(distance=10 / 1000); coreg.fit_icp(n_iterations=40, nasion_weight=2.0, verbose=False)
    tb = SUBJECTS_DIR / "sample" / "bem"
    return mne.make_forward_solution(inf, trans=coreg.trans, src=mne.read_source_spaces(tb / "sample-oct-6-src.fif"),
                                     bem=mne.read_bem_solution(tb / "sample-1280-1280-1280-bem-sol.fif"),
                                     eeg=True, meg=False, mindist=5.0, n_jobs=1)

def signed_stcs(inf, fwd):
    '''Per-participant signed eLORETA estimate of LD - LS (linear, so = difference of the estimates).'''
    out = []
    for s in subs:
        z = np.load(ERP_DIR / f"{s}.npz")
        inv = mne.minimum_norm.make_inverse_operator(inf, fwd, mne.read_cov(SRC_DIR / f"{s}-cov.fif"), loose=LOOSE, depth=DEPTH)
        ev = mne.EvokedArray((z["topo__LD"] - z["topo__LS"]) * 1e-6, inf, tmin=z["times"][0])
        out.append(mne.minimum_norm.apply_inverse(ev, inv, LAMBDA2, method="eLORETA", pick_ori="normal",
                                                  method_params=dict(max_iter=60)))
    return out

have_template = (SUBJECTS_DIR / "sample" / "bem" / "sample-oct-6-src.fif").exists()
print("template head available:", have_template)
need_stcs = RECOMPUTE_SOURCE or RUN_SPREAD_CHECK
if have_template and need_stcs:
    t0 = time.time()
    inf = src_info(); fwd = make_forward(inf); src = fwd["src"]
    stcs = signed_stcs(inf, fwd)
    labels = {lb.name: lb for lb in mne.read_labels_from_annot("sample", "aparc", subjects_dir=SUBJECTS_DIR)}
    print(f"forward + 15 inverse solutions: {time.time()-t0:.0f} s")
""")

code(r"""
if RECOMPUTE_SOURCE and have_template:
    X = np.stack([s.copy().crop(*WIN_SRC).data.mean(1) for s in stcs])
    T, clusters, pv, _ = mne.stats.permutation_cluster_1samp_test(
        X, threshold=stats.t.ppf(1 - 0.01 / 2, n - 1), n_permutations=N_PERM, adjacency=mne.spatial_src_adjacency(src),
        tail=0, seed=SEED, out_type="mask", n_jobs=1, verbose=False)
    mask = np.zeros_like(T, bool)
    for c, p in zip(clusters, pv):
        if p < 0.05: mask |= c
    vert = [s["vertno"] for s in src]; cluster_p = np.sort(pv)
else:
    z = np.load(SRC_DIR / "local_tmap.npz")          # saved by 19_figure3.py (same settings)
    T, mask, vert, cluster_p = z["T"], z["mask"], [z["vert_lh"], z["vert_rh"]], z["cluster_p"]
print(f"local effect: cluster p = {np.round(cluster_p[:3], 4)}, {mask.sum()} significant sources, peak |t| = {np.abs(T).max():.1f}")
report(3, "eLORETA cluster p", cluster_p[0]); report(3, "eLORETA n sources", int(mask.sum())); report(3, "eLORETA peak |t|", np.abs(T).max())

if have_template:
    labs = mne.read_labels_from_annot("sample", "aparc", subjects_dir=SUBJECTS_DIR)
    rows = []
    for lb in labs:
        if "unknown" in lb.name: continue
        h = 0 if lb.hemi == "lh" else 1
        idx = np.where(np.isin(vert[h], lb.vertices))[0] + (0 if h == 0 else len(vert[0]))
        if len(idx): rows.append(dict(label=lb.name, frac_sig=mask[idx].mean(), mean_abs_t=np.abs(T[idx]).mean()))
    inv_tab = pd.DataFrame(rows).sort_values("mean_abs_t", ascending=False)
    print(inv_tab[inv_tab.frac_sig > 0].head(12).round(2).to_string(index=False))
    for l in ["transversetemporal-lh", "superiortemporal-lh", "superiortemporal-rh", "insula-lh", "parahippocampal-lh"]:
        report(3, f"{l} fraction significant", float(inv_tab.set_index("label").loc[l, "frac_sig"]))
""")

md(r"""
### 3.1 Auditory-cortex source time course (Figure 3B)

Sign-flip-aligned mean over the aparc labels transverse temporal + superior temporal (`mean_flip`), per
hemisphere; window-mean t-test. Source polarity is arbitrary (plotted with the MMN negative).
""")

code(r"""
if have_template and need_stcs:
    ac = {h: sum((labels[f"{nm}-{h}"] for nm in AC[1:]), labels[f"{AC[0]}-{h}"]) for h in ("lh", "rh")}
    tc = {h: np.stack([s.extract_label_time_course(ac[h], src, mode="mean_flip")[0] for s in stcs]) for h in ac}
    tsrc = stcs[0].times
else:
    z = np.load(SRC_DIR / "local_tmap.npz"); tc = {"lh": z["ac_lh"], "rh": z["ac_rh"]}; tsrc = z["times"]
w = (tsrc >= WIN_SRC[0]) & (tsrc <= WIN_SRC[1])
flip = -1.0 if tc["lh"].mean(0)[w].mean() > 0 else 1.0
for h in ["lh", "rh"]:
    X = flip * tc[h]; m = X[:, w].mean(1); t = stats.ttest_1samp(m, 0)
    sel = (tsrc >= 0.05) & (tsrc <= 0.2); pk = tsrc[sel][np.argmin(X.mean(0)[sel])] * 1e3
    print(f"auditory cortex {h}: t(14) = {t.statistic:.2f}, p = {t.pvalue:.2g}, {int((np.sign(m)==np.sign(m.mean())).sum())}/15, peak {pk:.0f} ms")
    report(3, f"auditory cortex {h} t", t.statistic)
""")

md(r"""
### 3.2 Is the insula / inferior-frontal part of the cluster a separate generator?

Two checks (as `20_frontal_generator_check.py`): (i) **timing** — a second (frontal) MMN generator should peak
*after* auditory cortex; jackknife test of the peak-latency difference (Miller et al., 1998). (ii) **leakage** — a
source confined to left auditory cortex is projected through the forward model and inverted with each participant's
operator; the insula/IFG amplitude relative to auditory cortex is what spread alone produces.
""")

code(r"""
if have_template and RUN_SPREAD_CHECK:
    ifg = labels["insula-lh"] + labels["parsopercularis-lh"]; acl = ac["lh"]
    sa = lambda X: X * np.sign(X.mean(0)[(tsrc >= .08) & (tsrc <= .14)].mean())
    A_ = sa(tc["lh"]); I_ = sa(np.stack([s.extract_label_time_course(ifg, src, mode="mean_flip")[0] for s in stcs]))
    wpk = (tsrc >= 0.06) & (tsrc <= 0.16); tw = tsrc[wpk]
    pk = lambda X: tw[np.argmax(X.mean(0)[wpk])]
    dfull = pk(I_) - pk(A_)
    jk = np.array([pk(np.delete(I_, i, 0)) - pk(np.delete(A_, i, 0)) for i in range(n)])
    se = np.sqrt((n - 1) / n * ((jk - jk.mean()) ** 2).sum()); tj = dfull / se
    print(f"window t (insula+IFG, left): {stats.ttest_1samp(I_[:, w].mean(1), 0).statistic:.2f}")
    print(f"peak latency: auditory cortex {pk(A_)*1e3:.0f} ms, insula/IFG {pk(I_)*1e3:.0f} ms; difference {dfull*1e3:+.0f} ms, "
          f"jackknife t({n-1}) = {tj:.2f}, p = {2*stats.t.sf(abs(tj), n-1):.2f}")
    ratio_obs = I_.mean(0)[wpk].max() / A_.mean(0)[wpk].max()
    # leakage simulation: unit, sign-aligned currents in left auditory cortex only
    fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True)
    idx = np.where(np.isin(vert[0], acl.vertices))[0]
    x = np.zeros(sum(len(v) for v in vert)); x[idx] = 1e-9 * mne.label_sign_flip(acl, src)
    data = fwd_fixed["sol"]["data"] @ x
    rat = []
    for s in subs:
        inv = mne.minimum_norm.make_inverse_operator(inf, fwd, mne.read_cov(SRC_DIR / f"{s}-cov.fif"), loose=LOOSE, depth=DEPTH)
        ev = mne.EvokedArray(np.tile(data[:, None], (1, 3)), inf, tmin=0); ev.set_eeg_reference("average", projection=True)
        e = mne.minimum_norm.apply_inverse(ev, inv, LAMBDA2, method="eLORETA", pick_ori="normal", method_params=dict(max_iter=60))
        rat.append(abs(e.extract_label_time_course(ifg, src, mode="mean_flip")[0][0]) / abs(e.extract_label_time_course(acl, src, mode="mean_flip")[0][0]))
    print(f"insula/IFG amplitude relative to auditory cortex: observed {100*ratio_obs:.0f}%, leakage alone {100*np.mean(rat):.0f}% "
          f"(range {100*min(rat):.0f}-{100*max(rat):.0f}%)")
    report(3, "frontal latency jackknife t", tj); report(3, "leakage ratio (%)", 100 * np.mean(rat))
""")

md(r"""
### 3.3 Global effect: no reliable focal source (summary from `11_source_eloreta.py`)

The 300–700 ms GD − GS signed difference was tested the same way in 11 (with sLORETA as a check). Saved summary:
""")

code(r"""
gs = json.loads((SRC_DIR / "group_summary.json").read_text())["global"]
print(f"global effect: cluster p = {gs['cluster_p'][:3]}, {gs['n_sig_sources']} significant sources; "
      f"peak of mean |current|: {gs['peak_label_of_mean_abs_current']}")
print(pd.DataFrame(gs["top_labels_by_mean_abs_t"]).to_string(index=False))
report(3, "global n sources", gs["n_sig_sources"])
""")

# =====================================================================================================
md(r"""
## 4. Validation of the methods on simulated data

Each check simulates data where the answer is known and confirms the method returns it.

| Check | Method | Expected |
|---|---|---|
| V1 | leaky Beta-Bernoulli learner | closed-form posterior; reset and forgetting behave as specified |
| V2 | random-effects BMS / PXP / BOR | finds the true majority model; BOR ≈ 1 and PXP ≈ 1/K under the null |
| V3 | nested CV | selects the generating *model family*; shows how well half-life / reset are identifiable at the real SNR |
| V4 | temporal cluster test | false-positive rate ≈ 5% on null data with ERP-like autocorrelation |
| V5 | 2 × 2 rm-ANOVA | each F equals the squared t of its contrast |
| V6 | added-variable slope | ≈ 0 for a regressor with no unique effect, despite a raw correlation |
| V7 | eLORETA localisation | a simulated left auditory-cortex source is localised there, and how far it leaks |
""")

code(r"""
# V1: learner against the closed form. No forgetting: p(change) before trial t = (k + 1) / (t + 2).
rng = np.random.default_rng(1)
y = rng.integers(0, 2, 200)
S = leaky_surprise(y, np.inf, None)
k = np.concatenate([[0], np.cumsum(y)[:-1]]); tt_ = np.arange(200)
mu = (k + 1) / (tt_ + 2)
assert np.allclose(S, -np.log2(np.where(y == 1, mu, 1 - mu))), "learner != closed form"
# reset: surprise after a reset equals surprise of a fresh learner on the same outcomes
r_ = np.zeros(200, bool); r_[100] = True
assert np.allclose(leaky_surprise(y, np.inf, r_)[100:], leaky_surprise(y[100:], np.inf, None))
# forgetting: half-life h halves the weight of an observation after h further observations
h = 10; lam = 2 ** (-1 / h); assert np.isclose(lam ** h, 0.5)
# block structure: the first deviant of a 'repeat' block is maximally surprising after a reset
print("V1 OK: learner matches the closed-form posterior; reset and forgetting behave as specified")
""")

code(r"""
# V2: BMS. Simulate 15 subjects x 4 models; each subject's true model drawn from frequencies r,
# log evidence = true model gets +d, others 0, plus noise.
def sim_L(freqs, d=4.0, sd=1.0, n_sub=15, seed=0):
    g = np.random.default_rng(seed)
    true = g.choice(len(freqs), n_sub, p=freqs)
    L = g.normal(0, sd, (n_sub, len(freqs))); L[np.arange(n_sub), true] += d
    return L
wins, bors = [], []
for s in range(200):
    b = rfx_bms(sim_L([0.6, 0.2, 0.1, 0.1], seed=s), n_samples=20000, seed=s)
    wins.append(np.argmax(b["pxp"]) == 0)
null = [rfx_bms(np.random.default_rng(s).normal(0, 1, (15, 4)), n_samples=20000, seed=s) for s in range(200)]
print(f"V2: majority model (60%) has the highest PXP in {100*np.mean(wins):.0f}% of 200 simulations")
print(f"    null (no differences): mean BOR = {np.mean([b['bor'] for b in null]):.2f}, mean max PXP = {np.mean([b['pxp'].max() for b in null]):.2f} (1/K = 0.25), "
      f"max PXP > 0.9 in {100*np.mean([b['pxp'].max() > 0.9 for b in null]):.1f}%")
""")

code(r"""
# V3: nested CV recovery on a real sequence (sub-02). Generate single-trial data from M2 with a known
# half-life and reset (truth = 50-quintlet half-life, reset at block start), at the empirical noise level
# (SD of that participant's single-trial amplitudes), with an effect size x1, x3 and x10 the empirical
# one (-0.6 µV per SD of surprise), and ask which configuration the inner CV selects.
s0 = "sub-02"; tr0, bank0 = load_trials(s0), regressor_bank(s0)
truth = ("loc", 50, True)
X = design_matrix(tr0, bank0, [truth])
noise_sd = tr0["local_80-140ms"].std()
# (a) how different are the candidate regressors at all? correlation of the truth with every local config
loc_keys = [k for k in bank0 if k[0] == "loc"]
R = np.corrcoef(np.array([bank0[k][tr0["seq_idx"].to_numpy()] for k in loc_keys]))
print("V3a: correlation of the generating regressor with each local configuration (half-life, reset):")
print("     " + ", ".join(f"{k[1]}/{'R' if k[2] else '-'}: {R[loc_keys.index(truth), j]:.3f}" for j, k in enumerate(loc_keys)))
# (b) recovery as a function of SNR (10 simulations per level; the code is correct if recovery -> 100%)
for scale in [1, 3, 10]:
    sel = []
    for rep in range(10):
        y = X @ np.array([0, 0, -0.6 * scale]) + np.random.default_rng(rep).normal(0, noise_sd, len(tr0))
        _, chosen = nested_cv(tr0, y, bank0, "M2 local")
        sel.append(pd.Series([c[0] for c in chosen]).mode()[0])
    print(f"V3b: effect x{scale:<2}: CV's modal choice = truth in {sum(c == truth for c in sel)}/10; "
          f"choices: {pd.Series(sel).value_counts().to_dict()}")
# (c) model identity at the empirical SNR: M2 should still beat M0, M1 and M3 on held-out data
y = X @ np.array([0, 0, -0.6]) + np.random.default_rng(99).normal(0, noise_sd, len(tr0))
print("V3c: held-out log-lik at empirical SNR:", {m.split()[0]: round(nested_cv(tr0, y, bank0, m)[0], 1) for m in MODELS[:4]})
""")

code(r"""
# V4: false-positive rate of the temporal cluster test on null ERP-like data (15 subjects, 1/f-ish noise
# smoothed to ~30 Hz, sampled like the real data 0-700 ms).
from scipy.ndimage import gaussian_filter1d
import warnings; warnings.filterwarnings("ignore", message="No clusters found")
tt0 = times[(times >= 0) & (times <= 0.7)]
fp = 0; n_sim = 200
for s in range(n_sim):
    g = np.random.default_rng(1000 + s)
    Xn = gaussian_filter1d(g.normal(0, 1, (15, len(tt0))), sigma=5, axis=1)
    _, cl, pv, _ = mne.stats.permutation_cluster_1samp_test(Xn, threshold=stats.t.ppf(0.975, 14), n_permutations=500,
                                                           tail=0, seed=s, out_type="indices", verbose=False)
    fp += np.any(pv < 0.05)
print(f"V4: family-wise false-positive rate = {fp/n_sim:.3f} (nominal 0.05; 95% MC interval ~0.02-0.08)")
""")

code(r"""
# V5: 2x2 rm-ANOVA vs contrast t-tests (real cell means): F(1,14) = t^2 for each effect
for comp in ["local", "global"]:
    aov, piv = anova[comp]; aov = aov.set_index("Source")
    con = {"local": (piv.LDGS + piv.LDGD) - (piv.LSGS + piv.LSGD),
           "global": (piv.LSGD + piv.LDGD) - (piv.LSGS + piv.LDGS),
           "local * global": (piv.LDGD - piv.LSGD) - (piv.LDGS - piv.LSGS)}
    for eff, c in con.items():
        t2_ = stats.ttest_1samp(c, 0).statistic ** 2
        assert np.isclose(t2_, aov.loc[eff, "F"], rtol=1e-6), (comp, eff)
print("V5 OK: every ANOVA F equals the squared contrast t")

# V6: added-variable slope. x1 drives y; x2 correlates with x1 (r ~ .2, as S_loc vs S_glob) but has no effect.
g = np.random.default_rng(5); raw, part = [], []
for s in range(15):
    x1 = g.normal(size=900); x2 = 0.2 * x1 + g.normal(size=900); y = -0.5 * x1 + g.normal(size=900) * 3
    raw.append(np.polyfit(x2, y, 1)[0]); part.append(np.polyfit(resid(x2, x1), resid(y, x1), 1)[0])
print(f"V6: raw slope of the inert regressor t = {stats.ttest_1samp(raw,0).statistic:.1f} (spurious), "
      f"added-variable slope t = {stats.ttest_1samp(part,0).statistic:.2f}, p = {stats.ttest_1samp(part,0).pvalue:.2f} (correctly null)")
""")

code(r"""
# V7: eLORETA localisation of a simulated source in left auditory cortex (template head, one covariance)
if have_template and need_stcs:
    fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True)
    idx = np.where(np.isin(vert[0], ac["lh"].vertices))[0]
    x = np.zeros(sum(len(v) for v in vert)); x[idx] = 1e-9 * mne.label_sign_flip(ac["lh"], src)
    ev = mne.EvokedArray(np.tile((fwd_fixed["sol"]["data"] @ x)[:, None], (1, 3)), inf, tmin=0)
    ev.set_eeg_reference("average", projection=True)
    inv = mne.minimum_norm.make_inverse_operator(inf, fwd, mne.read_cov(SRC_DIR / f"{subs[0]}-cov.fif"), loose=LOOSE, depth=DEPTH)
    e = mne.minimum_norm.apply_inverse(ev, inv, LAMBDA2, method="eLORETA", pick_ori="normal", method_params=dict(max_iter=60))
    amp = np.abs(e.data[:, 0]); rows = []
    for lb in labs:
        if "unknown" in lb.name: continue
        h = 0 if lb.hemi == "lh" else 1
        ii = np.where(np.isin(vert[h], lb.vertices))[0] + (0 if h == 0 else len(vert[0]))
        if len(ii): rows.append(dict(label=lb.name, mean_amp=amp[ii].mean()))
    v7 = pd.DataFrame(rows).sort_values("mean_amp", ascending=False)
    v7["rel"] = v7.mean_amp / v7.mean_amp.max()
    peak = int(np.argmax(amp)); hp = 0 if peak < len(vert[0]) else 1
    vpk = vert[hp][peak - (0 if hp == 0 else len(vert[0]))]
    peak_lab = next((lb.name for lb in labs if lb.hemi == ["lh", "rh"][hp] and vpk in lb.vertices), "?")
    in_ac = amp[idx].sum() / amp.sum()
    print(f"V7: peak of the estimate in {peak_lab}; {100*in_ac:.0f}% of total |current| inside the simulated AC patch")
    print("    mean |current| by region (top 8, relative to the strongest region):")
    print(v7.head(8)[["label", "rel"]].round(2).to_string(index=False))
""")

# =====================================================================================================
md(r"""
## 5. Every Results number vs this notebook

`draft` = the value in the Results draft (rounded as reported); `notebook` = recomputed above.
`match` = equal at the reported precision.
""")

code(r"""
DRAFT = {   # (section, quantity) -> (value in the Results draft, decimals as reported)
    (1, "MMN mean (µV)"): (-1.22, 2), (1, "MMN t"): (-9.4, 1), (1, "MMN 95% CI half-width"): (0.28, 2), (1, "MMN peak (ms)"): (106, 0),
    (1, "P3b mean (µV)"): (2.35, 2), (1, "P3b t"): (5.4, 1), (1, "P3b 95% CI half-width"): (0.94, 2), (1, "P3b peak (ms)"): (432, 0),
    (1, "ANOVA local-window local F"): (79.2, 1), (1, "ANOVA local-window global F"): (31.5, 1), (1, "ANOVA local-window local * global F"): (12.9, 1),
    (1, "ANOVA global-window local F"): (0.8, 1), (1, "ANOVA global-window global F"): (28.8, 1), (1, "ANOVA global-window local * global F"): (3.6, 1),
    (1, "local simple effect LD: GD vs GS t"): (-5.0, 1), (1, "local simple effect LS: GD vs GS t"): (-2.5, 1),
    (1, "global simple effect LS: GD vs GS t"): (4.9, 1), (1, "global simple effect LD: GD vs GS t"): (5.7, 1),
    (1, "MMN Fz linked mastoids (µV)"): (-3.17, 2), (1, "Mastoid inversion (µV)"): (1.77, 2),
    (2, "PXP M2 local (early)"): (0.94, 2), (2, "BOR early"): (0.022, 3), (2, "PXP M3 global (late)"): (0.95, 2), (2, "BOR late"): (0.035, 3),
    (2, "PXP M2 local M0-M3 only (early)"): (0.90, 2), (2, "PXP M3 global M0-M3 only (late)"): (0.97, 2),
    (2, "early M2-M3 mean dLL"): (18.8, 1), (2, "early M2-M1 mean dLL"): (4.4, 1), (2, "early M2-M0 mean dLL"): (2.2, 1),
    (2, "late M3-M2 mean dLL"): (30.2, 1), (2, "late M3-M1 mean dLL"): (34.4, 1), (2, "late M3-M0 mean dLL"): (1.5, 1),
    (2, "recovery M2 early (%)"): (100, 0), (2, "recovery M2 late (%)"): (90, 0), (2, "recovery M3 early (%)"): (100, 0), (2, "recovery M3 late (%)"): (100, 0),
    (2, "cond-mean r local window local learner"): (0.99, 2), (2, "cond-mean r local window global learner"): (0.48, 2),
    (2, "cond-mean r global window global learner"): (0.99, 2), (2, "cond-mean r global window local learner"): (0.28, 2),
    (2, "binned slope t local raw"): (-5.4, 1), (2, "binned slope t global raw"): (5.2, 1),
    (2, "binned slope t local within"): (-1.8, 1), (2, "binned slope t global within"): (5.0, 1),
    (2, "single-trial slope t local own learner, within condition"): (-2.7, 1), (2, "single-trial slope t global own learner, within condition"): (6.1, 1),
    (2, "single-trial slope t local other learner beyond own"): (-2.6, 1), (2, "single-trial slope t global other learner beyond own"): (0.2, 1),
    (2, "clocks Wilcoxon p"): (0.055, 3), (2, "reset test1 BF01"): (3.8, 1), (2, "reset group-best rho"): (0.0, 1),
    (3, "eLORETA cluster p"): (0.0005, 4), (3, "eLORETA n sources"): (1153, 0), (3, "eLORETA peak |t|"): (10.2, 1),
    (3, "transversetemporal-lh fraction significant"): (0.73, 2), (3, "superiortemporal-lh fraction significant"): (0.80, 2),
    (3, "superiortemporal-rh fraction significant"): (0.13, 2),
    (3, "auditory cortex lh t"): (-7.5, 1), (3, "auditory cortex rh t"): (-2.7, 1),
    (3, "frontal latency jackknife t"): (-1.0, 1), (3, "leakage ratio (%)"): (33, 0), (3, "global n sources"): (278, 0),
}
rep = pd.DataFrame(REPORT).drop_duplicates(["section", "quantity"], keep="last")
rows = []
for (sec, q), (v, dec) in DRAFT.items():
    hit = rep[(rep.section == sec) & (rep.quantity == q)]
    nb = float(hit.notebook.iloc[0]) if len(hit) else np.nan
    rows.append(dict(section=sec, quantity=q, draft=v, notebook=round(nb, dec + 1) if np.isfinite(nb) else nb,
                     match=bool(np.isfinite(nb) and abs(round(nb, dec) - v) <= 10 ** -dec * 1.01)))
summary = pd.DataFrame(rows)
pd.set_option("display.width", 200, "display.max_rows", 200)
print(summary.to_string(index=False))
print(f"\n{summary.match.sum()}/{len(summary)} Results numbers reproduced at the reported precision")
""")

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).parent / "results_analyses.ipynb"
nbf.write(nb, out); print(out)
