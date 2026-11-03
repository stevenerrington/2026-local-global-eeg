"""
03_ideal_observer.py — trial-by-trial perceptual-inference variables for the local–global task.

Two leaky Bayesian (Beta–Bernoulli) observers are run over the *full presented* sound sequence
(from sourcedata, so unrecorded/rejected quintlets still shape the observer's history):

  LOCAL observer   position-agnostic transition statistics: "does the next sound repeat the
                   previous one?", updated at every within-quintlet transition (4 per quintlet).
                   -> local surprise / prediction change / precision at the 5th sound.
  GLOBAL observer  position-specific statistics: "does the 5th sound differ from the first four?",
                   updated once per quintlet. This is the level at which the block rule lives.
                   -> global surprise / prediction change / precision at the 5th sound.

Each observer keeps leaky counts (memory half-life h, in observations) with a Beta(1, 1) prior:
    n <- lambda * n + outcome,  lambda = 2 ** (-1 / h)
Before each observation the predicted probability of a change is mu = (n1 + 1) / (n0 + n1 + 2).

Variables per quintlet (all for the 5th sound, i.e. the epoch time-locking event):
    p_*        predicted probability of the observed outcome
    S_*        Shannon surprise, -log2 p(outcome)
    PE_*       signed prediction error, outcome - mu
    dmu_*      prediction change |mu_after - mu_before|
    BS_*       Bayesian surprise, KL(posterior || prior) of the Beta (bits)
    prec_*     precision of the prior belief, 1 / Var[Beta] (log-scaled: logprec_*)

Computed for a grid of half-lives (columns suffixed _h{h}) and for the default half-lives
(unsuffixed columns). Output: derivatives/observer/<sub>_observer.tsv, one row per presented
quintlet, with seq_idx to join onto epochs.metadata.

Usage:  python scripts/03_ideal_observer.py [sub-01 ...]
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.special import betaln, digamma

# ----------------------------------------------------------------------------- CONFIG
ROOT = Path(__file__).resolve().parents[1]
BIDS_ROOT = ROOT / "data" / "bids"
OUT_DIR = ROOT / "derivatives" / "observer"
SESSION = "LGeffect"
SOUNDS = {1: "A", 2: "B"}

H_LOCAL_DEFAULT = 50        # transitions (~12 quintlets)
H_GLOBAL_DEFAULT = 10       # quintlets
H_LOCAL_GRID = [5, 10, 20, 50, 100, 200, 500, np.inf]
H_GLOBAL_GRID = [2, 3, 5, 10, 20, 50, 100, np.inf]
RESET_AT_BLOCKS = True      # observers forget everything at each block start. Default because the EEG
                            # shows no surprise to a new standard that was the previous block's
                            # deviant, and a large P3b to the first deviant of each block.
                            # The opposite variant is also written (default half-lives only, suffix
                            # _noreset / _reset) for model comparison.
# ------------------------------------------------------------------------------------


def load_sequence(subject):
    src = BIDS_ROOT / "sourcedata" / subject / f"ses-{SESSION}"
    seq = loadmat(next(src.glob("GlobLoc_stimulation_SEQUENCE*.mat")), squeeze_me=True)["SEQUENCE"]
    rows = []
    for b, blk in enumerate(seq, start=1):
        blk = np.atleast_2d(blk).astype(int)
        std = "".join(SOUNDS[x] for x in blk[0])
        for i, q in enumerate(blk):
            rows.append(dict(block=b, trial_in_block=i,
                             quintlet="".join(SOUNDS[x] for x in q), block_standard=std))
    qs = pd.DataFrame(rows)
    qs.insert(0, "seq_idx", np.arange(len(qs)))
    qs["local"] = np.where(qs["quintlet"].str[3] == qs["quintlet"].str[4], "LS", "LD")
    qs["global"] = np.where(qs["quintlet"] == qs["block_standard"], "GS", "GD")
    return qs


def kl_beta(a1, b1, a0, b0):
    """KL( Beta(a1,b1) || Beta(a0,b0) ) in bits."""
    kl = (betaln(a0, b0) - betaln(a1, b1) + (a1 - a0) * digamma(a1) + (b1 - b0) * digamma(b1)
          + (a0 - a1 + b0 - b1) * digamma(a1 + b1))
    return kl / np.log(2)


def leaky_beta(outcomes, half_life, resets=None):
    """Run a leaky Beta–Bernoulli observer. outcomes: 1 = change, 0 = repeat.

    Returns dict of arrays (one value per observation, describing the prediction *before* it).
    """
    lam = 0.0 if half_life == 0 else (1.0 if np.isinf(half_life) else 2.0 ** (-1.0 / half_life))
    n1 = n0 = 0.0
    out = {k: np.empty(len(outcomes)) for k in ["mu", "p", "S", "PE", "dmu", "BS", "logprec"]}
    for t, y in enumerate(outcomes):
        if resets is not None and resets[t]:
            n1 = n0 = 0.0
        a, b = n1 + 1, n0 + 1
        mu = a / (a + b)
        var = a * b / ((a + b) ** 2 * (a + b + 1))
        p = mu if y else 1 - mu
        n1, n0 = lam * n1 + y, lam * n0 + (1 - y)
        a2, b2 = n1 + 1, n0 + 1
        mu2 = a2 / (a2 + b2)
        out["mu"][t], out["p"][t] = mu, p
        out["S"][t] = -np.log2(p)
        out["PE"][t] = y - mu
        out["dmu"][t] = abs(mu2 - mu)
        out["BS"][t] = kl_beta(a2, b2, a, b)
        out["logprec"][t] = np.log(1 / var)
    return out


def observer_table(subject, reset=RESET_AT_BLOCKS, grid=True, suffix=""):
    qs = load_sequence(subject)
    first4 = qs["quintlet"].str[0].to_numpy()
    fifth = qs["quintlet"].str[4].to_numpy()
    block_start = (qs["trial_in_block"] == 0).to_numpy()

    # local: 4 transitions per quintlet (1->2, 2->3, 3->4 are always repeats; 4->5 varies)
    trans = np.zeros((len(qs), 4), int)
    trans[:, 3] = (fifth != first4).astype(int)
    loc_out = trans.ravel()
    loc_reset = np.zeros_like(loc_out, bool)
    loc_reset[::4] = block_start if reset else False
    idx5 = np.arange(3, len(loc_out), 4)                   # observation index of the 4->5 transition

    # global: one observation per quintlet — does the 5th sound differ from the first four?
    glob_out = (fifth != first4).astype(int)
    glob_reset = block_start if reset else None

    cols = {}

    def add(prefix, res, sel, suffix):
        for k, v in res.items():
            cols[f"{k}_{prefix}{suffix}"] = v[sel]

    lr = loc_reset if reset else None
    if grid:
        for h in H_LOCAL_GRID:
            add("loc", leaky_beta(loc_out, h, lr), idx5, f"_h{h:g}")
        for h in H_GLOBAL_GRID:
            add("glob", leaky_beta(glob_out, h, glob_reset), slice(None), f"_h{h:g}")
    add("loc", leaky_beta(loc_out, H_LOCAL_DEFAULT, lr), idx5, suffix)
    add("glob", leaky_beta(glob_out, H_GLOBAL_DEFAULT, glob_reset), slice(None), suffix)
    out = pd.concat([qs, pd.DataFrame(cols)], axis=1)
    if grid:   # also the opposite reset variant at default half-lives, for model comparison
        alt = observer_table(subject, reset=not reset, grid=False,
                             suffix="_noreset" if reset else "_reset")
        out = pd.concat([out, alt[[c for c in alt.columns if c not in out.columns]]], axis=1)
    return out


def main(subjects):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subjects = subjects or sorted(p.name for p in BIDS_ROOT.glob("sub-*") if p.is_dir())
    for sub in subjects:
        qs = observer_table(sub)
        qs.to_csv(OUT_DIR / f"{sub}_observer.tsv", sep="\t", index=False)
        g = qs.groupby(["local", "global"])[["S_loc", "S_glob"]].mean().round(2)
        print(f"{sub}: {len(qs)} quintlets; mean surprise (bits) by condition:\n"
              + g.to_string().replace("\n", "\n    "))
    print(f"Saved -> {OUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
