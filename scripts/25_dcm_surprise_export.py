"""
25_dcm_surprise_export.py — condition-mean model surprise per participant, for surprise-coded DCM.

For each participant: local (M2) and global (M3) learner surprise per trial, with that participant's
own selected parameters (06_model_comparison.py: M2 chosen on the early window, M3 on the late window),
averaged over the trials entering each 2x2 condition. For DCM the values are expressed relative to
LSGS (baseline trial type = 0) and scaled so the largest condition is 1 within each participant, i.e.
the same range as the 0/1 condition labels, so priors on the modulatory parameters are comparable.

Output: derivatives/dcm/surprise_by_condition.csv
        columns: subject, condition, S_loc, S_glob (bits), x_loc, x_glob (DCM regressors)
Used by scripts/dcm/dcm_05_surprise.m

Usage:  python scripts/25_dcm_surprise_export.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

# ----------------------------------------------------------------------------- CONFIG
ROOT = Path(__file__).resolve().parents[1]
BIDS_ROOT = ROOT / "data" / "bids"
MC_DIR = ROOT / "derivatives" / "model_comparison"
OUT = ROOT / "derivatives" / "dcm" / "surprise_by_condition.csv"
SOUNDS = {1: "A", 2: "B"}
CONDS = ["LSGS", "LDGS", "LSGD", "LDGD"]                  # DCM trial order
# ------------------------------------------------------------------------------------


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
    src = BIDS_ROOT / "sourcedata" / subject / "ses-LGeffect"
    seq = loadmat(next(src.glob("GlobLoc_stimulation_SEQUENCE*.mat")), squeeze_me=True)["SEQUENCE"]
    rows = []
    for b, blk in enumerate(seq, start=1):
        for i, q in enumerate(np.atleast_2d(blk).astype(int)):
            rows.append(dict(block=b, trial_in_block=i, quintlet="".join(SOUNDS[x] for x in q)))
    qs = pd.DataFrame(rows)
    qs.insert(0, "seq_idx", np.arange(len(qs)))
    c = clocks[clocks["subject"] == subject]
    p_loc = c[(c["component"] == "local_80-140ms") & (c["model"] == "M2 local")].iloc[0]
    p_glo = c[(c["component"] == "global_300-700ms") & (c["model"] == "M3 global")].iloc[0]
    first, fifth = qs["quintlet"].str[0].to_numpy(), qs["quintlet"].str[4].to_numpy()
    start = (qs["trial_in_block"] == 0).to_numpy()
    lo = np.zeros((len(qs), 4), int)
    lo[:, 3] = fifth != first
    lo = lo.ravel()
    lr = np.zeros(len(lo), bool)
    lr[::4] = start
    qs["S_loc"] = leaky_surprise(lo, float(p_loc["param"]), lr if bool(p_loc["reset"]) else None)[3::4]
    qs["S_glob"] = leaky_surprise((fifth != first).astype(int), float(p_glo["param"]),
                                  start if bool(p_glo["reset"]) else None)
    return qs[["seq_idx", "S_loc", "S_glob"]]


def main():
    clocks = pd.read_csv(MC_DIR / "clocks.tsv", sep="\t")
    out = []
    for f in sorted((MC_DIR / "trials").glob("sub-*.tsv")):
        sub = f.stem
        tr = pd.read_csv(f, sep="\t").merge(learner_surprise(sub, clocks), on="seq_idx", validate="one_to_one")
        m = tr.groupby("condition")[["S_loc", "S_glob"]].mean().reindex(CONDS)
        for col, x in (("S_loc", "x_loc"), ("S_glob", "x_glob")):
            rel = m[col] - m.loc["LSGS", col]
            m[x] = rel / np.abs(rel).max()
        m["subject"] = sub
        out.append(m.reset_index())
    df = pd.concat(out)[["subject", "condition", "S_loc", "S_glob", "x_loc", "x_glob"]]
    df.to_csv(OUT, index=False)
    print(df.groupby("condition")[["S_loc", "S_glob", "x_loc", "x_glob"]].agg(["mean", "std"]).round(2).reindex(CONDS))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
