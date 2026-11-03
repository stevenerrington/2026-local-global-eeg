"""
27_results_checks.py — statistics needed for the Results that were not yet formally tested.

1  2 x 2 repeated-measures ANOVA (local x global) on window amplitude at each a-priori ROI/window
   (frontocentral 80–140 ms; parietal 300–700 ms), with simple effects (paired t-tests): does the MMN
   depend on global status (LD/GD vs LD/GS), and does the P3b depend on local status?
2  Extent of the spatio-temporal clusters shown in Figure 1C (same test, same data): onset/offset,
   number of channels and peak |t| of every significant cluster, for both contrasts — e.g. when the
   global effect first reaches significance anywhere on the scalp.

Inputs : derivatives/header1_erps/sub-*_cond4.tsv (07_header1_erps.py), sub-*.npz + info.fif
Outputs: derivatives/header1_erps/results_checks.json (printed too)

Usage:  python scripts/27_results_checks.py
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pingouin as pg
import mne
from scipy import stats

# ----------------------------------------------------------------------------- CONFIG
ROOT = Path(__file__).resolve().parents[1]
ERP_DIR = ROOT / "derivatives" / "header1_erps"
COMPONENTS = {"local": "frontocentral 80–140 ms", "global": "parietal 300–700 ms"}
CONTRASTS = {"local": ("LD", "LS"), "global": ("GD", "GS")}
TMAX, N_PERM, SEED = 0.7, 2000, 42             # identical to 14_figure1.py
# ------------------------------------------------------------------------------------

mne.set_log_level("ERROR")


def anova_2x2():
    c4 = pd.concat([pd.read_csv(f, sep="\t") for f in sorted(ERP_DIR.glob("sub-*_cond4.tsv"))])
    d = c4[c4["source"] == "data"].copy()
    d["local"] = d["condition"].str[:2]
    d["global"] = d["condition"].str[2:]
    out = {}
    for comp, lab in COMPONENTS.items():
        q = d[d["component"] == comp]
        aov = pg.rm_anova(data=q, dv="amp", within=["local", "global"], subject="subject", detailed=True)
        piv = q.pivot(index="subject", columns="condition", values="amp")
        simple = {}
        for name, a, b in [("LD: GD vs GS", "LDGD", "LDGS"), ("LS: GD vs GS", "LSGD", "LSGS"),
                           ("GS: LD vs LS", "LDGS", "LSGS"), ("GD: LD vs LS", "LDGD", "LSGD")]:
            t = stats.ttest_rel(piv[a], piv[b])
            simple[name] = dict(diff=float((piv[a] - piv[b]).mean()), t=float(t.statistic), p=float(t.pvalue),
                                n_same_sign=int((np.sign(piv[a] - piv[b]) == np.sign((piv[a] - piv[b]).mean())).sum()))
        out[comp] = dict(window=lab, cell_means={c: float(piv[c].mean()) for c in ["LSGS", "LDGS", "LSGD", "LDGD"]},
                         cell_sd={c: float(piv[c].std(ddof=1)) for c in ["LSGS", "LDGS", "LSGD", "LDGD"]},
                         anova=[dict(effect=r.Source, F=float(r.F), df1=int(r["ddof1"]), df2=int(r["ddof2"]),
                                     p=float(r["p_unc"] if "p_unc" in r else r["p-unc"]), eta_g2=float(r.ng2))
                                for _, r in aov.iterrows() if r.Source != "Error" and pd.notna(r.F)],
                         simple_effects=simple)
    return out


def cluster_extents():
    files = sorted(ERP_DIR.glob("sub-*.npz"))
    info = mne.io.read_info(ERP_DIR / "info.fif")
    z0 = np.load(files[0])
    times, chs = z0["times"], list(z0["ch_names"])
    sel = (times >= 0) & (times <= TMAX)
    ts = times[sel]
    adjacency, _ = mne.channels.find_ch_adjacency(info, "eeg")
    n = len(files)
    out = {}
    for name, (dev, std) in CONTRASTS.items():
        X = np.stack([np.load(f)[f"topo__{dev}"] - np.load(f)[f"topo__{std}"] for f in files])[:, :, sel]
        T, clusters, pv, _ = mne.stats.spatio_temporal_cluster_1samp_test(
            X.transpose(0, 2, 1), threshold=stats.t.ppf(0.975, n - 1), n_permutations=N_PERM,
            adjacency=adjacency, tail=0, seed=SEED, out_type="mask", verbose=False)
        rows = []
        for c, p in zip(clusters, pv):
            if p >= 0.05:
                continue
            tt = np.where(c.any(1))[0]
            ch = np.where(c.any(0))[0]
            rows.append(dict(p=float(p), sign="+" if T[c].mean() > 0 else "−", onset_ms=float(ts[tt[0]] * 1e3),
                             offset_ms=float(ts[tt[-1]] * 1e3), n_channels=int(len(ch)),
                             peak_abs_t=float(np.abs(T[c]).max()),
                             channels_at_onset=[chs[i] for i in np.where(c[tt[0]])[0]]))
        out[name] = sorted(rows, key=lambda r: r["onset_ms"])
    return out


def main():
    res = dict(anova_2x2=anova_2x2(), cluster_extents=cluster_extents())
    (ERP_DIR / "results_checks.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
