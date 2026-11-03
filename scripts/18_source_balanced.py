"""
18_source_balanced.py — trial-count-matched eLORETA region time courses (bias-free source strength).

Why: source *strength* (vector magnitude) is never negative, so noise adds to it. The condition with
fewer trials (e.g. global deviants, ~20%) has a noisier average and therefore larger strength, even
with no real effect — the earlier region time courses (11_source_eloreta.py) were positive
everywhere, including a "global effect" at 80–140 ms where the scalp data show none. Baseline
correction does not remove this, because the noise-floor bias differs between conditions.

Fix: for each contrast, the larger condition is randomly subsampled to the trial count of the smaller
one (N_DRAWS draws). Each draw's average is projected with eLORETA, converted to strength, baseline-
corrected and averaged within regions; strengths are then averaged over draws. Both sides of every
contrast therefore have the same number of trials and the same noise floor. Everything else (template
head, coregistration, noise covariance, lambda², loose/depth, regions) is identical to 11.

MODES
    --extract   (needs the epochs; run where they are) per subject: full average of the smaller
                condition + N_DRAWS subsampled averages of the larger one, 30 Hz low-pass, decimated
                to 250 Hz -> derivatives/source/balanced/<sub>.npz
    (default)   forward + inverse per subject, region strength time courses, statistics, figure
                (needs derivatives/source/subjects/sample, <sub>-cov.fif, header1_erps/info.fif)

Outputs: derivatives/source/balanced_region_timecourses.tsv.gz, balanced_region_stats.tsv,
         figures/source_eloreta_regions_balanced.png

Usage:  python scripts/18_source_balanced.py --extract [sub-01 ...]
        python scripts/18_source_balanced.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- CONFIG
ROOT = Path(__file__).resolve().parents[1]
PRE_DIR = ROOT / "derivatives" / "preproc"
CACHE_DIR = ROOT / "derivatives" / "header1_erps"
SRC_DIR = ROOT / "derivatives" / "source"
BAL_DIR = SRC_DIR / "balanced"
SUBJECTS_DIR = SRC_DIR / "subjects"
FIG_DIR = ROOT / "figures"
STEM = "ses-LGeffect_task-LocalGlobal"
TEMPLATE = "sample"
LOWPASS, DECIM = 30.0, 2                 # 500 Hz -> 250 Hz after the 30 Hz low-pass
N_DRAWS, SEED = 20, 42
LAMBDA2 = 1.0 / 9.0
LOOSE, DEPTH = 0.2, 0.8
BASELINE = (-0.1, 0.0)
EFFECTS = {"local": ("local", "LD", "LS", (0.08, 0.14)), "global": ("global", "GD", "GS", (0.30, 0.70))}
REGIONS = {   # aparc labels pooled into regions, per hemisphere (as in 11)
    "Auditory cortex (HG + STG)": ["transversetemporal", "superiortemporal"],
    "Inferior frontal gyrus": ["parsopercularis", "parstriangularis"],
    "Middle frontal gyrus": ["caudalmiddlefrontal", "rostralmiddlefrontal"],
    "Anterior cingulate": ["caudalanteriorcingulate", "rostralanteriorcingulate"],
    "Supramarginal / inf. parietal": ["supramarginal", "inferiorparietal"],
    "Superior parietal / precuneus": ["superiorparietal", "precuneus"],
}
# ------------------------------------------------------------------------------------


def extract(subjects=None):
    import mne
    mne.set_log_level("ERROR")
    BAL_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for sub in subjects or sorted(p.name for p in PRE_DIR.glob("sub-*") if p.is_dir()):
        out = BAL_DIR / f"{sub}.npz"
        if out.exists():
            continue
        ep = mne.read_epochs(PRE_DIR / sub / f"{sub}_{STEM}_epo.fif", preload=True)
        ep.pick("eeg")
        meta = ep.metadata.reset_index(drop=True)
        X = ep.get_data()                                         # trials x ch x time (V)
        sfreq = ep.info["sfreq"]
        save = {}
        for eff, (col, dev, std, _) in EFFECTS.items():
            idx = {c: np.where(meta[col].eq(c).to_numpy())[0] for c in (dev, std)}
            nmin = min(len(v) for v in idx.values())
            for c, ii in idx.items():
                if len(ii) == nmin:                               # smaller condition: full average once
                    avgs = X[ii].mean(0)[None]
                else:                                             # larger condition: N_DRAWS subsamples
                    avgs = np.stack([X[rng.choice(ii, nmin, replace=False)].mean(0) for _ in range(N_DRAWS)])
                avgs = mne.filter.filter_data(avgs, sfreq, None, LOWPASS, verbose=False)[..., ::DECIM]
                save[f"{eff}__{c}"] = avgs.astype(np.float32)
            save[f"{eff}__nave"] = np.array(nmin)
        np.savez_compressed(out, times=ep.times[::DECIM], ch_names=np.array(ep.ch_names), **save)
        print(f"{sub}: " + ", ".join(f"{e} n={int(save[f'{e}__nave'])}" for e in EFFECTS), flush=True)
        del X, ep


def make_forward(info):
    """Same coregistration and BEM forward model as 11_source_eloreta.py."""
    import mne
    coreg = mne.coreg.Coregistration(info, TEMPLATE, subjects_dir=SUBJECTS_DIR, fiducials="auto")
    coreg.fit_fiducials(verbose=False)
    coreg.fit_icp(n_iterations=40, nasion_weight=2.0, verbose=False)
    coreg.omit_head_shape_points(distance=10 / 1000)
    coreg.fit_icp(n_iterations=40, nasion_weight=2.0, verbose=False)
    tb = SUBJECTS_DIR / TEMPLATE / "bem"
    src = mne.read_source_spaces(tb / f"{TEMPLATE}-oct-6-src.fif")
    bem = mne.read_bem_solution(tb / f"{TEMPLATE}-1280-1280-1280-bem-sol.fif")
    return mne.make_forward_solution(info, trans=coreg.trans, src=src, bem=bem, eeg=True, meg=False,
                                     mindist=5.0, n_jobs=1)


def main():
    import mne
    from scipy import stats
    mne.set_log_level("ERROR")
    subs = sorted(f.stem for f in BAL_DIR.glob("sub-*.npz"))
    info = mne.io.read_info(CACHE_DIR / "info.fif")
    info = mne.pick_info(info, mne.pick_types(info, eeg=True))
    with info._unlock():
        info["projs"] = []
    raw0 = mne.io.RawArray(np.zeros((len(info["ch_names"]), 1)), info)
    raw0.set_eeg_reference("average", projection=True)
    info = raw0.info
    fwd = make_forward(info)
    src = fwd["src"]
    labels = {lb.name: lb for lb in mne.read_labels_from_annot(TEMPLATE, "aparc", subjects_dir=SUBJECTS_DIR)}
    regs = {(reg, h): sum((labels[f"{n}-{h}"] for n in names[1:]), labels[f"{names[0]}-{h}"])
            for reg, names in REGIONS.items() for h in ("lh", "rh")}

    rows = []
    for sub in subs:
        z = np.load(BAL_DIR / f"{sub}.npz")
        assert list(z["ch_names"]) == info["ch_names"]
        times = z["times"]
        info_d = info.copy()                                       # data were decimated: match sfreq
        with info_d._unlock():
            info_d["sfreq"] = 1.0 / float(np.median(np.diff(times)))
        cov = mne.read_cov(SRC_DIR / f"{sub}-cov.fif")
        inv = mne.minimum_norm.make_inverse_operator(info, fwd, cov, loose=LOOSE, depth=DEPTH)
        for eff, (col, dev, std, _) in EFFECTS.items():
            nave = int(z[f"{eff}__nave"])
            prep = mne.minimum_norm.prepare_inverse_operator(inv, nave, LAMBDA2, method="eLORETA",
                                                            method_params=dict(max_iter=60))
            strength = {}
            for c in (dev, std):
                acc = []
                for a in z[f"{eff}__{c}"]:
                    ev = mne.EvokedArray(a.astype(float), info_d, tmin=times[0], nave=nave)
                    assert np.allclose(ev.times, times)
                    stc = mne.minimum_norm.apply_inverse(ev, prep, LAMBDA2, method="eLORETA", pick_ori=None,
                                                         prepared=True)
                    stc = stc.magnitude() if hasattr(stc, "magnitude") else stc
                    stc.apply_baseline(BASELINE)
                    acc.append(np.array([stc.extract_label_time_course(lb, src, mode="mean")[0]
                                         for lb in regs.values()]))
                strength[c] = np.mean(acc, axis=0)                 # regions x time, averaged over draws
            diff = strength[dev] - strength[std]
            for k, (reg, h) in enumerate(regs):
                for t, v, vd, vs in zip(times, diff[k], strength[dev][k], strength[std][k]):
                    rows.append(dict(effect=eff, subject=sub, region=reg, hemi=h, time=t, value=v, dev=vd, std=vs))
        print(f"{sub}: done", flush=True)
    tc = pd.DataFrame(rows)
    tc.to_csv(SRC_DIR / "balanced_region_timecourses.tsv.gz", sep="\t", index=False)

    # window statistics: each effect in both windows (the other window is the bias check)
    st = []
    for eff in EFFECTS:
        for wname, (*_, (t0, t1)) in EFFECTS.items():
            d = tc[(tc["effect"] == eff) & (tc["time"] >= t0) & (tc["time"] <= t1)]
            g = d.groupby(["region", "hemi", "subject"])["value"].mean().reset_index()
            for (reg, h), q in g.groupby(["region", "hemi"]):
                r = stats.ttest_1samp(q["value"], 0)
                st.append(dict(effect=eff, window=wname, region=reg, hemi=h, mean=q["value"].mean(),
                               t=r.statistic, p=r.pvalue))
    st = pd.DataFrame(st)
    from statsmodels.stats.multitest import multipletests
    st["p_fdr"] = np.nan
    for (eff, w), q in st.groupby(["effect", "window"]):
        st.loc[q.index, "p_fdr"] = multipletests(q["p"], method="fdr_bh")[1]
    st.to_csv(SRC_DIR / "balanced_region_stats.tsv", sep="\t", index=False)
    print(st.round(3).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--extract":
        extract(sys.argv[2:] or None)
    else:
        main()
