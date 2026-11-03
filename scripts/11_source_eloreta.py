"""
11_source_eloreta.py — cortical source estimates of the local and global effects (eLORETA).

No individual MRIs exist, so a template head is used:
  * template: MNE's "sample" subject (FreeSurfer anatomy with a realistic 3-layer BEM,
    1280 triangles per layer; oct-6 cortical source space, 2 x 4,098 sources). The usual fsaverage
    BEM is hosted on OSF, which is not reachable from here; MNE's GitHub test data provide the
    complete "sample" head model. (A spherical model on fsaverage was tried first and rejected:
    a sphere under all 62 electrodes excludes occipital and inferior temporal cortex.)
  * coregistration: standard 10-05 electrode positions -> sample head: fiducial fit, then ICP to
    the scalp (nasion weighted); electrodes are projected onto the scalp in the forward model
  * orientation: loose = 0.2, normal component (signed currents -> differences testable across
    subjects; sign depends on local cortical orientation, so read location, not colour)
  * noise covariance: per subject, silent pre-quintlet period −800 to −650 ms (re-baselined to
    that window), Ledoit–Wolf shrinkage
  * inverse: eLORETA (lambda² = 1/9), applied to each subject's LD − LS and GD − GS difference
    waves (linear, so identical to differencing the source estimates); sLORETA as a check of the
    peak regions
  * spatial precision: template anatomy + standard electrode positions — interpret at the level of
    regions (aparc labels), not peaks

Group statistics: one-sample cluster permutation over subjects on the signed source difference,
averaged within the a-priori windows (80–140 ms local; 300–700 ms global), spatial adjacency of
the source space, cluster-forming p < .01, 2000 permutations.

Region time courses: source strength (vector magnitude, loose orientation) for deviant and standard
separately, each baseline-corrected (−100–0 ms, removes most of the noise-floor bias from unequal
trial counts), contrast = deviant − standard, averaged within aparc regions. Positive = more
activity for deviants.

MODES
    --fetch   download the template files from MNE's GitHub test data
              -> derivatives/source/subjects/sample
    --cov     (needs the epochs) per-subject noise covariance -> derivatives/source/<sub>-cov.fif
    (default) coregistration, forward, inverses, group statistics, figures
              (needs derivatives/header1_erps/<sub>.npz from 07 and the covariances)

Usage:  python scripts/11_source_eloreta.py --fetch
        python scripts/11_source_eloreta.py --cov
        python scripts/11_source_eloreta.py
"""
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- CONFIG
ROOT = Path(__file__).resolve().parents[1]
PRE_DIR = ROOT / "derivatives" / "preproc"
CACHE_DIR = ROOT / "derivatives" / "header1_erps"
SRC_DIR = ROOT / "derivatives" / "source"
SUBJECTS_DIR = SRC_DIR / "subjects"
FIG_DIR = ROOT / "figures"
STEM = "ses-LGeffect_task-LocalGlobal"
TEMPLATE = "sample"
TPL_URL = "https://raw.githubusercontent.com/mne-tools/mne-testing-data/master/subjects/sample"
TPL_FILES = ["bem/sample-1280-1280-1280-bem-sol.fif", "bem/sample-oct-6-src.fif", "bem/sample-fiducials.fif",
             "bem/sample-head.fif", "bem/sample-head-dense.fif", "label/lh.aparc.annot", "label/rh.aparc.annot",
             "surf/lh.inflated", "surf/rh.inflated", "surf/lh.white", "surf/rh.white", "surf/lh.curv",
             "surf/rh.curv", "surf/lh.sphere", "surf/rh.sphere", "surf/lh.sphere.reg", "surf/rh.sphere.reg",
             "mri/T1.mgz"]

COV_WIN = (-0.80, -0.65)       # silent period before the 1st sound (≥ −0.63 s, or −0.49 s at 122 ms SOA)
LAMBDA2 = 1.0 / 9.0
LOOSE, DEPTH = 0.2, 0.8
EFFECTS = {"local": ("LD", "LS", (0.08, 0.14)), "global": ("GD", "GS", (0.30, 0.70))}
CLUSTER_P, N_PERM, SEED = 0.01, 2000, 42
REGIONS = {   # aparc labels pooled into regions, per hemisphere
    "Auditory cortex (HG + STG)": ["transversetemporal", "superiortemporal"],
    "Inferior frontal gyrus": ["parsopercularis", "parstriangularis"],
    "Middle frontal gyrus": ["caudalmiddlefrontal", "rostralmiddlefrontal"],
    "Anterior cingulate": ["caudalanteriorcingulate", "rostralanteriorcingulate"],
    "Supramarginal / inf. parietal": ["supramarginal", "inferiorparietal"],
    "Superior parietal / precuneus": ["superiorparietal", "precuneus"],
}
# ------------------------------------------------------------------------------------


def fetch():
    import urllib.request
    for f in TPL_FILES:
        out = SUBJECTS_DIR / TEMPLATE / f
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{TPL_URL}/{f}", out)
        print(f"fetched {f}")


def compute_covs():
    import mne
    mne.set_log_level("ERROR")
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    for sub in sorted(p.name for p in PRE_DIR.glob("sub-*") if p.is_dir()):
        out = SRC_DIR / f"{sub}-cov.fif"
        if out.exists():
            continue
        ep = mne.read_epochs(PRE_DIR / sub / f"{sub}_{STEM}_epo.fif", preload=True)
        ep.pick("eeg").set_eeg_reference("average", projection=True)
        ep.apply_baseline(COV_WIN)
        cov = mne.compute_covariance(ep, tmin=COV_WIN[0], tmax=COV_WIN[1], method="shrunk", rank="info")
        cov.save(out, overwrite=True)
        print(f"{sub}: noise covariance from {len(ep)} epochs")


def make_forward(info):
    """Coregister standard electrode positions to the template head; BEM forward model."""
    import mne
    coreg = mne.coreg.Coregistration(info, TEMPLATE, subjects_dir=SUBJECTS_DIR, fiducials="auto")
    coreg.fit_fiducials(verbose=False)
    coreg.fit_icp(n_iterations=40, nasion_weight=2.0, verbose=False)
    coreg.omit_head_shape_points(distance=10 / 1000)
    coreg.fit_icp(n_iterations=40, nasion_weight=2.0, verbose=False)
    dist = coreg.compute_dig_mri_distances() * 1e3
    mne.write_trans(SRC_DIR / f"{TEMPLATE}-coreg-trans.fif", coreg.trans, overwrite=True)
    tb = SUBJECTS_DIR / TEMPLATE / "bem"
    src = mne.read_source_spaces(tb / f"{TEMPLATE}-oct-6-src.fif")
    bem = mne.read_bem_solution(tb / f"{TEMPLATE}-1280-1280-1280-bem-sol.fif")
    fwd = mne.make_forward_solution(info, trans=coreg.trans, src=src, bem=bem, eeg=True, meg=False,
                                    mindist=5.0, n_jobs=1)
    return fwd, dist


def main():
    import mne
    import matplotlib
    matplotlib.use("Agg")
    from scipy import stats
    mne.set_log_level("ERROR")
    subs = sorted(f.stem for f in CACHE_DIR.glob("sub-*.npz"))
    info = mne.io.read_info(CACHE_DIR / "info.fif")
    info = mne.pick_info(info, mne.pick_types(info, eeg=True))
    with info._unlock():
        info["projs"] = []
    raw0 = mne.io.RawArray(np.zeros((len(info["ch_names"]), 1)), info)
    raw0.set_eeg_reference("average", projection=True)
    info = raw0.info
    fwd, dist = make_forward(info)
    n_src = sum(s["nuse"] for s in fwd["src"])
    print(f"coregistration: electrode-to-scalp distance median {np.median(dist):.1f} mm, max {dist.max():.1f} mm; "
          f"forward: {n_src} cortical sources")
    src = fwd["src"]
    labels = mne.read_labels_from_annot(TEMPLATE, "aparc", subjects_dir=SUBJECTS_DIR)
    adjacency = mne.spatial_src_adjacency(src)

    stcs = {m: {e: [] for e in EFFECTS} for m in ["eLORETA", "sLORETA"]}
    mag_contrast = {e: [] for e in EFFECTS}
    for sub in subs:
        z = np.load(CACHE_DIR / f"{sub}.npz")
        times = z["times"]
        cov = mne.read_cov(SRC_DIR / f"{sub}-cov.fif")
        inv = mne.minimum_norm.make_inverse_operator(info, fwd, cov, loose=LOOSE, depth=DEPTH)
        for eff, (dev, std, _) in EFFECTS.items():
            ev = mne.EvokedArray((z[f"topo__{dev}"] - z[f"topo__{std}"]) * 1e-6, info, tmin=times[0],
                                 comment=f"{dev}-{std}")
            for meth in stcs:
                kw = dict(method_params=dict(max_iter=60)) if meth == "eLORETA" else {}
                stc = mne.minimum_norm.apply_inverse(ev, inv, LAMBDA2, method=meth, pick_ori="normal", **kw)
                stcs[meth][eff].append(stc)
            # magnitude contrast for region time courses: |dev| − |std|, each baseline-corrected
            mags = {}
            for c in (dev, std):
                evc = mne.EvokedArray(z[f"topo__{c}"] * 1e-6, info, tmin=times[0])
                sm = mne.minimum_norm.apply_inverse(evc, inv, LAMBDA2, method="eLORETA", pick_ori=None,
                                                    method_params=dict(max_iter=60))
                mags[c] = sm.apply_baseline((-0.1, 0.0))
            mag_contrast[eff].append(mags[dev] - mags[std])
        print(f"{sub}: done", flush=True)

    # ---------------- group statistics in the a-priori windows (eLORETA)
    results, tmaps, sigmasks = {}, {}, {}
    for eff, (dev, std, (t0, t1)) in EFFECTS.items():
        X = np.stack([s.copy().crop(t0, t1).data.mean(1) for s in stcs["eLORETA"][eff]])   # subj x src
        thr = stats.t.ppf(1 - CLUSTER_P / 2, len(subs) - 1)
        T, clusters, pv, _ = mne.stats.permutation_cluster_1samp_test(
            X, threshold=thr, n_permutations=N_PERM, adjacency=adjacency, tail=0, seed=SEED,
            out_type="mask", n_jobs=1, verbose=False)
        mask = np.zeros_like(T, bool)
        for c, p in zip(clusters, pv):
            if p < 0.05:
                mask |= c
        tmaps[eff], sigmasks[eff] = T, mask
        # summarise significant sources by aparc label (|t|-weighted), plus peak label per method
        vert = [s["vertno"] for s in src]
        lab_rows = []
        for lab in labels:
            if "unknown" in lab.name:
                continue
            hemi = 0 if lab.hemi == "lh" else 1
            idx = np.where(np.isin(vert[hemi], lab.vertices))[0] + (0 if hemi == 0 else len(vert[0]))
            if len(idx) == 0:
                continue
            lab_rows.append(dict(label=lab.name, n_src=len(idx), n_sig=int(mask[idx].sum()),
                                 frac_sig=float(mask[idx].mean()), mean_abs_t=float(np.abs(T[idx]).mean())))
        lab_df = pd.DataFrame(lab_rows).sort_values("mean_abs_t", ascending=False)
        lab_df.to_csv(SRC_DIR / f"group_{eff}_labels.tsv", sep="\t", index=False)
        peaks = {}
        for meth in stcs:
            ga = np.mean([np.abs(s.copy().crop(t0, t1).data).mean(1) for s in stcs[meth][eff]], axis=0)
            i = int(np.argmax(ga))
            hemi = 0 if i < len(vert[0]) else 1
            v = vert[hemi][i - (0 if hemi == 0 else len(vert[0]))]
            lab = [lb.name for lb in labels if lb.hemi == ["lh", "rh"][hemi] and v in lb.vertices]
            peaks[meth] = lab[0] if lab else "unlabelled"
        results[eff] = dict(window_ms=[int(t0 * 1e3), int(t1 * 1e3)],
                            cluster_p=[float(p) for p in sorted(pv)[:4]],
                            n_sig_sources=int(mask.sum()),
                            top_labels_by_mean_abs_t=lab_df.head(8)[["label", "mean_abs_t", "frac_sig"]]
                            .round(3).to_dict("records"),
                            peak_label_of_mean_abs_current=peaks)
    (SRC_DIR / "group_summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    # ---------------- region time courses (eLORETA, mean_flip)
    tc_rows = []
    lab_by_name = {lb.name: lb for lb in labels}
    for eff in EFFECTS:
        for si, (sub, stc) in enumerate(zip(subs, mag_contrast[eff])):
            for reg, names in REGIONS.items():
                for hemi in ["lh", "rh"]:
                    lab = sum((lab_by_name[f"{n}-{hemi}"] for n in names[1:]), lab_by_name[f"{names[0]}-{hemi}"])
                    tc = stc.extract_label_time_course(lab, src, mode="mean")[0]
                    for t, v in zip(stc.times, tc):
                        tc_rows.append(dict(effect=eff, subject=sub, region=reg, hemi=hemi, time=t, value=v))
    tc = pd.DataFrame(tc_rows)
    tc.to_csv(SRC_DIR / "region_timecourses.tsv.gz", sep="\t", index=False)

    # ---------------- figures
    FIG_DIR.mkdir(exist_ok=True)
    plot_brains(stcs, tmaps, sigmasks, src, subs)
    plot_regions(tc)
    print(f"Figures -> {FIG_DIR}")


def plot_brains(stcs, tmaps, sigmasks, src, subs):
    """Group t-maps (eLORETA, window means) on the inflated surface; matplotlib backend."""
    import mne
    import matplotlib.pyplot as plt
    views = [("lh", "lat"), ("rh", "lat"), ("lh", "med"), ("rh", "med")]
    fig, axes = plt.subplots(len(EFFECTS), len(views) + 1, figsize=(16, 7.2),
                             gridspec_kw=dict(width_ratios=[1] * len(views) + [0.05]))
    vert = [s["vertno"] for s in src]
    for r, (eff, (dev, std, (t0, t1))) in enumerate(EFFECTS.items()):
        T = np.abs(tmaps[eff]) * sigmasks[eff]               # |t| (sign is orientation-dependent), significant only
        lim = max(np.abs(tmaps[eff]).max(), 1e-6)
        stc_t = mne.SourceEstimate(T[:, None], vertices=vert, tmin=0, tstep=1, subject=TEMPLATE)
        for c, (hemi, view) in enumerate(views):
            ax = axes[r, c]
            img = _render(stc_t, hemi, view, lim)
            ax.imshow(img)
            ax.set_axis_off()
            if r == 0:
                ax.set_title(f"{'Left' if hemi == 'lh' else 'Right'} {'lateral' if view == 'lat' else 'medial'}", fontsize=10)
        axes[r, 0].text(-0.06, 0.5, f"{'Local' if eff == 'local' else 'Global'}  {dev} − {std}\n"
                        f"{int(t0 * 1e3)}–{int(t1 * 1e3)} ms", transform=axes[r, 0].transAxes,
                        rotation=90, ha="center", va="center", fontsize=10, weight="bold")
        sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(0, lim))
        cb = fig.colorbar(sm, cax=axes[r, -1])
        cb.set_label("|t|", fontsize=9)
    fig.suptitle(f"eLORETA source estimates of the local and global effects (n = {len(subs)}; template head "
                 "with realistic 3-layer BEM)\nGroup t, shown only within significant clusters (p < .05, "
                 "cluster permutation) on the signed source difference; |t| shown because the sign depends on "
                 "cortical orientation.",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "source_eloreta_tmaps.png", dpi=150, bbox_inches="tight")


def _render(stc, hemi, view, lim):
    """Render one hemisphere/view with MNE's matplotlib backend; return a cropped RGB image on white."""
    import mne
    import matplotlib.pyplot as plt
    f = mne.viz.plot_source_estimates(stc, subject=TEMPLATE, subjects_dir=SUBJECTS_DIR, surface="inflated",
                                      hemi=hemi, views=view, backend="matplotlib", colormap="YlOrRd",
                                      clim=dict(kind="value", lims=[lim * 0.01, lim * 0.4, lim]),
                                      transparent=True, colorbar=False, time_label=None, spacing="oct6",
                                      verbose=False)
    f.canvas.draw()
    img = np.asarray(f.canvas.buffer_rgba())[..., :3].copy()
    plt.close(f)
    # drop the time-slider strip at the bottom, turn the black background white, crop to the brain
    white_rows = np.where((img > 240).all(-1).mean(1) > 0.25)[0]
    if len(white_rows):
        img = img[: white_rows[0] - 5]
    bg = (img < 12).all(-1)
    img[bg] = 255
    ys, xs = np.where(~bg[: img.shape[0]])
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def plot_regions(tc):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    sns.set_theme(context="notebook", style="ticks")
    regs = list(REGIONS)
    fig, axes = plt.subplots(len(EFFECTS), len(regs), figsize=(3.1 * len(regs), 6.4), sharex=True)
    hcol = {"lh": "#4C72B0", "rh": "#C44E52"}
    for r, (eff, (dev, std, (t0, t1))) in enumerate(EFFECTS.items()):
        d = tc[tc["effect"] == eff]
        for c, reg in enumerate(regs):
            ax = axes[r, c]
            for hemi in ["lh", "rh"]:
                q = d[(d["region"] == reg) & (d["hemi"] == hemi)].pivot(index="subject", columns="time",
                                                                          values="value")
                t = q.columns.to_numpy() * 1e3
                X = q.to_numpy() * 1e12                     # eLORETA units are arbitrary; scaled for display
                m, se = X.mean(0), X.std(0, ddof=1) / np.sqrt(len(X))
                ax.plot(t, m, color=hcol[hemi], lw=1.5, label={"lh": "left", "rh": "right"}[hemi])
                ax.fill_between(t, m - se, m + se, color=hcol[hemi], alpha=0.2, lw=0)
                w = (q.columns >= t0) & (q.columns <= t1)
                tv = stats.ttest_1samp(X[:, w].mean(1), 0)
                ax.text(0.98, 0.97 - 0.09 * (hemi == "rh"), f"{'L' if hemi == 'lh' else 'R'}: t={tv.statistic:.1f}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=hcol[hemi])
            ax.axvspan(t0 * 1e3, t1 * 1e3, color="0.92", zorder=0)
            ax.axvline(0, color="k", lw=0.7)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlim(-100, 700)
            if r == 0:
                ax.set_title(reg, fontsize=9.5)
            if c == 0:
                ax.set_ylabel(f"{'Local' if eff == 'local' else 'Global'}\n|{dev}| − |{std}| (a.u.)",
                              fontsize=9)
            if r == len(EFFECTS) - 1:
                ax.set_xlabel("Time re 5th sound (ms)")
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=8, loc="lower left")
    sns.despine(fig)
    fig.suptitle("eLORETA source strength, deviant minus standard (each baseline-corrected −100–0 ms; mean ± SEM, "
                 "n = 15; aparc regions; shaded = a-priori window; t = window mean vs 0). Positive = more activity "
                 "for deviants.", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "source_eloreta_regions.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    arg = sys.argv[1:]
    if arg == ["--fetch"]:
        fetch()
    elif arg == ["--cov"]:
        compute_covs()
    else:
        main()
