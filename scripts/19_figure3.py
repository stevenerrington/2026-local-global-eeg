"""
19_figure3.py — Figure 3 (single column, 84 x 74 mm): the local effect localises to left auditory cortex.

  A  eLORETA group t-map of the local effect (LD − LS, 80–140 ms window mean), shown only within
     significant clusters (one-sample cluster permutation on the signed source difference,
     cluster-forming p < .01, 2000 permutations); |t| because the sign depends on cortical orientation.
     Left and right lateral views of the template (inflated) surface; auditory cortex outlined.
  B  Time course of the signed source difference (LD − LS) in auditory cortex (transverse + superior
     temporal gyrus; aparc), left and right: eLORETA normal orientation, sign-flip-aligned label mean
     ("mean_flip"), mean ± SE, a-priori window shaded; window-mean t-test per hemisphere. A signed
     difference has zero expectation under the null, so — unlike source strength — it is not biased
     by unequal trial counts.

Methods identical to 11_source_eloreta.py (template "sample" head, 3-layer BEM, standard electrode
positions, loose 0.2, depth 0.8, lambda² = 1/9, per-subject noise covariance).
Inputs: derivatives/header1_erps/<sub>.npz + info.fif; derivatives/source/{<sub>-cov.fif, subjects/}
Outputs: figures/Figure3.pdf / .svg / .png; derivatives/source/figure3_stats.json;
         derivatives/source/local_tmap.npz (t-map, cluster mask, auditory-cortex time courses; cached so
         the figure can be re-styled without recomputing)
Rendering: the inflated template surface is drawn directly with matplotlib (painter's algorithm,
back-face culling, curvature shading; overlay morphed to the full-resolution surface), because the
MNE matplotlib 3D backend clips the lateral view.

Usage:  python scripts/19_figure3.py            (computes the t-map if local_tmap.npz is missing)
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ----------------------------------------------------------------------------- CONFIG
ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "derivatives" / "header1_erps"
SRC_DIR = ROOT / "derivatives" / "source"
SUBJECTS_DIR = SRC_DIR / "subjects"
FIG_DIR = ROOT / "figures"
TEMPLATE = "sample"
SIZE_MM = (84.0, 74.0)
WIN = (0.08, 0.14)
LAMBDA2, LOOSE, DEPTH = 1.0 / 9.0, 0.2, 0.8
CLUSTER_P, N_PERM, SEED = 0.01, 2000, 42
AC_LABELS = ["transversetemporal", "superiortemporal"]
REGION = "Auditory cortex (HG + STG)"
HCOL = {"lh": "#C44E52", "rh": "0.6"}
TEXT_RC = {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
           "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
           "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
           "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
           "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
           "axes.labelpad": 2}
# ------------------------------------------------------------------------------------


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


def local_tmap():
    """Group t-map + cluster mask of the local effect (signed eLORETA, window mean); cached."""
    out = SRC_DIR / "local_tmap.npz"
    if out.exists():
        z = np.load(out)
        return (z["T"], z["mask"], [z["vert_lh"], z["vert_rh"]], z["cluster_p"],
                dict(lh=z["ac_lh"], rh=z["ac_rh"]), z["times"])
    import mne
    mne.set_log_level("ERROR")
    subs = sorted(f.stem for f in CACHE_DIR.glob("sub-*.npz"))
    info = mne.io.read_info(CACHE_DIR / "info.fif")
    info = mne.pick_info(info, mne.pick_types(info, eeg=True))
    with info._unlock():
        info["projs"] = []
    raw0 = mne.io.RawArray(np.zeros((len(info["ch_names"]), 1)), info)
    raw0.set_eeg_reference("average", projection=True)
    info = raw0.info
    fwd = make_forward(info)
    src = fwd["src"]
    labs = {lb.name: lb for lb in mne.read_labels_from_annot(TEMPLATE, "aparc", subjects_dir=SUBJECTS_DIR)}
    ac = {h: sum((labs[f"{n}-{h}"] for n in AC_LABELS[1:]), labs[f"{AC_LABELS[0]}-{h}"]) for h in ("lh", "rh")}
    X, tcs = [], {"lh": [], "rh": []}
    for sub in subs:
        z = np.load(CACHE_DIR / f"{sub}.npz")
        cov = mne.read_cov(SRC_DIR / f"{sub}-cov.fif")
        inv = mne.minimum_norm.make_inverse_operator(info, fwd, cov, loose=LOOSE, depth=DEPTH)
        ev = mne.EvokedArray((z["topo__LD"] - z["topo__LS"]) * 1e-6, info, tmin=z["times"][0])
        stc = mne.minimum_norm.apply_inverse(ev, inv, LAMBDA2, method="eLORETA", pick_ori="normal",
                                             method_params=dict(max_iter=60))
        for h in ("lh", "rh"):
            tcs[h].append(stc.extract_label_time_course(ac[h], src, mode="mean_flip")[0])
        times = stc.times
        X.append(stc.copy().crop(*WIN).data.mean(1))
        print(f"{sub}: done", flush=True)
    X = np.stack(X)
    T, clusters, pv, _ = mne.stats.permutation_cluster_1samp_test(
        X, threshold=stats.t.ppf(1 - CLUSTER_P / 2, len(X) - 1), n_permutations=N_PERM,
        adjacency=mne.spatial_src_adjacency(src), tail=0, seed=SEED, out_type="mask", n_jobs=1, verbose=False)
    mask = np.zeros_like(T, bool)
    for c, p in zip(clusters, pv):
        if p < 0.05:
            mask |= c
    vert = [s["vertno"] for s in src]
    tcs = {h: np.stack(v) for h, v in tcs.items()}
    np.savez_compressed(out, T=T, mask=mask, vert_lh=vert[0], vert_rh=vert[1], cluster_p=np.sort(pv),
                        ac_lh=tcs["lh"], ac_rh=tcs["rh"], times=times)
    return T, mask, vert, np.sort(pv), tcs, times


def render(ax, values, vert, hemi, lim, cmap):
    """Lateral view of the inflated surface: back-face culling + depth sort, curvature shading,
    overlay (|t| within significant clusters) morphed to all vertices; auditory cortex outlined."""
    import mne
    import nibabel
    from matplotlib.collections import PolyCollection, LineCollection
    sd = SUBJECTS_DIR / TEMPLATE
    rr, tris = mne.read_surface(sd / "surf" / f"{hemi}.inflated")
    curv = nibabel.freesurfer.read_morph_data(sd / "surf" / f"{hemi}.curv")
    stc = mne.SourceEstimate(values[:, None], vertices=vert, tmin=0, tstep=1, subject=TEMPLATE)
    full = mne.compute_source_morph(stc, TEMPLATE, TEMPLATE, subjects_dir=SUBJECTS_DIR, spacing=None,
                                    smooth=5, verbose=False).apply(stc)
    val = full.data[: len(full.vertices[0]), 0] if hemi == "lh" else full.data[len(full.vertices[0]):, 0]
    vfull = np.zeros(len(rr))
    vfull[full.vertices[0 if hemi == "lh" else 1]] = val
    sgn = -1 if hemi == "lh" else 1                     # camera on the lateral side
    tri = rr[tris]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vis = sgn * nrm[:, 0] > 0
    order = np.argsort(sgn * tri[:, :, 0].mean(1)[vis])  # far (medial) first
    fi = np.where(vis)[0][order]
    u = (-rr[:, 1] if hemi == "lh" else rr[:, 1])          # anterior to the left for lh, right for rh
    xy = np.column_stack([u, rr[:, 2]])
    base = np.where(curv[tris[fi]].mean(1) > 0, 0.55, 0.78)
    col = np.repeat(base[:, None], 3, 1)
    fv = vfull[tris[fi]].mean(1)
    on = fv > lim * 0.02
    over = cmap(np.clip(fv / lim, 0, 1))[:, :3]
    a = np.clip(fv / (lim * 0.25), 0, 1)[:, None]
    col = np.where(on[:, None], (1 - a) * col + a * over, col)
    ax.add_collection(PolyCollection(xy[tris[fi]], facecolors=col, edgecolors="none", antialiased=False,
                                     rasterized=True))
    # auditory-cortex outline: boundary edges of the label's visible faces
    labs = mne.read_labels_from_annot(TEMPLATE, "aparc", hemi=hemi, subjects_dir=SUBJECTS_DIR, verbose=False)
    inlab = np.zeros(len(rr), bool)
    for lb in labs:
        if lb.name.split("-")[0] in AC_LABELS:
            inlab[lb.vertices] = True
    lf = tris[vis][inlab[tris[vis]].all(1)]
    e = np.sort(np.concatenate([lf[:, [0, 1]], lf[:, [1, 2]], lf[:, [2, 0]]]), axis=1)
    ue, cnt = np.unique(e, axis=0, return_counts=True)
    ax.add_collection(LineCollection(xy[ue[cnt == 1]], colors="k", linewidths=0.5))
    ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
    ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
    ax.set_aspect("equal")
    ax.set_axis_off()


def main():
    import mne
    mne.set_log_level("ERROR")
    T, mask, vert, cluster_p, tcs, times = local_tmap()
    absT = np.abs(T) * mask
    lim = float(np.ceil(np.abs(T).max()))
    cmap = plt.get_cmap("YlOrRd")
    n = len(tcs["lh"])

    sns.set_theme(style="ticks", rc=TEXT_RC)
    mm = 1 / 25.4
    fig = plt.figure(figsize=(SIZE_MM[0] * mm, SIZE_MM[1] * mm))
    out = dict(cluster_p=[float(p) for p in cluster_p[:3]], n_sig_sources=int(mask.sum()),
               t_max=float(np.abs(T).max()))
    # ---------------- A: brains
    for k, h in enumerate(("lh", "rh")):
        ax = fig.add_axes([0.03 + k * 0.42, 0.57, 0.40, 0.33])
        render(ax, absT, vert, h, lim, cmap)
        ax.set_title("Left" if h == "lh" else "Right", pad=1)
    cax = fig.add_axes([0.89, 0.62, 0.022, 0.22])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, lim)), cax=cax)
    cb.set_label("|t|", labelpad=1)
    cb.ax.tick_params(labelsize=5.5)
    fig.text(0.02, 0.99, "A", fontsize=9, weight="bold", va="top")
    fig.text(0.09, 0.985, f"Local effect (LD − LS), {int(WIN[0] * 1e3)}–{int(WIN[1] * 1e3)} ms", fontsize=7, va="top")

    # ---------------- B: signed source difference in auditory cortex
    ax = fig.add_axes([0.17, 0.135, 0.79, 0.36])
    ax.axvspan(WIN[0] * 1e3, WIN[1] * 1e3, color="0.92", lw=0, zorder=0)
    t = times * 1e3
    w = (times >= WIN[0]) & (times <= WIN[1])
    flip = -1.0 if tcs["lh"].mean(0)[w].mean() > 0 else 1.0   # mean_flip sign is arbitrary: plot MMN as negative
    scale = np.abs(tcs["lh"].mean(0)).max()
    for h, lab in (("lh", "left"), ("rh", "right")):
        X = flip * tcs[h] / scale
        m, se = X.mean(0), X.std(0, ddof=1) / np.sqrt(n)
        wm = X[:, w].mean(1)
        r = stats.ttest_1samp(wm, 0)
        same = int((np.sign(wm) == np.sign(wm.mean())).sum())
        ptxt = "p < .001" if r.pvalue < 0.001 else f"p = {r.pvalue:.3f}".replace("0.", ".")
        ax.plot(t, m, color=HCOL[h], lw=1.1, label=f"{lab}  t({n - 1}) = {r.statistic:.1f}, {ptxt}".replace("-", "−"))
        ax.fill_between(t, m - se, m + se, color=HCOL[h], alpha=0.25, lw=0)
        pk = t[(t >= 50) & (t <= 200)][np.argmin(m[(t >= 50) & (t <= 200)])]
        out[f"auditory_cortex_{h}"] = dict(t=float(r.statistic), p=float(r.pvalue), n_same_sign=same,
                                           group_peak_ms=float(pk))
    ax.axvline(0, color="k", lw=0.5)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlim(-100, 500)
    ax.set_xlabel("Time from 5th sound (ms)")
    ax.set_ylabel("LD − LS source current\n(a.u., normalised)")
    ax.legend(frameon=False, loc="lower right", handlelength=1.2, borderaxespad=0.1, title="Auditory cortex",
              title_fontsize=6)
    sns.despine(ax=ax)
    fig.text(0.02, 0.55, "B", fontsize=9, weight="bold", va="top")

    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "Figure3.pdf", dpi=600)
    fig.savefig(FIG_DIR / "Figure3.svg", dpi=600)
    svg = FIG_DIR / "Figure3.svg"                     # SVG names fonts: Liberation Sans -> Arial (same metrics)
    svg.write_text(svg.read_text().replace("Liberation Sans", "Arial"))
    fig.savefig(FIG_DIR / "Figure3.png", dpi=600)
    (SRC_DIR / "figure3_stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
