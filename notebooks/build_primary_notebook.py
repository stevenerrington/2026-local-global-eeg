"""
build_primary_notebook.py — builds notebooks/primary_analysis.ipynb: every analysis AND every figure.

The notebook starts from the saved derivatives (no preprocessing) and contains, in order:
  0     configuration and shared functions (inline)
  1–3   the statistics behind Figures 1–3, then the code that draws each figure
  4     DCM: reads the SPM outputs, tables, and draws the DCM figures (fitting stays in MATLAB)
  5     validation of every method on simulated data
  6     every Results number vs this notebook
  7     inventory of the figure files written

The analysis cells are taken from build_results_notebook.py (same folder). The figure cells are built
from the CURRENT figure scripts in scripts/: each script's code is placed inline in the notebook,
inside one function per figure, so its settings and helper functions cannot overwrite the notebook's
variables. Rebuild after editing a figure script:

    python notebooks/build_primary_notebook.py

Output: notebooks/primary_analysis.ipynb
"""
import ast
import re
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCRIPTS = ROOT / "scripts"

# ----------------------------------------------------------------------------- analysis cells
# Run the results-notebook builder up to (not including) its write step to obtain its cell list.
src = (HERE / "build_results_notebook.py").read_text()
src = src[: src.index("nb = nbf.v4.new_notebook()")] if "nb = nbf.v4.new_notebook()" in src else src
ns = {}
exec(compile(src, "build_results_notebook.py", "exec"), ns)
cells = ns["cells"]
md = lambda s: nbf.v4.new_markdown_cell(s.strip("\n"))
code = lambda s: nbf.v4.new_code_cell(s.strip("\n"))


def heading_index(prefix):
    for i, c in enumerate(cells):
        if c.cell_type == "markdown" and c.source.lstrip().startswith(prefix):
            return i
    raise KeyError(prefix)


def figure_cells(script, fn_name, calls, title, outputs, edits=(), inner="main()"):
    """Markdown (the script's own docstring) + one code cell holding the script inline.

    The script body is wrapped in `def fn_name(...)`, so its CONFIG constants and helpers are local to
    the figure. `ROOT` is set to the notebook's project root, the Agg backend line is removed, the
    `if __name__ == "__main__"` block is replaced by `calls`, and the saved PNGs are displayed.
    """
    text = (SCRIPTS / script).read_text()
    tree = ast.parse(text)
    doc = ast.get_docstring(tree) or ""
    first = tree.body[0]
    body = text.splitlines()[first.end_lineno:] if isinstance(first, ast.Expr) else text.splitlines()
    body = "\n".join(body)
    body = body[: body.index('if __name__ == "__main__":')].rstrip()
    body = re.sub(r"^matplotlib\.use\([^)]*\)\n", "", body, flags=re.M)
    body = re.sub(r"^ROOT = Path\(__file__\)\.resolve\(\)\.parents\[1\]", "ROOT = PROJECT_ROOT", body, flags=re.M)
    for old, new in edits:
        assert old in body, f"{script}: '{old}' not found"
        body = body.replace(old, new)
    indented = "\n".join(("    " + ln) if ln.strip() else "" for ln in body.splitlines())
    args, call_lines = calls
    cell = (f"# ---- {script} (inline; rebuild the notebook after editing the script)\n"
            f"def {fn_name}({args}):\n{indented}\n\n    return {inner}\n\n\n{call_lines}\n\n"
            "plt.close('all')\n"
            f"for f in {outputs!r}:\n"
            "    p = PROJECT_ROOT / 'figures' / f\n"
            "    display(Image(filename=str(p), width=900) if p.exists() else f'missing: {p}')")
    note = f"### {title}\n\n`scripts/{script}`. From the script's header:\n\n```text\n{doc}\n```"
    return [md(note), code(cell)]


SETUP = code(r"""
# ---------------------------------------------------------------- figures: shared settings
from IPython.display import Image, display
import sys
PROJECT_ROOT = ROOT                      # every figure function reads its inputs relative to this
(PROJECT_ROOT / "figures").mkdir(exist_ok=True)
FIG_LOG = []                             # figures made in this session (section 7)

def run_figure(name, fn, *args, **kwargs):
    '''Run one figure function; report (rather than stop the notebook on) a missing input.'''
    t0 = time.time()
    try:
        fn(*args, **kwargs)
        FIG_LOG.append((name, "ok", round(time.time() - t0, 1)))
    except Exception as e:                # e.g. template head or DCM fits not available on this computer
        FIG_LOG.append((name, f"FAILED: {type(e).__name__}: {e}", round(time.time() - t0, 1)))
        print(f"{name} failed: {type(e).__name__}: {e}")
""")

FIG1 = figure_cells(
    "14_figure1.py", "make_figure1", ("lean=False, erp_only=False", "\n".join([
        "run_figure('Figure 1', make_figure1)                         # full figure (A–F)",
        "run_figure('Figure 1 lean', make_figure1, lean=True)         # lean version (fewer topography bins)",
        "run_figure('Figure 1 D–F print', make_figure1, erp_only=True)  # D–F at print size, editable text"])),
    "Figure 1 — design, topographies, ERPs and participant effects",
    ["Figure1.png", "Figure1_lean.png", "Figure1_DE.png"],
    edits=[('LEAN = "--lean" in sys.argv', "LEAN = lean"),
           ('ERP_ONLY = "--erp" in sys.argv', "ERP_ONLY = erp_only")])
FIG2A = figure_cells("15_figure2_schematic.py", "make_figure2_schematic",
                     ("", "run_figure('Figure 2A–B', make_figure2_schematic)"),
                     "Figure 2A–B — model schematic and regressors", ["Figure2_schematic.png"],
                     inner="main(SUBJECT)")
FIG2 = figure_cells("16_figure2_results.py", "make_figure2_results",
                    ("", "run_figure('Figure 2C–F', make_figure2_results)"),
                    "Figure 2C–F — model comparison results", ["Figure2_results.png"])
FIG2R = figure_cells("26_figure2_recovery.py", "make_figure2_recovery",
                     ("", "run_figure('Figure 2 recovery', make_figure2_recovery)"),
                     "Figure 2B (alternative) — model recovery", ["Figure2_recovery.png"])
FIG3 = figure_cells("19_figure3.py", "make_figure3", ("", "run_figure('Figure 3', make_figure3)"),
                    "Figure 3 — auditory-cortex localisation of the local effect", ["Figure3.png"])
DCMW = figure_cells("28_figure_dcm_windows.py", "make_figure_dcm_windows",
                    ("", "run_figure('DCM windows', make_figure_dcm_windows)"),
                    "DCM figure — full response (0–600 ms) vs MMN window (0–250 ms)", ["Figure_DCM_windows.png"])
DCMR = figure_cells("24_figure_dcm_robust.py", "make_figure_dcm_robust",
                    ("", "run_figure('DCM robustness', make_figure_dcm_robust)"),
                    "Supplementary — DCM robustness across five analyses (0/1 coding)", ["Figure_DCM_robust.png"])
DCMM = figure_cells("22_figure_dcm.py", "make_figure_dcm_main",
                    ("", "run_figure('DCM main model', make_figure_dcm_main)"),
                    "Supplementary — main DCM (0/1 coding, 0–600 ms)", ["Figure_DCM.png"])
DCMP = figure_cells("23_dcm_predicted_erps.py", "make_figure_dcm_predicted",
                    ("", "run_figure('DCM predicted ERPs', make_figure_dcm_predicted)"),
                    "Supplementary — ERPs predicted by the DCM vs observed", ["DCM_predicted_ERPs.png"])

DCM_INTRO = [md(r"""
## 4. DCM — effective connectivity (results read from SPM)

The DCMs are fitted in MATLAB/SPM12 and cannot run here. Run order (from `scripts/dcm/`):
`dcm_01_prepare.m` → `dcm_02_fit.m` → `dcm_03_peb.m` → `dcm_04_robustness.m` → (after
`python scripts/25_dcm_surprise_export.py`) `dcm_05_surprise.m` (0–600 ms) → `dcm_06_early.m` (0–250 ms) →
`dcm_07_search_full.m`. This section reads their outputs.

**Codings.** *Surprise*: each participant's condition-mean learner surprise (M2 local, M3 global), relative to
LS/GS, scaled to a maximum of 1. *Interaction*: 0/1 labels plus a local × global term. **Credible** = posterior
probability of being present > .95 after per-connection Bayesian model reduction (`Pp_search`; falls back to
the sign probability `Pp` where that column is missing). **Robust** = credible with the same sign in both codings.
"""), code(r"""
# ---------------------------------------------------------------- DCM: connection tables per window x coding
DCM_DIR = ROOT / "derivatives" / "dcm"
dcm_tabs = []
for win, sub in [("0–600 ms", "surprise_coded"), ("0–250 ms", "early_window")]:
    for cod in ["surprise", "interaction"]:
        f = DCM_DIR / sub / cod / "connections.csv"
        if not f.exists():
            print("missing:", f); continue
        d = pd.read_csv(f)
        pp = "Pp_search" if "Pp_search" in d else "Pp"
        ep = "Ep_search" if "Ep_search" in d else "Ep"
        dcm_tabs.append(d.assign(window=win, coding=cod, Ep_use=d[ep], credible=d[pp] > 0.95, stat=pp))
dcm = pd.concat(dcm_tabs, ignore_index=True)
print("credibility statistic used:", dcm.groupby(["window", "coding"])["stat"].first().to_dict())

# robust = credible with the same sign in both codings
rob = (dcm[dcm.effect.isin(["local", "global"])]
       .groupby(["window", "effect", "connection"])
       .apply(lambda g: pd.Series(dict(robust=bool(g.credible.all() and np.sign(g.Ep_use).nunique() == 1),
                                       mean_log_gain=g.Ep_use.mean(),
                                       **{f"Ep_{c}": float(g.loc[g.coding == c, "Ep_use"].iloc[0]) for c in g.coding})),
              include_groups=False)
       .reset_index())
for (win, eff), g in rob[rob.robust].groupby(["window", "effect"], sort=False):
    print(f"\n{win}, {eff}: robust connection changes (log gain)")
    print(g[["connection", "mean_log_gain", "Ep_surprise", "Ep_interaction"]].round(2).to_string(index=False))
""")]

INVENTORY = [md(r"""
## 7. Figure inventory

Every figure function run in this session, whether it succeeded, and the files now in `figures/`.
"""), code(r"""
print(pd.DataFrame(FIG_LOG, columns=["figure", "status", "seconds"]).to_string(index=False))
inv = sorted((PROJECT_ROOT / "figures").glob("*.pdf"))
print(f"\n{len(inv)} PDF figures in figures/ (each also as .png, most as .svg):")
for p in inv:
    print(f"  {p.name:34s} {time.strftime('%Y-%m-%d %H:%M', time.localtime(p.stat().st_mtime))}")
""")]

# ----------------------------------------------------------------------------- assemble
cells[0] = md(r"""
# Local–global EEG: primary analysis notebook

Every analysis in the Results and every figure, in one place, starting from the saved derivatives.

| Section | What it does | Figure files |
|---|---|---|
| 0 | configuration, shared functions (inline) | — |
| 1 | MMN / P3b window tests, clusters, 2 × 2 ANOVA, reference control | `Figure1`, `Figure1_lean`, `Figure1_DE` |
| 2 | learners, cross-validated model comparison, recovery, surprise–amplitude relations, reset | `Figure2_schematic`, `Figure2_results`, `Figure2_recovery` |
| 3 | eLORETA cluster, auditory-cortex time course, spread checks, global null | `Figure3` |
| 4 | DCM: tables from the SPM outputs, DCM figures | `Figure_DCM_windows`, `Figure_DCM_robust`, `Figure_DCM`, `DCM_predicted_ERPs` |
| 5 | validation of every method on simulated data | — |
| 6 | every Results number vs this notebook | — |
| 7 | figure inventory | — |

**Not in this notebook:** preprocessing (`scripts/01`–`02`), the single-trial exports (`07`), the full
model-comparison and recovery fits (`06`; loaded here unless `RECOMPUTE_ALL_CV = True`), source extraction
(`11`, `18 --extract`) and the DCM fits (MATLAB, `scripts/dcm/`).

**How the figure cells work.** Each figure's code is the current figure script, placed inline inside one function
(`make_figure1()`, …) so its settings cannot change the notebook's variables. Figures are written to `figures/`
(PDF/SVG with editable text, PNG) and shown below each cell. If an input is missing on this computer (e.g. the
template head for Figure 3, or the DCM fits), that figure reports the problem and the notebook continues.
To change a figure, edit its script and rebuild this notebook (`python notebooks/build_primary_notebook.py`).

Run top to bottom with the `lg-eeg` environment. Total time: about 15–20 minutes.
""")

i2, i3, i4, i5 = (heading_index(p) for p in ("## 2.", "## 3.", "## 4.", "## 5."))
cells[i4].source = cells[i4].source.replace("## 4.", "## 5.", 1)
cells[i5].source = cells[i5].source.replace("## 5.", "## 6.", 1)
new = (cells[:i2] + [SETUP] + FIG1 + cells[i2:i3] + FIG2A + FIG2 + FIG2R + cells[i3:i4] + FIG3
       + DCM_INTRO + DCMW + DCMR + DCMM + DCMP + cells[i4:] + INVENTORY)
nb = nbf.v4.new_notebook()
nb.cells = new
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (lg-eeg)", "language": "python"}
out = HERE / "primary_analysis.ipynb"
nbf.write(nb, out)
print(f"{out}  ({len(new)} cells)")
