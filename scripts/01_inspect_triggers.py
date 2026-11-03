"""
01_inspect_triggers.py — check that every LG run can be labelled before preprocessing.

ds005429's .vmrk files mark every sound with the same code ('Response/R128'), so they carry no
sound identity. Labels come from sourcedata/<sub>/ses-LGeffect:
  GlobLoc_stimulation_SEQUENCE*.mat  -> SEQUENCE{block}: n_quintlets x 5 (1 = A, 2 = B)
  GlobLoc_critical_events*.mat       -> tz: PTB onset time of every sound
Each run's *_events.tsv rows (FieldTrip trl, t = 0 at the 5th sound) are aligned to the
sequence by matching inter-quintlet intervals (jittered 800–1100 ms, so the match is unique).

Per subject this reports: sequence size, blocks and their standards, SOA, trigger counts,
per-run alignment offset and error (plus margin to the next-best offset), and unrecorded
quintlets. Output: derivatives/triggers/alignment.tsv and trigger_report.txt

Usage:  python scripts/01_inspect_triggers.py [sub-01 sub-02 ...]
"""
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from scipy.io import loadmat

# ----------------------------------------------------------------------------- CONFIG
ROOT = Path(__file__).resolve().parents[1]
BIDS_ROOT = ROOT / "data" / "bids"
OUT_DIR = ROOT / "derivatives" / "triggers"
TASK, SESSION = "LocalGlobal", "LGeffect"
SOUNDS = {1: "A", 2: "B"}
MAX_ALIGN_ERR = 0.010    # s
# ------------------------------------------------------------------------------------


def as_ranges(idx):
    """[1, 2, 3, 7] -> '1–3, 7'"""
    out, start = [], None
    for i, x in enumerate(idx):
        if start is None:
            start = x
        if i == len(idx) - 1 or idx[i + 1] != x + 1:
            out.append(f"{start}–{x}" if x != start else f"{x}")
            start = None
    return ", ".join(out)


def vhdr_sfreq(vhdr):
    si = float(re.search(r"SamplingInterval=([\d.]+)", vhdr.read_text(encoding="latin-1")).group(1))
    return 1e6 / si


def vmrk_counts(vmrk):
    txt = vmrk.read_text(encoding="latin-1")
    marks = re.findall(r"^Mk\d+=([^,]*),([^,]*),", txt, re.M)
    return pd.Series([f"{t}/{d}" for t, d in marks]).value_counts().to_dict()


def main(subjects=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subjects = subjects or sorted(p.name for p in BIDS_ROOT.glob("sub-*") if p.is_dir())
    lines, rows = [], []
    for sub in subjects:
        src = BIDS_ROOT / "sourcedata" / sub / f"ses-{SESSION}"
        seq = loadmat(next(src.glob("GlobLoc_stimulation_SEQUENCE*.mat")), squeeze_me=True)["SEQUENCE"]
        tz = loadmat(next(src.glob("GlobLoc_critical_events*.mat")), squeeze_me=True)["tz"]
        tz = tz[tz > 0]
        blocks = [np.atleast_2d(b).astype(int) for b in seq]
        nq = sum(len(b) for b in blocks)
        stds = ["".join(SOUNDS[x] for x in b[0]) for b in blocks]
        gd = [int(sum((q != b[0]).any() for q in b)) for b in blocks]
        t5 = tz[4::5]
        soa = np.median(np.diff(tz.reshape(-1, 5), axis=1)) if len(tz) == 5 * nq else np.nan
        lines.append(f"\n=== {sub}: {len(blocks)} blocks, {nq} quintlets, {len(tz)} sound timestamps, "
                     f"SOA {soa * 1e3:.1f} ms")
        lines.append(f"  block standards: {stds}")
        lines.append(f"  global deviants per block: {gd}")

        covered = []
        eeg_dir = BIDS_ROOT / sub / f"ses-{SESSION}" / "eeg"
        for vhdr in sorted(eeg_dir.glob(f"{sub}_ses-{SESSION}_task-{TASK}_run-*_eeg.vhdr")):
            run = int(vhdr.name.split("_run-")[1].split("_")[0])
            sf = vhdr_sfreq(vhdr)
            ev = pd.read_csv(str(vhdr).replace("_eeg.vhdr", "_events.tsv"), sep="\t")
            a = (ev["begsample"].to_numpy() - ev["offset"].to_numpy() - 1) / sf
            n = len(a)
            errs = np.array([np.max(np.abs(np.diff(t5[o:o + n]) - np.diff(a)))
                             for o in range(nq - n + 1)])
            o = int(np.argmin(errs))
            nxt = np.partition(errs, 1)[1] if len(errs) > 1 else np.nan
            ok = errs[o] <= MAX_ALIGN_ERR
            covered += range(o, o + n)
            counts = vmrk_counts(Path(str(vhdr).replace(".vhdr", ".vmrk")))
            lines.append(f"  run-{run}: sfreq {sf:g} Hz, {n} quintlets, markers {counts}")
            lines.append(f"     -> sequence {o}–{o + n - 1}, max error {errs[o] * 1e3:.1f} ms, "
                         f"next best {nxt * 1e3:.0f} ms  {'OK' if ok else 'FAILED'}")
            rows.append(dict(subject=sub, run=run, sfreq=sf, n_quintlets=n, seq_offset=o,
                             max_err_ms=errs[o] * 1e3, next_best_ms=nxt * 1e3, ok=ok,
                             soa_ms=soa * 1e3))
        missing = sorted(set(range(nq)) - set(covered))
        if missing:
            lines.append(f"  unrecorded sequence quintlets ({len(missing)}): {as_ranges(missing)}")

    report = "\n".join(lines)
    (OUT_DIR / "trigger_report.txt").write_text(report)
    pd.DataFrame(rows).to_csv(OUT_DIR / "alignment.tsv", sep="\t", index=False)
    print(report)
    print(f"\nSaved -> {OUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
