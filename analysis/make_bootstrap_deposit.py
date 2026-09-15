"""Write the bootstrap replicate values that produce the intervals in Tables 4 and 5.

The intervals are the one reported quantity that cannot be recomputed from the
confusion matrices, because resampling is over cases. Depositing the replicate
values closes that gap without releasing a case-level record.

Why this discloses nothing beyond what is already deposited: a replicate
accuracy is (number of correct cases in the resample)/n. The resample indices
are not written, the correctness indicators are 0/1, and resampling is uniform
with replacement, so the distribution of the replicate values is a function of
the number correct alone - which is the trace of the confusion matrix already in
analysis/deposit/. The same argument applies to macro-F1 through the class-wise
counts, which are the matrix itself. No row corresponds to a patient.

Run:  REFERENCE_XLSX=... JSON_DIRS=... python3 analysis/make_bootstrap_deposit.py
Env:  OUT_CSV   destination (default analysis/deposit/bootstrap_replicates.csv)
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_results as A  # noqa: E402

AXES = [("T", "cT"), ("N", "cN"), ("M", "cM"), ("S", "stage group")]


def main() -> None:
    out = os.environ.get("OUT_CSV") or os.path.join(HERE, "deposit", "bootstrap_replicates.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    gt = A.load_gt()
    runs = A.discover()
    if not runs:
        sys.exit("no runs discovered - set JSON_DIRS")

    rows = 0
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["edition", "model", "configuration", "component",
                    "statistic", "replicate", "value"])
        for r in sorted(runs, key=lambda x: (x["ed"], x["model"], A.MODE_ORDER.index(x["mode"]))):
            ed = f"AJCC{r['ed']}th"
            mo, md = A.MODEL_LABEL[r["model"]], A.MODE_LABEL[r["mode"]]
            for ax, lab in AXES:
                key = f"{r['ed']}|{r['model']}|{r['mode']}|{ax}"
                p = np.array(A.preds(r, ax), dtype=object)
                t = np.array(A.gts(gt, r, ax), dtype=object)
                m = t != ""
                p, t = p[m], t[m]
                n = len(t)
                correct = (p == t)

                rng = A.cell_rng("acc|" + key)
                for i in range(A.N_BOOT):
                    idx = rng.integers(0, n, n)
                    w.writerow([ed, mo, md, lab, "accuracy", i, f"{correct[idx].mean() * 100:.4f}"])
                    rows += 1
                rng = A.cell_rng("f1|" + key)
                for i in range(A.N_BOOT):
                    idx = rng.integers(0, n, n)
                    w.writerow([ed, mo, md, lab, "macro_f1", i,
                                f"{A.macro_f1(list(p[idx]), list(t[idx])):.4f}"])
                    rows += 1

    # Verify on the written file: the reported intervals must fall out of it.
    got = {}
    for r in csv.DictReader(open(out)):
        got.setdefault((r["edition"], r["model"], r["configuration"],
                        r["component"], r["statistic"]), []).append(float(r["value"]))
    AXK = {"cT": "T", "cN": "N", "cM": "M", "stage group": "S"}
    rep = {(x["edition"], x["model"], x["mode"], x["axis"]): (float(x["ci_lo"]), float(x["ci_hi"]))
           for x in csv.DictReader(open(os.path.join(HERE, "results", "table4.csv")))}
    bad = 0
    for k, v in got.items():
        if k[4] != "accuracy":
            continue
        lo, hi = np.percentile(v, [2.5, 97.5])
        want = rep.get((k[0].replace("AJCC", "").replace("th", ""), k[1], k[2], AXK[k[3]]))
        if want and (abs(lo - want[0]) > 0.05 or abs(hi - want[1]) > 0.05):
            bad += 1
    print(f"rows                 : {rows:,}")
    print(f"file                 : {out} ({os.path.getsize(out):,} bytes)")
    print(f"cells                : {len(got) // 2}")
    print(f"intervals recomputed : {len(rep)} reported, {bad} disagree")
    print("verification         : " + ("PASS" if bad == 0 else "FAIL"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
