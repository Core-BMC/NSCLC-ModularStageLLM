"""Paired baseline-vs-MAA significance per model x edition x component, for Figure 2.
Accuracy: McNemar exact test. Macro-F1: paired bootstrap of the F1 difference.
Writes results/sig.csv (model,edition,axis,acc_p,f1_p).
"""
from __future__ import annotations
import glob, hashlib, json, os, re
from math import comb
from collections import Counter
import numpy as np
import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
# Reference-standard labels (case identifier + cT/cN/cM/stage per edition).
# Not distributed with the repository: the file is held under institutional
# data governance. Override with the REFERENCE_XLSX environment variable.
REFERENCE_XLSX = os.environ.get("REFERENCE_XLSX", "reference_labels.xlsx")
REPO = os.path.dirname(HERE)
GT = REFERENCE_XLSX if os.path.isabs(REFERENCE_XLSX) else os.path.join(REPO, "input", REFERENCE_XLSX)
# Same search rule as analyze_results.py, so that the figure inputs and the
# table inputs cannot come from different run sets.
JSON_DIRS = [os.path.join(REPO, d) for d in
             os.environ.get("JSON_DIRS", "output/json:output").split(":") if d]
RES = os.environ.get("RESULTS_DIR") or os.path.join(HERE, "results")
RES = RES if os.path.isabs(RES) else os.path.join(HERE, RES)
os.makedirs(RES, exist_ok=True)
OUT = os.path.join(RES, "sig.csv")
SEED = 42; NB = 1000


def cell_rng(key: str) -> "np.random.Generator":
    """A generator determined by the cell alone, as in analyze_results.py.

    The paired difference intervals plotted in Figure 2 were previously drawn
    from one module-level generator consumed in run order, so an interval
    depended on how many cells had been computed before it. The Methods state
    that every interval is a function of its own cell; this makes that true of
    the figure as well as of Tables 4 and 5.
    """
    h = hashlib.sha256(f"{SEED}|{key}".encode()).digest()[:8]
    return np.random.default_rng(int.from_bytes(h, "big"))
RUN = re.compile(r"(llama3-70b|phi4|gpt4o)-(mma|base)-ajcc([89])")


def load_gt():
    wb = openpyxl.load_workbook(GT, read_only=True, data_only=True); ws = wb[wb.sheetnames[0]]
    rows = ws.iter_rows(values_only=True); hdr = [str(h) for h in next(rows)]; ix = {h: i for i, h in enumerate(hdr)}
    out = []
    for r in rows:
        if r is None or all(c is None for c in r): continue
        g = lambda c: ("" if r[ix[c]] is None else str(r[ix[c]]).strip())
        out.append({"8": {"T": g("8th-cT"), "N": g("8th-cN"), "M": g("8th-cM")},
                    "9": {"T": g("9th-cT"), "N": g("9th-cN"), "M": g("9th-cM")}})
    return out


def preds(fp, axis):
    d = json.load(open(fp)); d.sort(key=lambda r: r.get("id", 0))
    return [(r.get("aiTnm") or {}).get(axis, "") or "" for r in d]


def macro_f1(pred, truth, idx):
    classes = sorted({truth[i] for i in idx if truth[i] != ""})
    tp = Counter(); fp = Counter(); fn = Counter()
    for i in idx:
        t = truth[i]
        if t == "": continue
        p = pred[i]
        if p == t: tp[t] += 1
        else:
            fn[t] += 1
            if p: fp[p] += 1
    f1s = []
    for c in classes:
        pr = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rc = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def mcnemar_p(base_ok, maa_ok):
    b = sum(1 for x, y in zip(base_ok, maa_ok) if x and not y)
    c = sum(1 for x, y in zip(base_ok, maa_ok) if not x and y)
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def main():
    gt = load_gt()
    files = {}
    found = [fp for d in JSON_DIRS for fp in sorted(glob.glob(os.path.join(d, "*.json")))
             if "repro" not in os.path.basename(fp).lower()]
    if not found:
        raise SystemExit("no run files found under " + ":".join(JSON_DIRS)
                         + " - set JSON_DIRS")
    for fp in found:
        m = RUN.search(os.path.basename(fp))
        if m and "repro" not in os.path.basename(fp).lower():
            files[(m.group(1), m.group(2), m.group(3))] = fp
    rows = ["model,edition,axis,acc_delta,acc_dlo,acc_dhi,acc_p,f1_delta,f1_dlo,f1_dhi,f1_p"]
    lab = {"llama3-70b": "LLaMA 3.3 70B", "phi4": "Phi-4 14B", "gpt4o": "GPT-4o"}
    for model in ["gpt4o", "llama3-70b", "phi4"]:
        for ed in ["8", "9"]:
            base_fp, maa_fp = files.get((model, "base", ed)), files.get((model, "mma", ed))
            if not base_fp or not maa_fp: continue
            for axis in ["T", "N", "M"]:
                pb, pm = preds(base_fp, axis), preds(maa_fp, axis)
                truth = [g[ed][axis] for g in gt]
                n = min(len(pb), len(pm), len(truth))
                valid = [i for i in range(n) if truth[i] != ""]
                base_ok = np.array([pb[i] == truth[i] for i in valid])
                maa_ok = np.array([pm[i] == truth[i] for i in valid])
                accp = mcnemar_p(list(base_ok), list(maa_ok))
                acc_delta = maa_ok.mean() - base_ok.mean()
                # paired bootstrap of accuracy diff and F1 diff (MAA - base)
                arr = np.array(valid); acc_d, f1_d = [], []
                rng = cell_rng(f"diff|{ed}|{lab[model]}|{axis}")
                for _ in range(NB):
                    sel = rng.integers(0, len(arr), len(arr))
                    acc_d.append(maa_ok[sel].mean() - base_ok[sel].mean())
                    idx = arr[sel]
                    f1_d.append(macro_f1(pm, truth, idx) - macro_f1(pb, truth, idx))
                acc_d = np.array(acc_d); f1_d = np.array(f1_d)
                f1_delta = macro_f1(pm, truth, arr) - macro_f1(pb, truth, arr)
                f1p = min(1.0, 2 * min((f1_d <= 0).mean(), (f1_d >= 0).mean()))
                a_lo, a_hi = np.percentile(acc_d, [2.5, 97.5])
                f_lo, f_hi = np.percentile(f1_d, [2.5, 97.5])
                rows.append(f"{lab[model]},{ed}th,{axis},{acc_delta:.4f},{a_lo:.4f},{a_hi:.4f},"
                            f"{accp:.4g},{f1_delta:.4f},{f_lo:.4f},{f_hi:.4f},{f1p:.4g}")
    open(OUT, "w").write("\n".join(rows) + "\n")
    print("\n".join(rows)); print("\nwrote", OUT)


if __name__ == "__main__":
    main()
