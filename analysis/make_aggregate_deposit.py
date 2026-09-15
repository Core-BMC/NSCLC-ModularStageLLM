"""Build the aggregate deposit tables (JMIR #101332, editorial comment 6).

Institutional data governance does not permit release of case-level records for
this cohort - neither the reference labels nor the model predictions - so the
per-case file described in make_deposit.py stays inside the institution and this
script builds the aggregate tables that are released in its place.

Two tables are written:

  confusion_matrices.csv   reference label x predicted label counts, for every
                           edition, model, configuration and component. A
                           confusion matrix determines accuracy exactly
                           (trace / total) and macro-F1 exactly (per-class TP,
                           FP and FN are all recoverable from it), so Tables 4
                           and 5 and the Figure 3 matrices can be recomputed
                           from this file alone.

  mcnemar_contingency.csv  the paired 2x2 discordance counts for every pair of
                           configurations, per edition, model and component.
                           The exact binomial test depends only on b and c, so
                           every reported P value can be recomputed from this
                           file alone.

What cannot be reproduced from these tables is the bootstrap confidence
interval, which resamples cases and therefore needs case-level data. The
bootstrap code is deposited; its input is not releasable, and the response to
the Editor says so rather than implying otherwise.

Neither table contains a case identifier, a per-case row, a hospital number,
report text, model reasoning or raw model output. That is enforced below, on
the written files, not asserted.

Run:   JSON_DIRS=... REFERENCE_XLSX=... python3 analysis/make_aggregate_deposit.py
Env:   MIN_CELL=n   suppress confusion cells with a count below n (default 0,
                    no suppression). Suppression breaks exact recomputation of
                    the reported metrics, so use it only if required.
       DEPOSIT_DIR=path  where the two tables are written (default
                    analysis/deposit, which is tracked; analysis/results is not).
Out -> analysis/deposit/confusion_matrices.csv
       analysis/deposit/mcnemar_contingency.csv
"""
from __future__ import annotations

import csv
import os
import re
import sys
from collections import Counter
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_results as A  # noqa: E402
import make_deposit as D  # noqa: E402

AXES = [("T", "cT"), ("N", "cN"), ("M", "cM"), ("S", "stage")]
AXIS_LABEL = {"T": "cT", "N": "cN", "M": "cM", "S": "stage group"}
MIN_CELL = int(os.environ.get("MIN_CELL", "0"))
_dep = os.environ.get("DEPOSIT_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "deposit")
DEPOSIT = _dep if os.path.isabs(_dep) else os.path.abspath(_dep)
HANGUL = re.compile(r"[가-힣]")
# Column names that would indicate a case-level record had reached the output.
FORBIDDEN_COLS = ("case_id", "case_number", "hospital_id", "hospitalNumber", "id")


def _correct_map(run: dict, gt: list, axis: str) -> dict:
    """id -> bool, for cases with a reference label on this axis."""
    p = A.preds(run, axis)
    t = A.gts(gt, run, axis)
    return {rec["id"]: (pp == tt)
            for rec, pp, tt in zip(run["data"], p, t) if tt != ""}


def main() -> None:
    gt = A.load_gt()
    runs = A.discover()
    if not runs:
        sys.exit("no runs discovered - check JSON_DIRS")
    os.makedirs(DEPOSIT, exist_ok=True)
    conf_path = os.path.join(DEPOSIT, "confusion_matrices.csv")
    mcn_path = os.path.join(DEPOSIT, "mcnemar_contingency.csv")

    bad: list[str] = []
    suppressed = 0
    conf_rows = 0
    # Self-check: accuracy recomputed from each matrix against analyze_results.
    acc_checks: list[tuple[str, float, float]] = []

    with open(conf_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["edition", "model", "configuration", "component",
                    "reference_label", "predicted_label", "n"])
        for r in sorted(runs, key=lambda x: (x["ed"], x["model"],
                                             A.MODE_ORDER.index(x["mode"]))):
            for axis, vocab_key in AXES:
                pred = A.preds(r, axis)
                truth = A.gts(gt, r, axis)
                cells = Counter()
                for p, t in zip(pred, truth):
                    if t == "":
                        continue
                    p = str(p).strip() or "(no label)"
                    t = str(t).strip()
                    for v, kind in ((t, "reference"), (p, "predicted")):
                        if v != "(no label)" and v not in D.VOCAB[vocab_key]:
                            bad.append(f"{r['file']} {axis} {kind}={v!r}")
                    cells[(t, p)] += 1
                total = sum(cells.values())
                diag = sum(n for (t, p), n in cells.items() if t == p)
                acc_checks.append((
                    f"AJCC{r['ed']}th {A.MODEL_LABEL[r['model']]} "
                    f"{A.MODE_LABEL[r['mode']]} {AXIS_LABEL[axis]}",
                    100 * diag / total if total else float("nan"),
                    A.accuracy_ci(pred, truth)[0]))
                for (t, p), n in sorted(cells.items()):
                    if MIN_CELL and n < MIN_CELL:
                        suppressed += 1
                        n_out = f"<{MIN_CELL}"
                    else:
                        n_out = n
                    w.writerow([f"AJCC{r['ed']}th", A.MODEL_LABEL[r["model"]],
                                A.MODE_LABEL[r["mode"]], AXIS_LABEL[axis],
                                t, p, n_out])
                    conf_rows += 1

    by_key: dict = {}
    for r in runs:
        by_key.setdefault((r["ed"], r["model"]), {})[r["mode"]] = r

    mcn_rows = 0
    with open(mcn_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["edition", "model", "component",
                    "configuration_1", "configuration_2", "n_paired",
                    "both_correct", "only_1_correct", "only_2_correct",
                    "both_incorrect", "accuracy_1_pct", "accuracy_2_pct",
                    "p_exact_mcnemar"])
        for (ed, model), modes in sorted(by_key.items()):
            present = [m for m in A.MODE_ORDER if m in modes]
            for m1, m2 in combinations(present, 2):
                for axis, _ in AXES:
                    c1 = _correct_map(modes[m1], gt, axis)
                    c2 = _correct_map(modes[m2], gt, axis)
                    ids = sorted(set(c1) & set(c2))
                    n = len(ids)
                    both = sum(1 for i in ids if c1[i] and c2[i])
                    only1 = sum(1 for i in ids if c1[i] and not c2[i])
                    only2 = sum(1 for i in ids if not c1[i] and c2[i])
                    neither = n - both - only1 - only2
                    _, _, p = A.mcnemar({i: c1[i] for i in ids},
                                        {i: c2[i] for i in ids})
                    w.writerow([f"AJCC{ed}th", A.MODEL_LABEL[model], AXIS_LABEL[axis],
                                A.MODE_LABEL[m1], A.MODE_LABEL[m2], n,
                                both, only1, only2, neither,
                                f"{100*(both+only1)/n:.1f}" if n else "",
                                f"{100*(both+only2)/n:.1f}" if n else "",
                                f"{p:.3g}"])
                    mcn_rows += 1

    # ---- verification on the written files ----------------------------------
    problems = []
    for path in (conf_path, mcn_path):
        with open(path) as fh:
            text = fh.read()
        if HANGUL.search(text):
            problems.append(f"{os.path.basename(path)}: Korean characters present")
        for k in D.FORBIDDEN_KEYS:
            if k in text:
                problems.append(f"{os.path.basename(path)}: forbidden key {k!r} present")
        header = text.splitlines()[0].split(",")
        for c in FORBIDDEN_COLS:
            if c in header:
                problems.append(f"{os.path.basename(path)}: case-level column {c!r} present")
    if bad:
        problems.append(f"{len(bad)} label(s) outside the vocabulary: {bad[:5]}")
    drift = [(k, a, b) for k, a, b in acc_checks if abs(a - b) > 0.05]
    if drift:
        problems.append(f"{len(drift)} matrix/accuracy mismatch(es): {drift[:3]}")

    print(f"runs                 : {len(runs)}")
    print(f"confusion cells      : {conf_rows} rows -> {conf_path}")
    print(f"mcnemar comparisons  : {mcn_rows} rows -> {mcn_path}")
    print(f"cells suppressed     : {suppressed}"
          + (f" (MIN_CELL={MIN_CELL})" if MIN_CELL else " (no suppression)"))
    print(f"accuracy self-check  : {len(acc_checks)} matrices recomputed from "
          f"their own cells; {len(drift)} disagree with analyze_results")
    print("verification         : " + ("PASS - aggregate counts only, no case-level "
                                       "record, all labels in vocabulary"
                                       if not problems else "FAIL"))
    for p in problems:
        print("   ! " + p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
