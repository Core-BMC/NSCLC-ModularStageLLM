"""Recompute the reported tables from the deposited aggregate files.

The three files in ``deposit/`` are the aggregate release described in
``README.md`` (Deposit). They are sufficient to reproduce every number in
Tables 4 to 6 of the article without any case-level record:

    confusion_matrices.csv    reference label x predicted label counts
    bootstrap_replicates.csv  the 1,000 resample values behind each interval
    mcnemar_contingency.csv   paired 2x2 discordance counts and the exact P

This script recomputes those numbers and prints them, so that a reader can set
the printed tables beside the article. It also checks the deposited files
against each other and exits non-zero if they disagree; it makes no comparison
with the article, because the article is not one of its inputs.

    python3 analysis/recompute_tables.py

Standard library only. No environment variables, no network, no case-level data.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEP = os.path.join(HERE, "deposit")

CONF = os.path.join(DEP, "confusion_matrices.csv")
BOOT = os.path.join(DEP, "bootstrap_replicates.csv")
MCN = os.path.join(DEP, "mcnemar_contingency.csv")

COMPONENTS = ("cT", "cN", "cM", "stage group")
CONFIGS = ("baseline", "decomposition-only", "MAA")
CONFIG_LABEL = {"baseline": "single prompt",
                "decomposition-only": "decomposition only",
                "MAA": "MAA"}

PROBLEMS: list[str] = []


def note(msg: str) -> None:
    PROBLEMS.append(msg)


def percentile(values, q):
    """The linear-interpolation percentile, matching numpy's default method.

    Written out so that this script needs nothing outside the standard library;
    it returns the same numbers as the numpy call that produced the reported
    intervals.
    """
    v = sorted(values)
    if not v:
        return float("nan")
    pos = (len(v) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return v[int(pos)]
    return v[lo] * (hi - pos) + v[hi] * (pos - lo)


def mcnemar_exact(b, c):
    """Two-sided exact binomial McNemar test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def read_confusion():
    """cell -> {(reference, predicted): n}"""
    cells = defaultdict(dict)
    for r in csv.DictReader(open(CONF, encoding="utf-8")):
        key = (r["edition"], r["model"], r["configuration"], r["component"])
        cells[key][(r["reference_label"], r["predicted_label"])] = int(r["n"])
    return cells


def accuracy(counts):
    total = sum(counts.values())
    hit = sum(n for (ref, pred), n in counts.items() if ref == pred)
    return 100.0 * hit / total if total else float("nan")


def macro_f1(counts):
    """Macro-averaged F1 over the classes present in the reference standard.

    One-vs-rest, unweighted mean over reference classes - the definition used
    to produce the reported values (analyze_results.py, _per_class). A predicted
    label that never appears as a reference label (Tx/Nx/Mx, Unknown,
    '(no label)', T0, bare T1/T2) contributes false positives to itself only,
    so it lowers accuracy and recall without being averaged as a class.
    """
    classes = sorted({ref for ref, _ in counts})
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for (ref, pred), n in counts.items():
        if ref == pred:
            tp[ref] += n
        else:
            fn[ref] += n
            fp[pred] += n
    f1s = []
    for c in classes:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return 100.0 * sum(f1s) / len(f1s) if f1s else float("nan")


def read_replicates():
    """(edition, model, configuration, component, statistic) -> [1000 values]"""
    series = defaultdict(list)
    for r in csv.DictReader(open(BOOT, encoding="utf-8")):
        series[(r["edition"], r["model"], r["configuration"],
                r["component"], r["statistic"])].append(float(r["value"]))
    return series


def main() -> int:
    for path in (CONF, BOOT, MCN):
        if not os.path.exists(path):
            sys.exit("missing deposited file: %s" % path)

    cells = read_confusion()
    series = read_replicates()
    mcn = list(csv.DictReader(open(MCN, encoding="utf-8")))

    # ---- the deposited files, checked against each other -------------------
    if len(cells) != 72:
        note("expected 72 confusion matrices (2 editions x 3 models x 3 "
             "configurations x 4 components), found %d" % len(cells))
    for key, counts in cells.items():
        if sum(counts.values()) != 495:
            note("confusion matrix %s sums to %d, not the cohort size 495"
                 % ("/".join(key), sum(counts.values())))
    if len(series) != 144:
        note("expected 144 replicate series, found %d" % len(series))
    for key, vals in series.items():
        if len(vals) != 1000:
            note("replicate series %s has %d values, not 1,000"
                 % ("/".join(key), len(vals)))
    if len(mcn) != 72:
        note("expected 72 paired comparisons, found %d" % len(mcn))

    models = sorted({k[1] for k in cells})
    editions = sorted({k[0] for k in cells})

    # ---- Table 4 and Table 5 ----------------------------------------------
    for stat, title in (("accuracy", "Table 4. Accuracy, % (95% CI)"),
                        ("macro_f1", "Table 5. Macro-averaged F1, % (95% CI)")):
        print("\n" + title)
        print("(recomputed from deposit/; point estimate from the confusion "
              "matrices, interval from the 2.5th and 97.5th percentiles of the "
              "1,000 deposited resample values)")
        for ed in editions:
            print("\n  %s" % ed.replace("AJCC", "AJCC "))
            print("    %-14s %-20s %s" % ("Model", "Configuration",
                                          "   ".join("%-22s" % c for c in COMPONENTS)))
            for model in models:
                for cfg in CONFIGS:
                    row = []
                    for comp in COMPONENTS:
                        counts = cells.get((ed, model, cfg, comp))
                        vals = series.get((ed, model, cfg, comp, stat))
                        if not counts or not vals:
                            note("missing performance cell: %s/%s/%s/%s/%s" % (ed, model, cfg, comp, stat))
                            row.append("%-22s" % "-")
                            continue
                        point = accuracy(counts) if stat == "accuracy" else macro_f1(counts)
                        lo = percentile(vals, 2.5)
                        hi = percentile(vals, 97.5)
                        if not (lo - 0.05 <= point <= hi + 0.05):
                            note("%s %s/%s/%s/%s: the point estimate %.1f lies "
                                 "outside its own interval %.1f-%.1f"
                                 % (stat, ed, model, cfg, comp, point, lo, hi))
                        row.append("%-22s" % ("%.1f (%.1f-%.1f)" % (point, lo, hi)))
                    print("    %-14s %-20s %s"
                          % (model, CONFIG_LABEL[cfg], "   ".join(row)))

    # ---- Table 6 -----------------------------------------------------------
    print("\n\nTable 6. Paired comparisons between configurations")
    print("(McNemar exact P recomputed from the deposited discordance counts; "
          "b and c are the discordant pairs)")
    print("\n  %-9s %-14s %-13s %-22s %8s %5s %5s %9s"
          % ("Edition", "Model", "Component", "Comparison", "n", "b", "c", "P"))
    for r in sorted(mcn, key=lambda r: (r["edition"], r["model"], r["component"])):
        b = int(r["only_1_correct"])
        c = int(r["only_2_correct"])
        n = int(r["n_paired"])
        cells_sum = (int(r["both_correct"]) + b + c + int(r["both_incorrect"]))
        if cells_sum != n:
            note("%s/%s/%s %s vs %s: the four cells sum to %d, not n_paired %d"
                 % (r["edition"], r["model"], r["component"],
                    r["configuration_1"], r["configuration_2"], cells_sum, n))
        p = mcnemar_exact(b, c)
        dep_p = float(r["p_exact_mcnemar"])
        if abs(p - dep_p) > max(1e-3 * max(p, 1e-12), 5e-4):
            note("%s/%s/%s %s vs %s: recomputed P %.4g against deposited %.4g"
                 % (r["edition"], r["model"], r["component"],
                    r["configuration_1"], r["configuration_2"], p, dep_p))
        # the two accuracies in this file must agree with the confusion matrices
        for which, cfg in (("accuracy_1_pct", r["configuration_1"]),
                           ("accuracy_2_pct", r["configuration_2"])):
            counts = cells.get((r["edition"], r["model"], cfg, r["component"]))
            if counts and abs(accuracy(counts) - float(r[which])) > 0.1:
                note("%s/%s/%s %s: %s is %.1f, the confusion matrix gives %.1f"
                     % (r["edition"], r["model"], r["component"], cfg, which,
                        float(r[which]), accuracy(counts)))
        print("  %-9s %-14s %-13s %-22s %8d %5d %5d %9.3g"
              % (r["edition"].replace("AJCC", "AJCC "), r["model"],
                 r["component"],
                 "%s vs %s" % (CONFIG_LABEL.get(r["configuration_1"],
                                                r["configuration_1"]),
                               CONFIG_LABEL.get(r["configuration_2"],
                                                r["configuration_2"])),
                 n, b, c, p))

    sig = sum(1 for r in mcn if float(r["p_exact_mcnemar"]) < .05)
    print("\n  %d of the %d comparisons have P<.05, uncorrected for multiplicity."
          % (sig, len(mcn)))

    print("\n" + "=" * 78)
    if PROBLEMS:
        print("the deposited files disagree with each other - %d finding(s):"
              % len(PROBLEMS))
        for m in PROBLEMS:
            print("  - " + m)
        return 1
    print("%d confusion matrices, %d replicate series and %d paired "
          "comparisons are mutually consistent." % (len(cells), len(series), len(mcn)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
