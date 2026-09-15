"""Regression guard: a bootstrap interval must not depend on what else was run.

The intervals reported in the first revision were computed from a single
module-level generator consumed sequentially, so adding the six decomposition-only
runs shifted the interval of every cell computed after them while leaving the
point estimates identical. Nothing failed and nothing warned. This test fails if
that coupling is ever reintroduced.

Run:  REFERENCE_XLSX=... JSON_DIRS=... python3 analysis/test_ci_stability.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_results as A


def ci_table(runs):
    gt = A.load_gt()
    out = {}
    for r in runs:
        for ax in ("T", "N", "M", "S"):
            key = f"{r['ed']}|{r['model']}|{r['mode']}|{ax}"
            p, t = A.preds(r, ax), A.gts(gt, r, ax)
            out[key] = (A.accuracy_ci(p, t, key), A.macro_f1_ci(p, t, key))
    return out


def main() -> None:
    runs = A.discover()
    if not runs:
        sys.exit("no runs discovered - set JSON_DIRS")
    full = ci_table(runs)
    subset_runs = [r for r in runs if r["mode"] != "deconly"]
    if not subset_runs or len(subset_runs) == len(runs):
        sys.exit("need both agent and decomposition-only runs present to test")
    subset = ci_table(subset_runs)

    bad = [k for k in subset if full[k] != subset[k]]
    print(f"cells compared        : {len(subset)}")
    print(f"runs, full vs subset  : {len(runs)} vs {len(subset_runs)}")
    print(f"intervals that moved  : {len(bad)}")
    for k in bad[:10]:
        print(f"   ! {k}\n     full   {full[k]}\n     subset {subset[k]}")

    # A second call with the same key must also be identical - no hidden state.
    again = ci_table(subset_runs)
    repeat = [k for k in again if again[k] != subset[k]]
    print(f"non-deterministic     : {len(repeat)}")

    if bad or repeat:
        sys.exit(f"FAIL - {len(bad)} interval(s) depend on the run set, "
                 f"{len(repeat)} not reproducible on repeat")
    print("PASS - every interval is a function of its own cell only")


if __name__ == "__main__":
    main()
