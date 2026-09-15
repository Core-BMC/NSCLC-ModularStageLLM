"""Build the per-case prediction file (JMIR #101332, editorial comment 6).

INTERNAL USE ONLY - THE OUTPUT OF THIS SCRIPT IS NOT DISTRIBUTED.
Institutional data governance treats a case-level record derived from this
cohort as patient data whether or not it carries the reference label, so
neither this file nor a predictions-only variant of it may be released. It is
built for internal verification, and `analysis/results/` is untracked. The
tables that are released are aggregate and are built by
`make_aggregate_deposit.py`.

The deposit carries labels only: an arbitrary sequential case index, the AJCC
edition, the model, the configuration, and the reference and predicted cT, cN, cM
and stage group. Nothing else from the run JSON is copied — in particular no
report text, no model reasoning, no raw model output and no hospital identifier.

That is enforced rather than assumed: every value written is checked against the
closed label vocabularies below, and the run JSON keys that carry free text are
listed explicitly so that adding one to the output would fail the check.

Run:  JSON_DIRS=... REFERENCE_XLSX=... python3 analysis/make_deposit.py
Outputs -> analysis/results/per_case_predictions.csv  (+ a verification report)
"""
from __future__ import annotations

import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_results as A  # noqa: E402

# Closed vocabularies: every value the deposit may contain. Anything else is a
# defect in this script, not a label, and aborts the build.
#
# Three groups are deliberately admitted alongside the canonical categories:
#   - Tx/Nx/Mx: the model judged the component indeterminate. This is a clinical
#     category, restored in the first revision when the earlier coercion of
#     TX/NX/MX to 0 was removed.
#   - "Unknown": the stage group could not be derived because a component was
#     indeterminate or unparseable.
#   - the umbrella categories T1, T2, N2 and M1c, and T0: outputs that are not
#     valid answers under the edition in question. They are kept in the deposit
#     because they are what the models produced, and counted in the report below.
CANON = {
    "cT": {"Tis", "T1mi", "T1a", "T1b", "T1c", "T2a", "T2b", "T3", "T4"},
    "cN": {"N0", "N1", "N2", "N2a", "N2b", "N3"},
    "cM": {"M0", "M1a", "M1b", "M1c", "M1c1", "M1c2"},
    "stage": {"0", "IA1", "IA2", "IA3", "IB", "IIA", "IIB",
              "IIIA", "IIIB", "IIIC", "IVA", "IVB"},
}
INDETERMINATE = {"cT": {"Tx", "TX"}, "cN": {"Nx", "NX"}, "cM": {"Mx", "MX"},
                 "stage": {"Unknown"}}
NON_CANONICAL = {"cT": {"T0", "T1", "T2"}, "cN": {"N2"}, "cM": {"M1c"}, "stage": set()}
# Which values are answers depends on the edition: bare N2 and M1c are valid under
# the 8th edition and are umbrella categories under the 9th, where the subcategory
# (N2a/N2b, M1c1/M1c2) is required.
VALID_BY_EDITION = {
    "8": {"cT": CANON["cT"], "cN": {"N0", "N1", "N2", "N3"},
          "cM": {"M0", "M1a", "M1b", "M1c"}, "stage": CANON["stage"]},
    "9": {"cT": CANON["cT"], "cN": {"N0", "N1", "N2a", "N2b", "N3"},
          "cM": {"M0", "M1a", "M1b", "M1c1", "M1c2"}, "stage": CANON["stage"]},
}
VOCAB = {k: CANON[k] | INDETERMINATE[k] | NON_CANONICAL[k] | {""} for k in CANON}
LABEL_COLS = ["ref_cT", "ref_cN", "ref_cM", "ref_stage",
              "pred_cT", "pred_cN", "pred_cM", "pred_stage"]

# Keys in the run JSON that carry free text or identifiers; never copied.
FORBIDDEN_KEYS = ("hospitalNumber", "reports", "reasonings", "raw_outputs", "histology")

HANGUL = re.compile(r"[가-힣]")


def main() -> None:
    gt = A.load_gt()
    runs = A.discover()
    out_path = os.path.join(A.OUT, "per_case_predictions.csv")
    rows = 0
    bad: list[str] = []
    noted = {"indeterminate": 0, "non_canonical": 0}

    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "edition", "model", "configuration",
                    "ref_cT", "ref_cN", "ref_cM", "ref_stage",
                    "pred_cT", "pred_cN", "pred_cM", "pred_stage"])
        for r in sorted(runs, key=lambda x: (x["ed"], x["model"], A.MODE_ORDER.index(x["mode"]))):
            for rec, g in zip(r["data"], gt):
                ai = rec.get("aiTnm") or {}
                ref = g[r["ed"]]
                vals = {
                    "ref_cT": ref["T"], "ref_cN": ref["N"], "ref_cM": ref["M"], "ref_stage": ref["S"],
                    "pred_cT": ai.get("T", "") or "", "pred_cN": ai.get("N", "") or "",
                    "pred_cM": ai.get("M", "") or "", "pred_stage": ai.get("Stage", "") or "",
                }
                for k, v in vals.items():
                    axis = k.split("_", 1)[1]
                    v = str(v).strip()
                    if v not in VOCAB[axis]:
                        bad.append(f"{r['file']} case {rec.get('id')} {k}={v!r}")
                    elif k.startswith("pred_"):
                        # For the 9th edition the umbrella categories are not answers.
                        if v in INDETERMINATE[axis]:
                            noted["indeterminate"] += 1
                        elif v and v not in VALID_BY_EDITION[r["ed"]][axis]:
                            noted["non_canonical"] += 1
                            noted.setdefault("detail", []).append(
                                f"AJCC{r['ed']}th {A.MODEL_LABEL[r['model']]} "
                                f"{A.MODE_LABEL[r['mode']]} case {rec.get('id')} {k}={v}")
                    vals[k] = v
                w.writerow([f"case_{rec.get('id'):03d}", f"AJCC{r['ed']}th",
                            A.MODEL_LABEL[r["model"]], A.MODE_LABEL[r["mode"]],
                            vals["ref_cT"], vals["ref_cN"], vals["ref_cM"], vals["ref_stage"],
                            vals["pred_cT"], vals["pred_cN"], vals["pred_cM"], vals["pred_stage"]])
                rows += 1

    # ---- verification on the written file, not on the intermediate objects ----
    with open(out_path) as fh:
        text = fh.read()
    problems = []
    if HANGUL.search(text):
        problems.append("Korean characters present (report text would contain them)")
    for k in FORBIDDEN_KEYS:
        if k in text:
            problems.append(f"forbidden key name {k!r} present")
    # Length is checked on the label columns only; the metadata columns carry
    # configuration names such as "decomposition-only".
    lines = [l.split(",") for l in text.splitlines()]
    head = lines[0]
    cols = [head.index(c) for c in LABEL_COLS]
    longest = max(len(l[i]) for l in lines[1:] for i in cols)
    if longest > 8:
        problems.append(f"a label field is {longest} characters long; labels are short")
    if bad:
        problems.append(f"{len(bad)} value(s) outside the label vocabulary: {bad[:5]}")

    print(f"rows written : {rows}  ({len(runs)} runs x {len(gt)} cases)")
    print(f"file         : {out_path}  ({os.path.getsize(out_path):,} bytes)")
    print(f"columns      : 12, labels only")
    print(f"longest label: {longest} characters")
    print(f"indeterminate predictions (Tx/Nx/Mx, Unknown stage): {noted['indeterminate']}")
    print(f"non-canonical predictions (T0, or an umbrella category "
          f"where the edition requires a subcategory): {noted['non_canonical']}")
    for d in noted.get("detail", []):
        print("   - " + d)
    print("verification : " + ("PASS - no report text, no identifiers, all values in vocabulary"
                               if not problems else "FAIL"))
    for p in problems:
        print("   ! " + p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
