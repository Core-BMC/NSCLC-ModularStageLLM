"""Parse-path audit for JMIR #101332 — editorial comment 7.

The Editor asked for the JSON parse-failure rate and the regex fallback
invocation rate separately, by component and configuration, and noted that for N
and M the fallback path could not fail in the code as released, because both
extractors terminated by returning the majority class. The rate therefore cannot
be recovered from the reported runs by inspecting their outputs alone.

It can be recovered by replay. Every run stores the model's raw output per
component, so this script re-runs the shipped parsers over those stored outputs
and records which path produced each result:

    json      - a JSON object carrying "classification" was recovered
    regex     - JSON parsing failed and the regex extractor produced a label
    none      - neither produced a label; the deposited matrices record it as
                (no label), which is not the same event as an undeducible stage
                group (Unknown). Before the parser correction such an output was
                silently returned as N0 or M0.

The parsers are imported from src/, not reimplemented, so the audit reflects the
code as released rather than a description of it.

Run:  JSON_DIRS=... REFERENCE_XLSX=... python3 analysis/parse_audit.py
Outputs -> analysis/results/parse_audit.{md,csv}
"""
from __future__ import annotations

import csv
import logging
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import analyze_results as A  # noqa: E402
from src.parsers.tnm_parsers import (  # noqa: E402
    TNM_T_Parser, TNM_N_Parser, TNM_M_Parser, TNMOutputParser,
)
import json as _json  # noqa: E402
import re as _re  # noqa: E402

logging.disable(logging.CRITICAL)   # the parsers log every attempt

PARSERS = {"T": TNM_T_Parser(), "N": TNM_N_Parser(), "M": TNM_M_Parser()}
COMBINED = TNMOutputParser()


def classify_path(parser, text: str) -> str:
    """Which path yields a label for this raw output: json, regex or none."""
    if not text:
        return "none"
    try:
        if parser._try_json_parsing(text):
            return "json"
    except Exception:
        pass
    try:
        if parser._try_extract(text):
            return "regex"
    except Exception:
        pass
    return "none"


def classify_combined(text: str) -> str:
    """Path taken by the single-prompt parser on one combined response.

    The single-prompt configuration emits one response carrying t_/n_/m_classification
    and is parsed once by TNMOutputParser, not three times by the component parsers.
    Auditing it with the component parsers would be a description of a pipeline that
    does not exist, so it is replayed with the parser the pipeline actually uses.
    """
    if not text:
        return "none"
    t = _re.sub(r"```json\s*", "", str(text).strip())
    t = _re.sub(r"\s*```", "", t)
    m = _re.search(r"{[^{}]*}", t)
    if m:
        try:
            data = _json.loads(m.group(0))
            COMBINED._validate_data(data)
            return "json"
        except Exception:
            pass
    try:
        COMBINED.parse(text)
        return "regex"
    except Exception:
        return "none"


def main() -> None:
    runs = A.discover()
    rows = []
    for r in sorted(runs, key=lambda x: (x["ed"], x["model"], A.MODE_ORDER.index(x["mode"]))):
        counts = {ax: Counter() for ax in ("T", "N", "M", "combined")}
        present = Counter()
        baseline = r["mode"] == "base"
        for rec in r["data"]:
            raw = rec.get("raw_outputs") or {}
            if baseline:
                txt = raw.get("T")               # all three keys hold the same response
                present["combined"] += 1
                counts["combined"][classify_combined(txt)] += 1
                continue
            for ax in ("T", "N", "M"):
                txt = raw.get(ax)
                if txt is None:
                    continue                      # component not run in this configuration
                present[ax] += 1
                counts[ax][classify_path(PARSERS[ax], str(txt))] += 1
        for ax in (("combined",) if baseline else ("T", "N", "M")):
            if not present[ax]:
                continue
            c = counts[ax]
            rows.append({
                "edition": f"AJCC{r['ed']}th",
                "model": A.MODEL_LABEL[r["model"]],
                "configuration": A.MODE_LABEL[r["mode"]],
                "component": "combined (cT+cN+cM)" if ax == "combined" else f"c{ax}",
                "n_outputs": present[ax],
                "json": c["json"], "regex": c["regex"], "none": c["none"],
                "json_fail_pct": round(100 * (c["regex"] + c["none"]) / present[ax], 2),
                "regex_pct": round(100 * c["regex"] / present[ax], 2),
                "none_pct": round(100 * c["none"] / present[ax], 2),
            })

    os.makedirs(A.OUT, exist_ok=True)
    csv_path = os.path.join(A.OUT, "parse_audit.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md_path = os.path.join(A.OUT, "parse_audit.md")
    with open(md_path, "w") as fh:
        fh.write("# Parse-path audit (replay of the shipped parsers over the stored raw outputs)\n\n")
        fh.write("json = JSON object recovered; regex = JSON failed, regex extractor produced a label; ")
        fh.write("none = neither; recorded as (no label), not as Unknown.\n\n")
        fh.write("| Edition | Model | Configuration | Component | n | json | regex | none | JSON-fail % | regex % | none % |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for d in rows:
            fh.write("| {edition} | {model} | {configuration} | {component} | {n_outputs} | {json} | "
                     "{regex} | {none} | {json_fail_pct} | {regex_pct} | {none_pct} |\n".format(**d))

    tot = Counter()
    for d in rows:
        for k in ("n_outputs", "json", "regex", "none"):
            tot[k] += d[k]
    print(f"component outputs audited : {tot['n_outputs']:,}")
    print(f"  JSON path                : {tot['json']:,} ({100*tot['json']/tot['n_outputs']:.2f}%)")
    print(f"  regex fallback           : {tot['regex']:,} ({100*tot['regex']/tot['n_outputs']:.2f}%)")
    print(f"  neither                  : {tot['none']:,} ({100*tot['none']/tot['n_outputs']:.2f}%)")
    print(f"\nWrote {md_path} and {csv_path}")


if __name__ == "__main__":
    main()
