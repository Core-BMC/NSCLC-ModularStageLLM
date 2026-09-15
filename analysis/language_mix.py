"""Language composition of the input documents, under a stated rule.

The manuscript describes the reports as mixed Korean-English. Three things have
to be said explicitly or the figure is not checkable: the unit, the denominator,
and the character class.

  unit          the per-patient document - the concatenation of that patient's
                reports, which is what the model receives
  denominator   non-whitespace characters
  class         Hangul syllables and Jamo

Report-level figures are given alongside, because the two levels differ a great
deal and an earlier version of the manuscript applied a patient-level denominator
to a report-level statement.

Run:  REFERENCE_XLSX=... python3 analysis/language_mix.py
"""
from __future__ import annotations

import os
import re
import statistics

import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
XLSX = os.environ.get("REFERENCE_XLSX", "reference_labels.xlsx")
XLSX = XLSX if os.path.isabs(XLSX) else os.path.join(REPO, "input", XLSX)

REPORT_COLUMNS = ["Pathology", "Chest CT", "Brain MR", "PET", "EBUS",
                  "neck biopsy", "Bone scan", "Abdomen&Pelvis CT", "Adrenal CT"]
HANGUL = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")
NONSPACE = re.compile(r"\S")


def main() -> None:
    ws = openpyxl.load_workbook(XLSX, read_only=True, data_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    header = [str(c).strip() if c else "" for c in rows[0]]
    missing = [c for c in REPORT_COLUMNS if c not in header]
    if missing:
        raise SystemExit(f"columns not found: {missing}")
    idx = {c: header.index(c) for c in REPORT_COLUMNS}

    per_column, reports, no_hangul, shares, patients = {}, 0, 0, [], 0
    for row in rows[1:]:
        if all(row[i] in (None, "") for i in idx.values()):
            continue
        patients += 1
        document = []
        for col, i in idx.items():
            value = row[i]
            if value is None or not str(value).strip():
                continue
            text = str(value)
            reports += 1
            per_column[col] = per_column.get(col, 0) + 1
            if not HANGUL.search(text):
                no_hangul += 1
            document.append(text)
        joined = "".join(document)
        denominator = len(NONSPACE.findall(joined))
        if denominator:
            shares.append(100 * len(HANGUL.findall(joined)) / denominator)

    q1, _, q3 = statistics.quantiles(shares, n=4)
    print(f"patients                          {patients}")
    print(f"non-empty reports                 {reports}")
    for col in REPORT_COLUMNS:
        print(f"  {col:<30s} {per_column.get(col, 0)}")
    print(f"reports containing no Hangul      {no_hangul} "
          f"({100 * no_hangul / reports:.1f}%)")
    print(f"patients whose document has none  {sum(1 for s in shares if s == 0)}")
    print(f"Hangul share per patient document median {statistics.median(shares):.1f}%  "
          f"IQR {q1:.1f}%-{q3:.1f}%  range {min(shares):.1f}%-{max(shares):.1f}%")


if __name__ == "__main__":
    main()
