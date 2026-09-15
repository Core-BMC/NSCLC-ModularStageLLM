"""Compute every cell of Table 2 (cohort description) from auditable sources.

Demographics come from the internal demographic worksheet (identifiers, never
deposited); the histologic distribution from the clinician-verified histology
worksheet, which is case-level and is likewise not deposited; the stage
distributions from the deposited confusion matrices, so that Table 2 and
analysis/deposit/ cannot disagree.

Run:  REFERENCE_XLSX=... DEMOG_XLSX=<path> python3 analysis/make_table2.py
"""
from __future__ import annotations

import csv, json, os, statistics as st, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEP = os.path.join(HERE, "deposit")
RES = os.path.join(HERE, "results")
N = 495

def pct(k, n=N):
    return f"{100 * k / n:.1f}"

out = {"N": N, "rows": []}
add = lambda *a: out["rows"].append(list(a))

# ---- demographics --------------------------------------------------------
dem = os.environ.get("DEMOG_XLSX")
if dem:
    from openpyxl import load_workbook
    ws = load_workbook(dem, data_only=True)["입력"]
    ages, sexes = [], []
    for r in range(3, ws.max_row + 1):
        ages.append(int(ws.cell(row=r, column=4).value))
        sexes.append(str(ws.cell(row=r, column=5).value).strip())
    if len(ages) != N:
        sys.exit(f"demographics has {len(ages)} rows, expected {N}")
    q1, med, q3 = st.quantiles(ages, n=4, method="inclusive")
    add("Age, years — median (IQR)", f"{med:.0f} ({q1:.0f}–{q3:.0f})", "", f"{med:.0f} ({q1:.0f}–{q3:.0f})", "")
    c = Counter(sexes)
    if set(c) - {"M", "F"}:
        sys.exit(f"unexpected sex values: {set(c)}")
    add("Sex — male", str(c['M']), pct(c['M']), str(c['M']), pct(c['M']))
    add("Sex — female", str(c['F']), pct(c['F']), str(c['F']), pct(c['F']))

# ---- histology -----------------------------------------------------------
fin = json.load(open(os.path.join(RES, "histology_final.json")))
if len(fin) != N:
    sys.exit(f"histology has {len(fin)} cases, expected {N}")
h = Counter(v for v in fin.values() if v)
und = sum(1 for v in fin.values() if not v)
ORDER = ["Adenocarcinoma, NOS", "Squamous cell carcinoma, NOS", "Non-small cell carcinoma, NOS",
         "Invasive mucinous adenocarcinoma", "Small cell carcinoma", "Sarcomatoid carcinoma",
         "Adenosquamous carcinoma", "Combined small cell carcinoma", "Carcinoid tumour, NOS",
         "Mucoepidermoid carcinoma", "Large cell neuroendocrine carcinoma"]
if set(ORDER) != set(h):
    sys.exit(f"histology labels changed: {set(h) ^ set(ORDER)}")
for k in ORDER:
    add(f"Histology — {k}", str(h[k]), pct(h[k]), str(h[k]), pct(h[k]))
add("Histology — not determined", str(und), pct(und), str(und), pct(und))
if sum(h.values()) + und != N:
    sys.exit("histology does not sum to N")

# ---- stage distributions, from the deposited matrices --------------------
marg = defaultdict(Counter)
for r in csv.DictReader(open(os.path.join(DEP, "confusion_matrices.csv"))):
    if r["model"] == "GPT-4o" and r["configuration"] == "MAA":
        marg[(r["edition"], r["component"])][r["reference_label"]] += int(r["n"])
CT = ["Tis", "T1mi", "T1a", "T1b", "T1c", "T2a", "T2b", "T3", "T4"]
CN8, CN9 = ["N0", "N1", "N2", "N3"], ["N0", "N1", "N2a", "N2b", "N3"]
CM8, CM9 = ["M0", "M1a", "M1b", "M1c"], ["M0", "M1a", "M1b", "M1c1", "M1c2"]
STG = ["0", "IA1", "IA2", "IA3", "IB", "IIA", "IIB", "IIIA", "IIIB", "IIIC", "IVA", "IVB"]

def row_for(label, key, order, prefix="c"):
    """order is the display order, merging both editions; a label absent from an
    edition leaves that edition's cells blank rather than being moved."""
    a = marg[("AJCC8th", key)]; b = marg[("AJCC9th", key)]
    seen = set(a) | set(b)
    if seen - set(order):
        sys.exit(f"{key}: label not in display order: {seen - set(order)}")
    for lab in order:
        v8, v9 = a.get(lab), b.get(lab)
        if v8 is None and v9 is None:
            continue
        add(f"{label} — {prefix}{lab}",
            "" if v8 is None else str(v8), "" if v8 is None else pct(v8),
            "" if v9 is None else str(v9), "" if v9 is None else pct(v9))

row_for("Clinical T", "cT", CT)
row_for("Clinical N", "cN", ["N0", "N1", "N2", "N2a", "N2b", "N3"])
row_for("Clinical M", "cM", ["M0", "M1a", "M1b", "M1c", "M1c1", "M1c2"])
row_for("Stage group", "stage group", STG, prefix="")

for key, labs in (("cT", CT), ("stage group", STG)):
    for ed in ("AJCC8th", "AJCC9th"):
        if sum(marg[(ed, key)].values()) != N:
            sys.exit(f"{ed} {key} does not sum to {N}")

json.dump(out, open(os.path.join(RES, "table2.json"), "w"), ensure_ascii=False, indent=1)
w = max(len(r[0]) for r in out["rows"])
print(f"{'Characteristic':<{w}}  {'8th N':>6} {'%':>6}   {'9th N':>6} {'%':>6}")
for r in out["rows"]:
    print(f"{r[0]:<{w}}  {r[1]:>6} {r[2]:>6}   {r[3]:>6} {r[4]:>6}")
print(f"\n{len(out['rows'])} rows -> {RES}/table2.json")
