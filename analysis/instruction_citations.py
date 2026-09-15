"""Count how often the models quoted the 9th-edition revision instruction.

Comment 1 of the second round concerns an instruction that was present in the N
and M agent prompts of the AJCC 9th edition only, and within that edition only in
the modular configuration. The prompt text labels that instruction, so a stored
reasoning containing the label is unambiguous evidence that the model read it.

The rule is deliberately strict: only the label is matched, never a paraphrase. A
looser pattern that also counts phrases the model could have produced on its own
("from imaging alone", "modality-prioritization rules") gives counts several
times larger, and those counts cannot distinguish reading the instruction from
describing the same clinical reasoning independently.

Run:  JSON_DIRS=... python3 analysis/instruction_citations.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
JSON_DIRS = [os.path.join(REPO, d) for d in
             os.environ.get("JSON_DIRS", "output/json:output").split(":") if d]

LABEL = re.compile(r"REVISION 2026|Code review 2[56]")
RUN_RE = re.compile(r"(llama3-70b|phi4|gpt4o)-(mma|base|deconly)-ajcc([89])")
MODEL = {"gpt4o": "GPT-4o", "llama3-70b": "LLaMA 3.3 70B", "phi4": "Phi-4 14B"}
MODE = {"base": "single prompt", "deconly": "decomposition only", "mma": "MAA"}


def main() -> None:
    rows = []
    for d in JSON_DIRS:
        for fp in sorted(glob.glob(os.path.join(d, "*.json"))):
            base = os.path.basename(fp)
            if "repro" in base.lower():
                continue
            m = RUN_RE.search(base)
            if not m:
                continue
            model, mode, ed = m.group(1), m.group(2), m.group(3)
            with open(fp) as fh:
                data = json.load(fh)
            hits, seen = Counter(), 0
            for rec in data:
                r = rec.get("reasonings")
                if not isinstance(r, dict):
                    continue
                for k, v in r.items():
                    if k.upper() not in ("N", "M"):
                        continue
                    seen += 1
                    if v and LABEL.search(str(v)):
                        hits[k.upper()] += 1
            rows.append((f"AJCC {ed}th", MODEL[model], MODE[mode],
                         hits["N"], hits["M"], sum(hits.values()), seen))
    if not rows:
        sys.exit("no run files found - set JSON_DIRS")
    rows.sort(key=lambda r: (r[0], r[2], r[1]))
    print(f"{'Edition':9s} {'Model':15s} {'Configuration':19s} {'N':>4s} {'M':>4s} "
          f"{'total':>6s} {'of':>6s}")
    for r in rows:
        print(f"{r[0]:9s} {r[1]:15s} {r[2]:19s} {r[3]:4d} {r[4]:4d} {r[5]:6d} {r[6]:6d}")
    bad = [r for r in rows if r[5] and not (r[0] == "AJCC 9th" and r[2] == "MAA")]
    print()
    print("runs whose prompts did NOT contain the instruction but whose outputs quote it: "
          f"{len(bad)}")
    if bad:
        for r in bad:
            print("  !", r)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
