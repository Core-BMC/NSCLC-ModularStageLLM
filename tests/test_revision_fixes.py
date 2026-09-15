"""Checks for the corrections made in the 2026 revision.

Runnable either with pytest or directly::

    python3 tests/test_revision_fixes.py

Covers three behaviours that were reported as defects:

1. Unparseable N/M output is no longer coerced to the majority class.
2. The AJCC 9th-edition stage tables cover Tis and T1mi.
3. Input column lookup is case-insensitive.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.utils.data_utils import _report_field  # noqa: E402
from src.utils.stage_utils import determine_stage_from_tnm  # noqa: E402
from src.utils.tnm_extraction_utils import (  # noqa: E402
    extract_m_classification,
    extract_n_classification,
)

with open(os.path.join(REPO, "config", "tnm_config.yaml"), encoding="utf-8") as _fh:
    _CONFIG = yaml.safe_load(_fh)
RULES_8TH = _CONFIG["stage_rules"]
RULES_9TH = _CONFIG["ajcc9th_prompts"]["stage_rules"]


def test_unparseable_n_is_not_coerced_to_n0() -> None:
    """An N response matching no pattern yields no result, not N0."""
    assert extract_n_classification("no staging content here at all") is None


def test_unparseable_m_is_not_coerced_to_m0() -> None:
    """An M response matching no pattern yields no result, not M0."""
    assert extract_m_classification("no staging content here at all") is None


def test_recognised_categories_still_parse() -> None:
    """The change must not disturb responses that do match a pattern."""
    assert extract_n_classification(
        "no evidence of lymph node involvement"
    ) == {"classification": "N0"}
    assert extract_m_classification(
        "solitary brain metastasis"
    ) == {"classification": "M1b"}


def test_ninth_edition_covers_minimally_invasive_tumours() -> None:
    """T1mi and Tis map to the 9th-edition stage groups, not the T1_T2 default."""
    assert determine_stage_from_tnm("T1mi", "N1", "M0", RULES_9TH) == "IIA"
    assert determine_stage_from_tnm("T1mi", "N2a", "M0", RULES_9TH) == "IIB"
    assert determine_stage_from_tnm("T1mi", "N2b", "M0", RULES_9TH) == "IIIA"
    assert determine_stage_from_tnm("Tis", "N1", "M0", RULES_9TH) == "IIA"
    # T1mi must not be staged above the larger T1a in the same nodal category.
    assert determine_stage_from_tnm(
        "T1mi", "N1", "M0", RULES_9TH
    ) == determine_stage_from_tnm("T1a", "N1", "M0", RULES_9TH)


def test_eighth_edition_mapping_unchanged() -> None:
    """The 8th-edition tables collapse T1/T2 with N1/N2, so behaviour is stable."""
    assert determine_stage_from_tnm("T1mi", "N1", "M0", RULES_8TH) == "IIB"
    assert determine_stage_from_tnm("T1mi", "N2", "M0", RULES_8TH) == "IIIA"
    assert determine_stage_from_tnm("Tis", "N0", "M0", RULES_8TH) == "0"


def test_t0_is_not_an_allowed_output() -> None:
    """No prompt offers T0, which has no AJCC stage group with N0 M0."""
    for section in ("ajcc8th_prompts", "ajcc9th_prompts"):
        for key in ("t_classifier", "t_classifier_base", "tnm_classifier_base"):
            prompt = _CONFIG.get(section, {}).get(key, "")
            assert "ONE of: T0" not in prompt, f"{section}.{key} still offers T0"


def test_column_lookup_is_case_insensitive() -> None:
    """Either the README spelling or the template spelling resolves."""
    for column in ("Neck biopsy", "neck biopsy", " NECK BIOPSY "):
        row = pd.Series({"Pathology": "p", column: "neck report"})
        assert _report_field(row, "neck biopsy") is not None, column
    assert _report_field(pd.Series({"Pathology": "p"}), "neck biopsy") is None


if __name__ == "__main__":
    failures = 0
    for name, func in sorted(globals().items()):
        if not name.startswith("test_") or not callable(func):
            continue
        try:
            func()
            print(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
