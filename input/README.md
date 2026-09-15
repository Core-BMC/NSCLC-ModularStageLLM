# Synthetic input examples

These examples were created solely to demonstrate the input format. They are entirely fictional and are not derived from patient records.

- synthetic_single_excel_ajcc_8th.xlsx: SYNTHETIC_AJCC_8TH_001.
- synthetic_single_excel_ajcc_9th.xlsx: SYNTHETIC_AJCC_9TH_001.

Created on 2026-09-15 from the literal fictional text in generate_synthetic_examples.py, which reads no patient files. Both examples describe a solitary solid nodule greater than 1 cm and no greater than 2 cm, with no nodal or distant spread, and illustrative T1b/N0/M0/IA2 labels. They demonstrate input parsing, not clinical validation or measured model performance. Blank optional fields mean no report was supplied.

Run python input/generate_synthetic_examples.py from the repository root to regenerate the files. These fictional rows are separate from the three aggregate evaluation files described in [the analysis README](../analysis/README.md).
