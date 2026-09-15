# Analysis and scoring scripts

## Local context window and output limit

The local models were served with a context window of 8,192 tokens (num_ctx=8192), separate from the output generation cap of max_tokens=2048. The ctx8k model names in the preserved run configurations are local deployment aliases. Whether any evaluation inputs were truncated at the context limit has not been established; successful parsing of stored outputs does not demonstrate that the complete input was processed. This disclosure does not change the reported results or establish an effect of truncation on performance.

Scoring, statistics and figure generation for the evaluation reported in the
accompanying manuscript (JMIR Medical Informatics #101332). The inference
pipeline lives in `../src`; these scripts consume its outputs and produce the
reported tables and figures.

## Inputs

| Path | Contents | Distributed? |
|---|---|---|
| `../output/json/*.json` | One file per run (model x configuration x AJCC edition), per-case predictions | produced by running the pipeline |
| `../output/*.json` | Decomposition-only ablation runs written by `config/runs/run_deconly_matrix.sh` | produced by running the pipeline |
| `../output/repro/*.json` | Repeat runs on the fixed 50-case subset, for the reproducibility analysis | produced by running the pipeline |
| `../input/reference_labels.xlsx` | Reference-standard labels: case identifier and cT/cN/cM/stage group for both editions | **no** — see below |

The reference-label file is held under institutional data governance and is not
distributed with the repository. Point the scripts at your own file by setting
the `REFERENCE_XLSX` environment variable, or place a file with the expected
columns at `../input/reference_labels.xlsx`. Expected columns:

```
No., 8th-cT, 8th-cN, 8th-cM, 8th-cStage, 9th-cT, 9th-cN, 9th-cM, 9th-cStage
```

Predictions are aligned to the reference by the JSON `id` field (1..N), which
follows the data-row order of the input workbook. Empty or unparseable
predictions are scored as incorrect.

## Scripts

| Script | Produces |
|---|---|
| `analyze_results.py` | Per-component and stage accuracy with bootstrap 95% CI, macro-F1, McNemar tests, confusion matrices (Table 4, Table 5, Figure 3 inputs) |
| `compute_sig.py` | Paired single-prompt-vs-agent significance per model x edition x component (McNemar exact for accuracy; paired bootstrap for macro-F1) |
| `make_figures.py` | Figure 2 (performance) and Figure 3 (stage confusion). Configuration A is labelled "single prompt" in both, matching the manuscript; the word "baseline" survives only as an internal dictionary key |
| `make_figure2_main.py` | Alternative Figure 2 layout (radar + delta bars) |
| `make_repro_subset.py` | Builds the fixed 50-case reproducibility subset (seed 42) |
| `make_bootstrap_deposit.py` | Writes `deposit/bootstrap_replicates.csv` and verifies the reported intervals fall out of it |
| `analyze_repro.py` | Run-to-run label agreement and accuracy mean +/- SD across repeats |
| `parse_audit.py` | Replays the shipped parsers over the stored raw model outputs and reports the JSON-parse and regex-fallback rates by component and configuration |
| `make_table2.py` | Computes every cell of Table 2: demographics from the internal demographic worksheet, the histologic distribution from the clinician-verified histology worksheet, and the stage distributions from `deposit/confusion_matrices.csv` so that Table 2 cannot disagree with the deposit |
| `make_aggregate_deposit.py` | **Deposit tables**: confusion matrices and McNemar 2x2 contingency counts (see Deposit below) |
| `recompute_tables.py` | Recomputes Tables 4, 5 and 6 from `deposit/` alone and prints them, and checks the three deposited files against each other; exits non-zero if they disagree. Standard library only, no inputs beyond `deposit/` |
| `make_appendices.py` | Builds the multimedia appendix workbooks (component-level paired comparisons, per-class metrics, the parse audit, the exclusion cohorts) from `deposit/` and `results*/` |
| `instruction_citations.py` | Counts the stored N and M reasonings that quote the label of the AJCC 9th-edition revision instruction. The rule is strict - only the label is matched, never a paraphrase - and the script exits non-zero if a run whose prompt did not contain the instruction quotes it anyway |
| `language_mix.py` | Language composition of the 2,273 staging reports: how many contain no Hangul, and the per-patient Hangul share (median and IQR) |
| `test_ci_stability.py` | Regression guard: fails if a bootstrap interval depends on which other runs were analysed |
| `make_deposit.py` | Builder for the per-case prediction file. The script is published so that its contents can be inspected; **its output is not distributed** - see Deposit below |

## Deposit

Institutional data governance treats any case-level record derived from this
cohort as patient data, whether or not it carries the reference label, so no
per-case file is released. `make_deposit.py` builds one for internal
verification and its output stays inside the institution; `analysis/results/`
is untracked for the same reason.

What is released is aggregate, in `analysis/deposit/`:

| File | Contents |
|---|---|
| `confusion_matrices.csv` | reference label x predicted label counts for every edition, model, configuration and component |
| `mcnemar_contingency.csv` | paired 2x2 discordance counts and the exact P value for every pair of configurations |
| `bootstrap_replicates.csv` | the 1,000 resample values behind each reported interval, for accuracy and macro-F1 (144 series, 144,000 rows) |

### Aggregate data and patient-level data

Consistent with the institutional data-sharing policy described in our response to the Editor, the three deposited files contain aggregate data only. They do not include individual patient records, identifiers, or patient-by-patient reference and prediction data; releasing predictions alone would still constitute sharing patient-level data.

In `bootstrap_replicates.csv`, each row is an accuracy or macro-F1 statistic calculated for a bootstrap resample, not a patient record. The 1,000 values for each metric support recalculation of its percentile confidence interval. `confusion_matrices.csv` provides aggregate counts for recalculating accuracy and macro-F1, and `mcnemar_contingency.csv` provides paired discordance counts for recalculating exact McNemar P values. Together, these files allow verification of the reported tabulated statistics without access to individual patient data. They do not provide the patient-level inputs needed to rerun inference or independently regenerate the bootstrap samples.

Run `python3 analysis/recompute_tables.py` from the repository root to recompute Tables 4 to 6 from these three files.

### Label vocabulary in the deposited tables

Alongside the AJCC categories, the `predicted_label` column can carry the
following. All are scored as incorrect.

| Value | Meaning |
|---|---|
| `Tx`, `Nx`, `Mx` | the model judged the component indeterminate. A clinical category, not a parse failure |
| `Unknown` | the stage group could not be derived because a component was indeterminate or unparseable |
| `(no label)` | neither the JSON path nor the regex fallback produced a label for this component. Four cells, all from one case (Phi-4 14B, AJCC 9th, single prompt); see the parse audit under `parse_audit.py` |
| `T0` | emitted by the model although no AJCC stage group exists for T0 N0 M0. Removed from the permitted set in this revision; retained here because it is what the reported runs produced |
| bare `T1`, `T2` | an umbrella category where the edition requires a subcategory |
| bare `N2`, `M1c` under `AJCC9th` | 8th-edition categories emitted under a 9th-edition prompt, which requires `N2a`/`N2b` and `M1c1`/`M1c2`. Model behaviour, not a data defect; the `reference_label` column is edition-appropriate throughout |

`reference_label` carries only categories valid for its edition.

These are sufficient to recompute the reported point estimates: a confusion
matrix determines accuracy (trace / total) and macro-F1 (per-class TP, FP and
FN) exactly, and the exact binomial McNemar test depends only on the two
discordance counts. `make_aggregate_deposit.py` verifies this on the file it
has just written - it recomputes the accuracy of all 72 matrices from their own
cells and fails if any disagrees with `analyze_results.py`.

The confusion matrices give the point estimates. The bootstrap confidence
intervals cannot be recomputed from a matrix, because resampling is over cases,
so the replicate values themselves are deposited: `bootstrap_replicates.csv`
carries the 1,000 resample values behind each of the 144 reported intervals, and
taking the 2.5th and 97.5th percentiles of a series reproduces the interval
exactly (`make_bootstrap_deposit.py` verifies all 144 when it writes them, and
`recompute_tables.py` recomputes them from the deposited values).

Those values disclose nothing beyond the matrices. A replicate accuracy is the
number of correct cases in a resample over n; the resample indices are not
written, the correctness indicators are 0/1, and resampling is uniform with
replacement, so the distribution of a series depends only on the number correct
- the trace of a matrix already deposited. No row corresponds to a patient.

They do allow the intervals to be checked, if not reproduced. A closed-form
Wilson binomial interval computed from the cell counts of a matrix agrees with
the reported bootstrap interval to within 0.6 percentage points across all
seventy-two cells of Table 4. The largest deviation is 0.51 pp against the
intervals as printed in Table 4 and 0.54 pp against the unrounded percentiles of
`bootstrap_replicates.csv`:

| Run | Reported (bootstrap) | Wilson, from this deposit |
|---|---|---|
| 8th, GPT-4o, MAA, stage group | 70.9-78.6 | 70.7-78.4 |
| 8th, GPT-4o, single prompt, stage group | 66.5-74.6 | 66.3-74.4 |
| 9th, Phi-4 14B, MAA, stage group | 31.1-39.6 | 31.1-39.5 |
| 8th, LLaMA 3.3 70B, MAA, cM | 88.3-93.3 | 88.1-93.1 |
| 9th, GPT-4o, MAA, cT | 87.5-92.7 | 87.2-92.4 |

The reported intervals are percentile bootstrap intervals over 1,000 resamples
of the cases with a fixed seed; the Wilson interval is a different estimator,
so exact agreement is not expected and none is claimed.

One property of the bootstrap should be stated, because it changed between
revisions. Until this revision `RNG` was a single generator seeded once at
import and consumed sequentially across runs and components. An interval
therefore depended on how many cells had been computed before it, and adding the
six decomposition-only runs shifted the intervals of cells whose data had not
changed - silently, since no point estimate moved and nothing failed.

`cell_rng()` now seeds a generator from the identity of the cell
(`edition|model|mode|axis`, plus the statistic and `SEED`). An interval is a
function of that key, `SEED` and `N_BOOT` and of nothing else: reproducible in
isolation, and unchanged when runs are added or the analysis is restricted to a
subset. `test_ci_stability.py` enforces this - it recomputes every interval from
the full run set and again from a subset and fails on any difference.

Point estimates are unaffected by the change; interval bounds moved by up to
1.0 percentage point for accuracy and 1.4 for macro-F1 relative to the
first-revision tables.

The reference-label margins of these matrices disaggregate the categories
reported in Table 2 of the manuscript - cT1 into T1mi/T1a/T1b/T1c, cT2 into
T2a/T2b, 9th-edition cN2 into N2a/N2b and cM1c into M1c1/M1c2, and the five
stage groups into their twelve subgroups. The margins are identical across all
nine runs of an edition and roll up exactly to Table 2.

```bash
python3 analysis/make_aggregate_deposit.py     # -> analysis/deposit/
```

Outputs are written to `analysis/results/`.

`analyze_results.py` discovers runs in `../output/json` and `../output` (override with
`JSON_DIRS`, colon-separated, relative to the repository root) and recognises three
conditions from the file name: `base` (single prompt), `deconly` (decomposition only)
and `mma` (full modular agent architecture). `mcnemar.md` reports every pair of
conditions present for a given model and edition, on the cases scored in both runs, so
an incomplete run narrows its own comparisons instead of being scored against the full
set. For an interim analysis while some runs are still executing, restrict the scope
with `MODELS=gpt4o,phi4` and/or `EDITIONS=8,9`; runs shorter than the reference set are
flagged `<< INCOMPLETE` in the console listing. `NO_FIGURES=1` (or a host without
matplotlib) writes the tables and confusion CSVs and skips the PNGs.

Example, analysing on the inference host where the reported runs sit in a nested
directory and the label file is kept outside the repository:

```bash
JSON_DIRS="output/revised_experiments_202607/ajcc-8:output/revised_experiments_202607/ajcc-9:output" \
REFERENCE_XLSX="$HOME/nsclc_labels/reference_labels.xlsx" \
NO_FIGURES=1 .venv/bin/python analysis/analyze_results.py
```

## Exclusion cohorts

`analyze_results.py` accepts `EXCLUDE_CASES` - a comma-separated list of case
identifiers, or `@` followed by the path of a file holding them - to drop
before scoring, and `RESULTS_DIR` to write the outputs elsewhere. The two
histology sensitivity analyses reported in the manuscript were produced this way:

```bash
EXCLUDE_CASES=@analysis/results/exclude_nonnsclc_narrow.txt \
  RESULTS_DIR="$PWD/analysis/results_excl_narrow" python3 analysis/analyze_results.py   # N=480
EXCLUDE_CASES=@analysis/results/exclude_nonnsclc_broad.txt \
  RESULTS_DIR="$PWD/analysis/results_excl_broad"  python3 analysis/analyze_results.py   # N=477
```

`RESULTS_DIR` is resolved relative to `analysis/`, not to the working directory,
so give it as an absolute path or it lands in `analysis/analysis/...`.

### The reference file has two numberings. Join on row order, never on `No.`

`analyze_results.py` aligns each run record to its reference labels by
`gt[rec["id"] - 1]` — that is, by the reference worksheet's **row order**, with
the `id` carried in the run output as a 1-based row index. The worksheet also has
a column called `No.`, and it is a *different* permutation: the patient at row 214
carries `No.` 249. Joining on `No.` therefore silently pairs predictions with the
wrong patient, and re-sorting the worksheet would change every reported number
without failing anything.

`analyze_results.py` aligns all 495 records against the
`hospital_id` in each run output, and asserts that `No.` is *not* the row order,
so that if the file is ever re-sorted or the two are ever conflated the test fails
first. Those `id` values are analysis row indices; they are not published — no
case identifier appears in the manuscript, the appendices or the deposited
tables.

The two case lists follow from the histologic classification of the cohort,
which is not derived here: the thoracic oncologists established it from the
pathology reports and, for the cases those reports left open, from the clinical
record, and returned a verified worksheet. That worksheet is case-level and is
not distributed, and neither is the classification derived from it (Table 2
records the same in its footnote). `make_appendices.py` builds the two
exclusion-cohort tables and sets them beside the full-cohort tables (largest
accuracy change, and which comparisons cross P=.05); they are Multimedia
Appendix 8.

## Order

```bash
python3 analyze_results.py     # tables, CIs, McNemar, confusion matrices
python3 compute_sig.py         # paired significance for Figure 2
python3 make_figures.py        # Figure 2 (bars) and Figure 3 (confusion)
python3 make_figure2_main.py   # Figure 2 as published (radar + delta bars)
python3 analyze_repro.py       # reproducibility (after repeat runs exist)
python3 make_appendices.py     # Multimedia Appendices 5 to 8
```

Then the checks. `recompute_tables.py` needs nothing but this repository: it
reads the three files in `deposit/` and reproduces every number in Tables 4, 5
and 6 from them.

```bash
python3 recompute_tables.py            # Tables 4 to 6, recomputed and printed

# the three below read the stored run outputs, so they need JSON_DIRS, and the
# first two also need REFERENCE_XLSX; export them or repeat them on each line.
JSON_DIRS="/abs/path/to/repo/output/json:/abs/path/to/repo/output:/abs/path/to/repo/output/revised_experiments_202607/json" \
REFERENCE_XLSX=/path/to/reference_labels.xlsx \
python3 instruction_citations.py       # exits non-zero if a run quotes an instruction it never had
python3 test_ci_stability.py           # intervals must not depend on the analysis set
python3 parse_audit.py                 # JSON vs regex-fallback rates over the stored outputs
```

`JSON_DIRS` behaves like `RESULTS_DIR`: its entries are resolved from the
working directory, so give them as absolute paths. A relative entry that does
not resolve is not an error - the sections that need the run outputs are simply
skipped, and the run still reports `0 FAILED`.

A skipped section prints why it was skipped. Treat a run that skips sections as
an incomplete check, not a pass. `recompute_tables.py` is the exception: it runs
to completion on a clone of this repository, because everything it needs is in
`deposit/`. The other scripts need `analysis/results*/`, the reference workbook
or the run outputs, none of which are distributed (see Deposit above), and each
names the input it is missing.

## Requirements

Outside the standard library these scripts import `numpy` and `openpyxl`, both
declared in `../requirements.txt`. `matplotlib` is optional: only the PNG figure
steps use it, and a host without it writes the tables and CSVs and skips the
figures. `parse_audit.py` imports the repository's own `src` package, so run it
from the repository root. `recompute_tables.py` imports nothing outside the
standard library.
