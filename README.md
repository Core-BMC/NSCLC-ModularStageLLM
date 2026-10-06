# Lung Cancer Clinical TNM Staging with Modular Agent Architecture

A modular, agent-based system for automated clinical TNM staging of lung cancer using Large Language Models (LLMs). This system processes medical reports and automatically classifies cancer stages according to AJCC (American Joint Committee on Cancer) guidelines.

## What you can check

1. **Recompute the reported tables:** Run `python3 analysis/recompute_tables.py` from the repository root. It uses the three deposited aggregate files and the Python standard library; no model API or patient records are required. See the [analysis README](analysis/README.md).
2. **Run a fictional example:** Install the inference dependencies and configure your model/API, then use the included synthetic Excel files in the examples below. These examples demonstrate execution and do not reproduce the study results.
3. **Re-evaluate the original cohort:** This cannot be done from the public files alone because the patient-level inputs are not distributed. Aggregate table recomputation does not re-run patient-level inference.

## Default template and evaluated configurations

The default template in `config/tnm_config.yaml` has not been evaluated on the study cohort. Its performance on that cohort is unknown. The reported results apply to the preserved single-prompt and full MAA configurations and the deposited decomposition-only run configurations, not to the current default template. Sections 8 and 9 of Multimedia Appendix 1 describe the configuration differences and reproduce the preserved single-prompt and full MAA prompts with internal review annotations redacted.

## Analysis and aggregate data

See the [analysis README](analysis/README.md#aggregate-data-and-patient-level-data) for the institutional data-sharing policy, the three aggregate files (including bootstrap statistics), and instructions for reproducing Tables 4 to 6. These files contain no individual patient records or patient-by-patient reference and prediction data.

## Overview

This research workflow processes medical reports to predict clinical T, N and M categories, then assigns the overall stage group using deterministic AJCC rules. It supports Single Prompt, decomposition-only and MAA configurations. Performance depends on the model and configuration; generated rationales have not been validated as explanations for clinical decision-making.

### Key Features

- **Workflow configurations**: A combined T/N/M prompt or separate T, N and M classifiers, followed by deterministic stage grouping. An initial histology call is present in all configurations, but its output is not used by the staging components.
- **Consensus Mechanism**: Optional repeated sampling and voting for T/N/M classification; the evaluated comparisons do not isolate a benefit from voting.
- **AJCC Support**: Supports both AJCC 8th and 9th edition staging guidelines
- **Multiple LLM Backends**: Works with OpenAI, Azure OpenAI, or local LLM servers (e.g., Ollama)
- **Comprehensive Input Support**: Processes various medical report types (Pathology, CT, MRI, PET, EBUS, etc.)
- **Generated Rationales**: Records model-generated reasoning for review; this does not establish the correctness or clinical usefulness of the reasoning.
- **Batch Processing**: Handles multiple cases from CSV or Excel files
- **Performance Metrics**: Calculates accuracy and confusion matrices when true TNM values are provided

## Project Structure

```bash
NSCLC-ModularStageLLM/
├── config/
│   ├── ajcc8th/
│   │   └── tnm_classification.json    # AJCC 8th edition classification rules
│   ├── ajcc9th/
│   │   └── tnm_classification.json    # AJCC 9th edition classification rules
│   └── tnm_config.yaml                # Main configuration file
├── src/
│   ├── agents/
│   │   └── consensus.py               # Consensus mechanism for multiple responses
│   ├── histology/
│   │   ├── classification.py          # Histology classification logic
│   │   ├── parser.py                  # Histology result parser
│   │   └── workflow.py                # Histology workflow
│   ├── models/
│   │   ├── config.py                  # Configuration model
│   │   └── data_models.py             # Data models (InputData, MedicalReport)
│   ├── parsers/
│   │   ├── base_parser.py             # Base parser class
│   │   └── tnm_parsers.py             # T, N, M classification parsers
│   ├── utils/
│   │   ├── data_utils.py              # Data processing utilities
│   │   ├── file_operations.py         # File I/O operations
│   │   ├── llm_utils.py               # LLM setup and utilities
│   │   ├── logging_utils.py           # Logging configuration
│   │   ├── metrics_utils.py           # Accuracy and metrics calculation
│   │   ├── stage_utils.py             # Stage determination logic
│   │   └── workflow_utils.py          # Workflow helper functions
│   ├── workflow/
│   │   ├── setup.py                   # Workflow graph setup
│   │   └── state.py                   # Workflow state definition
│   └── tnm_workflow.py                # Main workflow class
├── input/                              # Sample input files
├── output/                             # Output directory
├── run_workflow.py                     # Command-line entry point
├── sample.env                          # Environment variables template
└── README.md                           # This file
```

## How It Works

### Workflow Overview

The modular path executes the following nodes sequentially. T, N and M classifiers receive all available reports in component-specific context order; the initial histology label is excluded from their input. Single Prompt replaces the three T/N/M classifier nodes with one combined query. Stage grouping is deterministic in both paths:

```bash
1. Histology Classification
   ↓
2. T Classification (Tumor size and extent)
   ↓
3. N Classification (Lymph node involvement)
   ↓
4. M Classification (Metastasis)
   ↓
5. Rule-based Stage Grouping (no LLM call)
   ↓
6. Final Save (Results output)
```

### Detailed Workflow Steps

#### 1. Histology Classification

- Analyzes pathology reports to identify cancer type
- Classifies according to WHO Classification of Lung Tumors
- Determines category, subcategory, and type
- Records model-generated confidence and reasoning
- The histology classifier was not validated against a reference standard; its label is excluded from staging context. The manuscript cohort histology was established separately from pathology reports and clinician review.

#### 2. T Classification

- Analyzes tumor size, location, and local invasion
- Reviews CT scans, pathology reports, and other relevant imaging
- Permitted clinical T categories include Tis, T1mi, T1a, T1b, T1c, T2a, T2b, T3 and T4; T0 has been removed from the prompt output set. Indeterminate Tx is distinct from a parsing failure.
- Uses repeated sampling and voting when MAA is selected; decomposition-only disables voting.

#### 3. N Classification

- Evaluates lymph node involvement
- Reviews CT scans, PET scans, EBUS reports, and biopsy results
- Uses N0, N1, N2 and N3 under the 8th edition; the 9th edition subdivides N2 into N2a and N2b. Indeterminate Nx is retained.
- Considers regional lymph node stations

#### 4. M Classification

- Assesses distant metastasis
- Reviews brain MRI, PET scans, bone scans, and other imaging
- Uses M0, M1a, M1b and M1c under the 8th edition; the 9th edition subdivides M1c into M1c1 and M1c2. Indeterminate Mx is retained.
- Distinguishes between different metastatic sites

#### 5. Stage Classification

- Rule-based stage determination using TNM combinations
- Follows AJCC staging tables
- Supports both AJCC 8th and 9th editions
- Assigns stage 0, IA1, IA2, IA3, IB, IIA, IIB, IIIA, IIIB, IIIC, IVA or IVB where the TNM combination has a defined rule; an unresolved combination is not a valid stage assignment.

#### 6. Final Save

- Saves results to CSV and JSON files
- Includes all classifications, reasoning, and metadata
- Calculates accuracy metrics if true TNM values are provided

### Consensus Mechanism

In MAA, each T/N/M classification is sampled at temperature 0.5 until one parsed label receives two concordant votes, with a maximum of 50 attempts. Unparseable responses count toward the attempt limit but not toward votes. A three-way disagreement can therefore require more than three attempts. At the limit, the most frequent parsed label is returned; if no response is parsed, no classification is returned.

Single Prompt issues one combined staging query without pipeline retry. Decomposition-only uses separate T/N/M queries without voting; two component calls required a retry in the reported evaluation. All three configurations also make an initial histology call. The comparisons involve decomposition, voting and output handling; ninth-edition comparisons additionally differ in prompt wording and do not isolate the effect of architecture or voting.

The stored-output audit covered 17,820 component outputs from the two agent configurations, all yielding labels through the agent JSON path. That path includes direct classification-field extraction and does not establish strict JSON validity. Among 2,970 combined Single Prompt outputs, 2,951 used the JSON path and 18 used regex fallback. One further output had no stored generated text, so its original parsing path could not be verified; it was recorded as no label and scored as incorrect.

Discarded generations were not retained, so this audit cannot establish the parsing-failure or resampling rate across all attempted generations. Explicit Tx/Nx/Mx labels and an undetermined overall stage are distinct from a missing extracted label. The N/M extraction functions return no result on an unmatched output rather than silently assigning N0 or M0.

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/Core-BMC/NSCLC-ModularStageLLM.git
cd NSCLC-ModularStageLLM
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install manually:

```bash
pip install pandas openpyxl pyyaml python-dotenv
pip install langchain langchain-core langchain-openai langchain-community langchain-experimental langgraph
pip install openai pydantic
```

### Step 4: Configure Environment Variables

Copy the sample environment file and edit with your credentials:

```bash
cp sample.env .env
```

Edit `.env` with your API credentials:

**For Azure OpenAI:**

```env
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_BASE=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-05-01-preview
AZURE_DEPLOYMENT_NAME=gpt-4o
```

**For OpenAI (Direct):**

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL_NAME=gpt-4o
```

**For Local LLM (e.g., Ollama):**
Configure in `config/tnm_config.yaml` (see Configuration section).

## Configuration

The main configuration file is `config/tnm_config.yaml`. Key settings include:

### AJCC Edition

```yaml
tnm_json: "config/ajcc8th/tnm_classification.json"  # or "config/ajcc9th/tnm_classification.json"
```

### LLM Settings

```yaml
model_settings:
  llm_choice: "azure"  # Options: "local", "openai", or "azure"

  # Standardized to a fixed temperature of 0.5 across all backends
  azure:
    name: "gpt-4o"
    temperature_low: 0.5
    temperature_high: 0.5

  openai:
    name: "gpt-4o"
    temperature_low: 0.5
    temperature_high: 0.5

  local:
    name: "llama3.3:70b"
    base_url: "https://local.server.ip.address:port"
    api_key: "your_bearer_token"
    temperature_low: 0.5
    temperature_high: 0.5
    max_tokens: 2048  # output-token cap for local backends
```

`max_tokens` caps the generated output for local backends (Ollama maps it to
`num_predict`). The evaluation reported in the accompanying manuscript was run
at **2048**; raising it changes generation behaviour and any re-run should be
described as such.

### Evaluation context window

The `*-ctx8k` names are local deployment aliases configured with `num_ctx=8192`. To use the provided configurations, create these aliases from the corresponding base models (`phi4:14b` and `llama3.3:70b`), or update the model names to your deployment names and configure the context window to 8,192 tokens. Downloading a base model does not automatically create these aliases.

The local models used a context window of 8,192 tokens (num_ctx=8192), separate from the output limit of 2,048 tokens (max_tokens=2048). Whether any evaluation inputs were truncated at the context limit has not been established. This clarification does not change the reported results. See [the analysis README](analysis/README.md#local-context-window-and-output-limit) for details.

### Configuration Selection

| Configuration | tnm_base | use_consensus |
|---|---|---|
| Single Prompt | true | false |
| Decomposition only | false | false |
| MAA | false | true |

Setting use_consensus to false alone does not select Single Prompt; tnm_base controls whether the combined node is used.

### Input/Output Files

```yaml
input_file: "input/synthetic_single_excel_ajcc_8th.xlsx"
output_file: "output/results"
json_output_file: "output/results.json"
log_file: "output/results.log"
```

## Usage

### Command Line Interface

The easiest way to run the workflow is using the command-line interface:

```bash
python run_workflow.py -i input/synthetic_single_excel_ajcc_8th.xlsx -o output/results
```

**Options:**

- `-i, --i`: Input file path (CSV or Excel)
- `-o, --o`: Output file prefix (without extension)
- `--config`: Path to config file (default: `config/tnm_config.yaml`)
- `--log`: Log file path (default: `<output_prefix>.log`)
- `-v, --verbose`: Enable verbose logging

**Examples:**

```bash
# Process Excel file
python run_workflow.py -i input/synthetic_single_excel_ajcc_8th.xlsx -o output/results --config config/tnm_config.yaml

# Process the AJCC 9th edition synthetic example (configure this model/API first)
python run_workflow.py -i input/synthetic_single_excel_ajcc_9th.xlsx -o output/results_ajcc9 --config config/runs/tnm_config_deconly_9_phi4.yaml

# Enable verbose logging
python run_workflow.py -i input/synthetic_single_excel_ajcc_8th.xlsx -o output/results -v
```

### Base Mode vs Agent Mode

You can switch between a single-pass base TNM workflow and the multi-agent chain.

**Base Mode (single-pass T+N+M):**

- Set `tnm_base: true` in `config/tnm_config.yaml`
- Provide `tnm_classifier_base` under the matching AJCC prompt section
- Stage is still rule-based after parsing T/N/M

```yaml
tnm_base: true

ajcc8th_prompts:
  tnm_classifier_base: |
    ...single-pass TNM prompt...
```

> **Note.** The single-prompt configuration reported in the accompanying manuscript
> is selected by `tnm_base: true`, not by `use_consensus: false`. Setting
> `use_consensus: false` while `tnm_base: false` gives a third configuration:
> the T/N/M agent chain with majority voting disabled. The manuscript calls these
> three the single combined prompt, decomposition-only, and the full MAA; the word
> "baseline" is not used for any of them.
>
> The current template differs from the configurations used in the reported evaluation. Its ninth-edition M prompt was corrected so that a finding on a single imaging study is not, by itself, a reason to default to M0. The six decomposition-only configurations in config/runs/ retain the earlier wording used for their reported results. The original MAA prompts also differ from the decomposition-only prompts in handling equivocal N/M findings. Multimedia Appendix 1, Section 8 documents these differences. The corrected template has not been evaluated on the study cohort, and no new performance claim follows from the code correction.

**Agent Mode (T -> N -> M chain):**

- Set `tnm_base: false`
- Provide `t_classifier`/`n_classifier`/`m_classifier` (or their `_base` variants)
- Consensus behavior is controlled by `use_consensus`

```yaml
tnm_base: false
use_consensus: true

ajcc8th_prompts:
  t_classifier: |
    ...T prompt...
  n_classifier: |
    ...N prompt...
  m_classifier: |
    ...M prompt...
```

### Input File Format

The input file should be a **CSV** or **Excel** file with the following columns:

**Required Columns:**

- `hospital_id`: Patient/hospital identifier
- `Pathology`: Pathology report content
- `Chest CT`: Chest CT scan report

**Optional Columns:**

- `Brain MR`: Brain MRI report
- `PET`: PET scan report
- `EBUS`: EBUS report
- `Neck biopsy`: Neck biopsy report (written `neck biopsy` in the bundled
  templates; column names are matched case-insensitively, so either works)
- `Bone scan`: Bone scan report
- `Abdomen&Pelvis CT`: Abdomen and pelvis CT report
- `Adrenal CT`: Adrenal CT report

**For Accuracy Evaluation (Optional):**

- `cT`, `cN`, `cM`, `cStage`: True TNM values for comparison

### Output Format

The system generates two output files:

#### CSV Output (`<prefix>.csv`)

Contains one row per case with the following columns:

- `pid`: Patient identifier
- `hospital_number`: Hospital identifier
- `histology_category`, `histology_subcategory`, `histology_type`: Histology classification
- `histology_confidence`, `histology_reason`: Histology confidence and reasoning
- `T_classification`, `N_classification`, `M_classification`, `Stage_classification`: Classifications
- `T_reasoning`, `N_reasoning`, `M_reasoning`, `Stage_reasoning`: Detailed reasoning for each classification
- `true_T`, `true_N`, `true_M`, `true_Stage`: True values (if provided)

#### JSON Output (`<prefix>.json`)

Contains detailed JSON structure with:

- Input data
- All classifications
- Reasoning for each step
- Metadata and timestamps

## Workflow Architecture

### State Graph

The system uses LangGraph to manage the workflow state:

```python
histology_classifier → t_classifier → n_classifier → m_classifier → stage_classifier → final_save
```

The graph above describes the modular path. Single Prompt uses a combined staging node instead of separate T/N/M nodes. The stage_classifier node applies deterministic rules and final_save writes outputs; neither is a separate LLM classification agent.

### Error Handling

The workflow includes comprehensive error handling:

- Each node catches exceptions and creates error states
- Errors are logged with full stack traces
- Workflow continues to next step even if one node fails
- Error information is included in output files

### Logging

The system provides detailed logging:

- Logs are written to both console and file
- Different log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Includes timing information and performance metrics
- Verbose mode provides additional debugging information

## Performance Metrics

When true TNM values are provided in the input file, the system automatically calculates:

- **Accuracy**: Percentage of correct classifications for T, N, M, and Stage
- **Confusion Metrics**: Over-staging and under-staging counts
- **Case-by-Case Analysis**: Detailed comparison for each case

Metrics are displayed in the console and included in log files.

## Examples

See the `input/` directory for sample input files:

- `single_csv_ajcc_9th.csv`: Single case CSV example
- `synthetic_single_excel_ajcc_8th.xlsx`: Single case Excel example
- `multiple_csv_cases_ajcc_8th.csv`: Multiple cases CSV example

See `Example_Notebook_Modular_cTNM_Staging.ipynb` for a Jupyter notebook example.

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your `.env` file is correctly configured and contains valid API keys
2. **File Not Found**: Check that input file paths are correct and files exist
3. **Import Errors**: Ensure all dependencies are installed (`pip install -r requirements.txt`)
4. **LLM Timeout**: Increase timeout settings in configuration or check network connectivity
5. **Consensus Failures**: Inspect parsing and backend errors in the logs. Changing temperature or voting thresholds defines a different configuration and does not reproduce the reported evaluation.

### Debug Mode

Enable verbose logging for detailed debugging:

```bash
python run_workflow.py -i input/synthetic_single_excel_ajcc_8th.xlsx -o output/results -v
```

## Tests

`tests/test_revision_fixes.py` checks the behaviours corrected in the 2026
revision: unparseable N/M output is recorded as no result rather than coerced to
the majority class, the AJCC 9th-edition stage tables cover Tis and T1mi, T0 is
not offered as an output, and input column lookup is case-insensitive.

```bash
python3 tests/test_revision_fixes.py

# Optional pytest runner:
python3 -m pip install pytest
python3 -m pytest tests/
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

**Paper**:  
[Citation will be added upon publication]

Please use the citation metadata in [CITATION.cff](CITATION.cff) or the "Cite this repository" button on GitHub.

## Contact

**Corresponding Author**:  
Shinkyo Yoon, MD, PhD  
Email: <shinkyoyoon@amc.seoul.kr>  
Affiliation: Asan Medical Center

**Developer**:  
Hwon Heo, PhD  
Email: <heoh@amc.seoul.kr>  
ORCID: [0000-0002-6103-4680](https://orcid.org/0000-0002-6103-4680)  
Affiliation: Asan Medical Center

## Acknowledgments

This work was supported by [Grant information to be added].
