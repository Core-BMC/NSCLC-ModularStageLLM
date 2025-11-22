# Lung Cancer Clinical TNM Staging with Modular Agent Architecture

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

A modular, agent-based system for automated TNM staging classification of Non-Small Cell Lung Cancer (NSCLC) using Large Language Models (LLMs). This system processes medical reports and automatically classifies cancer stages according to AJCC (American Joint Committee on Cancer) guidelines.

## Overview

This project provides an intelligent workflow that analyzes medical reports (pathology, CT scans, MRI, PET scans, etc.) and automatically determines the TNM staging for lung cancer patients. The system uses a modular architecture with specialized agents for each classification component (T, N, M, and Stage), ensuring accurate and explainable results.

### Key Features

- **Modular Agent Architecture**: Separate specialized agents for histology, T, N, M, and Stage classification
- **Consensus Mechanism**: Multiple LLM responses with majority voting for improved accuracy
- **AJCC Support**: Supports both AJCC 8th and 9th edition staging guidelines
- **Multiple LLM Backends**: Works with OpenAI, Azure OpenAI, or local LLM servers (e.g., Ollama)
- **Comprehensive Input Support**: Processes various medical report types (Pathology, CT, MRI, PET, EBUS, etc.)
- **Explainable Results**: Provides detailed reasoning for each classification decision
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

The system follows a sequential workflow where each step builds upon the previous one:

```bash
1. Histology Classification
   ↓
2. T Classification (Tumor size and extent)
   ↓
3. N Classification (Lymph node involvement)
   ↓
4. M Classification (Metastasis)
   ↓
5. Stage Classification (Final stage determination)
   ↓
6. Final Save (Results output)
```

### Detailed Workflow Steps

#### 1. Histology Classification

- Analyzes pathology reports to identify cancer type
- Classifies according to WHO Classification of Lung Tumors
- Determines category, subcategory, and type
- Provides confidence score and reasoning

#### 2. T Classification

- Analyzes tumor size, location, and local invasion
- Reviews CT scans, pathology reports, and other relevant imaging
- Classifies as T0, T1a, T1b, T1c, T2a, T2b, T3, or T4
- Uses consensus mechanism for improved accuracy

#### 3. N Classification

- Evaluates lymph node involvement
- Reviews CT scans, PET scans, EBUS reports, and biopsy results
- Classifies as N0, N1, N2, or N3
- Considers regional lymph node stations

#### 4. M Classification

- Assesses distant metastasis
- Reviews brain MRI, PET scans, bone scans, and other imaging
- Classifies as M0, M1a, M1b, or M1c
- Distinguishes between different metastatic sites

#### 5. Stage Classification

- Rule-based stage determination using TNM combinations
- Follows AJCC staging tables
- Supports both AJCC 8th and 9th editions
- Generates final stage (IA1, IA2, IA3, IB, IIA, IIB, IIIA, IIIB, IIIC, IVA, IVB)

#### 6. Final Save

- Saves results to CSV and JSON files
- Includes all classifications, reasoning, and metadata
- Calculates accuracy metrics if true TNM values are provided

### Consensus Mechanism

The system uses a consensus mechanism to improve classification accuracy:

1. **Multiple Responses**: Collects multiple responses from the LLM (typically 3-5 responses)
2. **Temperature Variation**: Uses different temperature settings for each response to introduce diversity
3. **Majority Voting**: Selects the most common classification among valid responses
4. **Validation**: Ensures all responses follow the correct format before voting
5. **Retry Logic**: Automatically retries if consensus cannot be reached

This approach significantly improves accuracy compared to single-response methods.

## Installation

### Prerequisites

- Python 3.8 or higher
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
  
  azure:
    name: "gpt-4o"
    temperature_low: 0.0
    temperature_high: 0.0
  
  openai:
    name: "gpt-4o"
    temperature_low: 0.0
    temperature_high: 0.0
  
  local:
    name: "llama3.3:70b"
    base_url: "https://local.server.ip.address:port"
    api_key: "your_bearer_token"
    temperature_low: 0.8
    temperature_high: 1.0
```

### Consensus Mode

```yaml
use_consensus: True  # True for consensus mode, False for single response
```

### Input/Output Files

```yaml
input_file: "input/single_excel_ajcc_8th.xlsx"
output_file: "output/results"
json_output_file: "output/results.json"
log_file: "output/results.log"
```

## Usage

### Command Line Interface

The easiest way to run the workflow is using the command-line interface:

```bash
python run_workflow.py -i input/sample_cases.csv -o output/results
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
python run_workflow.py -i input/single_excel_ajcc_9th.xlsx -o output/results

# Process CSV file with custom config
python run_workflow.py -i input/multiple_csv_cases_ajcc_8th.csv -o output/results --config config/custom_config.yaml

# Enable verbose logging
python run_workflow.py -i input/sample.csv -o output/results -v
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
- `Neck biopsy`: Neck biopsy report
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

Each node:

1. Receives the current state
2. Performs its classification task
3. Updates the state with results
4. Passes control to the next node

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
- `single_excel_ajcc_8th.xlsx`: Single case Excel example
- `multiple_csv_cases_ajcc_8th.csv`: Multiple cases CSV example

See `Example_Notebook_Modular_cTNM_Staging.ipynb` for a Jupyter notebook example.

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your `.env` file is correctly configured and contains valid API keys
2. **File Not Found**: Check that input file paths are correct and files exist
3. **Import Errors**: Ensure all dependencies are installed (`pip install -r requirements.txt`)
4. **LLM Timeout**: Increase timeout settings in configuration or check network connectivity
5. **Consensus Failures**: Try adjusting temperature settings or reducing consensus requirements

### Debug Mode

Enable verbose logging for detailed debugging:

```bash
python run_workflow.py -i input/sample.csv -o output/results -v
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

**Paper**:  
[Citation will be added upon publication]

**Code**:  
Heo, H. (2025). Lung Cancer Clinical TNM Staging with Modular Agent Architecture (Version 1.0.0) [Computer software]. Zenodo. <https://doi.org/10.5281/zenodo.XXXXXXX>

Or use the citation information in [CITATION.cff](CITATION.cff) or the "Cite this repository" button on GitHub.

## Contact

**Corresponding Author**:  
Shinkyo Yoon, MD, PhD  
Email: <shinkyoyoon82@gmail.com>  
Affiliation: Asan Medical Center

**Developer**:  
Hwon Heo  
Email: <heohwon@gmail.com>  
ORCID: [0000-0002-6103-4680](https://orcid.org/0000-0002-6103-4680)  
Affiliation: Asan Medical Center

## Acknowledgments

This work was supported by [Grant information to be added].
