# Run configurations — decomposition-only ablation

Six configurations for the ablation that separates the two components of the
modular agent architecture: **task decomposition** and **majority voting**.

| Condition | `tnm_base` | `use_consensus` | Isolates |
|---|---|---|---|
| A — single-pass baseline | `true` | (ignored) | — |
| **C — decomposition only** | `false` | `false` | A → C = effect of task decomposition |
| D — full agent architecture | `false` | `true` | C → D = effect of majority voting |

The files here are condition **C** for three models x two AJCC editions. They are
the configurations that produced the decomposition-only rows of Tables 4, 5 and 6,
and they preserve the prompt text as it was run. They were derived from
`../tnm_config.yaml`; only the backend, the AJCC edition, the consensus flag and
the input/output paths differ.

`../tnm_config.yaml` has since been corrected in one place: the 9th-edition
`m_classifier` prompt listed "single imaging" among the conditions that default to
M0, which overlapped the same prompt's rule that a clinical M1 may be assigned
from imaging alone. The template now states that a single imaging study is not by
itself a reason to default to M0. The six files here keep the uncorrected text,
because they are the artifacts that produced the reported rows; that one prompt is
now the only place where they and the template differ, and the corrected template
has not been re-evaluated. Section 8 of Multimedia Appendix 1 sets this out.

That is a statement about these six files, not about the comparison. The
baseline and full-architecture rows of Tables 4 and 5 were produced from twelve
earlier configuration files whose AJCC 9th-edition n_classifier and m_classifier
prompts carry the opposite instruction on the handling of equivocal findings
(see the Methods and Limitations, and editorial comment 1 of the second review).
Consequently:

  - Under the 8th edition the decomposition-only rows differ from the
    full-architecture rows in the voting setting and in the two stage-table
    corrections only, so that contrast estimates the effect of voting.
  - Under the 9th edition the same contrast additionally carries the prompt
    wording difference, so it does NOT estimate the effect of voting and must
    not be read as if it did.

| File | Model | Edition |
|---|---|---|
| `tnm_config_deconly_8_llama3.yaml` | LLaMA 3.3 70B | AJCC 8th |
| `tnm_config_deconly_8_phi4.yaml` | Phi-4 14B | AJCC 8th |
| `tnm_config_deconly_8_gpt4o.yaml` | GPT-4o (Azure) | AJCC 8th |
| `tnm_config_deconly_9_llama3.yaml` | LLaMA 3.3 70B | AJCC 9th |
| `tnm_config_deconly_9_phi4.yaml` | Phi-4 14B | AJCC 9th |
| `tnm_config_deconly_9_gpt4o.yaml` | GPT-4o (Azure) | AJCC 9th |

## Model builds

The local backends name the same derived builds used in the reported evaluation
(`llama3.3-ctx8k:latest`, `phi4-ctx8k:latest`), which set `num_ctx=8192` so the
model fits the GPU. Do not substitute the stock `llama3.3:70b` or `phi4:14b`
tags: a different context length is a different model for this purpose, and the
ablation would no longer be comparable to the reported baseline and full-agent
runs. Temperature (0.5) and `max_tokens` (2048) also match the reported runs.

## Credentials

The files keep the placeholder `base_url` and `api_key` from the template. Do not
edit them in place — supply the real values through the environment so the tracked
files stay clean:

```bash
export LOCAL_API_BASE_URL=http://your-local-llm-host:11434
export LOCAL_API_KEY=your_local_bearer_token
# Azure runs read AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY from .env
```

## Running

```bash
bash config/runs/run_deconly_matrix.sh              # all six
bash config/runs/run_deconly_matrix.sh llama3       # one lane
```

Or a single run:

```bash
python3 run_workflow.py \
  --i input/<your input file>.xlsx \
  --o output/<date>_llama3-70b-deconly-ajcc9-try1 \
  --config config/runs/tnm_config_deconly_9_llama3.yaml
```

## Note on scope

These are **new** configurations for the ablation. They are not the
configurations of the previously reported baseline and full-architecture runs,
which are described in the manuscript Methods.
