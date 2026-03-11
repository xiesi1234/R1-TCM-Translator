# R1-TCM-Translator

Open-source package for the paper **Reinforcement Learning for the Computational Interpretation of Classical Medical Heritage Texts**.

## Project Goal

This project targets ancient-to-modern Chinese translation for classical TCM texts, with emphasis on heritage-oriented interpretive fidelity (semantic faithfulness and cultural/epistemic consistency), not only surface fluency.

## Method Overview

- Task: Classical TCM Chinese -> Modern Chinese translation
- Training pipeline:
  - Stage I: SFT for domain alignment
  - Stage II: GRPO for multi-objective reward optimization
- Structured reasoning: six-step format
- Reward design: format reward + quality reward (BLEU + COMET)

## Data and Evaluation

- Training corpus: 15,387 sentence pairs from 8 representative classics
- Main evaluation metrics: BLEU / COMET
- Four challenge dimensions:
  - semantic shift
  - cultural-epistemic context
  - zero anaphora
  - parataxis

## 2.3 General Open-Source SFT Data (Hugging Face)

To systematically inject foundational TCM knowledge into base models, this study uses the
public large-scale instruction-tuning source `SylvanL/Traditional-Chinese-Medicine-Dataset-SFT`
from Hugging Face.

Following the paper-level accounting, the general open-source SFT data exceeds 1.2 million
samples and is organized into three source types:

| Source Type | Scale (paper accounting) | Core Contribution |
|---|---:|---|
| Structured knowledge text (authoritative databases/encyclopedic sources) | ~550k | Learn core concept attributes and relations |
| Terminology interpretation (national standards/professional dictionaries) | ~100k | Improve precise understanding of term scope and meaning |
| Unstructured classical text (TCM classics and teaching materials) | ~560k | Learn classical language patterns and cultural context |

In this repository, the following files are project-aligned processed SFT inputs derived
from that public source:

- `sft/data/sft_medical_knowledge_source1.json`
- `sft/data/sft_medical_knowledge_source2.json`
- `sft/data/sft_medical_knowledge_source3.json`

Source link:

- `https://huggingface.co/datasets/SylvanL/Traditional-Chinese-Medicine-Dataset-SFT`

## Main Reported Results

- R1-TCM-Translator-8B: BLEU 41.5 / COMET 79.6
- R1-TCM-Translator-0.6B: BLEU 29.5 / COMET 77.4

## Hugging Face Models

- R1-TCM-Translator-0.6B: `https://huggingface.co/R1-TCM_translator/R1-TCM-Translator-0.6B`
- R1-TCM-Translator-8B: `https://huggingface.co/R1-TCM_translator/R1-TCM-Translator-8B`

## Repository Structure

```text
.
├── README.md
├── README_en.md
├── README_zh.md
├── LICENSE
├── DATA_LICENSE.md
├── requirements.txt
├── setup_env.sh
├── reproduce_fresh_env.sh
├── sft/
├── grpo/
├── evaluation/
├── experiments/
├── merged_test_set/
└── reference_answers/
```

## Dataset Split Policy

- Main paper/evaluation protocol uses `reference_answers/` (3241 samples).
- `merged_test_set/` (3377 samples) is retained as archival raw split and is **not** used for main reported results.

## Environment

Recommended:

```bash
bash setup_env.sh
```

Or:

```bash
pip install -r requirements.txt
```

## Reproduction

Minimal run (no COMET / no API modules):

```bash
bash reproduce_fresh_env.sh --install-deps --venv-dir .venv_repro --result-dir repro_results/fresh_run
```

Full run (with COMET + API-dependent modules):

```bash
export ARK_API_KEY="your_ark_key"
export SILICONFLOW_API_KEY="your_siliconflow_key"
bash reproduce_fresh_env.sh --install-deps --venv-dir .venv_repro --result-dir repro_results/full_run --with-comet --with-tagging --with-rare-llm
```

## Training Entry

- GRPO notebook: `grpo/notebooks/grpo.ipynb`
- Notebook quick guide: `grpo/notebooks/run_notebook.md`
- SFT configs (LLaMA-Factory): `sft/llama_factory/`

## Evaluation Entry

See:

- `evaluation/README.md` (language switch)
- `evaluation/README_en.md`
- `evaluation/README_zh.md`

## License

- Code: Apache-2.0 (`LICENSE`)
- Data/model outputs usage boundary: `DATA_LICENSE.md`
