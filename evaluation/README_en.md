# Evaluation Guide (Reviewer Package)

This folder contains Chapter-5 evaluation modules.

## Structure

```text
evaluation/
├── 01_overall_and_4challenges/
├── 02_long_sentence/
├── 03_rare_word/
└── 04_challenge_tagging/
```

## Module Summary

- `01_overall_and_4challenges`
  - Computes overall BLEU/COMET and four challenge-specific BLEU/COMET.
  - Main output: `filtered_challenge_results.json`

- `02_long_sentence`
  - Computes BLEU/COMET by source-length bins.
  - Generates LaTeX plotting artifact for paper figures.

- `03_rare_word`
  - Builds word-frequency statistics
  - (Optional) re-runs LLM word alignment
  - Evaluates rare-word accuracy and exports bin-level analysis

- `04_challenge_tagging`
  - Scripts/prompts/caches for semantic-shift, cultural context, zero-anaphora, and parataxis tagging.
  - `cache_*.json` files are consumed by module `01`.

## Key Output Files

- `01_overall_and_4challenges/filtered_challenge_results.json`
- `02_long_sentence/bleu_by_length_results.json`
- `02_long_sentence/comet_by_length_results.json`
- `02_long_sentence/grpo_comparison.tex`
- `03_rare_word/rare_word_evaluation_results.json`
- `03_rare_word/rare_word_evaluation_report.txt`
- `03_rare_word/tcm_rare_word_sft_grpo_results.json`
- `03_rare_word/tcm_rare_word_sft_grpo_analysis.tex`

## Recommended Run Order

1. (Optional) rerun tagging in `04_challenge_tagging/*/*.py`
2. `01_overall_and_4challenges/calculate_bleu_comet_filtered.py`
3. `02_long_sentence/calculate_bleu_by_length_local.py`
4. `02_long_sentence/calculate_comet_by_length.py`
5. `02_long_sentence/generate_latex_plot.py`
6. `03_rare_word/build_word_frequency.py`
7. (Optional) `03_rare_word/llm_word_alignment.py`
8. `03_rare_word/evaluate_rare_word_translation.py`
9. `03_rare_word/analyze_rare_words_correct.py`

## Notes

- Main protocol uses `reference_answers/` (3241 samples).
- `merged_test_set/` is archival only, not for main reported results.
- `STRICT_INDEX_ALIGNMENT=1` is recommended for strict reproducibility.
- COMET scripts download models on first run; GPU environment is recommended.
