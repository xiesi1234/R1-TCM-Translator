# Evaluation 实验代码说明（审稿提交版）

本目录对应论文第五章评估，按 `evaluation/01~04` 四个模块组织。

## 目录结构

```text
evaluation/
├── 01_overall_and_4challenges/
│   ├── calculate_bleu_comet_filtered.py
│   └── filtered_challenge_results.json
├── 02_long_sentence/
│   ├── calculate_bleu_by_length_local.py
│   ├── calculate_comet_by_length.py
│   ├── generate_latex_plot.py
│   ├── bleu_by_length_results.json
│   ├── comet_by_length_results.json
│   └── grpo_comparison.tex
├── 03_rare_word/
│   ├── build_word_frequency.py
│   ├── llm_word_alignment.py
│   ├── evaluate_rare_word_translation.py
│   ├── analyze_rare_words_correct.py
│   ├── rare_word_evaluation_results.json
│   ├── rare_word_evaluation_report.txt
│   ├── tcm_rare_word_sft_grpo_results.json
│   └── tcm_rare_word_sft_grpo_analysis.tex
└── 04_challenge_tagging/
    ├── semantic_shift/
    ├── cultural_context/
    ├── zero_anaphora/
    └── parataxis/
```

## 模块功能

- `01_overall_and_4challenges`
  - 计算总体 BLEU/COMET 与四大挑战子集 BLEU/COMET。
  - 输出 `filtered_challenge_results.json`。

- `02_long_sentence`
  - 按源句长度分桶计算 BLEU/COMET，并生成论文用 LaTeX 曲线图。

- `03_rare_word`
  - `build_word_frequency.py`：构建词频统计。
  - `llm_word_alignment.py`：可选重跑 LLM 词对齐（需 API）。
  - `evaluate_rare_word_translation.py`：按词对齐结果计算生僻词准确率（默认 `--runs 1,2,3`）。
  - `analyze_rare_words_correct.py`：输出频段分析与论文图表所需结果（默认 `--runs 1,2,3`）。

- `04_challenge_tagging`
  - 四类挑战标注脚本与缓存（`cache_*.json`）。
  - `01` 模块默认读取这些缓存进行 challenge 过滤评估。

## 文件级说明（补充）

- `01_overall_and_4challenges/filtered_challenge_results.json`
  - 总体与四挑战的最终评估结果（含 `avg_*_macro` 与 `avg_*_micro`）。

- `02_long_sentence/bleu_by_length_results.json`
  - 长句实验 BLEU 分桶结果。
- `02_long_sentence/comet_by_length_results.json`
  - 长句实验 COMET 分桶结果。
- `02_long_sentence/grpo_comparison.tex`
  - 长句实验论文绘图的 LaTeX 产物。

- `03_rare_word/rare_word_evaluation_results.json`
  - 生僻词总体评估结果（准确率、分类型统计、三次 run 汇总）。
- `03_rare_word/rare_word_evaluation_report.txt`
  - 生僻词评估的人类可读报告。
- `03_rare_word/tcm_rare_word_sft_grpo_results.json`
  - 生僻词频段分析（用于画频段曲线）。
- `03_rare_word/tcm_rare_word_sft_grpo_analysis.tex`
  - 生僻词频段分析的 LaTeX 表图产物。

- `04_challenge_tagging/*/prompt.txt`
  - 对应挑战维度的判定提示词模板。
- `04_challenge_tagging/*/cache_*.json`
  - 已缓存的 challenge 标注结果（供 `01` 直接读取，避免重复 API 调用）。
- `04_challenge_tagging/semantic_shift/semantic_check.py`
  - 语义变迁维度标注脚本。
- `04_challenge_tagging/cultural_context/evaluate_cultural_context.py`
  - 文化语境维度标注脚本。
- `04_challenge_tagging/zero_anaphora/zero_anaphora_check.py`
  - 零形回指维度标注脚本。
- `04_challenge_tagging/parataxis/evaluate_parataxis.py`
  - 意合句法维度标注脚本。

## 推荐运行顺序

1. （可选）`04_challenge_tagging/*/*.py` 重跑挑战标注  
2. `01_overall_and_4challenges/calculate_bleu_comet_filtered.py`  
3. `02_long_sentence/calculate_bleu_by_length_local.py`  
4. `02_long_sentence/calculate_comet_by_length.py`  
5. `02_long_sentence/generate_latex_plot.py`  
6. `03_rare_word/build_word_frequency.py`  
7. （可选）`03_rare_word/llm_word_alignment.py`  
8. `03_rare_word/evaluate_rare_word_translation.py`  
9. `03_rare_word/analyze_rare_words_correct.py`

## 当前默认实验输出路径

- `../experiments/`
- 关键 checkpoint：
  - `Qwen3-0.6B (SFT+GRPO)`: `Qwen3-0.6B/SFT_GRPO/reward_bleu_comet/checkpoints/Qwen3-0.6B-checkpoint-4000`
  - `Qwen3-8B (SFT+GRPO)`: `Qwen3-8B/SFT_GRPO/reward_bleu_comet/checkpoints/Qwen3-8B-checkpoint-1000`

## 关键说明

- 主结果评估口径：`reference_answers/`（3241 条）。  
- `merged_test_set/` 仅为原始留档，不作为论文主结果输入。  
- `01` 默认 `STRICT_INDEX_ALIGNMENT=1`，遇到索引越界不静默丢样本。  
- `01` 同时输出宏平均与微平均：`avg_*_macro`、`avg_*_micro`。  
- COMET 首次运行会下载模型，建议在有 GPU 的环境运行。
