# R1-TCM-Translator

基于强化学习（GRPO）的中医古籍古今翻译项目。

## 研究概述（论文内容提炼）

本项目对应论文 **Reinforcement Learning for the Computational Interpretation of Classical Medical Heritage Texts**，核心目标是提升中医古籍古今翻译中的“语义忠实性与文化-知识一致性”（而不仅仅是表面流畅度）。

### 任务与方法

- 任务：中医古文到现代汉语翻译，强调 heritage-oriented interpretive fidelity（遗产文本语义与医理传承的一致性）。
- 两阶段训练：
  - Stage I：SFT 进行领域知识对齐；
  - Stage II：GRPO 进行多目标奖励优化。
- 推理框架：六步结构化推理（原文、术语解析、医理推演、句法解构、文化注释、现代译文）。
- 奖励设计：格式奖励 + 质量奖励，其中质量奖励包含 BLEU 与 COMET。

### 数据与评测

- 训练语料：TCM-Ancient-Modern 并行语料，共 **15,387** 条句对，覆盖 8 部代表性中医古籍。
- 评测协议：在统一测试集上报告 BLEU/COMET，并进行四类挑战维度分析：
  - 古今异义（semantic shift）
  - 文化语境（cultural-epistemic context）
  - 零形回指（zero-anaphora）
  - 意合句法（parataxis）

### 主要结果（论文报告）

- R1-TCM-Translator-8B：**BLEU 43.3 / COMET 79.6**
- R1-TCM-Translator-0.6B：**BLEU 29.5 / COMET 77.4**
- 论文还报告了术语相关细粒度分析中的明显提升（约 23 个百分点）。

### Hugging Face 模型链接

- R1-TCM-Translator-0.6B：`https://huggingface.co/R1-TCM_translator/R1-TCM-Translator-0.6B`
- R1-TCM-Translator-8B：`https://huggingface.co/R1-TCM_translator/R1-TCM-Translator-8B`

本目录是整理后的公开版本，重点提供：

- 可直接使用的数据文件（SFT + GRPO）
- 可复现的 GRPO 训练 notebook
- 已整理可直接复用的实验输出（`experiments/`）
- 从旧工程迁移并可直接运行的评估脚本（`evaluation/`）
- 项目工作说明与复现入口（本 README）

## 目录结构

```text
.
├── README.md
├── LICENSE
├── DATA_LICENSE.md
├── requirements.txt
├── setup_env.sh
├── sft/
│   ├── data/
│   │   ├── sft_medical_knowledge_source1.json
│   │   ├── sft_medical_knowledge_source2.json
│   │   ├── sft_medical_knowledge_source3.json
│   │   └── sft_translation_deepseek_12145.json
│   └── llama_factory/
│       ├── README.md
│       ├── dataset_info.json
│       ├── qwen3_8b_sft_paper.yaml
│       └── qwen3_0_6b_sft_paper.yaml
├── grpo/
│   ├── data/
│   │   └── tcm_ancient_modern_grpo_train_12145.json
│   └── notebooks/
│       ├── grpo.ipynb
│       └── run_notebook.md
├── evaluation/
│   ├── README.md
│   ├── 01_overall_and_4challenges/
│   │   └── calculate_bleu_comet_filtered.py
│   ├── 02_long_sentence/
│   │   ├── calculate_bleu_by_length_local.py
│   │   ├── calculate_comet_by_length.py
│   │   └── generate_latex_plot.py
│   ├── 03_rare_word/
│   │   ├── analyze_rare_words_correct.py
│   │   ├── build_word_frequency.py
│   │   ├── evaluate_rare_word_translation.py
│   │   ├── extract_word_alignments.py
│   │   └── llm_word_alignment.py
│   └── 04_challenge_tagging/
│       ├── semantic_shift/
│       ├── cultural_context/
│       ├── zero_anaphora/
│       └── parataxis/
├── experiments/
│   ├── Qwen3-0.6B/
│   ├── Qwen3-8B/
│   └── baselines/
├── merged_test_set/
│   └── *_test.json
└── reference_answers/
    ├── *_queries.txt
    ├── *_references.txt
    └── *_indices.txt
```

## 数据文件

### SFT 数据

| 文件 | 样本数 | 字段 |
|---|---:|---|
| `sft/data/sft_medical_knowledge_source1.json` | 203,803 | `prompt`, `response` |
| `sft/data/sft_medical_knowledge_source2.json` | 99,334 | `prompt`, `response` |
| `sft/data/sft_medical_knowledge_source3.json` | 556,540 | `prompt`, `response` |
| `sft/data/sft_translation_deepseek_12145.json` | 12,145 | `prompt`, `query`, `response` |

### 2.3 通用开源数据集（Hugging Face）

为系统性给基础模型注入中医药领域知识，本研究采用公开的大规模指令微调数据来源
`SylvanL/Traditional-Chinese-Medicine-Dataset-SFT`（Hugging Face）。

按论文口径统计，通用开源 SFT 数据总规模超过 120 万条，主要由三类数据构成：

| 数据来源类型 | 规模（论文口径） | 核心贡献 |
|---|---:|---|
| 结构化知识文本（权威数据库/百科） | ~550k | 使模型掌握核心概念的属性与关联 |
| 专业术语释义（国家标准/专业词典） | ~100k | 使模型精确理解术语内涵与外延 |
| 非结构化典籍文本（古籍与教材） | ~560k | 使模型学习古文语言范式与文化语境 |

本仓库中的以下文件为该开源数据在本项目中的整理版本（用于 SFT 训练）：

- `sft/data/sft_medical_knowledge_source1.json`
- `sft/data/sft_medical_knowledge_source2.json`
- `sft/data/sft_medical_knowledge_source3.json`

开源数据来源链接：

- `https://huggingface.co/datasets/SylvanL/Traditional-Chinese-Medicine-Dataset-SFT`

说明：

- 论文中的规模统计按原始开源数据口径；
- 本仓库发布的是与本项目训练流程对齐后的可复现处理文件。

### GRPO 数据

| 文件 | 样本数 | 字段 |
|---|---:|---|
| `grpo/data/tcm_ancient_modern_grpo_train_12145.json` | 12,145 | `id`, `ancient`, `modern` |

### 测试集与参考答案（评估口径说明）

- 论文主结果与默认评估脚本使用：`reference_answers/`（共 3,241 条）
- `merged_test_set/`（共 3,377 条）是原始拆分版本，包含后续清洗中剔除的脏样本，仅作数据留档，不作为主结果评估输入

`reference_answers/` 中：

- `*_queries.txt`：测试输入（古文）
- `*_references.txt`：参考译文
- `*_indices.txt`：索引对齐信息（用于从原始模型输出中筛选有效样本）

## 环境安装

### 推荐方式

```bash
bash setup_env.sh
```

### 已验证环境（unsloth_env）

- Python 3.12
- CUDA 12.x（验证环境为 12.8）
- PyTorch 2.8.0
- transformers 4.57.1 / trl 0.23.0 / unsloth 2025.10.11
- vllm 0.11.0（仅在 GPU Linux 环境建议启用）

### 手动安装

```bash
pip install -r requirements.txt
sudo apt-get update
sudo apt-get install -y fonts-noto-cjk
```

> 已移除 `uv` 安装步骤。

## 服务器新环境重跑

项目根目录提供一键脚本：`reproduce_fresh_env.sh`

最小复现（不含 COMET / API 评估）：

```bash
bash reproduce_fresh_env.sh --install-deps --venv-dir .venv_repro --result-dir repro_results/fresh_run
```

全流程（含 COMET、挑战标注、LLM 词对齐）：

```bash
export ARK_API_KEY="your_ark_key"
export SILICONFLOW_API_KEY="your_siliconflow_key"
bash reproduce_fresh_env.sh --install-deps --venv-dir .venv_repro --result-dir repro_results/full_run --with-comet --with-tagging --with-rare-llm
```

脚本输出：

- 运行日志：`repro_logs/<timestamp>/`
- 结果归档：`--result-dir` 指定目录（默认 `repro_results/<timestamp>/`）
- 对照摘要：`<result-dir>/result_summary.json`（含论文目标值与当前运行关键指标）

说明：
- `reproduce_fresh_env.sh` 默认开启严格索引对齐（`STRICT_INDEX_ALIGNMENT=1`），评估时不再静默丢弃越界样本；若输出缺行会以空预测补齐并在日志/结果中记录。

## GRPO 训练复现

Notebook：`grpo/notebooks/grpo.ipynb`

快速入口见：`grpo/notebooks/run_notebook.md`

关键参数：

- `MODEL_NAME`：如 `Qwen/Qwen3-0.6B`
- `DATA_FILE`：`../data/tcm_ancient_modern_grpo_train_12145.json`
- `OUTPUT_DIR`：训练输出目录
- `USE_SWANLAB`：是否启用 SwanLab（`0/1`）

## SFT 训练复现（LLaMA-Factory）

- 配置目录：`sft/llama_factory/`
- 论文参数对齐配置（8B）：`sft/llama_factory/qwen3_8b_sft_paper.yaml`
- 0.6B 对齐配置：`sft/llama_factory/qwen3_0_6b_sft_paper.yaml`
- 数据映射：`sft/llama_factory/dataset_info.json`（直接映射 `sft/data/*.json`）

运行示例（项目根目录）：

```bash
pip install llamafactory
llamafactory-cli train sft/llama_factory/qwen3_8b_sft_paper.yaml
```

详细说明见：`sft/llama_factory/README.md`

## 评估脚本说明

- 第五章评估代码已整理到：`evaluation/`
- 入口文档：`evaluation/README.md`
- 默认评估模型输出已放在：`experiments/`

包含模块：

- 总体 + 四大挑战 BLEU/COMET
- 长句分桶分析
- 生僻词分析
- 四大挑战样本标注（cache 生成）

总体/挑战指标口径说明：
- `evaluation/01_overall_and_4challenges/calculate_bleu_comet_filtered.py` 同时输出 `avg_*_macro` 与 `avg_*_micro`（micro 为按样本数加权的 book-level 平均）。
- 默认汇总字段 `avg_bleu` / `avg_comet` 使用宏平均（可通过 `OVERALL_AVG_MODE=micro` 切换）。

生僻词评估补充说明：

- `evaluation/03_rare_word/` 已提供 `*_word_alignments.json`，可直接运行 `evaluate_rare_word_translation.py`
- 若需重算词对齐，可先运行 `llm_word_alignment.py`（会输出 `*_word_alignments.json`，并兼容写入 `*_word_alignments_all_freq.json`）
- `evaluate_rare_word_translation.py` 与 `analyze_rare_words_correct.py` 默认按 `--runs 1,2,3` 统计，并按 `reference_answers/*_indices.txt` 对齐预测
- 论文中词对齐准确率图（如 22.36 / 54.80 / 80.68）对应的结果文件已归档到 `evaluation/03_rare_word/paper_llm_judgment/word_alignment_evaluation_results_single.json`（LLM judgment 口径）

## 仓库说明

- 当前公开仓库不包含手稿文件，仅保留复现实验所需代码、数据与说明文档。
- 如果需要论文版本信息或补充材料，请以投稿版本/通讯作者提供信息为准。

## 引用

若你使用本项目，请引用论文：  
**Reinforcement Learning for the Computational Interpretation of Classical Medical Heritage Texts**

## 数据可用性

核心训练数据与复现材料已在本目录公开。补充材料可联系通讯作者获取。

## 许可证与使用边界

- 代码许可：Apache-2.0（见 `LICENSE`）
- 数据与模型输出许可：见 `DATA_LICENSE.md`
- 若代码许可与数据许可条款冲突，以数据许可条款为准
