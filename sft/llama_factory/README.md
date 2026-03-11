# SFT with LLaMA-Factory

本目录提供可公开复现的 LLaMA-Factory 版 SFT 入口。

## 对齐论文参数

以下参数与论文附录 `R1_TCM_manuscript_submission/paper_latex.tex` 的 SFT 表一致（8B）：

- base model: `Qwen/Qwen3-8B`
- precision: `torch.bfloat16`
- LoRA rank: `32`
- LoRA alpha: `64`
- scheduler: `cosine`
- epochs: `2`
- batch size: `8`
- gradient accumulation steps: `2`
- learning rate: `1e-4`

对应配置文件：

- `qwen3_8b_sft_paper.yaml`
- `qwen3_0_6b_sft_paper.yaml`

## 数据映射

- 使用 `dataset_info.json` 直接映射 `sft/data/*.json`，不需要复制数据文件。
- 若你的 LLaMA-Factory 版本对 Qwen3 模板命名不同，可将 yaml 中 `template: qwen3` 改为该版本支持的模板名（例如 `qwen`）。

## 运行方式

在项目根目录执行：

```bash
pip install llamafactory
llamafactory-cli train sft/llama_factory/qwen3_8b_sft_paper.yaml
```

0.6B 版本：

```bash
llamafactory-cli train sft/llama_factory/qwen3_0_6b_sft_paper.yaml
```

快速冒烟（仅验证流程）：

```bash
llamafactory-cli train sft/llama_factory/qwen3_8b_sft_paper.yaml \
  --max_samples 64 \
  --num_train_epochs 0.01 \
  --output_dir outputs/llama_factory/smoke_qwen3_8b
```
