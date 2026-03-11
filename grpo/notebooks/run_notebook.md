# Run GRPO Notebook

Notebook file: `grpo.ipynb`

## 1) Go to notebook directory

```bash
cd grpo/notebooks
```

## 2) Set runtime variables (recommended)

```bash
export MODEL_NAME="Qwen/Qwen3-0.6B"
export DATA_FILE="../data/tcm_ancient_modern_grpo_train_12145.json"
export OUTPUT_DIR="../outputs/qwen3_0_6b_comet_grpo"
export USE_SWANLAB=0
```

If you want SwanLab tracking:

```bash
export USE_SWANLAB=1
```

## 3) Start notebook

```bash
jupyter notebook "grpo.ipynb"
```

Then execute cells from top to bottom.
