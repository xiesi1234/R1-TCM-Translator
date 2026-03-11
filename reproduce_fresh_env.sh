#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="python3"
VENV_DIR=""
RESULT_DIR=""
INSTALL_DEPS=0
WITH_COMET=0
WITH_TAGGING=0
WITH_RARE_LLM=0
STRICT_INDEX_ALIGNMENT=1

usage() {
  cat <<'EOF'
Usage:
  bash reproduce_fresh_env.sh [options]

Options:
  --python-bin PATH           Python executable (default: python3)
  --venv-dir DIR              Create/use venv at DIR (used with --install-deps)
  --result-dir DIR            Output folder for collected run artifacts
  --install-deps              Install requirements in current python/venv
  --with-comet                Enable COMET steps (01 + 02 COMET)
  --with-tagging              Run 04_challenge_tagging (requires API keys)
  --with-rare-llm             Re-run 03 llm_word_alignment.py (requires API key)
  --strict-index-alignment    Set STRICT_INDEX_ALIGNMENT=1 for 01 script (default)
  --non-strict-index-alignment Set STRICT_INDEX_ALIGNMENT=0 for 01 script
  -h, --help                  Show this help

Examples:
  # Minimal reproducible evaluation (no COMET/API):
  bash reproduce_fresh_env.sh --install-deps --venv-dir .venv_repro

  # Full run with COMET and API modules:
  bash reproduce_fresh_env.sh --install-deps --venv-dir .venv_repro --with-comet --with-tagging --with-rare-llm
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --venv-dir)
      VENV_DIR="$2"
      shift 2
      ;;
    --result-dir)
      RESULT_DIR="$2"
      shift 2
      ;;
    --install-deps)
      INSTALL_DEPS=1
      shift
      ;;
    --with-comet)
      WITH_COMET=1
      shift
      ;;
    --with-tagging)
      WITH_TAGGING=1
      shift
      ;;
    --with-rare-llm)
      WITH_RARE_LLM=1
      shift
      ;;
    --strict-index-alignment)
      STRICT_INDEX_ALIGNMENT=1
      shift
      ;;
    --non-strict-index-alignment)
      STRICT_INDEX_ALIGNMENT=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -n "$VENV_DIR" ]]; then
  if [[ ! -d "$PROJECT_ROOT/$VENV_DIR" ]]; then
    if [[ $INSTALL_DEPS -eq 1 ]]; then
      "$PYTHON_BIN" -m venv "$PROJECT_ROOT/$VENV_DIR"
    else
      echo "ERROR: venv not found: $PROJECT_ROOT/$VENV_DIR (add --install-deps to create it)." >&2
      exit 1
    fi
  fi
  # shellcheck disable=SC1090
  source "$PROJECT_ROOT/$VENV_DIR/bin/activate"
  PYTHON_BIN="python"
fi

if [[ $INSTALL_DEPS -eq 1 ]]; then
  "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
  "$PYTHON_BIN" -m pip install -r "$PROJECT_ROOT/requirements.txt"
fi

if [[ $WITH_TAGGING -eq 1 && -z "${ARK_API_KEY:-}" ]]; then
  echo "ERROR: --with-tagging requires ARK_API_KEY." >&2
  exit 1
fi
if [[ $WITH_TAGGING -eq 1 && -z "${SILICONFLOW_API_KEY:-}" ]]; then
  echo "ERROR: --with-tagging requires SILICONFLOW_API_KEY." >&2
  exit 1
fi
if [[ $WITH_RARE_LLM -eq 1 && -z "${SILICONFLOW_API_KEY:-}" ]]; then
  echo "ERROR: --with-rare-llm requires SILICONFLOW_API_KEY." >&2
  exit 1
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PROJECT_ROOT/repro_logs/$RUN_ID"
mkdir -p "$LOG_DIR"
if [[ -z "$RESULT_DIR" ]]; then
  RESULT_DIR="$PROJECT_ROOT/repro_results/$RUN_ID"
elif [[ "$RESULT_DIR" != /* ]]; then
  RESULT_DIR="$PROJECT_ROOT/$RESULT_DIR"
fi
mkdir -p "$RESULT_DIR"

run_step() {
  local name="$1"
  shift
  echo
  echo "==== [${name}] ===="
  set +e
  (
    cd "$PROJECT_ROOT"
    "$@"
  ) 2>&1 | tee "$LOG_DIR/${name}.log"
  local ec=${PIPESTATUS[0]}
  set -e
  if [[ $ec -ne 0 ]]; then
    echo "FAILED: ${name} (exit ${ec})"
    exit $ec
  fi
}

echo "Project root: $PROJECT_ROOT"
echo "Python: $($PYTHON_BIN -V)"
echo "Logs: $LOG_DIR"
echo "Result dir: $RESULT_DIR"

COMMON_ENV=()
COMMON_ENV+=("STRICT_INDEX_ALIGNMENT=$STRICT_INDEX_ALIGNMENT")
if [[ $WITH_COMET -eq 0 ]]; then
  COMMON_ENV+=("DISABLE_COMET=1")
fi

# 01: overall + four challenges
run_step "01_overall_and_4challenges" env "${COMMON_ENV[@]}" \
  "$PYTHON_BIN" evaluation/01_overall_and_4challenges/calculate_bleu_comet_filtered.py

# 02: long sentence
run_step "02_bleu_by_length" \
  "$PYTHON_BIN" evaluation/02_long_sentence/calculate_bleu_by_length_local.py
run_step "02_generate_latex_plot" \
  "$PYTHON_BIN" evaluation/02_long_sentence/generate_latex_plot.py
if [[ $WITH_COMET -eq 1 ]]; then
  run_step "02_comet_by_length" \
    "$PYTHON_BIN" evaluation/02_long_sentence/calculate_comet_by_length.py
fi

# 03: rare word
run_step "03_build_word_frequency" \
  "$PYTHON_BIN" evaluation/03_rare_word/build_word_frequency.py
run_step "03_analyze_rare_words_correct" \
  "$PYTHON_BIN" evaluation/03_rare_word/analyze_rare_words_correct.py
if [[ $WITH_RARE_LLM -eq 1 ]]; then
  run_step "03_llm_word_alignment" \
    "$PYTHON_BIN" evaluation/03_rare_word/llm_word_alignment.py
fi
run_step "03_evaluate_rare_word_translation" \
  "$PYTHON_BIN" evaluation/03_rare_word/evaluate_rare_word_translation.py

# 04: challenge tagging (optional, API dependent)
if [[ $WITH_TAGGING -eq 1 ]]; then
  run_step "04_semantic_shift" \
    "$PYTHON_BIN" evaluation/04_challenge_tagging/semantic_shift/semantic_check.py
  run_step "04_cultural_context" \
    "$PYTHON_BIN" evaluation/04_challenge_tagging/cultural_context/evaluate_cultural_context.py
  run_step "04_zero_anaphora" \
    "$PYTHON_BIN" evaluation/04_challenge_tagging/zero_anaphora/zero_anaphora_check.py
  run_step "04_parataxis" \
    "$PYTHON_BIN" evaluation/04_challenge_tagging/parataxis/evaluate_parataxis.py
fi

copy_artifact() {
  local rel="$1"
  local src="$PROJECT_ROOT/$rel"
  local dst="$RESULT_DIR/$rel"
  if [[ -f "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
  else
    echo "MISSING $rel" >> "$RESULT_DIR/missing_artifacts.txt"
  fi
}

copy_artifact "evaluation/01_overall_and_4challenges/filtered_challenge_results.json"
copy_artifact "evaluation/02_long_sentence/bleu_by_length_results.json"
copy_artifact "evaluation/02_long_sentence/comet_by_length_results.json"
copy_artifact "evaluation/02_long_sentence/grpo_comparison.tex"
copy_artifact "evaluation/03_rare_word/word_frequency_distribution.json"
copy_artifact "evaluation/03_rare_word/word_frequency_all.json"
copy_artifact "evaluation/03_rare_word/tcm_rare_word_sft_grpo_results.json"
copy_artifact "evaluation/03_rare_word/tcm_rare_word_sft_grpo_analysis.tex"
copy_artifact "evaluation/03_rare_word/rare_word_evaluation_results.json"
copy_artifact "evaluation/03_rare_word/rare_word_evaluation_report.txt"
copy_artifact "evaluation/03_rare_word/paper_llm_judgment/word_alignment_evaluation_results_single.json"
copy_artifact "evaluation/03_rare_word/paper_llm_judgment/word_alignment_evaluation_report_single.txt"
copy_artifact "evaluation/04_challenge_tagging/semantic_shift/cache_semantic.json"
copy_artifact "evaluation/04_challenge_tagging/cultural_context/cache_cultural.json"
copy_artifact "evaluation/04_challenge_tagging/zero_anaphora/cache_zero_anaphora.json"
copy_artifact "evaluation/04_challenge_tagging/parataxis/cache_parataxis.json"

mkdir -p "$RESULT_DIR/logs"
cp -r "$LOG_DIR/." "$RESULT_DIR/logs/"

cat > "$RESULT_DIR/run_config.txt" <<EOF
RUN_ID=$RUN_ID
PROJECT_ROOT=$PROJECT_ROOT
PYTHON_BIN=$PYTHON_BIN
VENV_DIR=$VENV_DIR
INSTALL_DEPS=$INSTALL_DEPS
WITH_COMET=$WITH_COMET
WITH_TAGGING=$WITH_TAGGING
WITH_RARE_LLM=$WITH_RARE_LLM
STRICT_INDEX_ALIGNMENT=$STRICT_INDEX_ALIGNMENT
EOF

"$PYTHON_BIN" - <<PY
import json
from pathlib import Path

result_dir = Path(r"$RESULT_DIR")
project_root = Path(r"$PROJECT_ROOT")

def read_json(path):
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None

summary = {
    "run_id": "$RUN_ID",
    "run_flags": {
        "with_comet": bool(int("$WITH_COMET")),
        "with_tagging": bool(int("$WITH_TAGGING")),
        "with_rare_llm": bool(int("$WITH_RARE_LLM")),
        "strict_index_alignment": bool(int("$STRICT_INDEX_ALIGNMENT")),
    },
    "paper_targets": {
        "overall": {
            "R1-TCM-Translator-0.6B": {"BLEU": 29.5, "COMET": 77.4},
            "R1-TCM-Translator-8B": {"BLEU": 43.3, "COMET": 79.6},
        },
        "rare_word_llm_judgment_accuracy": {
            "Qwen3-0.6B-SFT": 22.36,
            "Qwen3-0.6B-GRPO": 54.80,
            "Qwen3-8B-SFT": 57.65,
            "Qwen3-8B-GRPO": 80.68,
        },
    },
    "current_run": {},
}

overall_file = result_dir / "evaluation/01_overall_and_4challenges/filtered_challenge_results.json"
overall = read_json(overall_file)
if overall and "overall" in overall:
    cur = {}
    map_name = {
        "Qwen3-0.6B-SFT+GRPO": "R1-TCM-Translator-0.6B",
        "Qwen3-8B-SFT+GRPO": "R1-TCM-Translator-8B",
    }
    for src_name, dst_name in map_name.items():
        if src_name in overall["overall"]:
            item = overall["overall"][src_name]
            cur[dst_name] = {
                "BLEU": item.get("avg_bleu"),
                "COMET": item.get("avg_comet"),
                "bleu_by_book": item.get("bleu_by_book", {}),
                "comet_by_book": item.get("comet_by_book", {}),
            }
    summary["current_run"]["overall"] = cur

rare_file = result_dir / "evaluation/03_rare_word/rare_word_evaluation_results.json"
rare = read_json(rare_file)
if rare:
    summary["current_run"]["rare_word_open_source_pipeline_accuracy"] = {
        k: v.get("accuracy")
        for k, v in rare.items()
        if isinstance(v, dict) and "accuracy" in v
    }

rare_paper_file = result_dir / "evaluation/03_rare_word/paper_llm_judgment/word_alignment_evaluation_results_single.json"
rare_paper = read_json(rare_paper_file)
if rare_paper:
    summary["current_run"]["rare_word_paper_llm_judgment_accuracy"] = {
        k: v.get("accuracy")
        for k, v in rare_paper.items()
        if isinstance(v, dict) and "accuracy" in v
    }
    summary["current_run"]["rare_word_paper_llm_judgment_source"] = (
        "recomputed_in_this_run" if bool(int("$WITH_RARE_LLM")) else "archived_file"
    )

with (result_dir / "result_summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
PY

echo
echo "All selected steps finished."
echo "Logs saved to: $LOG_DIR"
echo "Artifacts saved to: $RESULT_DIR"
