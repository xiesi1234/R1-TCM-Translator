#!/usr/bin/env python3
"""
Rare-word prediction analysis for TCM classical translation.
Compute token frequency on the training corpus and evaluate model prediction
accuracy across frequency bins on the test set.
"""

import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, List

import jieba


def load_training_data(filepath: str) -> List[str]:
    """Load training data and extract modern-Chinese translations."""
    print(f"Loading training set: {filepath}")
    translations: List[str] = []

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check data schema.
    if data and isinstance(data, list):
        first_item = data[0]

        # Format 1: GRPO data with `modern` field.
        if "modern" in first_item:
            for item in data:
                modern_text = item.get("modern", "").strip()
                if modern_text:
                    translations.append(modern_text)

        # Format 2: SFT data with `output` field.
        elif "output" in first_item:
            for item in data:
                output = item.get("output", "")
                # Keep these Chinese markers for dataset compatibility.
                if "现代文：" in output:
                    modern_text = output.split("现代文：")[-1].strip()
                    translations.append(modern_text)
                elif output and not output.startswith("内容出自"):
                    translations.append(output.strip())

    print(f"Extracted {len(translations)} translation texts from training set")
    return translations


def load_text_file(filepath: str) -> List[str]:
    """Load a text file as stripped lines."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def load_predictions_from_detailed_json(filepath: str) -> List[str]:
    """Load predictions from a detailed JSON file."""
    if not os.path.exists(filepath):
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = []
    for item in data:
        # Prefer extracted_answer when available.
        if "extracted_answer" in item:
            predictions.append(item["extracted_answer"].strip())
        # Otherwise parse the <answer> field from full_model_output.
        elif "full_model_output" in item:
            output = item["full_model_output"]
            if "<answer>" in output and "</answer>" in output:
                answer = output.split("<answer>")[1].split("</answer>")[0].strip()
                predictions.append(answer)
            else:
                predictions.append(output.strip())
        else:
            predictions.append("")

    return predictions


def tokenize_chinese(text: str) -> List[str]:
    """Tokenize Chinese text with jieba."""
    return list(jieba.cut(text))


def calculate_word_frequency(texts: List[str]) -> Counter:
    """Compute token frequency from text list."""
    print("Computing token frequencies...")
    word_counter = Counter()
    for i, text in enumerate(texts):
        if i % 10000 == 0:
            print(f"  Progress: {i}/{len(texts)}")
        words = tokenize_chinese(text)
        word_counter.update(words)
    print(f"Frequency counting finished. Vocabulary size: {len(word_counter)}")
    return word_counter


def find_project_root(start_dir: str) -> str:
    """Search upward for the project root that contains reference_answers."""
    cur = start_dir
    for _ in range(8):
        if os.path.exists(os.path.join(cur, "reference_answers")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return start_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rare-word prediction analysis for TCM classical translation"
    )
    parser.add_argument(
        "--train-file",
        default=None,
        help="Training set file path (default: grpo/data/tcm_ancient_modern_grpo_train_12145.json)",
    )
    parser.add_argument(
        "--runs",
        default="1,2,3",
        help="Comma-separated decoding runs to evaluate (default: 1,2,3)",
    )
    return parser.parse_args()


def parse_runs(runs_text: str) -> List[int]:
    runs = []
    for part in runs_text.split(","):
        part = part.strip()
        if not part:
            continue
        runs.append(int(part))
    if not runs:
        raise ValueError("No valid run IDs were parsed; use --runs like 1,2,3")
    return sorted(set(runs))


def load_indices(project_root: str, book: str) -> List[int]:
    idx_file = os.path.join(project_root, f"reference_answers/{book}_indices.txt")
    if not os.path.exists(idx_file):
        return []
    with open(idx_file, "r", encoding="utf-8") as f:
        return [int(l.strip()) for l in f if l.strip()]


def align_predictions(
    predictions: List[str], indices: List[int], model_name: str, book: str, run: int
) -> List[str]:
    """Align predictions to reference_answers index protocol."""
    if not indices:
        return predictions
    if len(predictions) == len(indices):
        return predictions

    max_idx = max(indices)
    if max_idx < len(predictions):
        return [predictions[idx] for idx in indices]

    invalid_count = sum(1 for idx in indices if idx >= len(predictions))
    strict = os.getenv("STRICT_INDEX_ALIGNMENT", "0") == "1"
    msg = (
        f"{model_name} {book} run{run} index out of range: "
        f"indices={len(indices)}, max_idx={max_idx}, predictions={len(predictions)}, invalid={invalid_count}"
    )
    if strict:
        raise IndexError(msg)

    print(f"  Warning: {msg}; out-of-range indices were dropped automatically.")
    valid_indices = [idx for idx in indices if idx < len(predictions)]
    return [predictions[idx] for idx in valid_indices]


def analyze_by_frequency_bins(
    references: List[str], predictions: List[str], word_freq: Counter
) -> Dict:
    """Compute accuracy by token frequency bins."""
    # Frequency bins following Wu et al. 2016 style boundaries.
    freq_bins = [
        (1, 3, "1-2"),
        (3, 5, "2-4"),
        (5, 9, "4-8"),
        (9, 17, "8-16"),
        (17, 33, "16-32"),
        (33, 65, "32-64"),
        (65, 129, "64-128"),
        (129, 257, "128-256"),
        (257, 513, "256-512"),
    ]

    results = {}

    for freq_min, freq_max, label in freq_bins:
        total_words = 0
        correct_words = 0

        for ref, pred in zip(references, predictions):
            ref_words = tokenize_chinese(ref)
            pred_text = pred

            for word in ref_words:
                freq = word_freq.get(word, 0)
                # Exclude punctuation and single-character tokens.
                if (
                    freq_min <= freq < freq_max
                    and len(word.strip()) > 1
                    and not re.match(r"^[^\u4e00-\u9fa5]+$", word)
                ):
                    total_words += 1
                    if word in pred_text:
                        correct_words += 1

        accuracy = (correct_words / total_words * 100) if total_words > 0 else 0

        results[label] = {
            "freq_range": f"{freq_min}-{freq_max}",
            "count": total_words,
            "accuracy": accuracy,
        }

    return results


def generate_latex_table(all_results: Dict) -> str:
    """Generate the LaTeX table."""
    # Output order for models.
    model_configs = [
        ("Qwen3-0.6B-SFT", "Qwen3-0.6B-SFT"),
        ("Qwen3-0.6B-GRPO", "R1-TCM-Translator-0.6B"),
        ("Qwen3-8B-SFT", "Qwen3-8B-SFT"),
        ("Qwen3-8B-GRPO", "R1-TCM-Translator-8B"),
    ]

    # Get all frequency-bin labels.
    freq_labels = list(list(all_results.values())[0].keys()) if all_results else []

    latex = []
    latex.append(r"\begin{table}[!htbp]")
    latex.append(r"\centering")
    latex.append(
        r"\caption{Translation accuracy by training-corpus frequency bins for TCM classical translation. "
        r"\textit{Frequency range} indicates the occurrence interval in the training corpus, "
        r"\textit{Count} is the number of test tokens in that bin, and \textit{ACC} is token-level prediction accuracy (\%).}"
    )
    latex.append(r"\label{tab:rare_word_analysis}")
    latex.append(r"\begin{adjustbox}{max width=\textwidth}")

    # Header rows.
    latex.append(r"\begin{tabular}{lrrrrrrrr}")
    latex.append(r"\toprule")

    header1 = r"\multirow{2}{*}{\textbf{Frequency Bin}}"
    for _, display_name in model_configs:
        header1 += f" & \\multicolumn{{2}}{{c}}{{\\textbf{{{display_name}}}}}"
    header1 += r" \\"
    latex.append(header1)

    header2 = ""
    for _ in model_configs:
        header2 += r" & \textbf{Count} & \textbf{ACC(\%)}"
    header2 += r" \\"
    latex.append(
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}"
    )
    latex.append(header2)
    latex.append(r"\midrule")

    # Data rows.
    for label in freq_labels:
        row = label
        for model_key, _ in model_configs:
            if model_key in all_results and label in all_results[model_key]:
                stats = all_results[model_key][label]
                if stats["count"] > 0:
                    row += f" & {stats['count']} & {stats['accuracy']:.2f}"
                else:
                    row += " & - & -"
            else:
                row += " & - & -"
        row += r" \\"
        latex.append(row)

    latex.append(r"\midrule")

    # Low-frequency average (<16).
    low_freq_labels = ["1-2", "2-4", "4-8", "8-16"]
    row = r"\textbf{Low-Frequency Avg}"
    for model_key, _ in model_configs:
        if model_key in all_results:
            accs = [
                all_results[model_key][l]["accuracy"]
                for l in low_freq_labels
                if l in all_results[model_key] and all_results[model_key][l]["count"] > 0
            ]
            avg = sum(accs) / len(accs) if accs else 0
            row += f" & - & \\textbf{{{avg:.2f}}}"
        else:
            row += " & - & -"
    row += r" \\"
    latex.append(row)

    # Mid-frequency average (16-512).
    mid_freq_labels = ["16-32", "32-64", "64-128", "128-256", "256-512"]
    row = r"\textbf{Mid-Frequency Avg}"
    for model_key, _ in model_configs:
        if model_key in all_results:
            accs = [
                all_results[model_key][l]["accuracy"]
                for l in mid_freq_labels
                if l in all_results[model_key] and all_results[model_key][l]["count"] > 0
            ]
            avg = sum(accs) / len(accs) if accs else 0
            row += f" & - & \\textbf{{{avg:.2f}}}"
        else:
            row += " & - & -"
    row += r" \\"
    latex.append(row)

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{adjustbox}")
    latex.append(r"\end{table}")

    return "\n".join(latex)


def generate_analysis_paragraph(all_results: Dict) -> str:
    """Generate LaTeX analysis paragraph."""

    def calc_avg_acc(results, labels):
        accs = []
        for label in labels:
            if label in results and results[label]["count"] > 0:
                accs.append(results[label]["accuracy"])
        return sum(accs) / len(accs) if accs else 0

    low_freq_labels = ["1-2", "2-4", "4-8", "8-16"]
    mid_freq_labels = ["16-32", "32-64", "64-128", "128-256", "256-512"]

    paragraphs = []
    paragraphs.append(r"\paragraph{Key Findings}")
    paragraphs.append("")
    paragraphs.append(r"\begin{itemize}")

    # 0.6B model comparison.
    if "Qwen3-0.6B-SFT" in all_results and "Qwen3-0.6B-GRPO" in all_results:
        sft_low_06 = calc_avg_acc(all_results["Qwen3-0.6B-SFT"], low_freq_labels)
        grpo_low_06 = calc_avg_acc(all_results["Qwen3-0.6B-GRPO"], low_freq_labels)
        improvement_06 = (
            ((grpo_low_06 - sft_low_06) / sft_low_06 * 100) if sft_low_06 > 0 else 0
        )

        sft_mid_06 = calc_avg_acc(all_results["Qwen3-0.6B-SFT"], mid_freq_labels)
        grpo_mid_06 = calc_avg_acc(all_results["Qwen3-0.6B-GRPO"], mid_freq_labels)
        improvement_mid_06 = (
            ((grpo_mid_06 - sft_mid_06) / sft_mid_06 * 100) if sft_mid_06 > 0 else 0
        )

    # 8B model comparison.
    if "Qwen3-8B-SFT" in all_results and "Qwen3-8B-GRPO" in all_results:
        sft_low = calc_avg_acc(all_results["Qwen3-8B-SFT"], low_freq_labels)
        grpo_low = calc_avg_acc(all_results["Qwen3-8B-GRPO"], low_freq_labels)
        improvement = ((grpo_low - sft_low) / sft_low * 100) if sft_low > 0 else 0

        sft_mid = calc_avg_acc(all_results["Qwen3-8B-SFT"], mid_freq_labels)
        grpo_mid = calc_avg_acc(all_results["Qwen3-8B-GRPO"], mid_freq_labels)
        improvement_mid = ((grpo_mid - sft_mid) / sft_mid * 100) if sft_mid > 0 else 0

        if "Qwen3-0.6B-SFT" in all_results and "Qwen3-0.6B-GRPO" in all_results:
            paragraphs.append(
                f"    \\item \\textbf{{Low-frequency gains}}: For rare words (<16 occurrences), "
                f"the 0.6B model improves from {sft_low_06:.2f}\\% to {grpo_low_06:.2f}\\% "
                f"(relative +{improvement_06:.1f}\\%) after GRPO; the 8B model improves "
                f"from {sft_low:.2f}\\% to {grpo_low:.2f}\\% (relative +{improvement:.1f}\\%)."
            )
            paragraphs.append(
                f"    \\item \\textbf{{Mid-frequency translation}}: For words with 16--512 occurrences, "
                f"the 0.6B GRPO model reaches {grpo_mid_06:.2f}\\% "
                f"(relative +{improvement_mid_06:.1f}\\% vs SFT), while the 8B GRPO model reaches "
                f"{grpo_mid:.2f}\\% (relative +{improvement_mid:.1f}\\% vs SFT)."
            )
        else:
            paragraphs.append(
                f"    \\item \\textbf{{Low-frequency gains}}: For rare words (<16 occurrences), "
                f"the 8B model improves from {sft_low:.2f}\\% to {grpo_low:.2f}\\% "
                f"(relative +{improvement:.1f}\\%) after GRPO."
            )
            paragraphs.append(
                f"    \\item \\textbf{{Mid-frequency translation}}: For words with 16--512 occurrences, "
                f"the 8B GRPO model reaches {grpo_mid:.2f}\\% "
                f"(relative +{improvement_mid:.1f}\\% vs SFT)."
            )

    paragraphs.append(
        r"    \item \textbf{Frequency-accuracy trend}: Accuracy increases with token frequency, "
        r"which matches the behavior of causal language modeling where high-frequency tokens "
        r"are observed more often during training~\cite{radford2018improving}."
    )

    paragraphs.append(
        r"    \item \textbf{Rare-word challenge}: Even with SFT and GRPO gains, extremely low-frequency "
        r"tokens (<4 occurrences) remain difficult and still underperform relative to high-frequency bins, "
        r"consistent with reported MT challenges in the LLM era~\cite{pang2024mtchallenges}."
    )

    paragraphs.append(r"\end{itemize}")

    return "\n".join(paragraphs)


def main():
    args = parse_args()
    runs = parse_runs(args.runs)

    print("=" * 100)
    print("Rare-word prediction analysis based on training-set token frequencies")
    print("=" * 100)

    # Ensure output directory exists.
    output_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Output directory: {output_dir}\n")

    # 1) Load training set and compute token frequency.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = find_project_root(script_dir)
    default_train_file = os.path.join(
        project_root, "grpo/data/tcm_ancient_modern_grpo_train_12145.json"
    )
    training_file = os.path.abspath(args.train_file or default_train_file)

    if not os.path.exists(training_file):
        raise FileNotFoundError(
            f"Training file not found: {training_file}\n"
            "Please provide a valid path via --train-file."
        )

    print(f"Training file found: {training_file}")

    training_texts = load_training_data(training_file)
    word_freq = calculate_word_frequency(training_texts)

    # Print frequency-bin statistics.
    print("\nFrequency-bin statistics:")
    freq_ranges = [
        (1, 3, "1-2"),
        (3, 5, "2-4"),
        (5, 9, "4-8"),
        (9, 17, "8-16"),
        (17, 33, "16-32"),
        (33, 65, "32-64"),
        (65, 129, "64-128"),
        (129, 257, "128-256"),
        (257, 513, "256-512"),
    ]

    # Save words for each frequency bin.
    freq_words_by_bin = {}
    for freq_min, freq_max, label in freq_ranges:
        words_in_bin = [(w, f) for w, f in word_freq.items() if freq_min <= f < freq_max]
        count = len(words_in_bin)
        print(f"  {label}: {count} words")
        freq_words_by_bin[label] = sorted(words_in_bin, key=lambda x: x[1], reverse=True)

    # Save frequency distribution JSON.
    word_freq_dir = os.path.join(output_dir, "word_freq")
    os.makedirs(word_freq_dir, exist_ok=True)

    freq_distribution_output = os.path.join(word_freq_dir, "word_frequency_distribution.json")
    freq_distribution_data = {}
    for label, words in freq_words_by_bin.items():
        freq_distribution_data[label] = {
            "count": len(words),
            "words": [{"word": w, "frequency": f} for w, f in words],
        }

    with open(freq_distribution_output, "w", encoding="utf-8") as f:
        json.dump(freq_distribution_data, f, ensure_ascii=False, indent=2)
    print(f"\nFrequency distribution saved to: {freq_distribution_output}")

    # Save per-bin word lists as text files.
    for label, words in freq_words_by_bin.items():
        txt_filename = os.path.join(word_freq_dir, f"words_freq_{label.replace('-', '_')}.txt")
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(f"# Frequency bin: {label}\n")
            f.write(f"# Total words: {len(words)}\n")
            f.write("# Format: word\\tfrequency\n\n")
            for word, freq in words:
                f.write(f"{word}\t{freq}\n")
        print(f"  Saved words for bin {label}: {txt_filename}")

    # 2) Test sets.
    test_sets = [
        "shanghanlun",
        "huangdineijing",
        "jingkuiyaolue",
        "maijing",
        "nanjing",
        "bianquexinshu",
        "sishengxinyuan",
        "wenbingtiaobian",
    ]

    # 3) Model config.
    models = {
        "Qwen3-0.6B-SFT": "experiments/Qwen3-0.6B/SFT",
        "Qwen3-0.6B-GRPO": "experiments/Qwen3-0.6B/SFT_GRPO/reward_bleu_comet/checkpoints/Qwen3-0.6B-checkpoint-4000",
        "Qwen3-8B-SFT": "experiments/Qwen3-8B/SFT",
        "Qwen3-8B-GRPO": "experiments/Qwen3-8B/SFT_GRPO/reward_bleu_comet/checkpoints/Qwen3-8B-checkpoint-1000",
    }

    # 4) Evaluate on test sets.
    all_results: Dict[str, Dict] = {}

    for model_name, model_path in models.items():
        print(f"\nProcessing model: {model_name}")
        run_results = []
        run_counts = []

        for run in runs:
            all_refs = []
            all_preds = []
            for test_set in test_sets:
                ref_path = f"reference_answers/{test_set}_references.txt"
                detailed_json_path = f"{model_path}/{test_set}_test/{test_set}_test_detailed_{run}.json"
                pred_txt_path = f"{model_path}/{test_set}_test/{test_set}_test_answers_{run}.txt"

                if not os.path.exists(ref_path):
                    continue

                if os.path.exists(detailed_json_path):
                    preds = load_predictions_from_detailed_json(detailed_json_path)
                elif os.path.exists(pred_txt_path):
                    preds = load_text_file(pred_txt_path)
                else:
                    continue

                refs = load_text_file(ref_path)
                indices = load_indices(project_root, test_set)
                preds = align_predictions(preds, indices, model_name, test_set, run)

                min_len = min(len(refs), len(preds))
                all_refs.extend(refs[:min_len])
                all_preds.extend(preds[:min_len])

            if all_refs:
                run_results.append(analyze_by_frequency_bins(all_refs, all_preds, word_freq))
                run_counts.append(len(all_refs))
                print(f"  run{run}: {len(all_refs)} samples")

        if run_results:
            freq_labels = list(run_results[0].keys())
            merged_results = {}
            for label in freq_labels:
                counts = [rr[label]["count"] for rr in run_results if label in rr]
                accuracies = [rr[label]["accuracy"] for rr in run_results if label in rr]
                merged_results[label] = {
                    "freq_range": run_results[0][label]["freq_range"],
                    "count": int(round(sum(counts) / len(counts))) if counts else 0,
                    "accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
                }
            all_results[model_name] = merged_results
            print(
                f"  Mean sample count: {int(round(sum(run_counts) / len(run_counts)))} "
                f"(runs={runs})"
            )

    if not all_results:
        print("\nError: no valid model results were found")
        return

    # 5) Print detailed results.
    print("\n" + "=" * 120)
    print("Table: Test-set translation accuracy by training-set frequency bins")
    print("=" * 120)

    freq_labels = list(list(all_results.values())[0].keys())

    header = f"{'Freq Bin':<12}"
    for model_name in models.keys():
        header += f" | {model_name:>25}"
    print(header)
    print("-" * len(header))

    for label in freq_labels:
        row = f"{label:<12}"
        for model_name in models.keys():
            if model_name in all_results and label in all_results[model_name]:
                stats = all_results[model_name][label]
                if stats["count"] > 0:
                    row += f" | {stats['count']:>7} ({stats['accuracy']:>5.2f}%)"
                else:
                    row += f" | {'-':>25}"
            else:
                row += f" | {'-':>25}"
        print(row)

    # 6) Key summary.
    print("\n" + "=" * 100)
    print("Key summary")
    print("=" * 100)

    low_freq_labels = ["1-2", "2-4", "4-8", "8-16"]
    mid_freq_labels = ["16-32", "32-64", "64-128", "128-256", "256-512"]

    def calc_avg(results, labels):
        accs = []
        for label in labels:
            if label in results and results[label]["count"] > 0:
                accs.append(results[label]["accuracy"])
        return sum(accs) / len(accs) if accs else 0

    print("\nLow-frequency words (<16) average accuracy:")
    for model_name in models.keys():
        if model_name in all_results:
            avg = calc_avg(all_results[model_name], low_freq_labels)
            print(f"  {model_name}: {avg:.2f}%")

    print("\nMid-frequency words (16-512) average accuracy:")
    for model_name in models.keys():
        if model_name in all_results:
            avg = calc_avg(all_results[model_name], mid_freq_labels)
            print(f"  {model_name}: {avg:.2f}%")

    # 7) Generate LaTeX output.
    print("\n" + "=" * 100)
    print("Generating LaTeX output")
    print("=" * 100)

    latex_table = generate_latex_table(all_results)
    latex_analysis = generate_analysis_paragraph(all_results)

    output_path = os.path.join(output_dir, "tcm_rare_word_sft_grpo_analysis.tex")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("% Rare-word analysis for TCM classical translation\n")
        f.write("% Auto-generated LaTeX\n\n")
        f.write(latex_table)
        f.write("\n\n")
        f.write(latex_analysis)

    print(f"\nLaTeX saved to: {output_path}")

    # 8) Save JSON output.
    json_output = os.path.join(output_dir, "tcm_rare_word_sft_grpo_results.json")
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Detailed results saved to: {json_output}")


if __name__ == "__main__":
    main()
