#!/usr/bin/env python3
"""
Evaluate rare-word translation accuracy based on LLM word alignments.
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple


STRICT_INDEX_ALIGNMENT = os.getenv("STRICT_INDEX_ALIGNMENT", "0") == "1"


def load_alignment_results(alignment_file: str) -> List[Dict]:
    """Load word alignment results."""
    if not os.path.exists(alignment_file):
        return []

    with open(alignment_file, "r", encoding="utf-8") as f:
        return json.load(f)


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


def load_indices(project_root: str, book: str) -> List[int]:
    idx_file = os.path.join(project_root, f"reference_answers/{book}_indices.txt")
    if not os.path.exists(idx_file):
        return []
    with open(idx_file, "r", encoding="utf-8") as f:
        return [int(l.strip()) for l in f if l.strip()]


def align_predictions(predictions: List[str], indices: List[int], model_name: str, book: str, run: int) -> List[str]:
    """Align model outputs to the reference_answers index protocol."""
    if not indices:
        return predictions
    if len(predictions) == len(indices):
        return predictions

    max_idx = max(indices)
    if max_idx < len(predictions):
        return [predictions[idx] for idx in indices]

    invalid_count = sum(1 for idx in indices if idx >= len(predictions))
    msg = (
        f"{model_name} {book} run{run} index out of range: "
        f"indices={len(indices)}, max_idx={max_idx}, predictions={len(predictions)}, invalid={invalid_count}"
    )
    if STRICT_INDEX_ALIGNMENT:
        raise IndexError(msg)

    print(f"  Warning: {msg}; out-of-range indices were dropped automatically.")
    valid_indices = [idx for idx in indices if idx < len(predictions)]
    return [predictions[idx] for idx in valid_indices]


def load_model_predictions(pred_file: str) -> List[str]:
    """Load model predictions."""
    if not os.path.exists(pred_file):
        return []

    # Try detailed JSON first.
    if pred_file.endswith(".json"):
        with open(pred_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        predictions = []
        for item in data:
            if "extracted_answer" in item:
                predictions.append(item["extracted_answer"].strip())
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

    # Load plain-text output; keep blank lines to preserve index alignment.
    with open(pred_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def evaluate_rare_word_alignment(
    alignments: List[Dict],
    predictions: List[str],
    rare_vocab: Set[str],
) -> Dict:
    """Evaluate rare-word translation accuracy."""
    results = {
        "total_sentences": 0,
        "sentences_with_rare_words": 0,
        "total_rare_word_alignments": 0,
        "correct_translations": 0,
        "partial_translations": 0,
        "incorrect_translations": 0,
        "by_type": defaultdict(lambda: {"total": 0, "correct": 0}),
        "examples": {"correct": [], "incorrect": []},
    }

    for alignment in alignments:
        idx = alignment.get("index", 0)

        # Skip invalid indices.
        if idx >= len(predictions):
            continue

        pred_text = predictions[idx]
        rare_words = alignment.get("rare_words", [])
        alignment_pairs = alignment.get("alignments", [])

        # Backfill rare_words for legacy alignment files when possible.
        if not rare_words and rare_vocab:
            rare_words = [
                ((pair.get("ancient") or "").strip())
                for pair in alignment_pairs
                if ((pair.get("ancient") or "").strip()) in rare_vocab
            ]

        results["total_sentences"] += 1

        if not rare_words:
            continue

        results["sentences_with_rare_words"] += 1

        # Evaluate alignment for each rare word.
        for align_pair in alignment_pairs:
            ancient_word = align_pair.get("ancient", "")
            modern_word = align_pair.get("modern", "")
            align_type = align_pair.get("type", "")

            # Evaluate rare words only.
            if ancient_word not in rare_words:
                continue
            if not isinstance(modern_word, str) or not modern_word.strip():
                continue

            results["total_rare_word_alignments"] += 1
            results["by_type"][align_type]["total"] += 1

            # Exact match at token/string level.
            is_correct = modern_word in pred_text

            if is_correct:
                results["correct_translations"] += 1
                results["by_type"][align_type]["correct"] += 1

                # Keep up to 10 correct examples.
                if len(results["examples"]["correct"]) < 10:
                    results["examples"]["correct"].append(
                        {
                            "ancient_text": alignment["ancient_text"],
                            "modern_text": alignment["modern_text"],
                            "prediction": pred_text,
                            "ancient_word": ancient_word,
                            "modern_word": modern_word,
                            "type": align_type,
                        }
                    )
            else:
                # Partial match if any character of modern_word appears in prediction.
                if any(char in pred_text for char in modern_word):
                    results["partial_translations"] += 1
                else:
                    results["incorrect_translations"] += 1

                # Keep up to 10 incorrect examples.
                if len(results["examples"]["incorrect"]) < 10:
                    results["examples"]["incorrect"].append(
                        {
                            "ancient_text": alignment["ancient_text"],
                            "modern_text": alignment["modern_text"],
                            "prediction": pred_text,
                            "ancient_word": ancient_word,
                            "expected_modern_word": modern_word,
                            "type": align_type,
                        }
                    )

    # Compute accuracy metrics.
    if results["total_rare_word_alignments"] > 0:
        results["accuracy"] = (
            results["correct_translations"] / results["total_rare_word_alignments"] * 100
        )
        results["partial_accuracy"] = (
            (results["correct_translations"] + results["partial_translations"])
            / results["total_rare_word_alignments"]
            * 100
        )
    else:
        results["accuracy"] = 0
        results["partial_accuracy"] = 0

    # Compute per-alignment-type accuracy.
    for align_type, stats in results["by_type"].items():
        if stats["total"] > 0:
            stats["accuracy"] = stats["correct"] / stats["total"] * 100

    return results


def generate_report(all_results: Dict[str, Dict]) -> str:
    """Generate evaluation report text."""
    report = []
    report.append("=" * 100)
    report.append("Rare-word translation accuracy report based on LLM alignments")
    report.append("=" * 100)
    report.append("")

    # Overall summary.
    report.append("## Overall Summary")
    report.append("")

    total_sentences = sum(r["total_sentences"] for r in all_results.values())
    total_rare_alignments = sum(r["total_rare_word_alignments"] for r in all_results.values())
    total_correct = sum(r["correct_translations"] for r in all_results.values())

    report.append(f"Total sentences: {total_sentences}")
    report.append(
        f"Sentences with rare words: {sum(r['sentences_with_rare_words'] for r in all_results.values())}"
    )
    report.append(f"Total rare-word alignments: {total_rare_alignments}")
    report.append(f"Correct translations: {total_correct}")

    if total_rare_alignments > 0:
        overall_accuracy = total_correct / total_rare_alignments * 100
        report.append(f"Overall accuracy: {overall_accuracy:.2f}%")

    report.append("")

    # Per-model details.
    report.append("## Per-Model Details")
    report.append("")

    for model_name, results in all_results.items():
        report.append(f"### {model_name}")
        report.append("")
        report.append(f"- Total sentences: {results['total_sentences']}")
        report.append(f"- Sentences with rare words: {results['sentences_with_rare_words']}")
        report.append(f"- Rare-word alignments: {results['total_rare_word_alignments']}")
        report.append(f"- Correct translations: {results['correct_translations']}")
        report.append(f"- Partially correct: {results['partial_translations']}")
        report.append(f"- Incorrect translations: {results['incorrect_translations']}")
        report.append(f"- **Accuracy: {results['accuracy']:.2f}%**")
        report.append(f"- Partial accuracy: {results['partial_accuracy']:.2f}%")
        report.append("")

        # Per-type statistics.
        if results["by_type"]:
            report.append("By alignment type:")
            for align_type, stats in results["by_type"].items():
                report.append(
                    f"  - {align_type}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.2f}%)"
                )
            report.append("")

    report.append("=" * 100)

    return "\n".join(report)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate rare-word translation accuracy based on LLM word alignments"
    )
    parser.add_argument(
        "--alignment-dir",
        default=None,
        help="Directory containing alignment files (default: evaluation/03_rare_word)",
    )
    parser.add_argument(
        "--freq-file",
        default=None,
        help="Path to word frequency distribution JSON (default: evaluation/03_rare_word/word_frequency_distribution.json)",
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


def validate_alignment_files(test_sets: List[str], alignment_dir: str) -> Dict[str, str]:
    alignment_files = {}
    missing = []
    for test_set in test_sets:
        preferred = os.path.join(alignment_dir, f"{test_set}_word_alignments.json")
        fallback = os.path.join(alignment_dir, f"{test_set}_word_alignments_all_freq.json")
        if os.path.exists(preferred):
            alignment_files[test_set] = preferred
        elif os.path.exists(fallback):
            alignment_files[test_set] = fallback
        else:
            missing.append(preferred)
    if missing:
        missing_preview = "\n".join(f"  - {p}" for p in missing[:10])
        raise FileNotFoundError(
            "Missing alignment files; evaluation stopped to avoid misleading zero-score output.\n"
            f"Missing file count: {len(missing)}\n"
            f"{missing_preview}\n"
            "Please run llm_word_alignment.py first, or set --alignment-dir to the correct folder."
        )
    return alignment_files


def load_rare_vocab(freq_file: str) -> Set[str]:
    """Load low-frequency vocabulary (<16 occurrences) from distribution JSON."""
    if not os.path.exists(freq_file):
        raise FileNotFoundError(f"Word frequency distribution file not found: {freq_file}")
    with open(freq_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    rare_bins = {"1-2", "2-4", "4-8", "8-16"}
    vocab = set()
    for bin_label in rare_bins:
        bin_data = data.get(bin_label, {})
        for item in bin_data.get("words", []):
            word = (item.get("word") or "").strip()
            if word:
                vocab.add(word)
    if not vocab:
        raise RuntimeError(
            "No low-frequency vocabulary (<16 occurrences) parsed from frequency distribution JSON."
        )
    return vocab


def main():
    print("=" * 100)
    print("Evaluate rare-word translation accuracy (based on LLM alignments)")
    print("=" * 100)

    args = parse_args()
    runs = parse_runs(args.runs)

    # Resolve project paths.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = find_project_root(script_dir)
    alignment_dir = os.path.abspath(args.alignment_dir or script_dir)
    freq_file = os.path.abspath(args.freq_file or os.path.join(script_dir, "word_frequency_distribution.json"))

    # Test sets.
    test_sets = [
        "bianquexinshu",
        "huangdineijing",
        "jingkuiyaolue",
        "maijing",
        "nanjing",
        "shanghanlun",
        "sishengxinyuan",
        "wenbingtiaobian",
    ]

    alignment_files = validate_alignment_files(test_sets, alignment_dir)
    rare_vocab = load_rare_vocab(freq_file)

    # Model definitions.
    models = {
        "Qwen3-0.6B-SFT": "experiments/Qwen3-0.6B/SFT",
        "Qwen3-0.6B-GRPO": "experiments/Qwen3-0.6B/SFT_GRPO/reward_bleu_comet/checkpoints/Qwen3-0.6B-checkpoint-4000",
        "Qwen3-8B-SFT": "experiments/Qwen3-8B/SFT",
        "Qwen3-8B-GRPO": "experiments/Qwen3-8B/SFT_GRPO/reward_bleu_comet/checkpoints/Qwen3-8B-checkpoint-1000",
    }

    # Evaluate each model.
    all_model_results = {}
    global_rare_alignment_total = 0

    for model_name, model_path in models.items():
        print(f"\n{'=' * 80}")
        print(f"Evaluating model: {model_name}")
        print(f"{'=' * 80}")

        run_level_results: List[Tuple[int, Dict]] = []
        for run in runs:
            print(f"  -> run{run}")
            run_results = {
                "total_sentences": 0,
                "sentences_with_rare_words": 0,
                "total_rare_word_alignments": 0,
                "correct_translations": 0,
                "partial_translations": 0,
                "incorrect_translations": 0,
                "by_type": defaultdict(lambda: {"total": 0, "correct": 0}),
                "examples": {"correct": [], "incorrect": []},
            }
            used_any_test = False

            for test_set in test_sets:
                alignment_file = alignment_files[test_set]
                alignments = load_alignment_results(alignment_file)

                detailed_json = f"{model_path}/{test_set}_test/{test_set}_test_detailed_{run}.json"
                pred_txt = f"{model_path}/{test_set}_test/{test_set}_test_answers_{run}.txt"

                if os.path.exists(detailed_json):
                    predictions = load_model_predictions(detailed_json)
                elif os.path.exists(pred_txt):
                    predictions = load_model_predictions(pred_txt)
                else:
                    continue

                indices = load_indices(project_root, test_set)
                predictions = align_predictions(predictions, indices, model_name, test_set, run)
                results = evaluate_rare_word_alignment(alignments, predictions, rare_vocab)
                used_any_test = True

                run_results["total_sentences"] += results["total_sentences"]
                run_results["sentences_with_rare_words"] += results["sentences_with_rare_words"]
                run_results["total_rare_word_alignments"] += results["total_rare_word_alignments"]
                run_results["correct_translations"] += results["correct_translations"]
                run_results["partial_translations"] += results["partial_translations"]
                run_results["incorrect_translations"] += results["incorrect_translations"]

                for align_type, stats in results["by_type"].items():
                    run_results["by_type"][align_type]["total"] += stats["total"]
                    run_results["by_type"][align_type]["correct"] += stats["correct"]

            if not used_any_test:
                continue

            if run_results["total_rare_word_alignments"] > 0:
                run_results["accuracy"] = (
                    run_results["correct_translations"]
                    / run_results["total_rare_word_alignments"]
                    * 100
                )
                run_results["partial_accuracy"] = (
                    (run_results["correct_translations"] + run_results["partial_translations"])
                    / run_results["total_rare_word_alignments"]
                    * 100
                )
            else:
                run_results["accuracy"] = 0.0
                run_results["partial_accuracy"] = 0.0

            for align_type, stats in run_results["by_type"].items():
                if stats["total"] > 0:
                    stats["accuracy"] = stats["correct"] / stats["total"] * 100
            run_level_results.append((run, run_results))
            print(
                f"     run{run}: rare-word alignments={run_results['total_rare_word_alignments']}, "
                f"accuracy={run_results['accuracy']:.2f}%"
            )

        if not run_level_results:
            print(f"  Skipping {model_name}: no usable prediction files found")
            continue

        base_results = run_level_results[0][1]
        model_results = {
            "runs": [r for r, _ in run_level_results],
            "run_metrics": {},
            "total_sentences": base_results["total_sentences"],
            "sentences_with_rare_words": base_results["sentences_with_rare_words"],
            "total_rare_word_alignments": base_results["total_rare_word_alignments"],
            "correct_translations": int(
                round(
                    sum(r["correct_translations"] for _, r in run_level_results)
                    / len(run_level_results)
                )
            ),
            "partial_translations": int(
                round(
                    sum(r["partial_translations"] for _, r in run_level_results)
                    / len(run_level_results)
                )
            ),
            "incorrect_translations": int(
                round(
                    sum(r["incorrect_translations"] for _, r in run_level_results)
                    / len(run_level_results)
                )
            ),
            "by_type": {},
            "examples": base_results.get("examples", {"correct": [], "incorrect": []}),
            "accuracy": sum(r["accuracy"] for _, r in run_level_results)
            / len(run_level_results),
            "partial_accuracy": sum(r["partial_accuracy"] for _, r in run_level_results)
            / len(run_level_results),
        }
        global_rare_alignment_total += model_results["total_rare_word_alignments"]

        all_types = set()
        for _, run_result in run_level_results:
            all_types.update(run_result["by_type"].keys())
        for align_type in all_types:
            type_totals = [
                run_result["by_type"].get(align_type, {}).get("total", 0)
                for _, run_result in run_level_results
            ]
            type_corrects = [
                run_result["by_type"].get(align_type, {}).get("correct", 0)
                for _, run_result in run_level_results
            ]
            type_accuracies = [
                run_result["by_type"].get(align_type, {}).get("accuracy", 0.0)
                for _, run_result in run_level_results
            ]
            model_results["by_type"][align_type] = {
                "total": int(round(sum(type_totals) / len(type_totals))),
                "correct": int(round(sum(type_corrects) / len(type_corrects))),
                "accuracy": sum(type_accuracies) / len(type_accuracies),
            }

        for run, run_result in run_level_results:
            model_results["run_metrics"][f"run{run}"] = {
                "accuracy": run_result["accuracy"],
                "partial_accuracy": run_result["partial_accuracy"],
                "total_rare_word_alignments": run_result["total_rare_word_alignments"],
            }

        all_model_results[model_name] = model_results
        print(
            f"\n{model_name} mean accuracy (runs={model_results['runs']}): "
            f"{model_results['accuracy']:.2f}%"
        )

    if global_rare_alignment_total == 0:
        raise RuntimeError(
            "No evaluable rare-word alignments were found in alignment files "
            "(total_rare_word_alignments = 0).\n"
            "Check whether *_word_alignments.json contains both `rare_words` and `alignments`."
        )

    # Generate report text.
    report = generate_report(all_model_results)
    print("\n" + report)

    # Save report.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    report_file = os.path.join(script_dir, "rare_word_evaluation_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {report_file}")

    # Save detailed results.
    results_file = os.path.join(script_dir, "rare_word_evaluation_results.json")

    # Convert defaultdict to regular dict for JSON serialization.
    for model_results in all_model_results.values():
        model_results["by_type"] = dict(model_results["by_type"])

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_model_results, f, ensure_ascii=False, indent=2)
    print(f"Detailed results saved to: {results_file}")


if __name__ == "__main__":
    main()
