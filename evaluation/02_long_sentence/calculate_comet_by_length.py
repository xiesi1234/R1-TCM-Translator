#!/usr/bin/env python3
"""
Compute corpus-level COMET by source-length buckets.
"""

import os
import json
import numpy as np
from collections import defaultdict
from comet import download_model, load_from_checkpoint

# Project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Auto-detect project root (contains reference_answers and experiments)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
for _ in range(3):  # up to 3 levels upward
    if os.path.exists(os.path.join(PROJECT_ROOT, 'reference_answers')) and \
       os.path.exists(os.path.join(PROJECT_ROOT, 'experiments')):
        break
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

BOOK_NAMES = {
    'bianquexinshu': 'Bianque Xinshu',
    'huangdineijing': 'Huangdi Neijing',
    'jingkuiyaolue': 'Jingui Yaolue',
    'maijing': 'Maijing',
    'nanjing': 'Nanjing',
    'shanghanlun': 'Shanghanlun',
    'sishengxinyuan': 'Sisheng Xinyuan',
    'wenbingtiaobian': 'Wenbing Tiaobian'
}
BOOKS = list(BOOK_NAMES.keys())

# Model paths
MODELS = {
    'Qwen3-8B-checkpoint-1000': 'experiments/Qwen3-8B/SFT_GRPO/reward_bleu_comet/checkpoints/Qwen3-8B-checkpoint-1000',
    'Qwen3-0.6B-checkpoint-4000': 'experiments/Qwen3-0.6B/SFT_GRPO/reward_bleu_comet/checkpoints/Qwen3-0.6B-checkpoint-4000',
    'deepseek_v3': 'experiments/baselines/deepseek_v3_9.18',
    'DeepSeek-R1': 'experiments/baselines/DeepSeek-R1',
    'GPT-4.1': 'experiments/baselines/GPT-4.1',
    'GPT-4o': 'experiments/baselines/GPT-4o',
    'Qwen3-0.6B-Base': 'experiments/Qwen3-0.6B/Base',
    'Qwen3-0.6B-SFT': 'experiments/Qwen3-0.6B/SFT',
    'Qwen3-8B-Base': 'experiments/Qwen3-8B/Base',
    'Qwen3-8B-SFT': 'experiments/Qwen3-8B/SFT',
}

# Length bucket config
BUCKETS = [
    {'name': '1-10', 'min': 1, 'max': 10},
    {'name': '11-20', 'min': 11, 'max': 20},
    {'name': '21-30', 'min': 21, 'max': 30},
    {'name': '31-40', 'min': 31, 'max': 40},
    {'name': '41-50', 'min': 41, 'max': 50},
    {'name': '51-60', 'min': 51, 'max': 60},
    {'name': '61-70', 'min': 61, 'max': 70},
    {'name': '71-80', 'min': 71, 'max': 80},
    {'name': '81-90', 'min': 81, 'max': 90},
    {'name': '91-100', 'min': 91, 'max': 100},
    {'name': '101+', 'min': 101, 'max': 999},
]

# COMET model config
COMET_MODEL_NAME = "Unbabel/wmt22-comet-da"
# Optional models:
# "Unbabel/wmt22-comet-da" - general translation quality
# "Unbabel/wmt20-comet-da" - WMT20
# "Unbabel/wmt21-comet-da" - WMT21
# "Unbabel/XCOMET-XXL" - larger XCOMET model


def count_chinese_chars(text):
    """Count Chinese characters in text."""
    return len([c for c in text if '\u4e00' <= c <= '\u9fff'])


def get_bucket(length):
    """Return bucket index by source length."""
    for i, bucket in enumerate(BUCKETS):
        if bucket['min'] <= length <= bucket['max']:
            return i
    return len(BUCKETS) - 1


def load_file(filepath):
    """Load a text file as stripped lines."""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return [l.strip() for l in f.readlines()]
    return []


def load_indices(book):
    """Load index file."""
    idx_file = os.path.join(PROJECT_ROOT, f'reference_answers/{book}_indices.txt')
    if os.path.exists(idx_file):
        with open(idx_file, 'r', encoding='utf-8') as f:
            return [int(l.strip()) for l in f.readlines()]
    return []


def filter_by_indices(answers, indices):
    return [answers[idx] if idx < len(answers) else "" for idx in indices]


def load_comet_model():
    """Load COMET model."""
    print(f"Loading COMET model: {COMET_MODEL_NAME}")
    print("Note: first run downloads the model and may take time.")
    model_path = download_model(COMET_MODEL_NAME)
    model = load_from_checkpoint(model_path)
    return model


def calc_corpus_comet(model, srcs, hyps, refs, batch_size=128, gpus=1):
    """Compute corpus-level COMET."""
    if not hyps or not refs or not srcs:
        return 0.0

    valid_pairs = [(s, h, r) for s, h, r in zip(srcs, hyps, refs) if h and r]
    if not valid_pairs:
        return 0.0

    valid_srcs, valid_hyps, valid_refs = zip(*valid_pairs)

    # COMET input format
    data = [
        {"src": src, "mt": hyp, "ref": ref}
        for src, hyp, ref in zip(valid_srcs, valid_hyps, valid_refs)
    ]

    try:
        # Predict scores
        results = model.predict(data, batch_size=batch_size, gpus=gpus)
        # Return average score
        if hasattr(results, 'scores'):
            return float(np.mean(results.scores))
        elif hasattr(results, 'score'):
            return float(results.score)
        else:
            return 0.0
    except Exception as e:
        print(f"  COMET error: {e}")
        return 0.0


def process_model(model_name, model_path, comet_model):
    """Process one model."""
    print(f"\n{'='*80}")
    print(f"Processing model: {model_name}")
    print(f"Path: {model_path}")
    print(f"{'='*80}")

    full_path = os.path.join(PROJECT_ROOT, model_path)
    if not os.path.exists(full_path):
        print(f"  Warning: path not found, skipping.")
        return None

    # bucket_data[bucket_idx][run_idx] = {'srcs': [], 'refs': [], 'hyps': []}
    bucket_data = defaultdict(lambda: defaultdict(lambda: {'srcs': [], 'refs': [], 'hyps': []}))
    bucket_counts = defaultdict(int)

    print("\nCollecting data...")
    for book in BOOKS:
        srcs = load_file(os.path.join(PROJECT_ROOT, f'reference_answers/{book}_queries.txt'))
        refs = load_file(os.path.join(PROJECT_ROOT, f'reference_answers/{book}_references.txt'))
        indices = load_indices(book)

        hyps_list = []
        run_indices = []
        for run in [1, 2, 3]:
            hyp_file = os.path.join(full_path, f'{book}_test/{book}_test_answers_{run}.txt')
            all_hyps = load_file(hyp_file)
            if all_hyps:
                hyps = filter_by_indices(all_hyps, indices) if indices else all_hyps
                hyps_list.append(hyps)
                run_indices.append(run)

        if not hyps_list:
            continue

        min_len = min(len(srcs), len(refs), min(len(h) for h in hyps_list))

        for i in range(min_len):
            src = srcs[i]
            ref = refs[i]
            if not src or not ref:
                continue

            length = count_chinese_chars(src)
            bucket_idx = get_bucket(length)
            bucket_counts[bucket_idx] += 1

            for list_idx, hyps in enumerate(hyps_list):
                run_idx = run_indices[list_idx]
                if i < len(hyps) and hyps[i]:
                    bucket_data[bucket_idx][run_idx]['srcs'].append(src)
                    bucket_data[bucket_idx][run_idx]['refs'].append(ref)
                    bucket_data[bucket_idx][run_idx]['hyps'].append(hyps[i])

    # Compute COMET by bucket
    print("\nComputing corpus-level COMET by bucket...")
    bucket_comet = {}

    for bucket_idx in range(len(BUCKETS)):
        for run_idx in range(1, 4):
            data = bucket_data[bucket_idx][run_idx]
            if data['hyps'] and data['refs'] and data['srcs']:
                print(f"  Bucket {BUCKETS[bucket_idx]['name']}, Run {run_idx}: {len(data['hyps'])} samples")
                comet = calc_corpus_comet(comet_model, data['srcs'], data['hyps'], data['refs'],
                                         batch_size=128, gpus=1)
                bucket_comet[(bucket_idx, run_idx)] = comet
            else:
                bucket_comet[(bucket_idx, run_idx)] = 0.0

    return bucket_comet, bucket_counts


def print_results(bucket_comet, bucket_counts, model_name):
    """Print summary table."""
    print(f"\n[{model_name} - COMET by length bucket]")
    print("-" * 90)
    print(f"{'Length':<12} {'Samples':<8} | {'Run1':<10} {'Run2':<10} {'Run3':<10} {'Avg':<10}")
    print("-" * 90)

    all_comets = []
    for bucket_idx, bucket in enumerate(BUCKETS):
        count = bucket_counts.get(bucket_idx, 0)
        comet1 = bucket_comet.get((bucket_idx, 1), 0.0)
        comet2 = bucket_comet.get((bucket_idx, 2), 0.0)
        comet3 = bucket_comet.get((bucket_idx, 3), 0.0)

        valid_comets = [c for c in [comet1, comet2, comet3] if c > 0]
        avg_comet = np.mean(valid_comets) if valid_comets else 0.0
        all_comets.extend(valid_comets)

        print(f"{bucket['name']:<12} {count:<8} | {comet1:<10.2f} {comet2:<10.2f} {comet3:<10.2f} {avg_comet:<10.2f}")

    overall_comet = np.mean(all_comets) if all_comets else 0.0
    total_count = sum(bucket_counts.values())
    print("-" * 90)
    print(f"{'Overall Avg':<12} {total_count:<8} | {'':30} {overall_comet:<10.2f}")


def main():
    print("=" * 80)
    print("Corpus-Level COMET by Source-Length Buckets")
    print("=" * 80)

    # Load COMET model once
    comet_model = load_comet_model()

    all_results = {}

    for model_name, model_path in MODELS.items():
        result = process_model(model_name, model_path, comet_model)
        if result:
            bucket_comet, bucket_counts = result
            print_results(bucket_comet, bucket_counts, model_name)
            all_results[model_name] = {
                'bucket_comet': {f"{k[0]}_{k[1]}": v for k, v in bucket_comet.items()},
                'bucket_counts': dict(bucket_counts)
            }

    # Save results
    output_file = os.path.join(SCRIPT_DIR, 'comet_by_length_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
