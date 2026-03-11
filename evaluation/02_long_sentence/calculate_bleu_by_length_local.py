#!/usr/bin/env python3
"""
Local script: compute corpus-level BLEU for each source-length bucket.
COMET is not computed in this script.
"""

import os
import json
import numpy as np
import jieba
from sacrebleu.metrics import BLEU
from collections import defaultdict

# Project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

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

# Source-length buckets: step size = 10 chars
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
]

bleu_scorer = BLEU(tokenize="none")


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
    """Load reference index file."""
    idx_file = os.path.join(PROJECT_ROOT, f'reference_answers/{book}_indices.txt')
    if os.path.exists(idx_file):
        with open(idx_file, 'r', encoding='utf-8') as f:
            return [int(l.strip()) for l in f.readlines()]
    return []


def filter_by_indices(answers, indices):
    return [answers[idx] if idx < len(answers) else "" for idx in indices]


def calc_corpus_bleu(hyps, refs):
    """Compute corpus-level BLEU."""
    if not hyps or not refs:
        return 0.0

    valid_pairs = [(h, r) for h, r in zip(hyps, refs) if h and r]
    if not valid_pairs:
        return 0.0

    valid_hyps, valid_refs = zip(*valid_pairs)

    hyps_tok = [" ".join(jieba.cut(h)) for h in valid_hyps]
    refs_tok = [" ".join(jieba.cut(r)) for r in valid_refs]

    try:
        # Correct format: wrap each reference as a singleton list
        score = bleu_scorer.corpus_score(hyps_tok, [[r] for r in refs_tok]).score
        return float(score)
    except:
        return 0.0


def process_model(model_name, model_path):
    """Process one model."""
    print(f"\n{'='*80}")
    print(f"Processing model: {model_name}")
    print(f"Path: {model_path}")
    print(f"{'='*80}")

    full_path = os.path.join(PROJECT_ROOT, model_path)
    if not os.path.exists(full_path):
        print(f"  Warning: path not found, skipping.")
        return None

    # bucket_data[bucket_idx][run_idx] = {'refs': [], 'hyps': []}
    bucket_data = defaultdict(lambda: defaultdict(lambda: {'refs': [], 'hyps': []}))
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
                    bucket_data[bucket_idx][run_idx]['refs'].append(ref)
                    bucket_data[bucket_idx][run_idx]['hyps'].append(hyps[i])

    # Compute BLEU by bucket
    print("\nComputing corpus-level BLEU by bucket...")
    bucket_bleu = {}

    for bucket_idx in range(len(BUCKETS)):
        for run_idx in range(1, 4):
            data = bucket_data[bucket_idx][run_idx]
            if data['hyps'] and data['refs']:
                bleu = calc_corpus_bleu(data['hyps'], data['refs'])
                bucket_bleu[(bucket_idx, run_idx)] = bleu
            else:
                bucket_bleu[(bucket_idx, run_idx)] = 0.0

    return bucket_bleu, bucket_counts


def print_results(bucket_bleu, bucket_counts, model_name):
    """Print summary table."""
    print(f"\n[{model_name} - BLEU by length bucket]")
    print("-" * 90)
    print(f"{'Length':<12} {'Samples':<8} | {'Run1':<10} {'Run2':<10} {'Run3':<10} {'Avg':<10}")
    print("-" * 90)

    all_bleus = []
    for bucket_idx, bucket in enumerate(BUCKETS):
        count = bucket_counts.get(bucket_idx, 0)
        bleu1 = bucket_bleu.get((bucket_idx, 1), 0.0)
        bleu2 = bucket_bleu.get((bucket_idx, 2), 0.0)
        bleu3 = bucket_bleu.get((bucket_idx, 3), 0.0)
        
        valid_bleus = [b for b in [bleu1, bleu2, bleu3] if b > 0]
        avg_bleu = np.mean(valid_bleus) if valid_bleus else 0.0
        all_bleus.extend(valid_bleus)

        print(f"{bucket['name']:<12} {count:<8} | {bleu1:<10.2f} {bleu2:<10.2f} {bleu3:<10.2f} {avg_bleu:<10.2f}")

    overall_bleu = np.mean(all_bleus) if all_bleus else 0.0
    total_count = sum(bucket_counts.values())
    print("-" * 90)
    print(f"{'Overall Avg':<12} {total_count:<8} | {'':30} {overall_bleu:<10.2f}")


def main():
    print("=" * 80)
    print("Local BLEU Evaluation by Source-Length Buckets")
    print("=" * 80)

    all_results = {}

    for model_name, model_path in MODELS.items():
        result = process_model(model_name, model_path)
        if result:
            bucket_bleu, bucket_counts = result
            print_results(bucket_bleu, bucket_counts, model_name)
            all_results[model_name] = {
                'bucket_bleu': {f"{k[0]}_{k[1]}": v for k, v in bucket_bleu.items()},
                'bucket_counts': dict(bucket_counts)
            }

    # Save results
    output_file = os.path.join(SCRIPT_DIR, 'bleu_by_length_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
