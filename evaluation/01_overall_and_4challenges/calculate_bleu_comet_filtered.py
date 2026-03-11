#!/usr/bin/env python
"""
BLEU and COMET evaluation script.
1. Compute overall metrics on all valid samples.
2. Compute metrics on four challenge subsets (contains=true only).
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import gc
import json
import numpy as np
import jieba
from collections import defaultdict
from sacrebleu.metrics import BLEU

# ==================== Config ====================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_project_root(start_dir):
    """Find the project root (the directory that contains reference_answers)."""
    cur = start_dir
    for _ in range(8):
        if os.path.exists(os.path.join(cur, "reference_answers")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return start_dir


PROJECT_ROOT = find_project_root(SCRIPT_DIR)
COMET_ENABLED = os.getenv("DISABLE_COMET", "0") != "1"
STRICT_INDEX_ALIGNMENT = os.getenv("STRICT_INDEX_ALIGNMENT", "1") == "1"
OVERALL_AVG_MODE = os.getenv("OVERALL_AVG_MODE", "macro").strip().lower()
if OVERALL_AVG_MODE not in {"macro", "micro"}:
    OVERALL_AVG_MODE = "macro"

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

# Four challenge subset configs
CHALLENGES = {
    'semantic': ('Semantic Shift', 'evaluation/04_challenge_tagging/semantic_shift/cache_semantic.json'),
    'cultural': ('Cultural-Epistemic Context', 'evaluation/04_challenge_tagging/cultural_context/cache_cultural.json'),
    'zero_anaphora': ('Zero Anaphora', 'evaluation/04_challenge_tagging/zero_anaphora/cache_zero_anaphora.json'),
    'parataxis': ('Parataxis', 'evaluation/04_challenge_tagging/parataxis/cache_parataxis.json'),
}

# Model configs
MODELS = [
    ('Qwen3-0.6B-Base', 'experiments/Qwen3-0.6B/Base'),
    ('Qwen3-0.6B-SFT', 'experiments/Qwen3-0.6B/SFT'),
    ('Qwen3-0.6B-SFT+GRPO', 'experiments/Qwen3-0.6B/SFT_GRPO/reward_bleu_comet/checkpoints/Qwen3-0.6B-checkpoint-4000'),
    ('Qwen3-8B-Base', 'experiments/Qwen3-8B/Base'),
    ('Qwen3-8B-SFT', 'experiments/Qwen3-8B/SFT'),
    ('Qwen3-8B-SFT+GRPO', 'experiments/Qwen3-8B/SFT_GRPO/reward_bleu_comet/checkpoints/Qwen3-8B-checkpoint-1000'),
    ('GPT-4o', 'experiments/baselines/GPT-4o'),
    ('GPT-4.1', 'experiments/baselines/GPT-4.1'),
    ('DeepSeek-R1', 'experiments/baselines/DeepSeek-R1'),
    ('deepseek_v3_9.18', 'experiments/baselines/deepseek_v3_9.18'),
]

# ==================== Global Variables ====================

comet_model = None
bleu_scorer = BLEU(tokenize="none")
refs_cache = {}
srcs_cache = {}
indices_cache = {}
refs_tok_cache = {}
warned_alignment_issues = set()
alignment_fix_stats = defaultdict(int)
padded_alignment_issue_keys = set()


# ==================== Helpers ====================

def init_comet():
    global comet_model
    if not COMET_ENABLED:
        return None
    if comet_model is None:
        print("Loading COMET model...")
        from comet import download_model, load_from_checkpoint
        model_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(model_path)
        print("COMET model loaded.")
    return comet_model

def load_file(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return [l.strip() for l in f.readlines()]
    return []

def load_indices(book):
    idx_file = os.path.join(PROJECT_ROOT, f'reference_answers/{book}_indices.txt')
    if os.path.exists(idx_file):
        with open(idx_file, 'r', encoding='utf-8') as f:
            return [int(l.strip()) for l in f.readlines()]
    return []

def align_hypotheses(answers, indices, book, run, model_name):
    """
    Align model outputs with reference_answers.
    Supports two output formats:
    1) raw outputs (length close to merged_test_set), filtered by indices;
    2) pre-filtered outputs (length already equals index count), used directly.
    """
    if not indices:
        return answers

    if len(answers) == len(indices):
        return answers

    max_idx = max(indices)
    if max_idx < len(answers):
        return [answers[idx] for idx in indices]

    missing_needed = max_idx + 1 - len(answers)
    msg = (
        f"{model_name} {book} run{run} index out of range: "
        f"indices={len(indices)}, max_idx={max_idx}, answers={len(answers)}, missing_needed={missing_needed}"
    )
    issue_key = (model_name, book, run, len(indices), len(answers), max_idx)
    if issue_key not in warned_alignment_issues:
        mode = "strict mode" if STRICT_INDEX_ALIGNMENT else "non-strict mode"
        print(f"    Warning: {msg}; auto-padding empty predictions in {mode} to avoid dropping samples.")
        warned_alignment_issues.add(issue_key)
    if missing_needed > 0:
        answers = answers + [""] * missing_needed
        if issue_key not in padded_alignment_issue_keys:
            alignment_fix_stats["padded_predictions"] += missing_needed
            alignment_fix_stats["padded_files"] += 1
            padded_alignment_issue_keys.add(issue_key)
    return [answers[idx] for idx in indices]

def precompute_refs():
    print("Precomputing tokenized references...")
    for book in BOOKS:
        refs_cache[book] = load_file(os.path.join(PROJECT_ROOT, f'reference_answers/{book}_references.txt'))
        srcs_cache[book] = load_file(os.path.join(PROJECT_ROOT, f'reference_answers/{book}_queries.txt'))
        indices_cache[book] = load_indices(book)
        refs_tok_cache[book] = [" ".join(jieba.cut(r)) for r in refs_cache[book]]
    print(f"  Cached {sum(len(v) for v in refs_cache.values())} reference samples.")

def load_challenge_cache(challenge_key):
    _, cache_path = CHALLENGES[challenge_key]
    full_cache_path = os.path.join(PROJECT_ROOT, cache_path)
    if not os.path.exists(full_cache_path):
        print(f"  Warning: cache file not found: {full_cache_path}")
        return {}
    with open(full_cache_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_filtered_indices(challenge_key):
    cache = load_challenge_cache(challenge_key)
    if not cache:
        return {}

    # Prefer exact (source, reference) matching to avoid collisions with repeated sources.
    # Fall back to source-only matching when reference is missing.
    pair_to_indices = {}
    source_to_indices = {}
    for book in BOOKS:
        pair_map = defaultdict(list)
        source_map = defaultdict(list)
        refs = refs_cache.get(book, [])
        for idx, src in enumerate(srcs_cache[book]):
            ref = refs[idx] if idx < len(refs) else ""
            pair_map[(src, ref)].append(idx)
            source_map[src].append(idx)
        pair_to_indices[book] = pair_map
        source_to_indices[book] = source_map

    filtered = {book: set() for book in BOOKS}
    unmatched = 0
    for key, value in cache.items():
        if value.get('contains', False):
            book = value.get('test_set')
            source = (value.get('source') or "").strip()
            reference = (value.get('reference') or "").strip()
            if book not in BOOKS:
                unmatched += 1
                continue

            hit = False
            if (source, reference) in pair_to_indices.get(book, {}):
                for idx in pair_to_indices[book][(source, reference)]:
                    filtered[book].add(idx)
                hit = True
            elif source in source_to_indices.get(book, {}):
                for idx in source_to_indices[book][source]:
                    filtered[book].add(idx)
                hit = True

            if not hit:
                unmatched += 1

    if unmatched:
        print(f"  Warning: {challenge_key} has {unmatched} contains=true samples not mapped to reference_answers.")
    return filtered

def collect_bleu_tokenized_pairs(hyps, book, filtered_indices=None):
    refs_tok = refs_tok_cache[book]
    if not hyps or not refs_tok:
        return []

    min_len = min(len(hyps), len(refs_tok))
    pairs = []
    for i in range(min_len):
        if filtered_indices is not None and i not in filtered_indices.get(book, set()):
            continue
        hyp = hyps[i]
        ref_tok = refs_tok[i]
        if hyp and ref_tok:
            pairs.append((" ".join(jieba.cut(hyp)), ref_tok))
    return pairs


def calc_bleu_from_pairs(tok_pairs):
    if not tok_pairs:
        return 0.0
    hyps_tok = [p[0] for p in tok_pairs]
    refs_tok = [p[1] for p in tok_pairs]
    return bleu_scorer.corpus_score(hyps_tok, [[r] for r in refs_tok]).score

def batch_comet_predict(all_data):
    if not COMET_ENABLED or not all_data:
        return []
    import torch
    torch.cuda.empty_cache()
    gc.collect()
    
    model = init_comet()
    batch_size = 64
    print(f"  COMET batch_size={batch_size}, samples={len(all_data)}")
    output = model.predict(all_data, batch_size=batch_size, gpus=1, num_workers=2)
    
    torch.cuda.empty_cache()
    gc.collect()
    return output.scores


def eval_model_path(path, filtered_indices=None, books_for_macro=None, model_label=None):
    runs = [1, 2, 3]
    model_name = model_label or os.path.basename(path.rstrip("/"))
    macro_books = books_for_macro if books_for_macro is not None else BOOKS

    bleu_scores = {}
    comet_scores = {}
    bleu_run_scores_by_book = defaultdict(list)
    bleu_run_counts_by_book = defaultdict(list)
    bleu_samples_by_run = {run: 0 for run in runs}
    comet_data, data_map = [], []

    for book in BOOKS:
        refs = refs_cache[book]
        srcs = srcs_cache[book]
        indices = indices_cache[book]

        for run in runs:
            hyp_file = os.path.join(path, f'{book}_test/{book}_test_answers_{run}.txt')
            all_hyps = load_file(hyp_file)
            if not all_hyps:
                continue

            hyps = align_hypotheses(all_hyps, indices, book, run, model_name) if indices else all_hyps

            tok_pairs = collect_bleu_tokenized_pairs(hyps, book, filtered_indices)
            if tok_pairs:
                bleu_run_scores_by_book[book].append(calc_bleu_from_pairs(tok_pairs))
                bleu_run_counts_by_book[book].append(len(tok_pairs))
                bleu_samples_by_run[run] += len(tok_pairs)

            if COMET_ENABLED:
                min_len = min(len(hyps), len(refs), len(srcs))
                for i in range(min_len):
                    if filtered_indices is not None and i not in filtered_indices.get(book, set()):
                        continue
                    if hyps[i] and refs[i] and srcs[i]:
                        comet_data.append({"src": srcs[i], "mt": hyps[i], "ref": refs[i]})
                        data_map.append((book, run))

        bleu_scores[book] = np.mean(bleu_run_scores_by_book[book]) if bleu_run_scores_by_book[book] else 0.0

    avg_bleu_macro = np.mean([bleu_scores.get(b, 0.0) for b in macro_books]) if macro_books else 0.0
    bleu_weights = {
        b: (int(round(np.mean(bleu_run_counts_by_book[b]))) if bleu_run_counts_by_book[b] else 0)
        for b in BOOKS
    }
    weighted_bleu_sum = sum(bleu_scores.get(b, 0.0) * bleu_weights[b] for b in macro_books if bleu_weights[b] > 0)
    total_bleu_weight = sum(bleu_weights[b] for b in macro_books if bleu_weights[b] > 0)
    avg_bleu_micro = weighted_bleu_sum / total_bleu_weight if total_bleu_weight > 0 else 0.0

    comet_run_values = defaultdict(list)
    if comet_data:
        scores = batch_comet_predict(comet_data)
        comet_raw = defaultdict(list)
        for i, (book, run) in enumerate(data_map):
            score = scores[i] * 100
            comet_raw[(book, run)].append(score)
            comet_run_values[run].append(score)

        for book in BOOKS:
            run_scores = []
            for run in runs:
                key = (book, run)
                if key in comet_raw:
                    run_scores.append(np.mean(comet_raw[key]))
            comet_scores[book] = np.mean(run_scores) if run_scores else 0.0
    else:
        comet_scores = {book: 0.0 for book in BOOKS}

    avg_comet_macro = np.mean([comet_scores.get(b, 0.0) for b in macro_books]) if macro_books else 0.0
    comet_run_micro = [
        np.mean(comet_run_values[run])
        for run in runs
        if comet_run_values.get(run)
    ]
    avg_comet_micro = np.mean(comet_run_micro) if comet_run_micro else 0.0

    return {
        "bleu_by_book": bleu_scores,
        "comet_by_book": comet_scores,
        "avg_bleu_macro": avg_bleu_macro,
        "avg_bleu_micro": avg_bleu_micro,
        "avg_comet_macro": avg_comet_macro,
        "avg_comet_micro": avg_comet_micro,
        "avg_bleu": avg_bleu_micro if OVERALL_AVG_MODE == "micro" else avg_bleu_macro,
        "avg_comet": avg_comet_micro if OVERALL_AVG_MODE == "micro" else avg_comet_macro,
        "bleu_samples_by_run": {f"run{run}": bleu_samples_by_run[run] for run in runs},
        "bleu_samples_by_book": {b: bleu_weights[b] for b in BOOKS},
        "comet_samples_by_run": {f"run{run}": len(comet_run_values[run]) for run in runs},
    }


def main():
    print("=" * 70)
    print("BLEU + COMET Evaluation (Overall + Four Challenges)")
    print("=" * 70)
    print(f"Aggregation mode: {OVERALL_AVG_MODE} (both macro and micro are exported)")
    print("Micro definition: sample-weighted book-level average")
    print(f"Index alignment mode: {'strict no-drop (pad missing as empty)' if STRICT_INDEX_ALIGNMENT else 'compat mode (pad missing as empty)'}")
    
    precompute_refs()
    if COMET_ENABLED:
        init_comet()
    else:
        print("COMET is disabled (DISABLE_COMET=1); BLEU only.")
    
    # Build filtered indices for four challenge subsets
    print("\nBuilding filtered indices for the four challenge subsets...")
    challenge_indices = {}
    for challenge_key, (challenge_name, _) in CHALLENGES.items():
        filtered = build_filtered_indices(challenge_key)
        total = sum(len(v) for v in filtered.values())
        challenge_indices[challenge_key] = filtered
        print(f"  {challenge_name}: {total} samples (contains=true)")
    
    results = {
        'meta': {
            'aggregation_mode': OVERALL_AVG_MODE,
            'micro_definition': 'sample-weighted book-level average',
            'strict_index_alignment': STRICT_INDEX_ALIGNMENT,
        },
        'overall': {},
        'by_challenge': {k: {'name': v[0], 'models': {}} for k, v in CHALLENGES.items()},
    }
    
    # For each model: evaluate overall first, then challenge subsets
    for model_name, model_path in MODELS:
        full_model_path = os.path.join(PROJECT_ROOT, model_path)
        print(f"\n{'='*70}")
        print(f"Evaluating model: {model_name}")
        print(f"{'='*70}")
        
        if not os.path.exists(full_model_path):
            print(f"  Warning: path does not exist, skipping: {full_model_path}")
            continue
        
        # 1) Overall
        print(f"\n  [Overall]")
        overall_metrics = eval_model_path(
            full_model_path,
            filtered_indices=None,
            books_for_macro=BOOKS,
            model_label=model_name,
        )
        print(
            f"    BLEU: {overall_metrics['avg_bleu']:.2f} "
            f"(macro={overall_metrics['avg_bleu_macro']:.2f}, micro={overall_metrics['avg_bleu_micro']:.2f})"
        )
        print(
            f"    COMET: {overall_metrics['avg_comet']:.2f} "
            f"(macro={overall_metrics['avg_comet_macro']:.2f}, micro={overall_metrics['avg_comet_micro']:.2f})"
        )
        results['overall'][model_name] = overall_metrics
        
        # 2) Four challenge subsets
        for challenge_key, (challenge_name, _) in CHALLENGES.items():
            print(f"\n  [{challenge_name}]")
            valid_books = [b for b in BOOKS if challenge_indices[challenge_key].get(b)]
            metrics = eval_model_path(
                full_model_path,
                filtered_indices=challenge_indices[challenge_key],
                books_for_macro=valid_books,
                model_label=model_name,
            )
            print(
                f"    BLEU: {metrics['avg_bleu']:.2f} "
                f"(macro={metrics['avg_bleu_macro']:.2f}, micro={metrics['avg_bleu_micro']:.2f})"
            )
            print(
                f"    COMET: {metrics['avg_comet']:.2f} "
                f"(macro={metrics['avg_comet_macro']:.2f}, micro={metrics['avg_comet_micro']:.2f})"
            )
            results['by_challenge'][challenge_key]['models'][model_name] = metrics

    results['meta']['alignment_fixes'] = dict(alignment_fix_stats)
    
    output_file = os.path.join(SCRIPT_DIR, 'filtered_challenge_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 120)
    print("Summary (BLEU)")
    print("=" * 120)
    
    header = f"{'Model':<22} {'Overall':>8} {'Semantic':>10} {'Cultural':>10} {'ZeroAna':>10} {'Parataxis':>10}"
    print(header)
    print("-" * 120)
    
    for model_name in results['overall']:
        row = f"{model_name:<22}"
        row += f" {results['overall'][model_name]['avg_bleu']:>8.2f}"
        for ck in ['semantic', 'cultural', 'zero_anaphora', 'parataxis']:
            if model_name in results['by_challenge'][ck]['models']:
                row += f" {results['by_challenge'][ck]['models'][model_name]['avg_bleu']:>10.2f}"
            else:
                row += f" {'N/A':>10}"
        print(row)
    
    print("\n" + "=" * 120)
    print("Summary (COMET)")
    print("=" * 120)
    print(header)
    print("-" * 120)
    
    for model_name in results['overall']:
        row = f"{model_name:<22}"
        row += f" {results['overall'][model_name]['avg_comet']:>8.2f}"
        for ck in ['semantic', 'cultural', 'zero_anaphora', 'parataxis']:
            if model_name in results['by_challenge'][ck]['models']:
                row += f" {results['by_challenge'][ck]['models'][model_name]['avg_comet']:>10.2f}"
            else:
                row += f" {'N/A':>10}"
        print(row)
    
    if alignment_fix_stats:
        print("\nIndex alignment fix statistics:")
        for k, v in alignment_fix_stats.items():
            print(f"  {k}: {v}")

    print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()
