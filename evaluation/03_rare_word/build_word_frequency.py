#!/usr/bin/env python3
"""
Build classical-Chinese word frequency statistics with a TCM tokenizer.
Scope: training set + test set (classical text only).
"""

import json
import os
import argparse
from collections import Counter
from typing import List, Dict, Set
import jieba


class TCMTokenizer:
    """TCM classical-Chinese tokenizer."""
    
    def __init__(self, custom_dict_path: str = None):
        self.words = set()
        self.word_freq = Counter()
        self.custom_dict_path = custom_dict_path or "tcm_ancient_dict.txt"
        
    def extract_words_from_alignments(self, alignment_files: List[str]) -> Set[str]:
        """Extract classical words from alignment files."""
        for filepath in alignment_files:
            if not os.path.exists(filepath):
                print(f"File not found, skipping: {filepath}")
                continue
                
            print(f"  Processing: {os.path.basename(filepath)}")
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                for align in item.get('alignments', []):
                    if align is None:
                        continue
                    word = align.get('ancient', '') or ''
                    word = word.strip()
                    if self._is_valid_word(word):
                        self.words.add(word)
                        self.word_freq[word] += 1
        
        print(f"  Extracted {len(self.words)} classical words in total.")
        return self.words
    
    def _is_valid_word(self, word: str) -> bool:
        """Check whether a token is a valid word."""
        if not word or len(word) < 2:
            return False
        if all(c in '，。、；：？！""''（）【】' for c in word):
            return False
        return True
    
    def save_dict(self, output_path: str = None) -> str:
        """Save as a jieba user dictionary."""
        output_path = output_path or self.custom_dict_path
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for word in sorted(self.words):
                freq = max(self.word_freq[word] * 100, 500)
                f.write(f"{word} {freq} n\n")
        
        print(f"  Dictionary saved to: {output_path}")
        return output_path
    
    def load_dict(self, dict_path: str = None):
        """Load user dictionary into jieba."""
        dict_path = dict_path or self.custom_dict_path
        if os.path.exists(dict_path):
            jieba.load_userdict(dict_path)
            print(f"  Dictionary loaded: {dict_path}")
    
    def cut(self, text: str) -> List[str]:
        """Tokenize text."""
        return list(jieba.cut(text))


def load_training_ancient_texts(filepath: str) -> List[str]:
    """Load classical texts from training file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Training file not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = []
    for item in data:
        if isinstance(item, dict):
            ancient = (item.get("ancient") or item.get("query") or "").strip()
            if ancient:
                texts.append(ancient)
    return texts


def find_project_root(start_dir: str) -> str:
    """Find the project root (directory containing reference_answers)."""
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
    parser = argparse.ArgumentParser(description="Build classical-word frequency statistics with TCM tokenizer")
    parser.add_argument(
        "--alignment-dir",
        default=None,
        help="Alignment directory (default: evaluation/03_rare_word)",
    )
    parser.add_argument(
        "--train-file",
        default=None,
        help="Training file path (default: grpo/data/tcm_ancient_modern_grpo_train_12145.json)",
    )
    parser.add_argument(
        "--allow-test-only",
        action="store_true",
        help="Allow test-only frequency counting if training file is missing",
    )
    return parser.parse_args()


def load_test_ancient_texts(queries_file: str) -> List[str]:
    """Load classical test queries."""
    if not os.path.exists(queries_file):
        return []
    with open(queries_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def count_word_frequency(texts: List[str], tokenizer: TCMTokenizer) -> Counter:
    """Count token frequencies."""
    word_freq = Counter()
    for text in texts:
        words = tokenizer.cut(text)
        for word in words:
            word = word.strip()
            if len(word) >= 2:
                word_freq[word] += 1
    return word_freq


def build_frequency_distribution(word_freq: Counter) -> Dict:
    """Group words by frequency bins."""
    freq_bins = [
        (1, 2, "1-2"),
        (2, 4, "2-4"),
        (4, 8, "4-8"),
        (8, 16, "8-16"),
        (16, 32, "16-32"),
        (32, 64, "32-64"),
        (64, 128, "64-128"),
        (128, 256, "128-256"),
        (256, 512, "256-512"),
        (512, float('inf'), "512+"),
    ]
    
    distribution = {}
    for freq_min, freq_max, label in freq_bins:
        words_in_bin = [
            {'word': word, 'frequency': freq}
            for word, freq in word_freq.items()
            if freq_min <= freq < freq_max
        ]
        words_in_bin.sort(key=lambda x: x['frequency'], reverse=True)
        
        distribution[label] = {
            'range': f"{freq_min}-{freq_max if freq_max != float('inf') else '∞'}",
            'count': len(words_in_bin),
            'words': words_in_bin
        }
    
    return distribution


def main():
    print("=" * 60)
    print("Build Classical Word Frequency with TCM Tokenizer")
    print("=" * 60)

    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = find_project_root(script_dir)

    # By default, alignment files are read from evaluation/03_rare_word
    alignment_dir = os.path.abspath(args.alignment_dir or script_dir)
    default_train_file = os.path.join(project_root, "grpo/data/tcm_ancient_modern_grpo_train_12145.json")
    train_file = os.path.abspath(args.train_file or default_train_file)
    
    test_sets = [
        "bianquexinshu", "huangdineijing", "jingkuiyaolue", "maijing",
        "nanjing", "shanghanlun", "sishengxinyuan", "wenbingtiaobian"
    ]
    
    # 1) Initialize tokenizer
    print("\n[1] Initializing TCM tokenizer...")
    tokenizer = TCMTokenizer()
    
    alignment_files = [os.path.join(alignment_dir, f"{ts}_word_alignments.json") for ts in test_sets]
    alignment_files.append(os.path.join(alignment_dir, "train_word_alignments.json"))
    tokenizer.extract_words_from_alignments(alignment_files)
    
    dict_path = os.path.join(script_dir, "tcm_dict_temp.txt")
    tokenizer.save_dict(dict_path)
    tokenizer.load_dict(dict_path)
    
    print(f"  Tokenizer dictionary size: {len(tokenizer.words)}")
    
    # 2) Load classical texts
    print("\n[2] Loading classical texts...")
    all_ancient_texts = []
    
    # Training set
    if os.path.exists(train_file):
        train_texts = load_training_ancient_texts(train_file)
        all_ancient_texts.extend(train_texts)
        print(f"  Training set: {len(train_texts)} samples ({train_file})")
    elif args.allow_test_only:
        train_texts = []
        print("  Training set: 0 samples (missing file, --allow-test-only enabled)")
    else:
        raise FileNotFoundError(
            f"Training file not found: {train_file}\n"
            "Use --allow-test-only to run without training data."
        )
    
    # Test sets
    for test_set in test_sets:
        queries_file = os.path.join(project_root, f"reference_answers/{test_set}_queries.txt")
        test_texts = load_test_ancient_texts(queries_file)
        all_ancient_texts.extend(test_texts)
        print(f"  {test_set}: {len(test_texts)} samples")
    
    print(f"  Total classical samples: {len(all_ancient_texts)}")
    if not all_ancient_texts:
        raise RuntimeError("No classical text loaded; cannot build frequency statistics.")
    
    # 3) Count frequencies
    print("\n[3] Counting word frequencies...")
    word_freq = count_word_frequency(all_ancient_texts, tokenizer)
    print(f"  Unique words: {len(word_freq)}")
    print(f"  Total token count: {sum(word_freq.values())}")
    
    # 4) Build distribution
    print("\n[4] Building frequency distribution...")
    distribution = build_frequency_distribution(word_freq)
    
    print("\nFrequency distribution:")
    print(f"{'Range':<15} {'Words':<10}")
    print("-" * 30)
    for label in ["1-2", "2-4", "4-8", "8-16", "16-32", "32-64", "64-128", "128-256", "256-512", "512+"]:
        if label in distribution:
            print(f"{label:<15} {distribution[label]['count']:<10}")
    
    # 5) Save outputs
    print("\n[5] Saving outputs...")
    
    dist_file = os.path.join(script_dir, "word_frequency_distribution.json")
    with open(dist_file, 'w', encoding='utf-8') as f:
        json.dump(distribution, f, ensure_ascii=False, indent=2)
    print(f"  Distribution file: {dist_file}")
    
    freq_file = os.path.join(script_dir, "word_frequency_all.json")
    freq_list = [{'word': w, 'frequency': f} for w, f in word_freq.most_common()]
    with open(freq_file, 'w', encoding='utf-8') as f:
        json.dump(freq_list, f, ensure_ascii=False, indent=2)
    print(f"  Full frequency file: {freq_file}")
    
    print("\nTop 20 high-frequency words:")
    for word, freq in word_freq.most_common(20):
        print(f"  {word}: {freq}")
    
    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
