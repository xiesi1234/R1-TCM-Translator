#!/usr/bin/env python3
"""
LLM-based word alignment between classical and modern Chinese (concurrent version).
Uses the DeepSeek-V3 API to extract token-level alignments.
"""

import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import jieba
import requests
from tqdm import tqdm


class RateLimiter:
    """Simple per-minute request rate limiter."""

    def __init__(self, max_per_minute: int = 900):
        self.max_per_minute = max_per_minute
        self.calls = []
        self.lock = threading.Lock()

    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            self.calls = [t for t in self.calls if now - t < 60]

            if len(self.calls) >= self.max_per_minute:
                sleep_time = 60 - (now - self.calls[0]) + 0.1
                time.sleep(sleep_time)
                self.calls = []

            self.calls.append(now)


class LLMWordAligner:
    """Word aligner powered by an external LLM API."""

    def __init__(self, api_key: str, rate_limiter: RateLimiter = None):
        self.api_key = api_key
        self.url = "https://api.siliconflow.cn/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.rate_limiter = rate_limiter or RateLimiter()

    def create_alignment_prompt(
        self, ancient_text: str, modern_text: str, rare_words: List[str] = None
    ) -> str:
        """Build alignment prompt.

        Note: Chinese prompt content is intentionally preserved because it is
        part of the evaluation task definition.
        """
        prompt = f"""你是一个专业的中医古文翻译专家。请分析以下古文和现代文的对应关系，提取关键词的对齐。

古文：{ancient_text}
现代文：{modern_text}

"""

        if rare_words:
            # Limit count to avoid oversized prompts.
            prompt += f"\n请特别关注以下词的翻译：{', '.join(rare_words[:20])}\n"

        prompt += """
请按照以下格式输出词对齐结果（JSON格式）：

{
  "alignments": [
    {"ancient": "古文词", "modern": "现代词", "type": "exact/paraphrase/omitted/expansion"},
    ...
  ],
  "notes": "可选的说明"
}

对齐类型说明：
- exact: 完全匹配（如"脉浮数"→"脉浮数"）
- paraphrase: 意译或同义替换（如"法当"→"应当"）
- expansion: 古文词扩展为现代词（如"肾"→"肾脏"）
- omitted: 古文中有但现代文省略的词

注意事项：
1. 只提取实词和关键虚词的对齐
2. 重点关注医学术语、动词、名词的对齐
3. 方剂名、穴位名应保持完整

请直接输出JSON，不要有其他解释文字。
"""
        return prompt

    def call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Call LLM API with retries."""
        payload = {
            "model": "deepseek-ai/DeepSeek-V3",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": 2048,
            "temperature": 0.3,
        }

        for attempt in range(max_retries):
            try:
                self.rate_limiter.wait_if_needed()

                response = requests.post(
                    self.url,
                    json=payload,
                    headers=self.headers,
                    timeout=60,
                )
                response.raise_for_status()

                result = response.json()
                content = result["choices"][0]["message"]["content"]
                return content

            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise

    def parse_alignment_response(self, response: str) -> Dict:
        """Parse alignment JSON from raw LLM response."""
        try:
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                return result
            return {"alignments": [], "notes": "Parse failed"}

        except json.JSONDecodeError:
            return {"alignments": [], "notes": "JSON parse failed"}

    def align_sentence_pair(
        self, ancient_text: str, modern_text: str, rare_words: List[str] = None
    ) -> Dict:
        """Run alignment for one sentence pair."""
        prompt = self.create_alignment_prompt(ancient_text, modern_text, rare_words)
        response = self.call_llm(prompt)
        alignment = self.parse_alignment_response(response)

        alignment["ancient_text"] = ancient_text
        alignment["modern_text"] = modern_text
        alignment["rare_words"] = rare_words or []
        alignment["raw_response"] = response

        return alignment


def load_word_frequency(freq_file: str) -> Counter:
    """Load word frequency distribution."""
    if not os.path.exists(freq_file):
        return Counter()

    with open(freq_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    word_freq = Counter()
    for bin_data in data.values():
        for word_info in bin_data["words"]:
            word_freq[word_info["word"]] = word_info["frequency"]

    return word_freq


def identify_words_in_text(text: str, word_freq: Counter) -> List[str]:
    """Identify tokens to analyze from the input text."""
    words = list(jieba.cut(text))
    result = []

    for word in words:
        freq = word_freq.get(word, 0)
        # Keep tokens that appear in the frequency table and are longer than one character.
        if freq > 0 and len(word.strip()) > 1:
            result.append(word)

    return result


def load_parallel_corpus(queries_file: str, references_file: str) -> List[Tuple[str, str]]:
    """Load parallel corpus from query/reference files."""
    with open(queries_file, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]

    with open(references_file, "r", encoding="utf-8") as f:
        references = [line.strip() for line in f if line.strip()]

    return list(zip(queries, references))


def process_single_pair(args):
    """Process one sentence pair (for concurrent execution)."""
    idx, ancient, modern, word_freq, aligner = args

    words_to_analyze = identify_words_in_text(ancient, word_freq)

    if not words_to_analyze:
        return {
            "index": idx,
            "ancient_text": ancient,
            "modern_text": modern,
            "rare_words": [],
            "alignments": [],
            "notes": "No analyzable words; alignment skipped",
        }

    try:
        alignment = aligner.align_sentence_pair(ancient, modern, words_to_analyze)
        alignment["index"] = idx
        return alignment
    except Exception as e:
        return {
            "index": idx,
            "ancient_text": ancient,
            "modern_text": modern,
            "rare_words": words_to_analyze,
            "alignments": [],
            "notes": f"Processing failed: {str(e)[:100]}",
        }


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


def main():
    print("=" * 100)
    print("LLM-based classical-to-modern word alignment (concurrent)")
    print("=" * 100)

    # Configuration.
    api_key = os.getenv("SILICONFLOW_API_KEY", "")
    if not api_key:
        raise ValueError("Missing API key. Set SILICONFLOW_API_KEY first.")
    max_workers = 50

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load word frequency table.
    word_freq_file = os.path.join(script_dir, "word_frequency_distribution.json")
    print(f"\nLoading word frequencies: {word_freq_file}")
    word_freq = load_word_frequency(word_freq_file)
    print(f"Word-frequency vocabulary size: {len(word_freq)}")

    # Initialize aligner.
    rate_limiter = RateLimiter(max_per_minute=900)
    aligner = LLMWordAligner(api_key, rate_limiter)

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

    project_root = find_project_root(script_dir)

    for test_set in test_sets:
        print(f"\n{'=' * 80}")
        print(f"Processing test set: {test_set}")
        print(f"{'=' * 80}")

        queries_file = os.path.join(project_root, f"reference_answers/{test_set}_queries.txt")
        references_file = os.path.join(project_root, f"reference_answers/{test_set}_references.txt")

        if not os.path.exists(queries_file) or not os.path.exists(references_file):
            print(f"Skipping {test_set}: missing query/reference file")
            continue

        corpus = load_parallel_corpus(queries_file, references_file)
        print(f"Loaded {len(corpus)} sentence pairs")

        # Use a dedicated output file to avoid overwriting older outputs.
        output_file = os.path.join(script_dir, f"{test_set}_word_alignments_all_freq.json")

        # Resume from checkpoint when output already exists.
        processed_indices = set()
        alignments = []

        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                alignments = json.load(f)
            processed_indices = {a["index"] for a in alignments}
            print(f"Found {len(processed_indices)} already processed sentence pairs")

        # Build pending tasks.
        tasks = []
        for idx, (ancient, modern) in enumerate(corpus):
            if idx not in processed_indices:
                tasks.append((idx, ancient, modern, word_freq, aligner))

        print(f"Pending sentence pairs: {len(tasks)}")

        if not tasks:
            print("All sentence pairs are already processed")
            continue

        # Concurrent processing.
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_pair, task): task[0] for task in tasks}

            for future in tqdm(
                as_completed(futures), total=len(futures), desc=f"Aligning {test_set}"
            ):
                try:
                    result = future.result()
                    results.append(result)

                    # Save every 50 completed items.
                    if len(results) % 50 == 0:
                        all_alignments = alignments + results
                        all_alignments.sort(key=lambda x: x["index"])
                        with open(output_file, "w", encoding="utf-8") as f:
                            json.dump(all_alignments, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"Processing error: {e}")

        # Save final results.
        all_alignments = alignments + results
        all_alignments.sort(key=lambda x: x["index"])
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_alignments, f, ensure_ascii=False, indent=2)

        print(f"\n{test_set} alignment finished: {len(all_alignments)} sentence pairs")
        print(f"Saved to: {output_file}")

    print("\n" + "=" * 100)
    print("All test sets finished")
    print("=" * 100)


if __name__ == "__main__":
    main()
