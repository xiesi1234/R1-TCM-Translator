#!/usr/bin/env python3
"""
Zero-anaphora detection for TCM classical text

使用 LLM 判别原文中是否存在零形回指
Each sentence is evaluated 3 times with majority voting
"""

import json
import os
import re
import requests
import time
import threading
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, List, Tuple


# ==================== Custom log handler ====================

class InstantFileHandler(logging.FileHandler):
    """File handler with immediate flush after each write"""
    def emit(self, record):
        super().emit(record)
        self.flush()


# ==================== Configuration ====================

API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = os.getenv("SILICONFLOW_API_KEY", "")

# Evaluations per sentence
EVAL_TIMES = 3

PROMPT_TEMPLATE = """# Zero-anaphora detection for TCM classical text

## 定义
零形回指：句子中有一个与上文指称相同的对象但语法上没有词语表现形式。判断关键是：必须能从上文中找到先行词。

## 区别
- 零形回指：省略的内容从上文找
- 省略：省略的内容从上文/场景/常识找

## 评估重点
**只关注可能造成歧义或理解偏差的零形回指**。对于主语明确、不会产生误解的省略（如"头痛，发热"中的患者主语），不判定为contains zero anaphora。

## Few-shot示例

示例1（正例）
原文：服已，须臾，啜热稀粥一升余，以助药力。
参考译文：服药之后，过一会儿，喝热稀粥一升多，来辅助药力。
判断：contains = true
推理分析：原文'服'字前省略主语（患者），需从上文语境推断是服药。参考译文补充了"药"字，明确了是服药而非吃饭。这是典型的零形回指现象。
位置标注：["原文'服'字前省略主语（患者）"]

示例2（正例）
原文：食冷物，冰人齿。
参考译文：吃过分冰凉的东西，容易损坏人的牙齿。
判断：contains = true
推理分析：原文同时省略主语（吃冷物的人）和宾语（人），需从语境推断。参考译文补充了隐含的主语和宾语，使语义完整。
位置标注：["原文'食'字前省略主语（吃冷物的人）", "原文'人'字前省略宾语"]

示例3（正例）
原文：下利后脉绝，手足厥冷，晬时脉还，手足温者生。
参考译文：下利之后，患者出现脉搏消失，四肢冰凉，若经过一昼夜后脉搏复出、手足转温暖的，可以治愈。
判断：contains = true
推理分析：原文省略主语（患者），需从病证描述推断。参考译文明确补充了"患者"作为主语，使语义完整准确。
位置标注：["原文'下利后'前省略主语（患者）"]

示例4（正例）
原文：啬啬恶寒，淅淅恶风，翕翕发热，鼻鸣干呕者，桂枝汤主之。
参考译文：病人畏缩怕冷，瑟瑟畏风，象皮毛复盖身上一样发热，鼻塞气息不利，干呕的，应当用桂枝汤主治。
判断：contains = true
推理分析：原文省略主语（患太阳中风证的人），症状描述后接方剂，暗示是患者。参考译文补充了"病人"作为主语，使语义明确。
位置标注：["原文症状描述前省略主语（患太阳中风证的人）"]

示例5（正例）
原文：饮酒中风，则为漏风。
参考译文：饮酒之后感受风邪，成为漏风病。
判断：contains = true
推理分析：原文'饮酒'前省略主语（人），需从语境推断是饮酒者。参考译文补充了主语信息，明确了语义关系。
位置标注：["原文'饮酒'前省略主语（人）"]

示例6（反例）
原文：医生治病，先诊脉。
参考译文：医生治疗疾病，首先诊脉。
判断：contains = false
推理分析：原文中每个分句的主语明确：'医生'是'治病'的主语，也是'诊脉'的主语（承前省略但语义明确）。当主语明确且语义清晰时，不属于需要从远距离推断的零形回指。
位置标注：[]

示例7（反例）
原文：心者，君主之官，神明出焉。
参考译文：心是人体最重要的脏器，主持人体精神意识思维活动，就像国家的君主一样。
判断：contains = false
推理分析：原文是判断句，'心'是明确的主语，'君主之官'是谓语。不存在主语省略或需要从上下文推断的零形回指。
位置标注：[]

示例8（反例）
原文：头痛者，桂枝汤主之。
参考译文：头痛的人，用桂枝汤主治。
判断：contains = false
推理分析：原文'头痛者'明确指出了主体（头痛的人），虽是省略形式但语义明确。主体已明确给出，不属于需要从远距离上下文推断的零形回指。
位置标注：[]

示例9（反例）
原文：肝肺之气旺也；癫病者，安静而多悲恐。
参考译文：肝肺的气机偏旺；就癫病而言，病人多安静而悲伤恐惧。
判断：contains = false
推理分析：'肝肺之气旺也'中'肝肺之气'是明确的主语；'癫病者'明确指出了主体（癫病患者）。两句的主语都在句内明确给出，无需从上下文推断先行词。
位置标注：[]

## 评估任务
原文：{source_text}
参考译文：{reference_translation}

## 判断标准
- **contains = true**：原文中存在需要关注的零形回指（可能造成歧义或理解偏差）
- **contains = false**：原文中没有零形回指，或省略不会产生歧义

## 评估步骤
请先进行推理分析，再给出最终判断：

**第一步：推理分析**
1. 仔细分析原文的句法结构
2. 识别是否存在省略的主语、宾语或其他成分
3. 判断省略内容是否能从上文中找到先行词
4. 评估这种省略是否可能造成理解歧义

**第二步：给出判断**
根据推理分析结果，判断是否存在零形回指，并标注具体位置。

## 输出格式（JSON）
请直接输出 JSON，不要包含 markdown 标记：
{{
  "contains": <true 或 false>,
  "words": ["原文'服'字前省略主语（患者）'", "原文'人'字前省略宾语"],
  "reason": "<判断理由，说明是否存在零形回指>"
}}
注意：当contains为false时，words应为空数组[]。
"""

# ==================== Rate limiter ====================

class RateLimiter:
    """Rate limiter"""
    def __init__(self, max_per_minute=900):
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


# ==================== Data loading ====================

def load_queries(filepath: str) -> List[str]:
    """Load source text"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def load_references(filepath: str) -> List[str]:
    """Load reference translation"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def load_indices(filepath: str) -> List[int]:
    """Load valid index list"""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return [int(line.strip()) for line in f if line.strip()]


def find_project_root(start_dir: str) -> str:
    """Find project root containing reference_answers"""
    cur = start_dir
    for _ in range(8):
        if os.path.exists(os.path.join(cur, "reference_answers")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return start_dir


def extract_modern_translation(text: str) -> str:
    """Extract modern translation from model output"""
    if '<answer>' in text and '</answer>' in text:
        return text.split('<answer>')[1].split('</answer>')[0].strip()
    if '[现代译文]' in text:
        parts = text.split('[现代译文]')
        if len(parts) > 1:
            translation = parts[-1].strip()
            for marker in ['[', '\n\n', '</think>']:
                if marker in translation:
                    translation = translation.split(marker)[0].strip()
            if translation:
                return translation
    for pattern in ['现代译文：', '现代译文:']:
        if pattern in text:
            parts = text.split(pattern)
            if len(parts) > 1:
                translation = parts[-1].strip()
                for marker in ['\n\n', '\n[', '</think>']:
                    if marker in translation:
                        translation = translation.split(marker)[0].strip()
                if translation:
                    return translation
    return text.strip()


def load_model_predictions(model_path: str, test_set: str) -> List[str]:
    """Load model predictions (return all lines including blanks)"""
    detailed_json = os.path.join(model_path, f"{test_set}_test", f"{test_set}_test_detailed_1.json")
    pred_txt = os.path.join(model_path, f"{test_set}_test", f"{test_set}_test_answers_1.txt")
    
    if os.path.exists(detailed_json):
        with open(detailed_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        predictions = []
        for item in data:
            if 'extracted_answer' in item and item['extracted_answer'].strip():
                predictions.append(item['extracted_answer'].strip())
            elif 'full_model_output' in item:
                translation = extract_modern_translation(item['full_model_output'])
                predictions.append(translation)
            else:
                predictions.append("")
        return predictions
    elif os.path.exists(pred_txt):
        with open(pred_txt, 'r', encoding='utf-8') as f:
            # Keep all lines (including blanks) for index alignment
            return [line.strip() for line in f]
    
    return []


# ==================== LLM 评估 ====================

def evaluate_single_once(
    source_text: str,
    reference: str,
    rate_limiter: RateLimiter
) -> Dict:
    """Evaluate one sample once (single API call)"""
    prompt = PROMPT_TEMPLATE.format(
        source_text=source_text,
        reference_translation=reference
    )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-ai/DeepSeek-V3.2",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": 2000,
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }

    max_retries = 8
    for attempt in range(max_retries):
        try:
            rate_limiter.wait_if_needed()
            # Increase timeout to 180 seconds
            response = requests.post(API_URL, headers=headers, json=payload, timeout=180)

            if response.status_code != 200:
                logging.warning(f"API request failed: status={response.status_code}, response={response.text[:200]}")

            if response.status_code == 429:
                wait_time = min(30 * (attempt + 1), 120)
                logging.warning(f"HTTP 429 rate limit, wait {wait_time} seconds before retry (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content'].strip()

            # Clean content: remove markdown code fences
            clean_content = content.replace("```json", "").replace("```", "").strip()

            # Try parsing JSON; if it fails, extract valid JSON snippet
            try:
                result = json.loads(clean_content)
            except json.JSONDecodeError:
                # Try extracting a full JSON object
                # Find content between the first { and last }
                match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', clean_content, re.DOTALL)
                if match:
                    try:
                        result = json.loads(match.group(0))
                    except json.JSONDecodeError as e2:
                        logging.error(f"JSON still failed to parse after extraction: {str(e2)}, extracted={match.group(0)[:200]}")
                        raise
                else:
                    logging.error(f"Unable to extract valid JSON: content={clean_content[:300]}")
                    raise

            # Process contains field
            contains = result.get('contains', False)
            if isinstance(contains, str):
                contains = contains.lower() == 'true'

            # Process words field
            words = result.get('words', [])
            if not isinstance(words, list):
                words = []

            # API success log
            contains_str = "包含" if contains else "不包含"
            reason_preview = result.get('reason', '')[:50]
            logging.info(f"[API success] contains={contains_str}, reason={reason_preview}...")

            return {
                'contains': contains,
                'words': words,
                'reason': result.get('reason', ''),
                'success': True
            }
        except json.JSONDecodeError as e:
            content_preview = content[:300] if 'content' in locals() else 'empty'
            logging.error(f"JSON parse failed: {str(e)}, content={content_preview}")
            if attempt < max_retries - 1:
                wait_time = 3 * (attempt + 1)
                logging.info(f"等待{wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                return {'score': 0, 'error': f'JSON parse failed: {str(e)}', 'success': False}
        except requests.exceptions.Timeout as e:
            logging.error(f"Request timed out: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                logging.info(f"等待{wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                return {'score': 0, 'error': f'Request timed out: {str(e)}', 'success': False}
        except requests.exceptions.ConnectionError as e:
            logging.error(f"连接Error: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                logging.info(f"等待{wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                return {'score': 0, 'error': f'连接Error: {str(e)}', 'success': False}
        except Exception as e:
            logging.error(f"APIError: {str(e)}")
            if '429' in str(e) or 'rate' in str(e).lower():
                wait_time = min(30 * (attempt + 1), 120)
                logging.info(f"可能是限速，等待{wait_time}秒后重试...")
                time.sleep(wait_time)
                continue
            if attempt < max_retries - 1:
                wait_time = 3 * (attempt + 1)
                time.sleep(wait_time)
            else:
                return {'score': 0, 'error': f'APIError: {str(e)}', 'success': False}

    return {'score': 0, 'error': '未知Error', 'success': False}


def evaluate_single(
    source_text: str,
    reference: str,
    rate_limiter: RateLimiter,
    test_set: str = None
) -> Dict:
    """判别单samples（评估3次，投票决定结果）"""
    results = []

    for i in range(EVAL_TIMES):
        result = evaluate_single_once(source_text, reference, rate_limiter)
        if result.get('success'):
            results.append(result)
        else:
            logging.warning(f"run {i+1} failed: {result.get('error', 'unknown')}")

    if not results:
        return {'contains': False, 'error': f'所有评估都失败', 'success': False}

    # Majority vote on contains
    contains_true_count = sum(1 for r in results if r.get('contains') is True)
    contains_false_count = sum(1 for r in results if r.get('contains') is False)

    # Majority decides final contains value
    final_contains = contains_true_count > contains_false_count

    # Use reason from the first evaluation
    first_reason = results[0].get('reason', '') if results else ''

    return {
        'test_set': test_set,  # add test-set info
        'source': source_text,
        'reference': reference,
        'contains': final_contains,
        'all_evaluations': results,  # full results from three evaluations
        'success': True
    }


def process_sample(args: Tuple) -> Tuple[int, Dict]:
    """Process one sample (for concurrent execution)"""
    idx, source, reference, test_set, rate_limiter, cache, cache_lock, cache_file = args

    cache_key = f"{source[:50]}|{reference[:50]}"
    source_preview = source[:30] + "..." if len(source) > 30 else source

    with cache_lock:
        if cache_key in cache:
            cached = cache[cache_key]
            # Ensure cached result also includes test_set
            if 'test_set' not in cached:
                cached['test_set'] = test_set
            # Cache-hit log
            contains_str = "包含" if cached.get('contains') else "不包含"
            logging.info(f"[Cache hit] 样本#{idx} ({source_preview}) -> {contains_str}零形回指")
            return idx, cached

    result = evaluate_single(source, reference, rate_limiter, test_set)

    with cache_lock:
        cache[cache_key] = result
        # Save cache to disk in real time
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save cache: {e}")

    # New-evaluation completion log
    if result.get('success'):
        contains_str = "包含" if result.get('contains') else "不包含"
        logging.info(f"[Evaluation done] 样本#{idx} ({source_preview}) -> {contains_str}零形回指")
    else:
        logging.warning(f"[Evaluation failed] 样本#{idx} ({source_preview}) -> {result.get('error', 'unknown')}")

    return idx, result


# ==================== Main evaluation flow ====================

def evaluate_model(
    model_name: str,
    model_path: str,
    test_sets: List[str],
    project_root: str,
    cache_file: str,
    max_workers: int = 50,
    max_samples: int = None
) -> Dict:
    """Detect whether zero anaphora exists

    Args:
        max_samples: 最大处理样本数，None表示处理全部
    """

    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache)} cache entries")

    cache_lock = threading.Lock()
    rate_limiter = RateLimiter(max_per_minute=900)

    all_samples = []

    for test_set in test_sets:
        queries_file = os.path.join(project_root, f"reference_answers/{test_set}_queries.txt")
        references_file = os.path.join(project_root, f"reference_answers/{test_set}_references.txt")

        if not os.path.exists(queries_file) or not os.path.exists(references_file):
            print(f"  Warning: {test_set} data file missing")
            continue

        sources = load_queries(queries_file)
        references = load_references(references_file)

        min_len = min(len(sources), len(references))
        for i in range(min_len):
            all_samples.append({
                'source': sources[i],
                'reference': references[i],
                'test_set': test_set
            })

        print(f"  Loaded {test_set}: {min_len} samples")

    print(f"  Total {len(all_samples)} samples")

    # Test mode：限制处理样本数
    if max_samples is not None and len(all_samples) > max_samples:
        all_samples = all_samples[:max_samples]
        print(f"  [Test mode] process only first {max_samples} samples")

    tasks = [
        (i, s['source'], s['reference'], s['test_set'], rate_limiter, cache, cache_lock, cache_file)
        for i, s in enumerate(all_samples)
    ]

    results = [None] * len(all_samples)
    completed_count = 0
    progress_interval = 50  # 每50samples记录一次进度

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_sample, task): task[0] for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"判别 {model_name}"):
            try:
                idx, result = future.result()
                results[idx] = result
                completed_count += 1

                # 每处理 progress_interval samples记录一次进度
                if completed_count % progress_interval == 0:
                    progress_pct = completed_count / len(all_samples) * 100
                    logging.info(f"[Progress] completed {completed_count}/{len(all_samples)} ({progress_pct:.1f}%)")

            except Exception as e:
                idx = futures[future]
                results[idx] = {'contains': False, 'error': str(e), 'success': False}
                completed_count += 1
                logging.error(f"sample {idx} processing exception: {str(e)}")

    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    # Summary statistics
    valid_results = [r for r in results if r and r.get('success', False)]

    # Compute contains distribution
    contains_true_count = sum(1 for r in valid_results if r.get('contains') is True)
    contains_false_count = sum(1 for r in valid_results if r.get('contains') is False)

    # Debug: first 5 failed samples
    failed_samples = [(i, r) for i, r in enumerate(results) if r and not r.get('success', False)]
    if failed_samples:
        logging.info("\n  [Debug] Error info for first 5 failed samples:")
        for i, (idx, r) in enumerate(failed_samples[:5]):
            sample = all_samples[idx]
            source_preview = sample['source'][:50] + "..." if len(sample['source']) > 50 else sample['source']
            logging.info(f"    样本 {idx}: {source_preview}")
            logging.info(f"      Error: {r.get('error', 'unknown')}")
        logging.info(" ")

    # Group statistics by test_set
    stats_by_testset = {}
    idx = 0
    for test_set in test_sets:
        test_set_samples = [s for s in all_samples if s['test_set'] == test_set]
        test_set_valid = [r for r in results[idx:idx+len(test_set_samples)] if r and r.get('success', False)]
        test_set_true = sum(1 for r in test_set_valid if r.get('contains') is True)
        test_set_false = sum(1 for r in test_set_valid if r.get('contains') is False)
        stats_by_testset[test_set] = {
            'total': len(test_set_samples),
            'contains_true': test_set_true,
            'contains_false': test_set_false,
            'contains_rate': test_set_true / len(test_set_valid) * 100 if test_set_valid else 0
        }
        idx += len(test_set_samples)

    return {
        'model_name': model_name,
        'total_samples': len(all_samples),
        'evaluated_samples': len(valid_results),
        'contains_true': contains_true_count,
        'contains_false': contains_false_count,
        'contains_rate': contains_true_count / len(valid_results) * 100 if valid_results else 0,
        'stats_by_testset': stats_by_testset,
        'detailed_results': results
    }


def generate_report(all_results: Dict[str, Dict]) -> str:
    """Generate evaluation report"""
    report = []
    report.append("=" * 100)
    report.append("Zero-anaphora detection for TCM classical text报告")
    report.append("使用 DeepSeek V3 进行判别（每句评估3次，投票决定结果）")
    report.append("=" * 100)
    report.append("")

    report.append("## Evaluation summary")
    report.append("")
    report.append(f"{'模型':<25} {'总样本':<8} {'包含':<8} {'不包含':<8} {'Positive rate':<10}")
    report.append("-" * 100)

    for model_name, results in all_results.items():
        report.append(
            f"{model_name:<25} "
            f"{results['total_samples']:<8} "
            f"{results['contains_true']:<8} "
            f"{results['contains_false']:<8} "
            f"{results['contains_rate']:<10.2f}%"
        )

    report.append("")
    report.append("-" * 100)
    report.append("")

    # 添加按书籍分组的统计
    for model_name, results in all_results.items():
        if 'stats_by_testset' in results:
            report.append(f"## {model_name} - Per-book statistics")
            report.append("")
            report.append(f"{'书籍':<20} {'总样本':<8} {'包含':<8} {'不包含':<8} {'Positive rate':<10}")
            report.append("-" * 100)

            for test_set, stats in results['stats_by_testset'].items():
                report.append(
                    f"{test_set:<20} "
                    f"{stats['total']:<8} "
                    f"{stats['contains_true']:<8} "
                    f"{stats['contains_false']:<8} "
                    f"{stats['contains_rate']:<10.2f}%"
                )

            report.append("")
            report.append("-" * 100)
            report.append("")

    return "\n".join(report)


def main():
    # ==================== Test modeConfiguration ====================
    TEST_MODE = False  # Set False to process full dataset
    MAX_TEST_SAMPLES = 2 if TEST_MODE else None  # In test mode, process only first 2 samples
    # ===================================================

    # Configuration日志
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_filename = os.path.join(script_dir, f"evaluate_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            InstantFileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logging.info("=" * 100)
    logging.info("Zero-anaphora detection for TCM classical text")
    logging.info("Each sentence is evaluated 3 times with majority voting")
    logging.info("=" * 100)
    logging.info("")

    print("=" * 100)
    print("Zero-anaphora detection for TCM classical text")
    print("Each sentence is evaluated 3 times with majority voting")
    print("=" * 100)
    print("")

    project_root = find_project_root(script_dir)

    if not API_KEY:
        raise ValueError("Missing API key. Please set SILICONFLOW_API_KEY first")

    logging.info(f"Project root: {project_root}")
    logging.info(f"API endpoint: {API_URL}")
    logging.info(f"Concurrency: 50 threads")
    logging.info(f"Evaluations per sentence: {EVAL_TIMES}")
    logging.info(f"Log file: {log_filename}")
    logging.info("")

    print(f"Project root: {project_root}")
    print(f"API endpoint: {API_URL}")
    print(f"Concurrency: 50 threads")
    print(f"Evaluations per sentence: {EVAL_TIMES}")
    print(f"Log file: {log_filename}")
    print("")

    test_sets = [
        "bianquexinshu", "huangdineijing", "jingkuiyaolue", "maijing",
        "nanjing", "shanghanlun", "sishengxinyuan", "wenbingtiaobian"
    ]

    # Detect zero anaphora不需要模型路径，只需要数据集
    all_model_results = {}

    # 由于现在只需要判别是否存在零形回指，不需要模型路径
    # Use a generic identifier
    model_name = "数据集"
    dummy_path = ""  # not needed anymore

    logging.info("")
    logging.info("=" * 80)
    logging.info(f"Detect zero anaphora: {model_name}")
    logging.info("=" * 80)

    print(f"\n{'='*80}")
    print(f"Detect zero anaphora: {model_name}")
    print(f"{'='*80}")

    cache_file = os.path.join(script_dir, "cache_zero_anaphora.json")

    results = evaluate_model(
        model_name,
        dummy_path,
        test_sets,
        project_root,
        cache_file,
        max_workers=10,
        max_samples=MAX_TEST_SAMPLES
    )

    all_model_results[model_name] = results

    logging.info("")
    logging.info(f"{model_name} 判别结果:")
    logging.info(f"  Total samples: {results['total_samples']}")
    logging.info(f"  Evaluated samples: {results['evaluated_samples']}")
    logging.info(f"  contains zero anaphora: {results['contains_true']}")
    logging.info(f"  不contains zero anaphora: {results['contains_false']}")
    logging.info(f"  Positive rate: {results['contains_rate']:.2f}%")

    print(f"\n{model_name} 判别结果:")
    print(f"  Total samples: {results['total_samples']}")
    print(f"  Evaluated samples: {results['evaluated_samples']}")
    print(f"  contains zero anaphora: {results['contains_true']}")
    print(f"  不contains zero anaphora: {results['contains_false']}")
    print(f"  Positive rate: {results['contains_rate']:.2f}%")

    report = generate_report(all_model_results)
    print("\n" + report)

    report_file = os.path.join(script_dir, "zero_anaphora_evaluation_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved to: {report_file}")

    # 保存详细结果，包含原文和参考译文
    results_file = os.path.join(script_dir, "zero_anaphora_evaluation_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_model_results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {results_file}")
    # Save per-book results
    print("\n" + "=" * 100)
    print("Saving results by book...")
    print("=" * 100)

    detailed_results = results.get('detailed_results', [])
    all_samples_info = []
    for test_set in test_sets:
        queries_file = os.path.join(project_root, f"reference_answers/{test_set}_queries.txt")
        references_file = os.path.join(project_root, f"reference_answers/{test_set}_references.txt")

        if not os.path.exists(queries_file) or not os.path.exists(references_file):
            continue

        with open(queries_file, 'r', encoding='utf-8') as f:
            sources = [line.strip() for line in f if line.strip()]
        with open(references_file, 'r', encoding='utf-8') as f:
            references = [line.strip() for line in f if line.strip()]

        for i, (src, ref) in enumerate(zip(sources, references)):
            all_samples_info.append({
                'test_set': test_set,
                'index': i,
                'source': src,
                'reference': ref
            })

    # Create a separate result file for each book
    for test_set in test_sets:
        test_set_samples = [s for s in all_samples_info if s['test_set'] == test_set]

        if not test_set_samples:
            continue

        # Find detailed results for this book
        test_set_results = []
        contains_true_count = 0

        for sample in test_set_samples:
            # Find matching result in detailed_results
            result = None
            for r in detailed_results:
                if r and r.get('source') == sample['source'] and r.get('reference') == sample['reference']:
                    result = r
                    break

            if result:
                # Get reason: prefer all_evaluations when available
                reason = result.get('reason', '')
                if not reason and 'all_evaluations' in result:
                    evals = result.get('all_evaluations', [])
                    if evals and len(evals) > 0:
                        reason = evals[0].get('reason', '')

                item = {
                    'index': sample['index'],
                    'source': result.get('source'),
                    'reference': result.get('reference'),
                    'contains': result.get('contains'),
                    'reason': reason,
                    'success': result.get('success', True)
                }
                test_set_results.append(item)

                if result.get('contains'):
                    contains_true_count += 1

        # Save this book's results
        test_set_file = os.path.join(script_dir, f"results_{test_set}.json")
        test_set_data = {
            'test_set': test_set,
            'total_samples': len(test_set_samples),
            'contains_true': contains_true_count,
            'contains_false': len(test_set_results) - contains_true_count,
            'contains_rate': contains_true_count / len(test_set_results) * 100 if test_set_results else 0,
            'samples': test_set_results
        }

        with open(test_set_file, 'w', encoding='utf-8') as f:
            json.dump(test_set_data, f, ensure_ascii=False, indent=2)

        print(f"  {test_set}: {len(test_set_samples)} 样本, {contains_true_count} contains zero anaphora -> {test_set_file}")

    print("=" * 100)



if __name__ == "__main__":
    main()
