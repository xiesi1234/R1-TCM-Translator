#!/usr/bin/env python3
"""
Semantic-shift dataset classification

使用 LLM 判断原文中是否存在古今异义现象（数据集分类任务）
每个句子评估3次取投票结果
"""

import json
import os
import time
import threading
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, List, Tuple
from openai import OpenAI


# ==================== Configuration ====================

# 火山引擎方舟APIConfiguration
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
ARK_MODEL = "deepseek-v3-2-251201"
ARK_API_KEY = os.getenv("ARK_API_KEY", "")

# Evaluations per sentence
EVAL_TIMES = 3

PROMPT_TEMPLATE = """# 古今异义与语义变迁判断

## 定义
古今异义与语义变迁：同一词汇在古代汉语和现代汉语中含义不同。判断关键是：原文词汇在古汉语语境下的含义与现代汉语存在差异。一定要是在中医特定领域里面的。

## 区别
- 古今异义：同一词汇的历时语义变化
- 多义词：同一词汇在同一时代的多个义项

## 评估重点
只关注影响准确理解的古今异义现象。对于古今含义一致或差异微小的词汇，不判定为包含古今异义。

## Few-shot示例

### 示例1（正例）
原文：伤寒，胸中有热，胃中有邪气，腹中痛，欲呕吐者，黄连汤主之。
参考译文：伤寒病，胸中有热邪，胃中有寒邪，腹部疼痛，想要呕吐的，用黄连汤主治。

推理分析：原文'邪气'在此特定语境下指'寒邪'（与胸中'热'相对），而非现代汉语中泛指的致病因素。参考译文正确翻译为'寒邪'，说明这是古今异义词。
位置标注：邪气
参考文献：张晓. 认知翻译学视阈下《伤寒论》英译策略探究[J]. 现代语言学, 2023, 11(11): 5024-5029.
判断：存在

### 示例2（正例）
原文：脉浮热甚，反灸之，此为实。
参考译文：脉象浮，发热甚，这是太阳表实证。

推理分析：原文'实'指邪气盛（实证），与'虚'（正气虚）相对，是中医辨证的核心概念，非现代汉语的'真实、实在'。参考译文正确翻译为'实证'，说明这是古今异义词。
位置标注：实
参考文献：王育林. 标准医古文[M]. 北京: 外语教学与研究出版社, 2022: 218-220.
判断：存在

### 示例3（正例）
原文：老人滑肠困重，乃阳气虚脱，小便不禁。
参考译文：老人腹泻拉肚子，身体困重，是阳气虚脱的缘故，小便失控。

推理分析：原文'滑肠'指腹泻、大便滑泄不禁，是中医病名描述，非现代可能理解的'肠道光滑'。参考译文正确翻译为'腹泻'，说明这是古今异义词。
位置标注：滑肠
参考文献：王育林. 标准医古文[M]. 北京: 外语教学与研究出版社, 2022: 189-192.
判断：存在

### 示例4（正例）
原文：厥终不过五日，以热五日，故知自愈。
参考译文：四肢厥冷Total只有五天，而发热也是五天，四肢厥冷与发热时间相等，阴阳趋于平衡，所以知道会自行痊愈。

推理分析：原文'厥'指四肢厥冷，是阳气不能达于四末的表现，非现代汉语的'昏厥、休克'。参考译文正确翻译为'四肢厥冷'，说明这是古今异义词。
位置标注：厥
参考文献：王育林. 标准医古文[M]. 北京: 外语教学与研究出版社, 2022: 201-205.
判断：存在

### 示例5（正例）
原文：下利，脉数而渴者，今自愈。
参考译文：虚寒腹泻，出现脉数而口渴的，是阳气回复，其病将要痊愈。

推理分析：原文'下利'指腹泻、便泄，是中医病名，非现代汉语'向下获利'。参考译文正确翻译为'腹泻'，说明这是古今异义词。
位置标注：下利
参考文献：王育林. 标准医古文[M]. 北京: 外语教学与研究出版社, 2022: 176-179.
判断：存在

### 示例6（正例）
原文：小便少者，必苦里急也。
参考译文：如果小便短少不通畅的，是水停下焦，一定会出现小腹部胀满急迫不舒的症状。

推理分析：原文'里急'指小腹部胀满急迫不舒的感觉，是中医特有症状描述，非现代'里面急迫'的泛指理解。参考译文正确译出其解剖部位和症状特点，说明这是古今异义词。
位置标注：里急
参考文献：王育林. 标准医古文[M]. 北京: 外语教学与研究出版社, 2022: 234-237.
判断：存在

### 示例7（反例）
原文：太阳病，头痛发热。
参考译文：太阳病，头痛，发热。

推理分析：原文中'头痛'和'发热'均为基础症状描述，在古今汉语中含义一致，均指身体症状。这些词汇不涉及语义变迁，古今理解相同。
位置标注：无
参考文献：王育林. 标准医古文[M]. 北京: 外语教学与研究出版社, 2022: 42-45.
判断：不存在

### 示例8（反例）
原文：服药后，汗出而愈。
参考译文：服药之后，出汗就好了。

推理分析：原文中'服药'指吃药、'汗出'指出汗、'愈'指痊愈。这些词汇在古代和现代汉语中含义基本一致，不涉及古今异义问题。
位置标注：无
参考文献：王育林. 标准医古文[M]. 北京: 外语教学与研究出版社, 2022: 38-40.
判断：不存在

### 示例9（反例）
原文：病人腹痛，按之益甚。
参考译文：病人腹部疼痛，按压更加严重。

推理分析：'腹痛'古今均指腹部疼痛；'按'指按压；'甚'指严重。这些词汇在古代和现代汉语中含义基本一致，不涉及语义变迁。
位置标注：无
参考文献：王育林. 标准医古文[M]. 北京: 外语教学与研究出版社, 2022: 156-159.
判断：不存在

### 示例10（反例）
原文：桂枝、芍药、甘草，各三两。
参考译文：桂枝、芍药、甘草，各三两。

推理分析：'桂枝''芍药''甘草'均为中药名称，古今一致；'各'表示分别、各自；'三两'为剂量描述。这些词汇古今含义相同，不涉及语义变迁。
位置标注：无
参考文献：王育林. 标准医古文[M]. 北京: 外语教学与研究出版社, 2022: 78-82.
判断：不存在

### 示例11（正例）
原文：寸口脉缓而迟，缓则阳气长，其色鲜，其颜光。
参考译文：寸口脉缓而迟，缓脉是卫气调和之象，卫气充盛于外，所以其人皮肤颜色鲜明，有光泽。

推理分析：原文'寸口'指手腕桡动脉处，是中医脉诊专用部位，非现代'一寸宽的口'。参考译文保留了专业术语，说明这是古今异义词。
位置标注：寸口
参考文献：王育林. 标准医古文[M]. 北京: 外语教学与研究出版社, 2022: 298-302.
判断：存在

## 评估任务
原文：{source_text}
参考译文：{reference_translation}

## 评估步骤

1. 识别原文中可能存在古今异义的词汇
2. 确认该词汇在古汉语语境下的确切含义
3. 判断该词汇与现代汉语含义是否存在差异
4. 给出详细的推理分析（先推理）
5. 最后给出判断结论（存在/不存在）

## 输出格式（JSON）
请直接输出 JSON，不要包含 markdown 标记：
{{
  "contains": <true 或 false>,
  "words": ["识别出的古今异义词列表，如无则为空数组"],
  "reason": "<判断理由，说明是否存在古今异义问题>"
}}
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


# ==================== LLM 分类 ====================

def classify_single_once(
    source_text: str,
    reference: str,
    rate_limiter: RateLimiter,
    client: OpenAI
) -> Dict:
    """Classify one sample once (single API call)"""
    prompt = PROMPT_TEMPLATE.format(
        source_text=source_text,
        reference_translation=reference
    )

    max_retries = 8
    for attempt in range(max_retries):
        try:
            rate_limiter.wait_if_needed()

            completion = client.chat.completions.create(
                model=ARK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.0,
            )

            content = completion.choices[0].message.content.strip()

            clean_content = content.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean_content)

            # 成功日志
            logging.debug(f"API请求成功: contains={result.get('contains', False)}")

            # Process contains field
            contains = result.get('contains', False)
            if isinstance(contains, str):
                contains = contains.lower() == 'true'

            return {
                'contains': contains,
                'words': result.get('words', []) or [],
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
                return {'contains': False, 'error': f'JSON parse failed: {str(e)}', 'success': False}
        except Exception as e:
            error_str = str(e)
            logging.error(f"APIError: {error_str}")

            # 检查是否是限流Error
            if '429' in error_str or 'rate' in error_str.lower() or 'limit' in error_str.lower():
                wait_time = min(30 * (attempt + 1), 120)
                logging.warning(f"可能是限速，等待{wait_time}秒后重试...")
                time.sleep(wait_time)
                continue

            if attempt < max_retries - 1:
                wait_time = 3 * (attempt + 1)
                logging.info(f"等待{wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                return {'contains': False, 'error': f'APIError: {error_str}', 'success': False}

    return {'contains': False, 'error': '未知Error', 'success': False}


def classify_single(
    source_text: str,
    reference: str,
    rate_limiter: RateLimiter,
    test_set: str = None,
    client: OpenAI = None
) -> Dict:
    """Classify one sample (3 runs, majority vote)"""
    results = []

    for i in range(EVAL_TIMES):
        result = classify_single_once(source_text, reference, rate_limiter, client)
        if result.get('success'):
            results.append(result)
        else:
            logging.warning(f"run {i+1} failed: {result.get('error', 'unknown')}")

    if not results:
        logging.error(f"所有评估都失败: source={source_text[:50]}...")
        return {'contains': False, 'error': '所有评估都失败', 'success': False}

    # Majority vote on contains
    contains_true_count = sum(1 for r in results if r.get('contains') is True)
    contains_false_count = sum(1 for r in results if r.get('contains') is False)

    # Majority decides final contains value
    final_contains = contains_true_count > contains_false_count

    # 合并所有 words（去重）
    all_words = []
    for r in results:
        for w in r.get('words', []) or []:
            if w not in all_words:
                all_words.append(w)

    # Use reason from the first evaluation
    first_reason = results[0].get('reason', '') if results else ''

    return {
        'test_set': test_set,
        'source': source_text,
        'reference': reference,
        'contains': final_contains,
        'words': all_words,
        'reason': first_reason,
        'all_evaluations': results,
        'success': True
    }


def process_sample(args: Tuple) -> Tuple[int, Dict]:
    """Process one sample (for concurrent execution)"""
    idx, source, reference, rate_limiter, cache, cache_lock, cache_file, test_set, client = args

    cache_key = f"{source[:50]}|{reference[:50]}"

    with cache_lock:
        if cache_key in cache:
            cached = cache[cache_key]
            if 'test_set' not in cached:
                cached['test_set'] = test_set
            return idx, cached

    result = classify_single(source, reference, rate_limiter, test_set, client)

    with cache_lock:
        cache[cache_key] = result
        # Save cache to disk in real time
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save cache: {e}")

    return idx, result


# ==================== Main classification flow ====================

def classify_dataset(
    test_sets: List[str],
    project_root: str,
    cache_file: str,
    client: OpenAI,
    max_workers: int = 10,
    max_samples: int = None
) -> Dict:
    """对数据集进行分类

    Args:
        client: OpenAI 客户端实例
        max_samples: 最大处理样本数，None表示处理全部
    """

    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        logging.info(f"Loaded {len(cache)} cache entries")
        print(f"  Loaded {len(cache)} cache entries")

    cache_lock = threading.Lock()
    rate_limiter = RateLimiter(max_per_minute=900)

    all_samples = []

    for test_set in test_sets:
        queries_file = os.path.join(project_root, f"reference_answers/{test_set}_queries.txt")
        references_file = os.path.join(project_root, f"reference_answers/{test_set}_references.txt")

        if not os.path.exists(queries_file) or not os.path.exists(references_file):
            logging.warning(f"Warning: {test_set} data file missing")
            print(f"  Warning: {test_set} data file missing")
            continue

        sources = load_queries(queries_file)
        references = load_references(references_file)

        min_len = min(len(sources), len(references))
        for i in range(min_len):
            all_samples.append({
                'source': sources[i],
                'reference': references[i],
                'test_set': test_set,
                'index': i
            })

        logging.info(f"Loaded {test_set}: {min_len} samples")
        print(f"  Loaded {test_set}: {min_len} samples")

    logging.info(f"Total {len(all_samples)} samples")
    print(f"  Total {len(all_samples)} samples")

    # Test mode：限制处理样本数
    if max_samples is not None and len(all_samples) > max_samples:
        logging.info(f"[Test mode] process only first {max_samples} samples")
        print(f"  [Test mode] process only first {max_samples} samples")
        all_samples = all_samples[:max_samples]

    tasks = [
        (i, s['source'], s['reference'], rate_limiter, cache, cache_lock, cache_file, s['test_set'], client)
        for i, s in enumerate(all_samples)
    ]

    results = [None] * len(all_samples)
    completed_count = 0
    progress_interval = 50  # 每50samples记录一次进度

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_sample, task): task[0] for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="分类数据集"):
            try:
                idx, result = future.result()
                results[idx] = result
                completed_count += 1

                # 每处理 progress_interval samples记录一次进度
                if completed_count % progress_interval == 0:
                    progress_pct = completed_count / len(all_samples) * 100
                    logging.info(f"[Progress] completed {completed_count}/{len(all_samples)} ({progress_pct:.1f}%)")

                # 调试日志：记录每samples的处理结果
                if result and result.get('success'):
                    contains_status = "包含古今异义" if result.get('contains') else "不包含古今异义"
                    logging.debug(f"[样本 {idx}] {contains_status}")
                elif result:
                    logging.error(f"[样本 {idx}] 处理失败: {result.get('error', 'unknown')}")

            except Exception as e:
                idx = futures[future]
                results[idx] = {'contains': False, 'error': str(e), 'success': False}
                completed_count += 1
                logging.error(f"sample {idx} processing exception: {str(e)}")

    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    logging.info(f"分类完成，共处理 {completed_count} samples")

    # Summary statistics
    valid_results = [r for r in results if r and r.get('success', False)]

    # 打印调试信息：失败样本
    failed_samples = [(i, r) for i, r in enumerate(results) if r and not r.get('success', False)]
    if failed_samples:
        logging.warning(f"[调试] 发现 {len(failed_samples)} 个失败样本:")
        for i, (idx, r) in enumerate(failed_samples[:10]):  # 只显示前10个
            sample = all_samples[idx]
            source_preview = sample['source'][:50] + "..." if len(sample['source']) > 50 else sample['source']
            logging.warning(f"    样本 {idx}: {source_preview}")
            logging.warning(f"      Error: {r.get('error', 'unknown')}")
        if len(failed_samples) > 10:
            logging.warning(f"    ... 以及其他 {len(failed_samples) - 10} 个失败样本")
        logging.warning(" ")

    # Compute contains distribution
    contains_true_count = sum(1 for r in valid_results if r.get('contains') is True)
    contains_false_count = sum(1 for r in valid_results if r.get('contains') is False)

    # 日志记录Summary statistics
    logging.info(f"[统计] Evaluated samples: {len(valid_results)}/{len(all_samples)}")
    logging.info(f"[统计] 包含古今异义: {contains_true_count} ({contains_true_count/len(valid_results)*100:.1f}%)")
    logging.info(f"[统计] 不包含古今异义: {contains_false_count} ({contains_false_count/len(valid_results)*100:.1f}%)")

    # 统计所有识别出的古今异义词
    all_words_set = set()
    for r in valid_results:
        for w in r.get('words', []) or []:
            all_words_set.add(w)

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
        'total_samples': len(all_samples),
        'evaluated_samples': len(valid_results),
        'contains_true': contains_true_count,
        'contains_false': contains_false_count,
        'contains_rate': contains_true_count / len(valid_results) * 100 if valid_results else 0,
        'unique_words': sorted(list(all_words_set)),
        'stats_by_testset': stats_by_testset,
        'detailed_results': results
    }


def generate_report(results: Dict) -> str:
    """Generate classification report"""
    report = []
    report.append("=" * 100)
    report.append("Semantic-shift dataset classification报告")
    report.append("使用 DeepSeek-R1 进行判断（每句评估3次，投票决定结果）")
    report.append("=" * 100)
    report.append("")

    report.append("## Overall statistics")
    report.append("")
    report.append(f"Total samples: {results['total_samples']}")
    report.append(f"Evaluated samples: {results['evaluated_samples']}")
    report.append(f"包含古今异义: {results['contains_true']}")
    report.append(f"不包含古今异义: {results['contains_false']}")
    report.append(f"Positive rate: {results['contains_rate']:.2f}%")
    report.append(f"识别出的古今异义词数: {len(results['unique_words'])}")
    report.append("")

    report.append("=" * 100)
    report.append("## Per-dataset statistics")
    report.append("")
    report.append(f"{'数据集':<20} {'总样本':<8} {'包含':<8} {'不包含':<8} {'Positive rate':<10}")
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
    report.append("=" * 100)
    report.append("")
    report.append("## Summary of identified semantic-shift words")
    report.append("")

    words = results['unique_words']
    if words:
        report.append(f"共识别出 {len(words)} 个古今异义词:")
        for word in words:
            report.append(f"  - {word}")
    else:
        report.append("未识别出古今异义词")

    return "\n".join(report)


def main():
    # ==================== Test modeConfiguration ====================
    TEST_MODE = False   # Set False to process full dataset
    MAX_TEST_SAMPLES = 100 if TEST_MODE else None  # Test mode下只处理前100条
    # ===================================================

    # Configuration日志
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_filename = os.path.join(script_dir, f"evaluate_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # 文件记录所有级别

    # 文件handler - 记录DEBUG及以上级别
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(file_formatter)

    # 控制台handler - 只记录INFO及以上级别
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info("=" * 100)
    logging.info("Semantic-shift dataset classification")
    logging.info("Each sentence is evaluated 3 times with majority voting")
    logging.info("=" * 100)
    logging.info("")

    print("=" * 100)
    print("Semantic-shift dataset classification")
    print("Each sentence is evaluated 3 times with majority voting")
    print("=" * 100)
    print("")

    project_root = find_project_root(script_dir)

    logging.info(f"Project root: {project_root}")
    logging.info(f"API endpoint: {ARK_BASE_URL}")
    logging.info(f"Concurrency: 10 threads")
    logging.info(f"Evaluations per sentence: {EVAL_TIMES}")
    logging.info(f"Log file: {log_filename}")
    logging.info("")

    print(f"Project root: {project_root}")
    print(f"API endpoint: {ARK_BASE_URL}")
    print(f"Concurrency: 10 threads")
    print(f"Evaluations per sentence: {EVAL_TIMES}")
    print(f"Log file: {log_filename}")
    print("")

    test_sets = [
        "bianquexinshu", "huangdineijing", "jingkuiyaolue", "maijing",
        "nanjing", "shanghanlun", "sishengxinyuan", "wenbingtiaobian"
    ]

    cache_file = os.path.join(script_dir, "cache_semantic.json")

    # 初始化 OpenAI 客户端（火山引擎方舟）
    if not ARK_API_KEY:
        raise ValueError("缺少 API Key，请先设置环境变量 ARK_API_KEY")

    client = OpenAI(
        base_url=ARK_BASE_URL,
        api_key=ARK_API_KEY,
    )

    logging.info("")
    logging.info("=" * 80)
    logging.info("Start dataset classification")
    logging.info("=" * 80)
    print("")
    print("=" * 80)
    print("Start dataset classification")
    print("=" * 80)

    logging.info(f"Test mode: {TEST_MODE}")
    if TEST_MODE:
        logging.info(f"最大处理样本数: {MAX_TEST_SAMPLES}")

    results = classify_dataset(
        test_sets,
        project_root,
        cache_file,
        client,
        max_workers=10,
        max_samples=MAX_TEST_SAMPLES
    )

    logging.info("")
    logging.info("=" * 100)
    logging.info("Classification completed!")
    logging.info(f"  Total samples: {results['total_samples']}")
    logging.info(f"  Evaluated samples: {results['evaluated_samples']}")
    logging.info(f"  包含古今异义: {results['contains_true']}")
    logging.info(f"  不包含古今异义: {results['contains_false']}")
    logging.info(f"  Positive rate: {results['contains_rate']:.2f}%")
    logging.info(f"  Identified word count: {len(results['unique_words'])}")
    logging.info("=" * 100)

    print("")
    print("=" * 100)
    print("Classification completed!")
    print(f"  Total samples: {results['total_samples']}")
    print(f"  Evaluated samples: {results['evaluated_samples']}")
    print(f"  包含古今异义: {results['contains_true']}")
    print(f"  不包含古今异义: {results['contains_false']}")
    print(f"  Positive rate: {results['contains_rate']:.2f}%")
    print(f"  Identified word count: {len(results['unique_words'])}")
    print("=" * 100)

    report = generate_report(results)
    print("")
    print(report)

    report_file = os.path.join(script_dir, "semantic_change_evaluation_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    logging.info(f"Report saved to: {report_file}")
    print(f"\nReport saved to: {report_file}")

    # Save full results
    summary_results = {
        key: val for key, val in results.items() if key != 'detailed_results'
    }
    results_file = os.path.join(script_dir, "semantic_change_evaluation_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)
    logging.info(f"Results saved to: {results_file}")
    print(f"Results saved to: {results_file}")

    # Save per-book results
    logging.info("\n" + "=" * 100)
    logging.info("Saving results by book...")
    logging.info("=" * 100)
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
            result = None
            for r in detailed_results:
                if r and r.get('source') == sample['source'] and r.get('reference') == sample['reference']:
                    result = r
                    break

            if result:
                # 获取 reason
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
                    'words': result.get('words', []),
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

        logging.info(f"  {test_set}: {len(test_set_samples)} 样本, {contains_true_count} 包含古今异义 -> {test_set_file}")
        print(f"  {test_set}: {len(test_set_samples)} 样本, {contains_true_count} 包含古今异义 -> {test_set_file}")

    logging.info("=" * 100)
    print("=" * 100)


if __name__ == "__main__":
    main()
