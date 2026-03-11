#!/usr/bin/env python3
"""
文化语境与认识论差异数据集分类

使用 LLM 判断原文中是否存在文化语境与认识论差异现象（数据集分类任务）
每个句子评估3次取投票结果

用法:
    python evaluate_cultural_context.py --test-only     # 只测试2samples
    python evaluate_cultural_context.py --book shanghanlun  # 只评估伤寒论
    python evaluate_cultural_context.py                 # 评估所有书籍
"""

import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, List, Tuple
import logging
from datetime import datetime
from openai import OpenAI

# 所有书籍列表
ALL_BOOKS = [
    "bianquexinshu", "huangdineijing", "jingkuiyaolue", "maijing",
    "nanjing", "shanghanlun", "sishengxinyuan", "wenbingtiaobian"
]


# ==================== Configuration ====================

ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
ARK_API_KEY = os.getenv("ARK_API_KEY", "")

# 初始化OpenAI客户端，使用VolcEngine方舟API
client = OpenAI(
    base_url=ARK_BASE_URL,
    api_key=ARK_API_KEY,
)

MODEL_NAME = "deepseek-v3-2-251201"

# Evaluations per sentence
EVAL_TIMES = 3

PROMPT_TEMPLATE = """# 文化语境与认识论差异判断

## 定义
文化语境与认识论差异：中医古籍蕴含独特的理论体系和认识框架，通过隐喻性表达构建概念，与现代医学/日常认知存在根本差异。判断关键是：原文是否涉及中医核心理论概念或隐喻性表达，需要在中医自身理论体系内理解。

## 区别
- **认识论差异**：不同理论体系对同一现象的根本解释不同
- **术语翻译问题**：同一概念的不同表达方式
- **隐喻消解问题**：将中医隐喻性概念简单化/字面化理解

## 评估重点
只评估涉及中医核心理论概念的翻译。对于通用词汇或文化中性内容，不判定为包含文化语境差异。

---

## Few-shot示例

### 示例1（正例）
原文：心者，君主之官，神明出焉。
参考译文：心是人体最重要的脏器，主持人体精神意识思维活动，就像国家的君主一样。

推理分析：原文'君主之官'是古代政治隐喻，将心比作一国之君，体现中医'心主神明'理论。'神明出焉'指精神意识思维活动由心所主。这是中医藏象学说的核心概念，需要用中医自身理论体系理解。
识别概念：君主之官、神明
判断：存在

### 示例2（正例）
原文：胃者，水谷之海，六府之大源也。
参考译文：胃是受纳和腐熟水谷的器官，就像大海容纳百川一样，是六腑的源泉。

推理分析：原文'水谷之海'用'海'隐喻胃受纳万物之功能，体现中医取象比类思维。'六府之大源'指胃是六腑的源泉。这是中医藏象学说特有的隐喻性表达。
识别概念：水谷之海、六府、腐熟
判断：存在

### 示例3（正例）
原文：重阴必阳，重阳必阴。
参考译文：阴气过盛会转化为阳病，阳气过盛也会转变为阴病。

推理分析：原文体现中医阴阳理论的核心命题：物极必反，阴阳可相互转化。'重'意为'程度严重'，非'沉重'。这是中医阴阳学说的核心理论。
识别概念：重阴必阳、重阳必阴
判断：存在

### 示例4（正例）
原文：五行者，木、火、土、金、水。
参考译文：五行就是木、火、土、金、水。

推理分析：原文'五行'是中国哲学核心概念，将万物归类为五种属性及其生克制化关系，非化学元素。这是中医五行学说的基础理论。
识别概念：五行
判断：存在

### 示例5（正例）
原文：肾水下泄则虚火上炎，故多有夜热骨蒸。
参考译文：肾水下泄则虚火上炎，所以病人多伴有夜热骨蒸、五心烦热、唇口干燥等症状。

推理分析：原文'水火不济''虚火上炎'是中医特有病机描述，肾属水、心属火，水火失衡是中医核心理论。'骨蒸'是中医特有症状描述。这些概念属于中医病因病机学说。
识别概念：肾水下泄、虚火上炎、骨蒸
判断：存在

### 示例6（正例）
原文：三焦暖热方能腐熟水谷，若一刻无火则肌肤冰冷。
参考译文：三焦只有保持温热才能腐熟消化食物，如果有一刻失去火热的温暖就会肌肤冰冷。

推理分析：原文'三焦'是中医六腑之一，分上中下三焦，为气机升降之通道，是中医特有解剖概念。'腐熟水谷'是中医特有功能描述。这些是中医藏象学说核心概念。
识别概念：三焦、腐熟水谷
判断：存在

### 示例7（反例）
原文：病人发热，体温升高。
参考译文：病人发热，体温升高。

推理分析：原文'发热'、'体温升高'是对症状的描述性表达，不涉及中医特有的理论概念或隐喻性表达，古今中西医对发热的理解基本一致，属于通用医学描述。
识别概念：[]
判断：不存在

### 示例8（反例）
原文：患者感到口渴欲饮。
参考译文：患者感到口渴，想喝水。

推理分析：'口渴欲饮'是对生理反应的直接描述，属于通用表达，无中医特有文化负载或理论内涵。古今中西医对口渴的理解基本一致。
识别概念：[]
判断：不存在

### 示例9（反例）
原文：身体疼痛，活动不便。
参考译文：身体疼痛，活动不灵活。

推理分析：'身体疼痛''活动不便'是对身体状态的直接描述，属于通用表达，不涉及中医特有的理论概念或隐喻性表达，古今中西医理解一致。
识别概念：[]
判断：不存在

### 示例10（反例）
原文：服药之后，汗出而解。
参考译文：服药之后，出汗就好了。

推理分析：'服药''汗出''解除'是对治疗过程的直接描述，不涉及中医特有理论概念或隐喻性表达，古今理解一致，属于通用医学过程描述。
识别概念：[]
判断：不存在

### 示例11（正例）
原文：清升浊降，一定之位。
参考译文：清气要向上运动，浊气要向下运动，这是正常的生理规律。

推理分析：原文'清升浊降'是中医气机理论的核心概念，脾主升清、胃主降浊，体现中医独特的气机升降理论。这是中医气机学说的核心内容。
识别概念：清升浊降
判断：存在

---

## 评估任务
原文：{source_text}
参考译文：{reference_translation}

## 评估步骤

1. **识别原文中的中医核心概念**
   - 判断是否涉及阴阳、五行、藏象、气血津液、经络、病因病机等学说
   - 识别是否存在隐喻性表达（方位/实体/结构等）

2. **分析参考译文的理解方式**
   - 参考译文是否用中医自身理论概念来解释
   - 是否保留了原文的隐喻性和理论内涵

3. **先进行详细推理分析**
   - 说明原文涉及的中医理论体系
   - 解释这些概念在中医理论中的含义
   - 分析为何这些概念需要在中医理论体系内理解

4. **最后给出判断结论**
   - 如果涉及中医核心理论或隐喻表达 → 存在
   - 如果只是通用描述，不涉及中医特有概念 → 不存在

## 输出格式（JSON）
请直接输出 JSON，不要包含 markdown 标记：
{{
  "contains": <true 或 false>,
  "tcm_concepts": ["识别出的中医核心概念列表，如无则为空数组"],
  "reason": "<详细的推理分析和判断理由>"
}}
"""


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


# ==================== Custom log handler ====================

class InstantFileHandler(logging.FileHandler):
    """File handler with immediate flush after each write"""
    def emit(self, record):
        super().emit(record)
        self.flush()


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


# ==================== LLM 分类 ====================

def classify_single_once(
    source_text: str,
    reference: str,
    rate_limiter: RateLimiter
) -> Dict:
    """Classify one sample once (single API call)"""

    # ==================== 旧方式（requests + SiliconFlow）- 已弃用 ====================
    # prompt = PROMPT_TEMPLATE.format(
    #     source_text=source_text,
    #     reference_translation=reference
    # )
    # headers = {
    #     "Authorization": f"Bearer {API_KEY}",
    #     "Content-Type": "application/json"
    # }
    # payload = {
    #     "model": "deepseek-ai/DeepSeek-R1",
    #     "messages": [{"role": "user", "content": prompt}],
    #     "stream": False,
    #     "max_tokens": 2000,
    #     "temperature": 0.0,
    #     "response_format": {"type": "json_object"}
    # }
    # response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    # content = response.json()['choices'][0]['message']['content'].strip()
    # ================================================================================

    # 新模式（OpenAI SDK + VolcEngine 方舟）
    prompt = PROMPT_TEMPLATE.format(
        source_text=source_text,
        reference_translation=reference
    )

    max_retries = 5
    for attempt in range(max_retries):
        try:
            rate_limiter.wait_if_needed()

            # 使用OpenAI SDK调用方舟API
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是中医古文翻译评估专家，负责判断文本中是否存在文化语境与认识论差异。"},
                    {"role": "user", "content": prompt}
                ],
                stream=False,
                max_tokens=2000,
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            # 获取响应内容
            content = completion.choices[0].message.content.strip()

            clean_content = content.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean_content)

            # Process contains field
            contains = result.get('contains', False)
            if isinstance(contains, str):
                contains = contains.lower() == 'true'

            # API success log
            contains_str = "包含" if contains else "不包含"
            reason_preview = result.get('reason', '')[:50]
            logging.info(f"[API success] contains={contains_str}, reason={reason_preview}...")

            return {
                'contains': contains,
                'tcm_concepts': result.get('tcm_concepts', []) or [],
                'reason': result.get('reason', ''),
                'success': True
            }
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                return {'contains': False, 'error': f'JSON parse failed: {str(e)}', 'success': False}
        except Exception as e:
            error_str = str(e)
            # 处理速率限制Error
            if 'rate' in error_str.lower() or '429' in error_str or 'limit' in error_str.lower():
                wait_time = min(30 * (attempt + 1), 120)
                logging.warning(f"遇到速率限制，等待{wait_time} seconds before retry (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                return {'contains': False, 'error': f'APIError: {str(e)}', 'success': False}

    return {'contains': False, 'error': '未知Error', 'success': False}


def classify_single(
    source_text: str,
    reference: str,
    rate_limiter: RateLimiter
) -> Dict:
    """Classify one sample (3 runs, majority vote)"""
    results = []

    for _ in range(EVAL_TIMES):
        result = classify_single_once(source_text, reference, rate_limiter)
        if result.get('success'):
            results.append(result)

    if not results:
        return {'contains': False, 'error': '所有评估都失败', 'success': False}

    # Majority vote on contains
    contains_true_count = sum(1 for r in results if r.get('contains') is True)
    contains_false_count = sum(1 for r in results if r.get('contains') is False)

    # Majority decides final contains value
    final_contains = contains_true_count > contains_false_count

    # 合并所有 tcm_concepts（去重）
    all_concepts = []
    for r in results:
        for c in r.get('tcm_concepts', []) or []:
            if c not in all_concepts:
                all_concepts.append(c)

    # Use reason from the first evaluation
    first_reason = results[0].get('reason', '') if results else ''

    return {
        'source': source_text,
        'reference': reference,
        'contains': final_contains,
        'tcm_concepts': all_concepts,
        'reason': first_reason,
        'all_evaluations': results,
        'success': True
    }


def classify_single_with_book(
    source_text: str,
    reference: str,
    rate_limiter: RateLimiter,
    test_set: str
) -> Dict:
    """Classify one sample (3 runs, majority vote)，并add test-set info"""
    result = classify_single(source_text, reference, rate_limiter)
    result['test_set'] = test_set
    return result


def process_sample(args: Tuple) -> Tuple[int, Dict]:
    """Process one sample (for concurrent execution)"""
    idx, source, reference, test_set, rate_limiter, cache, cache_lock, cache_file = args

    cache_key = f"{source[:50]}|{reference[:50]}"
    source_preview = source[:30] + "..." if len(source) > 30 else source

    with cache_lock:
        if cache_key in cache:
            cached = cache[cache_key]
            # 确保缓存结果有 test_set 字段
            if 'test_set' not in cached:
                cached['test_set'] = test_set
            # Cache-hit log
            contains_str = "包含" if cached.get('contains') else "不包含"
            logging.info(f"[Cache hit] 样本#{idx} ({source_preview}) -> {contains_str}文化语境差异")
            return idx, cached

    result = classify_single_with_book(source, reference, rate_limiter, test_set)

    with cache_lock:
        cache[cache_key] = result
        # Save cache to disk in real time
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save cache: {e}")

    # 评估完成日志
    if result.get('success'):
        contains_str = "包含" if result.get('contains') else "不包含"
        logging.info(f"[Evaluation done] 样本#{idx} ({source_preview}) -> {contains_str}文化语境差异")
    else:
        logging.warning(f"[Evaluation failed] 样本#{idx} ({source_preview}) -> {result.get('error', 'unknown')}")

    return idx, result


# ==================== Main classification flow ====================

def classify_dataset(
    test_sets: List[str],
    project_root: str,
    cache_file: str,
    max_workers: int = 20  # 降低并发以避免触发 API 速率限制
) -> Dict:
    """对数据集进行分类"""

    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        logging.info(f"  Loaded {len(cache)} cache entries")

    cache_lock = threading.Lock()
    rate_limiter = RateLimiter(max_per_minute=900)

    all_samples = []

    for test_set in test_sets:
        queries_file = os.path.join(project_root, f"reference_answers/{test_set}_queries.txt")
        references_file = os.path.join(project_root, f"reference_answers/{test_set}_references.txt")

        if not os.path.exists(queries_file) or not os.path.exists(references_file):
            logging.warning(f"  Warning: {test_set} data file missing")
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

        logging.info(f"  Loaded {test_set}: {min_len} samples")

    logging.info(f"  Total {len(all_samples)} samples")

    tasks = [
        (i, s['source'], s['reference'], s['test_set'], rate_limiter, cache, cache_lock, cache_file)
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

            except Exception as e:
                idx = futures[future]
                results[idx] = {'contains': False, 'error': str(e), 'success': False}
                completed_count += 1

    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    # Summary statistics
    valid_results = [r for r in results if r and r.get('success', False)]

    # Compute contains distribution
    contains_true_count = sum(1 for r in valid_results if r.get('contains') is True)
    contains_false_count = sum(1 for r in valid_results if r.get('contains') is False)

    # 统计所有识别出的中医概念
    all_concepts_set = set()
    for r in valid_results:
        for c in r.get('tcm_concepts', []) or []:
            all_concepts_set.add(c)

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
        'unique_concepts': sorted(list(all_concepts_set)),
        'stats_by_testset': stats_by_testset,
        'detailed_results': results
    }


def generate_report(results: Dict) -> str:
    """Generate classification report"""
    report = []
    report.append("=" * 100)
    report.append("文化语境与认识论差异数据集分类报告")
    report.append("使用 DeepSeek-R1 进行判断（每句评估3次，投票决定结果）")
    report.append("=" * 100)
    report.append("")

    report.append("## Overall statistics")
    report.append("")
    report.append(f"Total samples: {results['total_samples']}")
    report.append(f"Evaluated samples: {results['evaluated_samples']}")
    report.append(f"包含文化语境差异: {results['contains_true']}")
    report.append(f"不包含文化语境差异: {results['contains_false']}")
    report.append(f"Positive rate: {results['contains_rate']:.2f}%")
    report.append(f"识别出的中医概念数: {len(results['unique_concepts'])}")
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
    report.append("## 识别出的中医核心概念汇总")
    report.append("")

    concepts = results['unique_concepts']
    if concepts:
        report.append(f"共识别出 {len(concepts)} 个中医核心概念:")
        for concept in concepts:
            report.append(f"  - {concept}")
    else:
        report.append("未识别出中医核心概念")

    return "\n".join(report)


def main():
    # 先Configuration脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ==================== Configuration日志 ====================
    log_filename = os.path.join(script_dir, f"evaluate_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            InstantFileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    # 使用 logging.info 记录启动信息
    logging.info("=" * 100)
    logging.info("文化语境与认识论差异数据集分类")
    logging.info("Each sentence is evaluated 3 times with majority voting")
    logging.info("=" * 100)
    logging.info("")

    project_root = find_project_root(script_dir)

    if not ARK_API_KEY:
        raise ValueError("缺少 API Key，请先设置环境变量 ARK_API_KEY")

    logging.info(f"Project root: {project_root}")
    logging.info(f"API 提供商: VolcEngine 方舟")
    logging.info(f"模型: {MODEL_NAME}")
    logging.info(f"Concurrency: 100 threads")
    logging.info(f"Evaluations per sentence: {EVAL_TIMES}")
    logging.info(f"Log file: {log_filename}")
    logging.info("")

    # ==================== Test mode ====================
    logging.info("=" * 100)
    logging.info("Test mode：先评估两条样本验证正确性")
    logging.info("=" * 100)
    logging.info("")

    test_samples = [
        {
            "source": "心者，君主之官，神明出焉。",
            "reference": "心是人体最重要的脏器，主持人体精神意识思维活动，就像国家的君主一样。",
            "expected_contains": True,
            "test_set": "huangdineijing"
        },
        {
            "source": "病人发热，体温升高。",
            "reference": "病人发热，体温升高。",
            "expected_contains": False,
            "test_set": "shanghanlun"
        }
    ]

    rate_limiter = RateLimiter(max_per_minute=900)
    test_passed = 0
    test_failed = 0

    for i, sample in enumerate(test_samples, 1):
        logging.info(f"测试样本 {i}:")
        logging.info(f"  test_set: {sample['test_set']}")
        logging.info(f"  原文: {sample['source']}")
        logging.info(f"  参考: {sample['reference']}")
        logging.info(f"  预期: {'包含' if sample['expected_contains'] else '不包含'}")

        result = classify_single_with_book(sample['source'], sample['reference'], rate_limiter, sample['test_set'])

        # 打印完整结果用于检查
        logging.info(f"  完整结果: {json.dumps({k: v for k, v in result.items() if k != 'all_evaluations'}, ensure_ascii=False)}")

        if result.get('success'):
            actual_contains = result.get('contains', False)
            has_test_set = 'test_set' in result
            logging.info(f"  实际: {'包含' if actual_contains else '不包含'}")
            logging.info(f"  识别概念: {result.get('tcm_concepts', [])}")
            logging.info(f"  test_set字段: {'存在 (' + result.get('test_set', '') + ')' if has_test_set else '缺失'}")

            if actual_contains == sample['expected_contains'] and has_test_set and result.get('test_set') == sample['test_set']:
                logging.info(f"  结果: ✓ 通过")
                test_passed += 1
            else:
                logging.info(f"  结果: ✗ 失败")
                test_failed += 1
        else:
            logging.warning(f"  结果: ✗ Error - {result.get('error', '未知Error')}")
            test_failed += 1
        logging.info("")

    logging.info("=" * 100)
    logging.info(f"测试结果: {test_passed} 通过, {test_failed} 失败")
    logging.info("=" * 100)
    logging.info("")

    if test_failed > 0:
        logging.warning("Warning: 测试样本有失败，请检查 prompt 和 API 是否正常")

    # ==================== 解析命令行参数 ====================
    test_only = False
    specified_book = None

    for arg in sys.argv[1:]:
        if arg == "--test-only":
            test_only = True
        elif arg.startswith("--book="):
            specified_book = arg.split("=", 1)[1]
        elif arg == "--book" and len(sys.argv) > sys.argv.index(arg) + 1:
            specified_book = sys.argv[sys.argv.index(arg) + 1]

    if test_only:
        logging.info("--test-only 模式：只运行测试，不进行全量评估")
        return

    # ==================== 全量分类 ====================
    if specified_book:
        if specified_book not in ALL_BOOKS:
            logging.error(f"Error: 书籍 '{specified_book}' 不在列表中")
            logging.info(f"可用书籍: {', '.join(ALL_BOOKS)}")
            return
        test_sets = [specified_book]
        logging.info(f"只评估书籍: {specified_book}")
    else:
        test_sets = ALL_BOOKS

    cache_file = os.path.join(script_dir, "cache_cultural.json")

    results = classify_dataset(
        test_sets,
        project_root,
        cache_file,
        max_workers=20  # 降低并发以避免触发 API 速率限制
    )

    logging.info("")
    logging.info("=" * 100)
    logging.info("Classification completed!")
    logging.info(f"  Total samples: {results['total_samples']}")
    logging.info(f"  Evaluated samples: {results['evaluated_samples']}")
    logging.info(f"  包含文化语境差异: {results['contains_true']}")
    logging.info(f"  不包含文化语境差异: {results['contains_false']}")
    logging.info(f"  Positive rate: {results['contains_rate']:.2f}%")
    logging.info(f"  识别概念数: {len(results['unique_concepts'])}")
    logging.info("=" * 100)

    report = generate_report(results)
    print("")
    print(report)

    report_file = os.path.join(script_dir, "cultural_context_evaluation_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved to: {report_file}")

    # Save full results
    summary_results = {
        key: val for key, val in results.items() if key != 'detailed_results'
    }
    results_file = os.path.join(script_dir, "cultural_context_evaluation_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {results_file}")

    # 保存包含文化语境差异的样本列表
    contains_true_samples = []
    for i, r in enumerate(results.get('detailed_results', [])):
        if r and r.get('success') and r.get('contains'):
            contains_true_samples.append({
                'index': i,
                'source': r.get('source'),
                'reference': r.get('reference'),
                'tcm_concepts': r.get('tcm_concepts', []),
                'reason': r.get('reason', '')
            })

    contains_file = os.path.join(script_dir, "contains_true_samples.json")
    with open(contains_file, 'w', encoding='utf-8') as f:
        json.dump(contains_true_samples, f, ensure_ascii=False, indent=2)
    logging.info(f"包含文化语境差异的样本已保存到: {contains_file}")

    # Save per-book results
    logging.info("")
    logging.info("=" * 100)
    logging.info("Saving results by book...")
    logging.info("=" * 100)

    detailed_results = results.get('detailed_results', [])
    all_samples = []
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
            all_samples.append({
                'test_set': test_set,
                'index': i,
                'source': src,
                'reference': ref
            })

    # Create a separate result file for each book
    for test_set in test_sets:
        test_set_samples = [s for s in all_samples if s['test_set'] == test_set]

        if not test_set_samples:
            continue

        # Find detailed results for this book
        test_set_results = []
        contains_true_count = 0

        for sample in test_set_samples:
            # Find matching result in detailed_results
            # 通过 source 和 reference 匹配
            result = None
            for r in detailed_results:
                if r and r.get('source') == sample['source'] and r.get('reference') == sample['reference']:
                    result = r
                    break

            if result:
                item = {
                    'index': sample['index'],
                    'test_set': test_set,
                    'source': result.get('source'),
                    'reference': result.get('reference'),
                    'contains': result.get('contains'),
                    'tcm_concepts': result.get('tcm_concepts', []),
                    'reason': result.get('reason', ''),
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

        logging.info(f"  {test_set}: {len(test_set_samples)} 样本, {contains_true_count} 包含文化语境差异 -> {test_set_file}")

    logging.info("=" * 100)


if __name__ == "__main__":
    main()
