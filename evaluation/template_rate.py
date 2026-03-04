"""
模板化率评测模块

检测模型回答的模板化程度，
方法：n-gram 重复率、句式多样性（distinct-n）、语义相似度聚类。
"""

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TemplateRateEvaluator:
    """
    模板化率评测器。

    检测一批回答中的模板化程度，综合使用以下方法：
    1. n-gram 重复率：跨回答的 n-gram 重复比例
    2. distinct-n：句子内部的词汇多样性指标
    3. 语义相似度聚类：检测语义层面的模板化

    Args:
        ngram_sizes: 用于计算的 n-gram 大小列表，默认 [2, 3, 4]
        cluster_threshold: 语义相似度聚类阈值，默认 0.85
        response_field: 回答字段名，默认 "response"
    """

    def __init__(
        self,
        ngram_sizes: Optional[List[int]] = None,
        cluster_threshold: float = 0.85,
        response_field: str = "response",
    ) -> None:
        self.ngram_sizes = ngram_sizes or [2, 3, 4]
        self.cluster_threshold = cluster_threshold
        self.response_field = response_field
        logger.info(
            "初始化 TemplateRateEvaluator: ngram_sizes=%s, cluster_threshold=%.2f",
            self.ngram_sizes,
            cluster_threshold,
        )

    def _tokenize(self, text: str) -> List[str]:
        """对中文文本进行分词，优先使用 jieba，回退到字符级分割。"""
        try:
            import jieba
            return list(jieba.cut(text.replace(" ", "").replace("\n", "")))
        except ImportError:
            # 回退：字符级分割（统计结果略有偏差，但功能正常）
            return list(text.replace(" ", "").replace("\n", ""))

    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple]:
        """提取 n-gram 列表。"""
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def _cross_response_repetition(self, responses: List[str], n: int = 3) -> float:
        """
        计算跨回答的 n-gram 重复率。

        统计所有回答中出现频率 > 1 的 n-gram 比例。

        Returns:
            float: 跨回答 n-gram 重复率（0-1）
        """
        if len(responses) < 2:
            return 0.0

        all_ngrams: List[Tuple] = []
        for response in responses:
            tokens = self._tokenize(response)
            all_ngrams.extend(self._get_ngrams(tokens, n))

        if not all_ngrams:
            return 0.0

        counter = Counter(all_ngrams)
        repeated = sum(count for count in counter.values() if count > 1)
        return repeated / len(all_ngrams)

    def _distinct_n(self, responses: List[str], n: int = 2) -> float:
        """
        计算 distinct-n 指标（词汇多样性）。

        distinct-n = 唯一 n-gram 数量 / 总 n-gram 数量

        Returns:
            float: distinct-n 值（越高越多样）
        """
        all_ngrams: List[Tuple] = []
        for response in responses:
            tokens = self._tokenize(response)
            all_ngrams.extend(self._get_ngrams(tokens, n))

        if not all_ngrams:
            return 1.0

        unique_ngrams = len(set(all_ngrams))
        return unique_ngrams / len(all_ngrams)

    def _sentence_pattern_rate(self, responses: List[str]) -> float:
        """
        检测句式模板化率。

        通过检测常见模板化句式（如"综上所述"、"总结一下"等）的出现频率。
        """
        template_patterns = [
            r"综上所述[，,]?",
            r"总体来说[，,]?",
            r"总结一下[，,]?",
            r"总的来说[，,]?",
            r"希望[以上]?.*帮助[到]?您",
            r"如有.*问题.*随时",
            r"祝您.*购物愉快",
            r"以下.*为您.*推荐",
            r"根据您的.*需求[，,]?",
            r"作为.*电商导购[，,]?",
        ]

        template_count = 0
        for response in responses:
            for pattern in template_patterns:
                if re.search(pattern, response):
                    template_count += 1
                    break  # 每条回答只计一次

        return template_count / max(len(responses), 1)

    def _semantic_cluster_rate(self, responses: List[str]) -> float:
        """
        基于 TF-IDF 余弦相似度的语义聚类模板化率。

        Returns:
            float: 语义层面的模板化率（0-1）
        """
        if len(responses) < 2:
            return 0.0

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            # 处理中文：字符级分词
            processed = [" ".join(list(r)) for r in responses]
            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), max_features=5000)
            tfidf_matrix = vectorizer.fit_transform(processed)
            sim_matrix = cosine_similarity(tfidf_matrix)

            # 计算高相似度对的比例
            n = len(responses)
            high_sim_pairs = 0
            total_pairs = n * (n - 1) // 2
            for i in range(n):
                for j in range(i + 1, n):
                    if sim_matrix[i, j] >= self.cluster_threshold:
                        high_sim_pairs += 1

            return high_sim_pairs / max(total_pairs, 1)
        except ImportError:
            logger.warning("未安装 scikit-learn，跳过语义聚类评测")
            return 0.0

    def compute_template_rate(self, responses: List[str]) -> Dict[str, float]:
        """
        计算一批回答的模板化率。

        Args:
            responses: 回答文本列表

        Returns:
            Dict: 包含各项模板化指标的字典
        """
        metrics: Dict[str, float] = {}

        # 跨回答 n-gram 重复率
        for n in self.ngram_sizes:
            metrics[f"cross_ngram_rep_{n}"] = self._cross_response_repetition(responses, n)

        # distinct-n
        for n in self.ngram_sizes[:2]:  # 只计算 distinct-2 和 distinct-3
            metrics[f"distinct_{n}"] = self._distinct_n(responses, n)

        # 句式模板化率
        metrics["sentence_pattern_rate"] = self._sentence_pattern_rate(responses)

        # 语义聚类模板化率
        metrics["semantic_cluster_rate"] = self._semantic_cluster_rate(responses)

        # 综合模板化率（加权平均）
        template_signals = [
            metrics.get("cross_ngram_rep_3", 0.0),
            metrics.get("sentence_pattern_rate", 0.0),
            metrics.get("semantic_cluster_rate", 0.0),
            1.0 - metrics.get("distinct_2", 1.0),  # distinct 越低越模板化
        ]
        metrics["overall_template_rate"] = float(np.mean(template_signals))

        return metrics

    def evaluate(self, dataset: List[Dict]) -> Dict[str, Any]:
        """
        对数据集进行批量模板化率评测。

        Args:
            dataset: 包含回答字段的字典列表

        Returns:
            Dict: 评测结果汇总
        """
        logger.info("开始模板化率评测，数据量: %d", len(dataset))

        responses = [item.get(self.response_field, "") for item in dataset]
        responses = [r for r in responses if r]

        metrics = self.compute_template_rate(responses)

        logger.info(
            "模板化率评测完成: 综合模板化率=%.1f%%, distinct-2=%.3f",
            metrics.get("overall_template_rate", 0.0) * 100,
            metrics.get("distinct_2", 0.0),
        )
        return {
            "metrics": metrics,
            "total_responses": len(responses),
            "template_rate_percentage": metrics.get("overall_template_rate", 0.0) * 100,
        }
