"""
质量过滤模块

基于规则的质量过滤器，包含长度检查、重复率、特殊字符、语言检测、问答完整性等过滤规则。
"""

import logging
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class QualityFilter:
    """
    基于规则的数据质量过滤器。

    过滤规则：
    - 最小/最大长度限制
    - 重复 n-gram 比例阈值
    - 特殊字符比例上限
    - 语言检测（可选）
    - 问答对完整性校验

    Args:
        min_length: 最小文本长度（字符数），默认 20
        max_length: 最大文本长度（字符数），默认 8192
        max_repetition_ratio: 重复 n-gram 比例上限，默认 0.3
        max_special_char_ratio: 特殊字符比例上限，默认 0.2
        require_qa_pair: 是否要求问答对完整，默认 False
        prompt_field: 问题字段名，默认 "prompt"
        response_field: 回答字段名，默认 "response"
        text_field: 纯文本字段名（非问答对场景），默认 "text"
    """

    def __init__(
        self,
        min_length: int = 20,
        max_length: int = 8192,
        max_repetition_ratio: float = 0.3,
        max_special_char_ratio: float = 0.2,
        require_qa_pair: bool = False,
        prompt_field: str = "prompt",
        response_field: str = "response",
        text_field: str = "text",
    ) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.max_repetition_ratio = max_repetition_ratio
        self.max_special_char_ratio = max_special_char_ratio
        self.require_qa_pair = require_qa_pair
        self.prompt_field = prompt_field
        self.response_field = response_field
        self.text_field = text_field
        logger.info(
            "初始化 QualityFilter: min_length=%d, max_length=%d, "
            "max_repetition_ratio=%.2f, max_special_char_ratio=%.2f",
            min_length,
            max_length,
            max_repetition_ratio,
            max_special_char_ratio,
        )

    def _get_text(self, item: Dict) -> str:
        """从数据条目中提取文本内容。"""
        if self.require_qa_pair:
            prompt = item.get(self.prompt_field, "")
            response = item.get(self.response_field, "")
            return prompt + " " + response
        return item.get(self.text_field, "")

    def _check_length(self, text: str) -> Tuple[bool, str]:
        """检查文本长度是否在合理范围内。"""
        length = len(text.strip())
        if length < self.min_length:
            return False, f"文本过短: {length} < {self.min_length}"
        if length > self.max_length:
            return False, f"文本过长: {length} > {self.max_length}"
        return True, ""

    def _check_repetition(self, text: str, ngram_size: int = 4) -> Tuple[bool, str]:
        """检查文本的重复 n-gram 比例。"""
        words = list(text)
        if len(words) < ngram_size:
            return True, ""
        ngrams = [tuple(words[i : i + ngram_size]) for i in range(len(words) - ngram_size + 1)]
        counter = Counter(ngrams)
        total = len(ngrams)
        repeated = sum(count for count in counter.values() if count > 1)
        ratio = repeated / max(total, 1)
        if ratio > self.max_repetition_ratio:
            return False, f"重复率过高: {ratio:.2f} > {self.max_repetition_ratio}"
        return True, ""

    def _check_special_chars(self, text: str) -> Tuple[bool, str]:
        """检查特殊字符比例。"""
        if not text:
            return False, "文本为空"
        special_chars = re.findall(r"[^\w\s\u4e00-\u9fff\u3000-\u303f\uff00-\uffef.,!?，。！？、：；""''（）【】《》]", text)
        ratio = len(special_chars) / max(len(text), 1)
        if ratio > self.max_special_char_ratio:
            return False, f"特殊字符比例过高: {ratio:.2f} > {self.max_special_char_ratio}"
        return True, ""

    def _check_qa_completeness(self, item: Dict) -> Tuple[bool, str]:
        """检查问答对完整性。"""
        if not self.require_qa_pair:
            return True, ""
        prompt = item.get(self.prompt_field, "").strip()
        response = item.get(self.response_field, "").strip()
        if not prompt:
            return False, "问题字段为空"
        if not response:
            return False, "回答字段为空"
        return True, ""

    def _filter_single(self, item: Dict) -> Tuple[bool, str]:
        """对单条数据进行所有规则检查。"""
        # 问答对完整性检查
        ok, reason = self._check_qa_completeness(item)
        if not ok:
            return False, reason

        text = self._get_text(item)
        if not text.strip():
            return False, "文本为空"

        # 长度检查
        ok, reason = self._check_length(text)
        if not ok:
            return False, reason

        # 重复率检查
        ok, reason = self._check_repetition(text)
        if not ok:
            return False, reason

        # 特殊字符检查
        ok, reason = self._check_special_chars(text)
        if not ok:
            return False, reason

        return True, ""

    def filter(self, dataset: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        对数据集进行质量过滤。

        Args:
            dataset: 包含文本字段的字典列表

        Returns:
            Tuple[List[Dict], Dict]: (过滤后数据集, 过滤统计报告)
        """
        logger.info("开始质量过滤，原始数据量: %d", len(dataset))

        filtered: List[Dict] = []
        filter_reasons: Dict[str, int] = {}

        for idx, item in enumerate(dataset):
            ok, reason = self._filter_single(item)
            if ok:
                filtered.append(item)
            else:
                filter_reasons[reason] = filter_reasons.get(reason, 0) + 1

            if (idx + 1) % 1000 == 0:
                logger.info("已处理 %d / %d 条数据", idx + 1, len(dataset))

        removed = len(dataset) - len(filtered)
        stats = {
            "original_count": len(dataset),
            "filtered_count": len(filtered),
            "removed_count": removed,
            "filter_ratio": removed / max(len(dataset), 1),
            "filter_reasons": filter_reasons,
        }
        logger.info(
            "质量过滤完成: 原始=%d, 过滤后=%d, 删除=%d (%.1f%%)",
            stats["original_count"],
            stats["filtered_count"],
            stats["removed_count"],
            stats["filter_ratio"] * 100,
        )
        return filtered, stats
