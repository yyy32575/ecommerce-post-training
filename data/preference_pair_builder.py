"""
偏好对数据构建模块

构建 DPO 所需的偏好对数据（chosen/rejected），
支持基于评分差异、规则对比（有无幻觉）、人工标注等多种策略。
"""

import logging
import random
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BuildStrategy(str, Enum):
    """偏好对构建策略枚举。"""

    SCORE_DIFF = "score_diff"          # 基于评分差异
    RULE_BASED = "rule_based"          # 基于规则（幻觉检测）
    HUMAN_ANNOTATION = "human"         # 基于人工标注
    MIXED = "mixed"                    # 混合策略


# 用于规则检测的幻觉/低质量信号
HALLUCINATION_SIGNALS = [
    r"据我所知[，,]?这款产品",
    r"我觉得应该|我猜测|可能是",
    r"具体价格[不]?[确详]定",
    r"库存[情况]?不[太清楚确定]",
    r"这个品牌[一般好坏]",
]

QUALITY_SIGNALS_NEGATIVE = [
    "不知道",
    "不清楚",
    "不确定",
    "无法确认",
    "可能",
    "也许",
    "大概",
]


class PreferencePairBuilder:
    """
    DPO 偏好对数据构建器。

    支持多种构建策略，输出格式：
    {"prompt": ..., "chosen": ..., "rejected": ...}

    Args:
        strategy: 构建策略，默认混合策略
        score_field: 评分字段名，默认 "score"
        response_field: 回答字段名，默认 "response"
        prompt_field: 问题字段名，默认 "prompt"
        min_score_diff: 最小评分差异（用于 score_diff 策略），默认 1.0
        responses_field: 多候选回答字段名，默认 "responses"
    """

    def __init__(
        self,
        strategy: BuildStrategy = BuildStrategy.MIXED,
        score_field: str = "score",
        response_field: str = "response",
        prompt_field: str = "prompt",
        min_score_diff: float = 1.0,
        responses_field: str = "responses",
        seed: Optional[int] = 42,
    ) -> None:
        self.strategy = strategy
        self.score_field = score_field
        self.response_field = response_field
        self.prompt_field = prompt_field
        self.min_score_diff = min_score_diff
        self.responses_field = responses_field
        self._rng = random.Random(seed)
        logger.info("初始化 PreferencePairBuilder: strategy=%s, seed=%s", strategy, seed)

    def _has_hallucination(self, text: str) -> bool:
        """检测文本是否含有幻觉/低质量信号。"""
        import re
        for signal in HALLUCINATION_SIGNALS:
            if re.search(signal, text):
                return True
        for signal in QUALITY_SIGNALS_NEGATIVE:
            if signal in text:
                return True
        return False

    def _build_from_scores(self, item: Dict) -> Optional[Dict]:
        """
        基于评分差异构建偏好对。

        期望数据格式：
        {
          "prompt": "...",
          "responses": [
            {"text": "...", "score": 4.5},
            {"text": "...", "score": 2.0},
          ]
        }
        """
        prompt = item.get(self.prompt_field, "")
        responses = item.get(self.responses_field, [])
        if len(responses) < 2:
            return None

        # 按评分排序
        scored = [(r.get("text", ""), r.get(self.score_field, 0)) for r in responses]
        scored = [(t, s) for t, s in scored if t]
        if len(scored) < 2:
            return None
        scored.sort(key=lambda x: x[1], reverse=True)

        best_text, best_score = scored[0]
        worst_text, worst_score = scored[-1]

        if best_score - worst_score < self.min_score_diff:
            return None

        return {
            "prompt": prompt,
            "chosen": best_text,
            "rejected": worst_text,
            "build_strategy": BuildStrategy.SCORE_DIFF,
            "score_diff": best_score - worst_score,
        }

    def _build_from_rules(self, item: Dict) -> Optional[Dict]:
        """
        基于规则（幻觉检测）构建偏好对。

        期望数据格式：
        {
          "prompt": "...",
          "responses": [{"text": "..."}, ...]
        }
        """
        prompt = item.get(self.prompt_field, "")
        responses = item.get(self.responses_field, [])
        if len(responses) < 2:
            return None

        texts = [r.get("text", "") for r in responses if r.get("text", "")]
        good_responses = [t for t in texts if not self._has_hallucination(t)]
        bad_responses = [t for t in texts if self._has_hallucination(t)]

        if not good_responses or not bad_responses:
            return None

        return {
            "prompt": prompt,
            "chosen": self._rng.choice(good_responses),
            "rejected": self._rng.choice(bad_responses),
            "build_strategy": BuildStrategy.RULE_BASED,
        }

    def _build_from_human(self, item: Dict) -> Optional[Dict]:
        """
        基于人工标注构建偏好对。

        期望数据格式：
        {
          "prompt": "...",
          "chosen": "...",
          "rejected": "..."
        }
        """
        prompt = item.get(self.prompt_field, "")
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")

        if not prompt or not chosen or not rejected:
            return None

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "build_strategy": BuildStrategy.HUMAN_ANNOTATION,
        }

    def _build_single(self, item: Dict) -> Optional[Dict]:
        """根据策略对单条数据构建偏好对。"""
        if self.strategy == BuildStrategy.SCORE_DIFF:
            return self._build_from_scores(item)
        elif self.strategy == BuildStrategy.RULE_BASED:
            return self._build_from_rules(item)
        elif self.strategy == BuildStrategy.HUMAN_ANNOTATION:
            return self._build_from_human(item)
        elif self.strategy == BuildStrategy.MIXED:
            # 混合策略：依次尝试各种方法
            for builder in [
                self._build_from_human,
                self._build_from_scores,
                self._build_from_rules,
            ]:
                result = builder(item)
                if result:
                    return result
            return None
        return None

    def build_pairs(self, dataset: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        对数据集批量构建偏好对。

        Args:
            dataset: 输入数据字典列表

        Returns:
            Tuple[List[Dict], Dict]: (偏好对数据集, 构建统计信息)
        """
        logger.info("开始构建偏好对，输入数据量: %d", len(dataset))

        pairs: List[Dict] = []
        strategy_counts: Dict[str, int] = {}
        skip_count = 0

        for idx, item in enumerate(dataset):
            pair = self._build_single(item)
            if pair:
                pairs.append(pair)
                strat = str(pair.get("build_strategy", "unknown"))
                strategy_counts[strat] = strategy_counts.get(strat, 0) + 1
            else:
                skip_count += 1

            if (idx + 1) % 500 == 0:
                logger.info("已处理 %d / %d 条数据", idx + 1, len(dataset))

        stats = {
            "input_count": len(dataset),
            "pairs_count": len(pairs),
            "skip_count": skip_count,
            "strategy_distribution": strategy_counts,
        }
        logger.info(
            "偏好对构建完成: 输入=%d, 偏好对=%d, 跳过=%d",
            stats["input_count"],
            stats["pairs_count"],
            stats["skip_count"],
        )
        return pairs, stats
