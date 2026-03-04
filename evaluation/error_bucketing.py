"""
错误分桶模块

将模型回答中的错误分类到预定义的错误类型桶中：
幻觉、遗漏、过度泛化、事实错误、格式问题。
"""

import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ErrorType(str, Enum):
    """错误类型枚举。"""

    HALLUCINATION = "hallucination"         # 幻觉：生成不存在的信息
    OMISSION = "omission"                   # 遗漏：忽略重要信息
    OVER_GENERALIZATION = "over_generalization"  # 过度泛化：用通用描述回避具体问题
    FACTUAL_ERROR = "factual_error"         # 事实错误：价格/日期/参数错误
    FORMAT_ISSUE = "format_issue"           # 格式问题：结构混乱、过长/过短
    NO_ERROR = "no_error"                   # 无错误


# 各错误类型的检测规则
ERROR_RULES = {
    ErrorType.HALLUCINATION: {
        "signals": [
            r"据我所知[，,]?这款",
            r"根据我的了解[，,]?",
            r"我了解到[，,]?",
            r"这个品牌(?:一向|一直|通常)很",
            r"(?:应该|大概|估计)(?:在|是|有)\s*[¥￥\d]",
        ],
        "description": "生成了无法验证或不存在的信息",
    },
    ErrorType.OMISSION: {
        "signals": [
            r"具体(?:价格|参数|库存)(?:请|可以)(?:咨询|查看|联系)",
            r"详细信息.*官网",
            r"(?:价格|库存|促销).*(?:以.*为准|请以实际为准)",
        ],
        "description": "回避了用户关心的核心信息",
        "min_length_check": True,  # 回答过短也可能是遗漏
        "min_length": 50,
    },
    ErrorType.OVER_GENERALIZATION: {
        "signals": [
            r"(?:总体来说|一般来说|通常情况下)[，,]?这类商品",
            r"(?:大多数|很多)[用消费]?者(?:都|会|认为)",
            r"根据市场(?:情况|行情)[，,]?",
            r"这个价位的商品(?:普遍|一般)",
        ],
        "description": "用通用描述回避了针对性回答",
    },
    ErrorType.FACTUAL_ERROR: {
        "signals": [
            r"(?:原价|售价)\s*[¥￥]\s*0",  # 价格为0
            r"2019年.*(?:上市|发布).*(?:2024|2025)年.*(?:新品|旗舰)",  # 时间逻辑错误
        ],
        "description": "包含可验证的错误事实",
    },
    ErrorType.FORMAT_ISSUE: {
        "signals": [
            r"^.{0,10}$",  # 极短回答（可能格式问题）
        ],
        "description": "格式/结构问题",
        "length_check": True,
        "max_length": 5000,
        "min_length": 20,
    },
}


class ErrorBucketing:
    """
    错误分桶分类器。

    将模型回答中的错误自动分类到预定义的错误类型桶，
    支持自动分类 + 人工校验流程。

    Args:
        response_field: 回答字段名，默认 "response"
        reference_field: 参考答案字段名，默认 "reference"
        question_field: 问题字段名，默认 "question"
    """

    def __init__(
        self,
        response_field: str = "response",
        reference_field: str = "reference",
        question_field: str = "question",
    ) -> None:
        self.response_field = response_field
        self.reference_field = reference_field
        self.question_field = question_field

        # 预编译正则
        self._compiled_rules: Dict[ErrorType, List[re.Pattern]] = {}
        for error_type, rule_cfg in ERROR_RULES.items():
            self._compiled_rules[error_type] = [
                re.compile(p) for p in rule_cfg.get("signals", [])
            ]
        logger.info("初始化 ErrorBucketing")

    def _classify_single(
        self, question: str, response: str, reference: str = ""
    ) -> List[Tuple[ErrorType, str]]:
        """
        对单条回答进行错误分类。

        Returns:
            List[Tuple[ErrorType, str]]: 错误类型及触发原因的列表
        """
        errors: List[Tuple[ErrorType, str]] = []

        for error_type, patterns in self._compiled_rules.items():
            rule_cfg = ERROR_RULES.get(error_type, {})

            # 长度检查（用于 OMISSION 和 FORMAT_ISSUE）
            if rule_cfg.get("min_length_check") and len(response.strip()) < rule_cfg.get("min_length", 50):
                errors.append((error_type, f"回答过短（{len(response.strip())}字）"))
                continue

            if rule_cfg.get("length_check"):
                resp_len = len(response.strip())
                if resp_len < rule_cfg.get("min_length", 20):
                    errors.append((error_type, f"回答过短（{resp_len}字）"))
                    continue
                if resp_len > rule_cfg.get("max_length", 5000):
                    errors.append((error_type, f"回答过长（{resp_len}字）"))
                    continue

            # 正则匹配
            for pattern in patterns:
                match = pattern.search(response)
                if match:
                    errors.append((error_type, f"匹配规则: {match.group()[:50]}"))
                    break

        return errors

    def bucket_single(
        self, question: str, response: str, reference: str = ""
    ) -> Dict[str, Any]:
        """
        对单条回答进行错误分桶。

        Args:
            question: 用户问题
            response: 模型回答
            reference: 参考答案（可选）

        Returns:
            Dict: 包含错误类型和详情的分桶结果
        """
        errors = self._classify_single(question, response, reference)
        error_types = [e[0] for e in errors]
        error_reasons = {str(err_type): reason for err_type, reason in errors}

        primary_error = error_types[0] if error_types else ErrorType.NO_ERROR

        return {
            "error_types": [str(e) for e in error_types],
            "primary_error": str(primary_error),
            "error_details": error_reasons,
            "has_error": len(error_types) > 0,
            "needs_human_review": any(e in [ErrorType.HALLUCINATION, ErrorType.FACTUAL_ERROR] for e in error_types),
        }

    def bucket(self, dataset: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        对数据集进行批量错误分桶。

        Args:
            dataset: 包含问题/回答/参考答案字段的字典列表

        Returns:
            Tuple[List[Dict], Dict]: (分桶后数据集, 统计信息)
        """
        logger.info("开始错误分桶，数据量: %d", len(dataset))

        results = []
        error_counts: Dict[str, int] = {str(e): 0 for e in ErrorType}
        human_review_needed = 0

        for item in dataset:
            bucket_result = self.bucket_single(
                question=item.get(self.question_field, item.get("prompt", "")),
                response=item.get(self.response_field, ""),
                reference=item.get(self.reference_field, ""),
            )
            result = {**item, **bucket_result}
            results.append(result)

            for error_type in bucket_result["error_types"]:
                error_counts[error_type] = error_counts.get(error_type, 0) + 1

            if bucket_result["needs_human_review"]:
                human_review_needed += 1

        # 收集各类型错误示例
        examples: Dict[str, List[Dict]] = {str(e): [] for e in ErrorType}
        for result in results:
            primary = result["primary_error"]
            if len(examples.get(primary, [])) < 3:
                examples[primary].append({
                    "question": result.get(self.question_field, ""),
                    "response": result.get(self.response_field, "")[:200],
                    "reason": result["error_details"],
                })

        stats = {
            "total": len(dataset),
            "error_counts": error_counts,
            "error_rate": sum(1 for r in results if r["has_error"]) / max(len(dataset), 1),
            "human_review_needed": human_review_needed,
            "examples": examples,
        }
        logger.info(
            "错误分桶完成: 总计=%d, 有错误=%d (%.1f%%), 需人工审核=%d",
            stats["total"],
            int(stats["error_rate"] * len(dataset)),
            stats["error_rate"] * 100,
            human_review_needed,
        )
        return results, stats
