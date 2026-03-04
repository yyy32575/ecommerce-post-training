"""
任务完成度评测模块

评估模型回答是否完成了用户的导购任务，
维度：需求理解、推荐相关性、信息完整性。
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

TASK_COMPLETION_PROMPT = """你是一位专业的电商导购服务质量评审员。请评估以下回答是否完成了用户的导购任务。

用户问题：{question}

模型回答：{response}

请从以下三个维度评估（每项1-5分）：
1. 需求理解（1-5分）：是否正确理解了用户的购买需求和偏好
2. 推荐相关性（1-5分）：推荐的商品/建议是否与需求匹配
3. 信息完整性（1-5分）：是否提供了足够的决策信息（价格/功能/优缺点等）

请按如下格式给出评分：
需求理解：[分数]
推荐相关性：[分数]
信息完整性：[分数]
综合评分：[分数]
评估理由：[简述]"""


class TaskCompletionEvaluator:
    """
    任务完成度评测器。

    评估模型回答是否完成用户的电商导购任务。
    维度：需求理解、推荐相关性、信息完整性。

    Args:
        judge_model: 用于评判的 LLM 模型，若为 None 则使用规则方法
        judge_tokenizer: 对应的分词器
        max_new_tokens: 评判生成最大 token 数，默认 256
    """

    def __init__(
        self,
        judge_model=None,
        judge_tokenizer=None,
        max_new_tokens: int = 256,
    ) -> None:
        self.judge_model = judge_model
        self.judge_tokenizer = judge_tokenizer
        self.max_new_tokens = max_new_tokens
        logger.info("初始化 TaskCompletionEvaluator")

    def _llm_judge(self, question: str, response: str) -> Dict[str, float]:
        """使用 LLM 评判任务完成度。"""
        if self.judge_model is None:
            return self._rule_based_score(question, response)

        prompt = TASK_COMPLETION_PROMPT.format(question=question, response=response)
        inputs = self.judge_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        inputs = {k: v.to(self.judge_model.device) for k, v in inputs.items()}

        import torch
        with torch.no_grad():
            outputs = self.judge_model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.judge_tokenizer.eos_token_id,
            )
        generated = self.judge_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return self._parse_scores(generated)

    def _parse_scores(self, text: str) -> Dict[str, float]:
        """从 LLM 输出中解析评分。"""
        scores = {}
        patterns = {
            "need_understanding": r"需求理解[：:]\s*([1-5])",
            "recommendation_relevance": r"推荐相关性[：:]\s*([1-5])",
            "information_completeness": r"信息完整性[：:]\s*([1-5])",
            "overall": r"综合评分[：:]\s*([1-5])",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            scores[key] = float(match.group(1)) if match else 3.0

        if "overall" not in scores or scores["overall"] == 3.0:
            dims = ["need_understanding", "recommendation_relevance", "information_completeness"]
            scores["overall"] = sum(scores.get(d, 3.0) for d in dims) / 3

        return scores

    def _rule_based_score(self, question: str, response: str) -> Dict[str, float]:
        """基于规则的任务完成度评分（无 LLM 时的 fallback）。"""
        scores: Dict[str, float] = {
            "need_understanding": 3.0,
            "recommendation_relevance": 3.0,
            "information_completeness": 3.0,
        }

        response_lower = response.lower()

        # 需求理解：是否包含问题中的关键词
        question_keywords = set(re.findall(r"[\u4e00-\u9fff]{2,}", question))
        response_keywords = set(re.findall(r"[\u4e00-\u9fff]{2,}", response))
        if question_keywords:
            overlap = len(question_keywords & response_keywords) / len(question_keywords)
            scores["need_understanding"] = 1.0 + 4.0 * overlap

        # 推荐相关性：是否包含推荐性词汇
        recommend_words = ["推荐", "建议", "适合", "可以考虑", "比较好", "不错"]
        if any(w in response for w in recommend_words):
            scores["recommendation_relevance"] += 1.0

        # 信息完整性：是否包含价格、参数等具体信息
        info_signals = [
            r"[¥￥]\d+|\d+元",  # 价格
            r"\d+[mM][bB]|\d+[gG][bB]",  # 存储参数
            r"\d+[wW]|\d+毫[安时]",  # 功率/电量
            r"优点|缺点|优势|劣势",  # 分析
        ]
        info_count = sum(1 for p in info_signals if re.search(p, response))
        scores["information_completeness"] = min(5.0, 2.0 + info_count)

        scores["overall"] = sum(scores.values()) / 3
        return scores

    def evaluate_single(self, question: str, response: str) -> Dict[str, Any]:
        """
        评估单条回答的任务完成度。

        Args:
            question: 用户问题
            response: 模型回答

        Returns:
            Dict: 包含各维度评分的评估结果
        """
        scores = self._llm_judge(question, response)
        return {
            "need_understanding_score": scores.get("need_understanding", 3.0),
            "recommendation_relevance_score": scores.get("recommendation_relevance", 3.0),
            "information_completeness_score": scores.get("information_completeness", 3.0),
            "task_completion_score": scores.get("overall", 3.0),
        }

    def evaluate(self, dataset: List[Dict]) -> Dict[str, Any]:
        """
        对数据集进行批量任务完成度评测。

        Args:
            dataset: 包含 question/response 字段的字典列表

        Returns:
            Dict: 评测结果汇总
        """
        logger.info("开始任务完成度评测，数据量: %d", len(dataset))

        results = []
        dim_scores: Dict[str, List[float]] = {
            "need_understanding_score": [],
            "recommendation_relevance_score": [],
            "information_completeness_score": [],
            "task_completion_score": [],
        }

        for item in dataset:
            result = self.evaluate_single(
                question=item.get("question", item.get("prompt", "")),
                response=item.get("response", ""),
            )
            result["id"] = item.get("id", len(results))
            results.append(result)
            for key in dim_scores:
                dim_scores[key].append(result[key])

        import numpy as np
        summary = {
            "mean_scores": {k: float(np.mean(v)) for k, v in dim_scores.items() if v},
            "total": len(dataset),
            "details": results,
        }
        logger.info(
            "任务完成度评测完成: 综合平均分=%.2f",
            summary["mean_scores"].get("task_completion_score", 0.0),
        )
        return summary
