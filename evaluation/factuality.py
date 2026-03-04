"""
事实性评测模块

基于 LLM-as-judge 的事实性评分（1-5分），支持实体级准确率计算。
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

FACTUALITY_PROMPT = """你是一位专业的电商导购内容质量评审员。请评估以下回答的事实性准确度。

用户问题：{question}

模型回答：{response}

参考答案（如有）：{reference}

请从以下维度评估回答的事实性（1-5分）：
1分：包含严重错误事实，如错误价格、虚假促销、错误品牌信息等
2分：有明显事实错误，影响用户决策
3分：基本准确，但有轻微偏差或不确定表述
4分：事实准确，信息可信
5分：完全准确，信息可靠，有据可查

请直接给出评分（数字1-5）和简短理由：
评分："""


class FactualityEvaluator:
    """
    事实性评测器。

    使用 LLM-as-judge 方法对模型回答进行事实性评分（1-5分），
    支持与 ground truth 对比的实体级准确率计算。

    Args:
        judge_model: 用于评判的 LLM 模型，若为 None 则使用规则方法
        judge_tokenizer: 对应的分词器
        max_new_tokens: 评判生成最大 token 数，默认 128
    """

    def __init__(
        self,
        judge_model=None,
        judge_tokenizer=None,
        max_new_tokens: int = 128,
    ) -> None:
        self.judge_model = judge_model
        self.judge_tokenizer = judge_tokenizer
        self.max_new_tokens = max_new_tokens
        logger.info("初始化 FactualityEvaluator")

    def _llm_judge(self, question: str, response: str, reference: str = "") -> Tuple[float, str]:
        """
        使用 LLM 进行事实性评判。

        Returns:
            Tuple[float, str]: (评分, 理由)
        """
        if self.judge_model is None:
            return self._rule_based_score(response), "规则评分"

        prompt = FACTUALITY_PROMPT.format(
            question=question,
            response=response,
            reference=reference if reference else "无",
        )
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

        # 提取评分
        score_match = re.search(r"[1-5]", generated)
        score = float(score_match.group()) if score_match else 3.0
        return score, generated.strip()

    def _rule_based_score(self, response: str) -> float:
        """基于规则的事实性评分（无 LLM 时的 fallback）。"""
        score = 3.0
        # 含不确定表述扣分
        uncertainty_words = ["可能", "也许", "大概", "我猜", "不确定", "不太清楚"]
        for word in uncertainty_words:
            if word in response:
                score -= 0.3

        # 含具体数字/价格加分（说明信息具体）
        if re.search(r"[¥￥]\d+|[\d,]+元", response):
            score += 0.5

        return max(1.0, min(5.0, score))

    def _extract_entities(self, text: str) -> List[str]:
        """提取文本中的关键实体（价格、品牌、型号等）。"""
        entities = []
        # 价格实体
        entities.extend(re.findall(r"[¥￥]\s*\d+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?\s*元", text))
        # 数字实体
        entities.extend(re.findall(r"\b\d+(?:\.\d+)?\b", text))
        return list(set(entities))

    def _entity_accuracy(self, response: str, reference: str) -> float:
        """
        计算实体级准确率。

        比较模型回答中的实体与参考答案中实体的重叠程度。
        """
        if not reference:
            return 1.0
        ref_entities = set(self._extract_entities(reference))
        if not ref_entities:
            return 1.0
        resp_entities = set(self._extract_entities(response))
        overlap = ref_entities & resp_entities
        return len(overlap) / len(ref_entities)

    def evaluate_single(
        self, question: str, response: str, reference: str = ""
    ) -> Dict[str, Any]:
        """
        评估单条回答的事实性。

        Args:
            question: 用户问题
            response: 模型回答
            reference: 参考答案（可选）

        Returns:
            Dict: 包含评分、理由、实体准确率的评估结果
        """
        score, reason = self._llm_judge(question, response, reference)
        entity_acc = self._entity_accuracy(response, reference)

        return {
            "factuality_score": score,
            "reason": reason,
            "entity_accuracy": entity_acc,
        }

    def evaluate(self, dataset: List[Dict]) -> Dict[str, Any]:
        """
        对数据集进行批量事实性评测。

        Args:
            dataset: 包含 question/response/reference 字段的字典列表

        Returns:
            Dict: 评测结果汇总
        """
        logger.info("开始事实性评测，数据量: %d", len(dataset))

        scores = []
        entity_accuracies = []
        results = []

        for item in dataset:
            result = self.evaluate_single(
                question=item.get("question", item.get("prompt", "")),
                response=item.get("response", ""),
                reference=item.get("reference", ""),
            )
            result["id"] = item.get("id", len(results))
            results.append(result)
            scores.append(result["factuality_score"])
            entity_accuracies.append(result["entity_accuracy"])

        import numpy as np
        summary = {
            "mean_factuality_score": float(np.mean(scores)) if scores else 0.0,
            "std_factuality_score": float(np.std(scores)) if scores else 0.0,
            "mean_entity_accuracy": float(np.mean(entity_accuracies)) if entity_accuracies else 0.0,
            "score_distribution": {
                str(i): scores.count(float(i)) for i in range(1, 6)
            },
            "total": len(dataset),
            "details": results,
        }
        logger.info(
            "事实性评测完成: 平均分=%.2f, 实体准确率=%.2f",
            summary["mean_factuality_score"],
            summary["mean_entity_accuracy"],
        )
        return summary
