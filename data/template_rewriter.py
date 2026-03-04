"""
模板多样化重写模块

使用 LLM 对模板化回答进行多样化重写，支持多种风格和批量处理。
"""

import logging
import random
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 内置重写风格 Prompt 模板
REWRITE_PROMPTS = {
    "colloquial": (
        "请将以下电商导购回答改写为更口语化、亲切自然的风格，"
        "就像朋友之间的对话，保持核心信息不变：\n\n{text}\n\n改写后："
    ),
    "professional": (
        "请将以下电商导购回答改写为更专业、严谨的风格，"
        "使用专业术语，保持核心信息不变：\n\n{text}\n\n改写后："
    ),
    "concise": (
        "请将以下电商导购回答改写为更简洁、精炼的风格，"
        "去掉冗余表达，保留最核心的信息：\n\n{text}\n\n改写后："
    ),
    "detailed": (
        "请将以下电商导购回答改写为更详细、丰富的风格，"
        "增加相关背景和使用场景，保持核心信息不变：\n\n{text}\n\n改写后："
    ),
    "casual": (
        "请将以下电商导购回答改写为轻松随意的风格，"
        "可以加入适当的网络用语，保持核心信息不变：\n\n{text}\n\n改写后："
    ),
}


class TemplateRewriter:
    """
    模板化回答多样化重写器。

    使用 LLM 将模板化、固定套路的回答改写为多样化风格，
    支持口语化、专业化、简洁化等多种风格。

    Args:
        model: 用于重写的 LLM 模型（兼容 transformers pipeline）
        tokenizer: 对应的分词器
        styles: 重写风格列表，默认使用所有内置风格
        response_field: 回答字段名，默认 "response"
        max_new_tokens: 生成最大 token 数，默认 512
        batch_size: 批处理大小，默认 8
        quality_threshold: 质量回检相似度阈值，默认 0.5
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        styles: Optional[List[str]] = None,
        response_field: str = "response",
        max_new_tokens: int = 512,
        batch_size: int = 8,
        quality_threshold: float = 0.5,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.styles = styles or list(REWRITE_PROMPTS.keys())
        self.response_field = response_field
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.quality_threshold = quality_threshold
        logger.info(
            "初始化 TemplateRewriter: styles=%s, batch_size=%d",
            self.styles,
            batch_size,
        )

    def _build_prompt(self, text: str, style: str) -> str:
        """构建指定风格的重写 Prompt。"""
        prompt_template = REWRITE_PROMPTS.get(style, REWRITE_PROMPTS["concise"])
        return prompt_template.format(text=text)

    def _generate(self, prompt: str) -> str:
        """调用 LLM 生成重写结果。"""
        if self.model is None or self.tokenizer is None:
            # 无模型时返回原文（用于测试/离线场景）
            logger.warning("未配置模型，返回原文")
            return prompt.split("改写后：")[-1].strip() if "改写后：" in prompt else prompt

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with __import__("torch").no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return generated.strip()

    def _quality_check(self, original: str, rewritten: str) -> bool:
        """
        对重写结果进行质量回检。

        检查重写后文本长度是否合理，避免生成过短或过长的结果。
        """
        if not rewritten or len(rewritten.strip()) < 10:
            return False
        length_ratio = len(rewritten) / max(len(original), 1)
        return 0.3 <= length_ratio <= 3.0

    def rewrite_single(self, text: str, style: Optional[str] = None) -> Tuple[str, str]:
        """
        重写单条文本。

        Args:
            text: 待重写文本
            style: 重写风格，若为 None 则随机选择

        Returns:
            Tuple[str, str]: (重写后文本, 使用的风格)
        """
        if style is None:
            style = random.choice(self.styles)
        prompt = self._build_prompt(text, style)
        rewritten = self._generate(prompt)
        if not self._quality_check(text, rewritten):
            logger.warning("重写质量不达标，使用原文")
            return text, style
        return rewritten, style

    def rewrite(self, dataset: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        对数据集进行批量重写。

        Args:
            dataset: 包含回答字段的字典列表

        Returns:
            Tuple[List[Dict], Dict]: (重写后数据集, 重写统计信息)
        """
        logger.info("开始批量重写，数据量: %d", len(dataset))

        rewritten_dataset: List[Dict] = []
        success_count = 0
        fail_count = 0
        style_counts: Dict[str, int] = {s: 0 for s in self.styles}

        for idx, item in enumerate(dataset):
            response = item.get(self.response_field, "")
            if not response:
                rewritten_dataset.append(item)
                fail_count += 1
                continue

            rewritten_text, used_style = self.rewrite_single(response)
            new_item = dict(item)
            new_item[self.response_field] = rewritten_text
            new_item["rewrite_style"] = used_style
            rewritten_dataset.append(new_item)

            if rewritten_text != response:
                success_count += 1
                style_counts[used_style] = style_counts.get(used_style, 0) + 1
            else:
                fail_count += 1

            if (idx + 1) % 100 == 0:
                logger.info("已重写 %d / %d 条数据", idx + 1, len(dataset))

        stats = {
            "total": len(dataset),
            "success_count": success_count,
            "fail_count": fail_count,
            "style_distribution": style_counts,
        }
        logger.info(
            "重写完成: 成功=%d, 失败=%d",
            success_count,
            fail_count,
        )
        return rewritten_dataset, stats
