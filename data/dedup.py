"""
数据去重模块

基于 MinHash + LSH 的近似去重，支持 n-gram 可配置和相似度阈值可配置。
"""

import logging
from typing import Dict, List, Optional, Tuple

from datasketch import MinHash, MinHashLSH

logger = logging.getLogger(__name__)


class MinHashDeduplicator:
    """
    基于 MinHash + LSH 的近似去重器。

    使用 MinHash 算法计算文本相似度，通过 LSH（局部敏感哈希）高效地
    找出近似重复文本，避免 O(n^2) 的暴力比较。

    Args:
        num_perm: MinHash 置换数量，越大精度越高，默认 128
        threshold: 相似度阈值，高于此值认为重复，默认 0.85
        ngram_size: n-gram 大小，默认 3（字符级 trigram）
        text_field: 数据集中文本字段名称，默认 "text"。
                    若该字段不存在，自动尝试拼接 prompt+response 字段。
    """

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.85,
        ngram_size: int = 3,
        text_field: str = "text",
    ) -> None:
        self.num_perm = num_perm
        self.threshold = threshold
        self.ngram_size = ngram_size
        self.text_field = text_field
        logger.info(
            "初始化 MinHashDeduplicator: num_perm=%d, threshold=%.2f, ngram_size=%d",
            num_perm,
            threshold,
            ngram_size,
        )

    def _get_text_from_item(self, item: Dict) -> str:
        """从数据条目提取文本，支持纯文本字段和 QA 对字段。"""
        text = item.get(self.text_field, "")
        if not text:
            # 尝试拼接 prompt + response（QA 对格式）
            prompt = item.get("prompt", "")
            response = item.get("response", "")
            text = (prompt + " " + response).strip()
        return text

    def _get_ngrams(self, text: str) -> List[str]:
        """提取文本的字符级 n-gram。"""
        text = text.strip()
        if len(text) < self.ngram_size:
            return [text]
        return [text[i : i + self.ngram_size] for i in range(len(text) - self.ngram_size + 1)]

    def _compute_minhash(self, text: str) -> MinHash:
        """计算文本的 MinHash 签名。"""
        mh = MinHash(num_perm=self.num_perm)
        for ngram in self._get_ngrams(text):
            mh.update(ngram.encode("utf-8"))
        return mh

    def dedup(self, dataset: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        对数据集进行近似去重。

        Args:
            dataset: 包含文本字段的字典列表

        Returns:
            Tuple[List[Dict], Dict]: (去重后数据集, 去重统计信息)
        """
        logger.info("开始去重，原始数据量: %d", len(dataset))

        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        deduplicated: List[Dict] = []
        duplicate_indices: List[int] = []

        for idx, item in enumerate(dataset):
            text = self._get_text_from_item(item)
            if not text:
                logger.warning("第 %d 条数据无有效文本内容，跳过", idx)
                duplicate_indices.append(idx)
                continue

            mh = self._compute_minhash(text)
            key = str(idx)

            # 查询 LSH 是否存在相似文本
            result = lsh.query(mh)
            if result:
                duplicate_indices.append(idx)
            else:
                lsh.insert(key, mh)
                deduplicated.append(item)

            if (idx + 1) % 1000 == 0:
                logger.info("已处理 %d / %d 条数据", idx + 1, len(dataset))

        stats = {
            "original_count": len(dataset),
            "deduplicated_count": len(deduplicated),
            "removed_count": len(duplicate_indices),
            "dedup_ratio": len(duplicate_indices) / max(len(dataset), 1),
        }
        logger.info(
            "去重完成: 原始=%d, 去重后=%d, 删除=%d (%.1f%%)",
            stats["original_count"],
            stats["deduplicated_count"],
            stats["removed_count"],
            stats["dedup_ratio"] * 100,
        )
        return deduplicated, stats
