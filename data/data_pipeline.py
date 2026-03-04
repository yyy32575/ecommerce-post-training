"""
统一数据管线入口

串联去重、质量过滤、模板重写、槽位标注、偏好对构建等所有数据处理步骤。
支持配置化的管线定义和完整的日志统计。
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .dedup import MinHashDeduplicator
from .preference_pair_builder import BuildStrategy, PreferencePairBuilder
from .quality_filter import QualityFilter
from .slot_annotator import SlotAnnotator
from .template_rewriter import TemplateRewriter

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    统一数据处理管线。

    串联所有数据处理步骤，支持配置化定义，
    提供完整的日志记录和统计信息输出。

    Args:
        config: 管线配置字典，定义各步骤的参数和开关
        model: （可选）用于模板重写的 LLM 模型
        tokenizer: （可选）对应的分词器
    """

    DEFAULT_CONFIG = {
        "dedup": {
            "enabled": True,
            "num_perm": 128,
            "threshold": 0.85,
            "ngram_size": 3,
        },
        "quality_filter": {
            "enabled": True,
            "min_length": 20,
            "max_length": 8192,
            "max_repetition_ratio": 0.3,
            "max_special_char_ratio": 0.2,
            "require_qa_pair": False,
        },
        "template_rewrite": {
            "enabled": False,  # 默认关闭（需要 LLM）
            "styles": ["colloquial", "professional", "concise"],
            "batch_size": 8,
        },
        "slot_annotate": {
            "enabled": True,
            "slot_types": ["price", "date", "stock", "promotion", "brand"],
        },
        "preference_pairs": {
            "enabled": False,  # 默认关闭（需要特定数据格式）
            "strategy": "mixed",
            "min_score_diff": 1.0,
        },
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model=None,
        tokenizer=None,
    ) -> None:
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.model = model
        self.tokenizer = tokenizer
        self._init_components()
        logger.info("初始化 DataPipeline，配置: %s", list(self.config.keys()))

    def _init_components(self) -> None:
        """根据配置初始化各处理组件。"""
        cfg = self.config

        # 去重组件
        if cfg["dedup"]["enabled"]:
            self.deduplicator = MinHashDeduplicator(
                num_perm=cfg["dedup"]["num_perm"],
                threshold=cfg["dedup"]["threshold"],
                ngram_size=cfg["dedup"]["ngram_size"],
            )
        else:
            self.deduplicator = None

        # 质量过滤组件
        if cfg["quality_filter"]["enabled"]:
            qf_cfg = cfg["quality_filter"]
            self.quality_filter = QualityFilter(
                min_length=qf_cfg["min_length"],
                max_length=qf_cfg["max_length"],
                max_repetition_ratio=qf_cfg["max_repetition_ratio"],
                max_special_char_ratio=qf_cfg["max_special_char_ratio"],
                require_qa_pair=qf_cfg["require_qa_pair"],
            )
        else:
            self.quality_filter = None

        # 模板重写组件
        if cfg["template_rewrite"]["enabled"]:
            tr_cfg = cfg["template_rewrite"]
            self.template_rewriter = TemplateRewriter(
                model=self.model,
                tokenizer=self.tokenizer,
                styles=tr_cfg["styles"],
                batch_size=tr_cfg["batch_size"],
            )
        else:
            self.template_rewriter = None

        # 槽位标注组件
        if cfg["slot_annotate"]["enabled"]:
            sa_cfg = cfg["slot_annotate"]
            self.slot_annotator = SlotAnnotator(
                slot_types=sa_cfg["slot_types"],
            )
        else:
            self.slot_annotator = None

        # 偏好对构建组件
        if cfg["preference_pairs"]["enabled"]:
            pp_cfg = cfg["preference_pairs"]
            self.preference_pair_builder = PreferencePairBuilder(
                strategy=BuildStrategy(pp_cfg["strategy"]),
                min_score_diff=pp_cfg["min_score_diff"],
            )
        else:
            self.preference_pair_builder = None

    def _load_data(self, input_path: str) -> List[Dict]:
        """加载数据文件（支持 jsonl 和 json 格式）。"""
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        data: List[Dict] = []
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                data = loaded if isinstance(loaded, list) else [loaded]
        else:
            raise ValueError(f"不支持的文件格式: {path.suffix}，请使用 .json 或 .jsonl")

        logger.info("加载数据: %d 条，来源: %s", len(data), input_path)
        return data

    def _save_data(self, data: List[Dict], output_path: str) -> None:
        """保存数据到文件（jsonl 格式）。"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info("保存数据: %d 条，目标: %s", len(data), output_path)

    def run(self, input_path: str, output_path: str) -> Dict:
        """
        运行完整数据管线。

        Args:
            input_path: 输入数据文件路径（.json 或 .jsonl）
            output_path: 输出数据文件路径（.jsonl）

        Returns:
            Dict: 完整的管线运行统计信息
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("数据管线开始运行")
        logger.info("输入: %s", input_path)
        logger.info("输出: %s", output_path)
        logger.info("=" * 60)

        pipeline_stats: Dict[str, Any] = {
            "input_path": input_path,
            "output_path": output_path,
            "steps": {},
        }

        # 加载数据
        data = self._load_data(input_path)
        pipeline_stats["input_count"] = len(data)

        # Step 1: 去重
        if self.deduplicator is not None:
            logger.info("Step 1: 去重")
            data, dedup_stats = self.deduplicator.dedup(data)
            pipeline_stats["steps"]["dedup"] = dedup_stats

        # Step 2: 质量过滤
        if self.quality_filter is not None:
            logger.info("Step 2: 质量过滤")
            data, filter_stats = self.quality_filter.filter(data)
            pipeline_stats["steps"]["quality_filter"] = filter_stats

        # Step 3: 模板重写
        if self.template_rewriter is not None:
            logger.info("Step 3: 模板重写")
            data, rewrite_stats = self.template_rewriter.rewrite(data)
            pipeline_stats["steps"]["template_rewrite"] = rewrite_stats

        # Step 4: 槽位标注
        if self.slot_annotator is not None:
            logger.info("Step 4: 槽位标注")
            data, annotate_stats = self.slot_annotator.annotate(data)
            pipeline_stats["steps"]["slot_annotate"] = annotate_stats

        # Step 5: 偏好对构建
        if self.preference_pair_builder is not None:
            logger.info("Step 5: 偏好对构建")
            data, pair_stats = self.preference_pair_builder.build_pairs(data)
            pipeline_stats["steps"]["preference_pairs"] = pair_stats

        # 保存结果
        self._save_data(data, output_path)

        elapsed = time.time() - start_time
        pipeline_stats["output_count"] = len(data)
        pipeline_stats["elapsed_seconds"] = round(elapsed, 2)

        logger.info("=" * 60)
        logger.info(
            "数据管线完成: 输入=%d, 输出=%d, 耗时=%.1fs",
            pipeline_stats["input_count"],
            pipeline_stats["output_count"],
            elapsed,
        )
        logger.info("=" * 60)

        # 保存统计报告
        stats_path = str(Path(output_path).with_suffix("")) + "_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(pipeline_stats, f, ensure_ascii=False, indent=2)
        logger.info("统计报告保存至: %s", stats_path)

        return pipeline_stats
