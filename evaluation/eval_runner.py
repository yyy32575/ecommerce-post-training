"""
统一评测运行器

串联四维评测（事实性、任务完成度、模板化率、安全性）和错误分桶，
生成综合评测报告，支持多模型对比。
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .error_bucketing import ErrorBucketing
from .factuality import FactualityEvaluator
from .safety import SafetyEvaluator
from .task_completion import TaskCompletionEvaluator
from .template_rate import TemplateRateEvaluator

logger = logging.getLogger(__name__)


class EvalRunner:
    """
    统一评测运行器。

    串联所有评测维度，生成综合评测报告（JSON + Markdown 格式）。
    支持多模型对比（SFT-only vs DPO vs GRPO）。

    Args:
        judge_model: LLM-as-judge 模型（用于事实性和任务完成度评测）
        judge_tokenizer: 对应分词器
        output_dir: 评测报告输出目录，默认 "./eval_results"
        enable_factuality: 是否启用事实性评测，默认 True
        enable_task_completion: 是否启用任务完成度评测，默认 True
        enable_template_rate: 是否启用模板化率评测，默认 True
        enable_safety: 是否启用安全性评测，默认 True
        enable_error_bucketing: 是否启用错误分桶，默认 True
    """

    def __init__(
        self,
        judge_model=None,
        judge_tokenizer=None,
        output_dir: str = "./eval_results",
        enable_factuality: bool = True,
        enable_task_completion: bool = True,
        enable_template_rate: bool = True,
        enable_safety: bool = True,
        enable_error_bucketing: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.factuality_eval = FactualityEvaluator(judge_model, judge_tokenizer) if enable_factuality else None
        self.task_eval = TaskCompletionEvaluator(judge_model, judge_tokenizer) if enable_task_completion else None
        self.template_eval = TemplateRateEvaluator() if enable_template_rate else None
        self.safety_eval = SafetyEvaluator() if enable_safety else None
        self.error_bucket = ErrorBucketing() if enable_error_bucketing else None

        logger.info("初始化 EvalRunner: output_dir=%s", output_dir)

    def run_single_model(
        self,
        dataset: List[Dict],
        model_name: str = "model",
    ) -> Dict[str, Any]:
        """
        对单个模型的输出运行所有评测维度。

        Args:
            dataset: 包含 question/response/reference 字段的字典列表
            model_name: 模型名称（用于报告标识）

        Returns:
            Dict: 该模型的完整评测结果
        """
        logger.info("=" * 60)
        logger.info("评测模型: %s，数据量: %d", model_name, len(dataset))
        logger.info("=" * 60)

        results: Dict[str, Any] = {
            "model_name": model_name,
            "total_samples": len(dataset),
            "eval_time": datetime.now().isoformat(),
        }

        # 1. 事实性评测
        if self.factuality_eval is not None:
            logger.info("Step 1: 事实性评测")
            fact_results = self.factuality_eval.evaluate(dataset)
            results["factuality"] = {
                "mean_score": fact_results["mean_factuality_score"],
                "mean_entity_accuracy": fact_results["mean_entity_accuracy"],
            }

        # 2. 任务完成度评测
        if self.task_eval is not None:
            logger.info("Step 2: 任务完成度评测")
            task_results = self.task_eval.evaluate(dataset)
            results["task_completion"] = task_results["mean_scores"]

        # 3. 模板化率评测
        if self.template_eval is not None:
            logger.info("Step 3: 模板化率评测")
            template_results = self.template_eval.evaluate(dataset)
            results["template_rate"] = {
                "overall_template_rate": template_results["template_rate_percentage"],
                "metrics": template_results["metrics"],
            }

        # 4. 安全性评测
        if self.safety_eval is not None:
            logger.info("Step 4: 安全性评测")
            safety_results = self.safety_eval.evaluate(dataset)
            results["safety"] = {
                "mean_score": safety_results["mean_safety_score"],
                "safety_rate": safety_results["safety_rate"],
                "unsafe_count": safety_results["unsafe_count"],
                "violation_counts": safety_results["violation_counts"],
            }

        # 5. 错误分桶
        if self.error_bucket is not None:
            logger.info("Step 5: 错误分桶")
            _, bucket_stats = self.error_bucket.bucket(dataset)
            results["error_bucketing"] = {
                "error_rate": bucket_stats["error_rate"],
                "error_counts": bucket_stats["error_counts"],
                "human_review_needed": bucket_stats["human_review_needed"],
            }

        # 计算综合评分
        results["overall_score"] = self._compute_overall_score(results)

        logger.info(
            "模型 %s 评测完成: 综合分=%.2f",
            model_name,
            results["overall_score"],
        )
        return results

    def _compute_overall_score(self, results: Dict[str, Any]) -> float:
        """
        计算综合评分（加权平均）。

        权重分配：事实性(30%) + 任务完成度(30%) + 安全性(25%) + 模板化率(15%)
        """
        weights = {
            "factuality": 0.30,
            "task_completion": 0.30,
            "safety": 0.25,
            "template_rate": 0.15,
        }
        scores = {}

        if "factuality" in results:
            scores["factuality"] = results["factuality"].get("mean_score", 3.0)

        if "task_completion" in results:
            scores["task_completion"] = results["task_completion"].get("task_completion_score", 3.0)

        if "safety" in results:
            scores["safety"] = results["safety"].get("mean_score", 5.0)

        if "template_rate" in results:
            # 模板化率越低越好，转化为分数
            template_rate = results["template_rate"].get("overall_template_rate", 0.0) / 100
            scores["template_rate"] = 5.0 * (1.0 - template_rate)

        if not scores:
            return 3.0

        weighted_sum = sum(scores.get(k, 3.0) * w for k, w in weights.items() if k in scores)
        weight_sum = sum(w for k, w in weights.items() if k in scores)
        return weighted_sum / max(weight_sum, 1e-8)

    def compare_models(
        self,
        model_datasets: Dict[str, List[Dict]],
    ) -> Dict[str, Any]:
        """
        多模型对比评测。

        Args:
            model_datasets: 字典，key 为模型名称，value 为对应的评测数据集

        Returns:
            Dict: 多模型对比评测结果
        """
        logger.info("开始多模型对比评测: %s", list(model_datasets.keys()))

        all_results: Dict[str, Any] = {}
        for model_name, dataset in model_datasets.items():
            all_results[model_name] = self.run_single_model(dataset, model_name)

        # 生成对比表格数据
        comparison_table = self._build_comparison_table(all_results)
        comparison = {
            "models": all_results,
            "comparison_table": comparison_table,
            "best_model": max(
                all_results.keys(),
                key=lambda k: all_results[k].get("overall_score", 0.0),
            ),
        }
        logger.info("多模型对比完成，最佳模型: %s", comparison["best_model"])
        return comparison

    def _build_comparison_table(self, all_results: Dict[str, Any]) -> List[Dict]:
        """构建对比表格数据。"""
        table = []
        for model_name, results in all_results.items():
            row = {"model": model_name}
            row["overall_score"] = results.get("overall_score", 0.0)
            row["factuality_score"] = results.get("factuality", {}).get("mean_score", "N/A")
            row["task_completion_score"] = results.get("task_completion", {}).get("task_completion_score", "N/A")
            row["safety_score"] = results.get("safety", {}).get("mean_score", "N/A")
            row["template_rate_%"] = results.get("template_rate", {}).get("overall_template_rate", "N/A")
            table.append(row)
        return sorted(table, key=lambda x: x["overall_score"], reverse=True)

    def generate_report(
        self,
        eval_results: Dict[str, Any],
        report_name: str = "eval_report",
    ) -> Dict[str, str]:
        """
        生成评测报告（JSON + Markdown 格式）。

        Args:
            eval_results: 评测结果字典（来自 run_single_model 或 compare_models）
            report_name: 报告文件名前缀

        Returns:
            Dict: 包含 JSON 和 Markdown 报告路径的字典
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON 报告
        json_path = self.output_dir / f"{report_name}_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2, default=str)
        logger.info("JSON 报告已保存: %s", json_path)

        # Markdown 报告
        md_path = self.output_dir / f"{report_name}_{timestamp}.md"
        md_content = self._generate_markdown_report(eval_results)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info("Markdown 报告已保存: %s", md_path)

        return {"json": str(json_path), "markdown": str(md_path)}

    def _generate_markdown_report(self, eval_results: Dict[str, Any]) -> str:
        """生成 Markdown 格式评测报告。"""
        lines = [
            "# 电商导购模型评测报告",
            f"\n**评测时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## 评测结果摘要\n",
        ]

        # 单模型报告
        if "model_name" in eval_results:
            results = eval_results
            lines.append(f"### 模型: {results['model_name']}")
            lines.append(f"\n**综合评分**: {results.get('overall_score', 'N/A'):.2f} / 5.00\n")
            lines.append("| 评测维度 | 得分 |")
            lines.append("|---------|------|")

            if "factuality" in results:
                lines.append(f"| 事实性 | {results['factuality'].get('mean_score', 'N/A'):.2f} |")
            if "task_completion" in results:
                lines.append(f"| 任务完成度 | {results['task_completion'].get('task_completion_score', 'N/A'):.2f} |")
            if "safety" in results:
                lines.append(f"| 安全性 | {results['safety'].get('mean_score', 'N/A'):.2f} |")
            if "template_rate" in results:
                lines.append(f"| 模板化率 | {results['template_rate'].get('overall_template_rate', 'N/A'):.1f}% |")

        # 多模型对比报告
        elif "comparison_table" in eval_results:
            lines.append("## 多模型对比\n")
            lines.append("| 模型 | 综合分 | 事实性 | 任务完成度 | 安全性 | 模板化率 |")
            lines.append("|------|--------|--------|-----------|--------|---------|")
            for row in eval_results["comparison_table"]:
                lines.append(
                    f"| {row['model']} | {row['overall_score']:.2f} | "
                    f"{row.get('factuality_score', 'N/A')} | "
                    f"{row.get('task_completion_score', 'N/A')} | "
                    f"{row.get('safety_score', 'N/A')} | "
                    f"{row.get('template_rate_%', 'N/A')} |"
                )
            lines.append(f"\n**最佳模型**: {eval_results.get('best_model', 'N/A')}")

        return "\n".join(lines)

    def run(
        self,
        dataset: List[Dict],
        model_name: str = "model",
        save_report: bool = True,
    ) -> Dict[str, Any]:
        """
        运行完整评测并可选生成报告。

        Args:
            dataset: 评测数据集
            model_name: 模型名称
            save_report: 是否保存报告，默认 True

        Returns:
            Dict: 评测结果
        """
        results = self.run_single_model(dataset, model_name)
        if save_report:
            report_paths = self.generate_report(results, f"eval_{model_name}")
            results["report_paths"] = report_paths
        return results
