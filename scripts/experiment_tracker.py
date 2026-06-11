"""实验追踪与结果记录工具。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(file_path: Path) -> Dict[str, Any]:
    """加载 JSON 文件。

    Args:
        file_path: JSON 文件路径。

    Returns:
        解析后的字典对象。
    """
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(file_path: Path, payload: Dict[str, Any]) -> None:
    """保存 JSON 文件。

    Args:
        file_path: 输出文件路径。
        payload: 待保存的字典对象。
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def collect_summary(benchmark_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从 benchmark 结果提取摘要表。

    Args:
        benchmark_data: `evaluation/benchmark_results.json` 的内容。

    Returns:
        可用于表格展示的模型摘要列表。
    """
    summaries: List[Dict[str, Any]] = []
    for model_key, model_data in benchmark_data.get("models", {}).items():
        metrics = model_data.get("metrics", {})
        training_cost = model_data.get("training_cost", {})
        inference = model_data.get("inference", {})
        summaries.append(
            {
                "model": model_key,
                "display_name": model_data.get("display_name", model_key),
                "overall_score": metrics.get("overall_score"),
                "factuality": metrics.get("factuality"),
                "task_completion": metrics.get("task_completion"),
                "safety": metrics.get("safety"),
                "template_rate": metrics.get("template_rate"),
                "gpu_hours": training_cost.get("gpu_hours"),
                "peak_vram_gb": training_cost.get("peak_vram_gb"),
                "latency_ms_p95": inference.get("latency_ms_p95"),
                "throughput_tok_per_s": inference.get("throughput_tok_per_s"),
            }
        )
    return sorted(summaries, key=lambda item: item.get("overall_score", 0), reverse=True)


def append_experiment_log(
    tracker_file: Path,
    experiment_id: str,
    method: str,
    seed: int,
    dataset_version: str,
    evaluator: str,
    hyperparameters: Dict[str, Any],
    notes: Optional[str] = None,
) -> None:
    """向实验日志文件追加一条记录。

    Args:
        tracker_file: 实验日志文件路径。
        experiment_id: 实验唯一标识。
        method: 训练方法名称。
        seed: 随机种子。
        dataset_version: 数据集版本。
        evaluator: 评测者或评测团队。
        hyperparameters: 超参数字典。
        notes: 可选补充说明。
    """
    data: Dict[str, Any]
    if tracker_file.exists():
        data = load_json(tracker_file)
    else:
        data = {"experiments": []}

    data.setdefault("experiments", []).append(
        {
            "experiment_id": experiment_id,
            "method": method,
            "seed": seed,
            "dataset_version": dataset_version,
            "evaluator": evaluator,
            "hyperparameters": hyperparameters,
            "notes": notes or "",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    save_json(tracker_file, data)


def generate_mock_update(
    benchmark_file: Path,
    output_file: Path,
    tracker_file: Path,
) -> Path:
    """根据现有 benchmark 结果生成可追踪快照。

    Args:
        benchmark_file: benchmark 结果文件路径。
        output_file: 摘要输出文件路径。
        tracker_file: 实验追踪日志路径。

    Returns:
        生成的摘要文件路径。
    """
    benchmark_data = load_json(benchmark_file)
    summary = {
        "meta": benchmark_data.get("meta", {}),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "leaderboard": collect_summary(benchmark_data),
    }
    save_json(output_file, summary)

    append_experiment_log(
        tracker_file=tracker_file,
        experiment_id="mock_summary_refresh",
        method="benchmark_summary",
        seed=42,
        dataset_version=str(benchmark_data.get("meta", {}).get("dataset_version", "unknown")),
        evaluator="experiment_tracker",
        hyperparameters={"source": str(benchmark_file)},
        notes="自动生成摘要快照，便于后续替换为真实实验更新。",
    )
    return output_file


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="实验追踪和结果记录工具")
    parser.add_argument(
        "--benchmark-file",
        type=Path,
        default=Path("evaluation/benchmark_results.json"),
        help="benchmark 结果 JSON 文件路径",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("evaluation/benchmark_summary.json"),
        help="摘要输出 JSON 文件路径",
    )
    parser.add_argument(
        "--tracker-file",
        type=Path,
        default=Path("evaluation/experiment_tracking_log.json"),
        help="实验追踪日志 JSON 文件路径",
    )
    return parser


def main() -> None:
    """命令行入口：生成摘要并更新实验追踪日志。"""
    parser = build_arg_parser()
    args = parser.parse_args()
    output_path = generate_mock_update(
        benchmark_file=args.benchmark_file,
        output_file=args.output_file,
        tracker_file=args.tracker_file,
    )
    print(f"实验摘要已生成: {output_path}")


if __name__ == "__main__":
    main()
