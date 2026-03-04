"""
评测模块

提供四维评测（事实性、任务完成度、模板化率、安全性）和错误分桶功能。
"""

from .factuality import FactualityEvaluator
from .task_completion import TaskCompletionEvaluator
from .template_rate import TemplateRateEvaluator
from .safety import SafetyEvaluator
from .error_bucketing import ErrorBucketing
from .eval_runner import EvalRunner

__all__ = [
    "FactualityEvaluator",
    "TaskCompletionEvaluator",
    "TemplateRateEvaluator",
    "SafetyEvaluator",
    "ErrorBucketing",
    "EvalRunner",
]
