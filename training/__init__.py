"""
训练模块

提供 SFT、DPO、GRPO 三种训练方式，以及训练工具函数。
"""

from .sft_trainer import SFTTrainingPipeline
from .dpo_trainer import DPOTrainingPipeline
from .grpo_trainer import GRPOTrainingPipeline
from .utils import (
    CheckpointManager,
    EarlyStopping,
    get_model_and_tokenizer,
    print_trainable_parameters,
    seed_everything,
    setup_wandb,
)

__all__ = [
    "SFTTrainingPipeline",
    "DPOTrainingPipeline",
    "GRPOTrainingPipeline",
    "EarlyStopping",
    "CheckpointManager",
    "seed_everything",
    "print_trainable_parameters",
    "get_model_and_tokenizer",
    "setup_wandb",
]
