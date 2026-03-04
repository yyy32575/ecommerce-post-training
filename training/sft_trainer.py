"""
SFT 训练器模块

支持 LoRA 和 QLoRA 两种训练模式，集成 PEFT、TRL 和 W&B。
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class SFTTrainingPipeline:
    """
    监督微调（SFT）训练管线。

    支持 LoRA 和 QLoRA 两种训练模式，通过配置文件切换。
    集成 PEFT LoraConfig、BitsAndBytesConfig，使用 trl.SFTTrainer 训练。
    记录 W&B 日志：loss 曲线、学习率调度、梯度范数。

    Args:
        config: 训练配置字典（对应 sft_lora.yaml 或 sft_qlora.yaml）
        mode: 训练模式，"lora" 或 "qlora"，默认 "lora"
    """

    def __init__(self, config: Dict[str, Any], mode: str = "lora") -> None:
        self.config = config
        self.mode = mode
        self.model = None
        self.tokenizer = None
        self.trainer = None
        logger.info("初始化 SFTTrainingPipeline: mode=%s", mode)

    def _load_model_and_tokenizer(self) -> None:
        """加载模型和分词器。"""
        from .utils import get_model_and_tokenizer

        # QLoRA 模式需要量化配置
        cfg = dict(self.config)
        if self.mode == "qlora" and "quantization" not in cfg:
            cfg["quantization"] = {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            }

        self.model, self.tokenizer = get_model_and_tokenizer(cfg)

    def _load_dataset(self) -> Any:
        """加载训练和验证数据集。"""
        from datasets import load_dataset

        data_cfg = self.config.get("data", {})
        train_file = data_cfg.get("train_file", "data/processed/train.jsonl")
        eval_file = data_cfg.get("eval_file", "data/processed/eval.jsonl")
        text_field = data_cfg.get("text_field", "text")

        logger.info("加载训练数据: %s", train_file)
        dataset = load_dataset(
            "json",
            data_files={"train": train_file, "eval": eval_file},
        )
        return dataset, text_field

    def _build_trainer(self, dataset: Any, text_field: str) -> Any:
        """构建 trl.SFTTrainer。"""
        from transformers import TrainingArguments
        from trl import SFTTrainer

        train_cfg = self.config.get("training", {})
        output_dir = train_cfg.get("output_dir", f"./outputs/sft_{self.mode}")

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=train_cfg.get("num_train_epochs", 3),
            per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
            per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
            gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
            learning_rate=train_cfg.get("learning_rate", 2e-4),
            warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
            lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
            weight_decay=train_cfg.get("weight_decay", 0.01),
            max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
            bf16=train_cfg.get("bf16", True),
            fp16=train_cfg.get("fp16", False),
            logging_steps=train_cfg.get("logging_steps", 10),
            eval_steps=train_cfg.get("eval_steps", 100),
            save_steps=train_cfg.get("save_steps", 200),
            save_total_limit=train_cfg.get("save_total_limit", 3),
            load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
            metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_loss"),
            dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
            remove_unused_columns=train_cfg.get("remove_unused_columns", False),
            report_to=train_cfg.get("report_to", "wandb"),
            optim=train_cfg.get("optim", "adamw_torch"),
            evaluation_strategy="steps",
        )

        model_cfg = self.config.get("model", {})
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            dataset_text_field=text_field,
            max_seq_length=model_cfg.get("max_seq_length", 2048),
            args=training_args,
        )
        return trainer

    def train(self) -> Dict[str, Any]:
        """
        运行 SFT 训练。

        Returns:
            Dict: 训练结果统计信息
        """
        from .utils import seed_everything, setup_wandb

        logger.info("=" * 60)
        logger.info("开始 SFT 训练（mode=%s）", self.mode)
        logger.info("=" * 60)

        # 设置随机种子
        seed_everything(42)

        # 初始化 W&B
        setup_wandb(self.config)

        # 加载模型
        self._load_model_and_tokenizer()

        # 加载数据集
        dataset, text_field = self._load_dataset()

        # 构建训练器
        self.trainer = self._build_trainer(dataset, text_field)

        # 开始训练
        logger.info("开始训练循环")
        train_result = self.trainer.train()

        # 保存最终模型
        output_dir = self.config.get("training", {}).get("output_dir", f"./outputs/sft_{self.mode}")
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info("模型已保存至: %s", output_dir)

        # 收集训练统计
        stats = {
            "mode": self.mode,
            "output_dir": output_dir,
            "train_loss": train_result.training_loss if hasattr(train_result, "training_loss") else None,
            "global_step": train_result.global_step if hasattr(train_result, "global_step") else None,
        }
        logger.info("SFT 训练完成: %s", stats)
        return stats

    def evaluate(self) -> Dict[str, float]:
        """
        运行评估。

        Returns:
            Dict: 评估指标
        """
        if self.trainer is None:
            raise RuntimeError("请先调用 train() 方法")
        logger.info("开始评估")
        metrics = self.trainer.evaluate()
        logger.info("评估结果: %s", metrics)
        return metrics
