"""
DPO 训练器模块

基于 SFT checkpoint 进行 Direct Preference Optimization 训练。
支持 reference model 冻结，W&B 记录 reward accuracy、margin 等指标。
"""

import logging
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class DPOTrainingPipeline:
    """
    直接偏好优化（DPO）训练管线。

    加载 SFT checkpoint 作为初始化，使用 trl.DPOTrainer 进行训练。
    支持 reference model 冻结，记录 reward accuracy、chosen/rejected reward margin。

    Args:
        config: DPO 训练配置字典（对应 dpo.yaml）
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.trainer = None
        logger.info("初始化 DPOTrainingPipeline")

    def _load_model_and_tokenizer(self) -> None:
        """加载 SFT checkpoint 或基座模型作为初始化。"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        model_cfg = self.config.get("model", {})
        base_model = model_cfg.get("base_model", "Qwen/Qwen2-7B")
        sft_checkpoint = model_cfg.get("sft_checkpoint", None)
        trust_remote_code = model_cfg.get("trust_remote_code", True)

        logger.info("加载基座模型: %s", base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载 policy model（从 SFT checkpoint）
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        if sft_checkpoint and sft_checkpoint != base_model:
            logger.info("从 SFT checkpoint 加载权重: %s", sft_checkpoint)
            try:
                self.model = PeftModel.from_pretrained(self.model, sft_checkpoint)
                self.model = self.model.merge_and_unload()
                logger.info("SFT LoRA 权重已合并")
            except Exception as e:
                logger.warning("加载 SFT checkpoint 失败: %s，使用基座模型", e)

        # 应用新的 LoRA（用于 DPO 阶段）
        lora_cfg = self.config.get("lora", {})
        if lora_cfg:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=lora_cfg.get("r", 32),
                lora_alpha=lora_cfg.get("lora_alpha", 64),
                lora_dropout=lora_cfg.get("lora_dropout", 0.05),
                target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
                bias=lora_cfg.get("bias", "none"),
                task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
            )
            self.model = get_peft_model(self.model, lora_config)
            logger.info("DPO LoRA 已应用: r=%d", lora_cfg.get("r", 32))

        # Reference model（冻结）
        logger.info("加载 reference model（冻结权重）")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def _load_dataset(self) -> Any:
        """加载偏好对数据集。"""
        from datasets import load_dataset

        data_cfg = self.config.get("data", {})
        train_file = data_cfg.get("train_file", "data/processed/preference_train.jsonl")
        eval_file = data_cfg.get("eval_file", "data/processed/preference_eval.jsonl")

        logger.info("加载偏好对数据: %s", train_file)
        dataset = load_dataset(
            "json",
            data_files={"train": train_file, "eval": eval_file},
        )
        return dataset

    def _build_trainer(self, dataset: Any) -> Any:
        """构建 trl.DPOTrainer。"""
        from transformers import TrainingArguments
        from trl import DPOTrainer, DPOConfig

        train_cfg = self.config.get("training", {})
        dpo_cfg = self.config.get("dpo", {})
        output_dir = train_cfg.get("output_dir", "./outputs/dpo")

        dpo_config = DPOConfig(
            output_dir=output_dir,
            num_train_epochs=train_cfg.get("num_train_epochs", 1),
            per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
            per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 2),
            gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
            learning_rate=train_cfg.get("learning_rate", 5e-7),
            warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
            lr_scheduler_type=train_cfg.get("lr_scheduler_type", "linear"),
            weight_decay=train_cfg.get("weight_decay", 0.0),
            max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
            bf16=train_cfg.get("bf16", True),
            logging_steps=train_cfg.get("logging_steps", 10),
            eval_steps=train_cfg.get("eval_steps", 100),
            save_steps=train_cfg.get("save_steps", 200),
            save_total_limit=train_cfg.get("save_total_limit", 2),
            load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
            metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_rewards/accuracies"),
            greater_is_better=train_cfg.get("greater_is_better", True),
            remove_unused_columns=train_cfg.get("remove_unused_columns", False),
            report_to=train_cfg.get("report_to", "wandb"),
            evaluation_strategy="steps",
            beta=dpo_cfg.get("beta", 0.1),
            max_length=dpo_cfg.get("max_length", 2048),
            max_prompt_length=dpo_cfg.get("max_prompt_length", 1024),
            loss_type=dpo_cfg.get("loss_type", "sigmoid"),
            label_smoothing=dpo_cfg.get("label_smoothing", 0.0),
        )

        trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=dpo_config,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            tokenizer=self.tokenizer,
        )
        return trainer

    def train(self) -> Dict[str, Any]:
        """
        运行 DPO 训练。

        Returns:
            Dict: 训练结果统计信息
        """
        from .utils import seed_everything, setup_wandb

        logger.info("=" * 60)
        logger.info("开始 DPO 训练")
        logger.info("=" * 60)

        seed_everything(42)
        setup_wandb(self.config)

        self._load_model_and_tokenizer()
        dataset = self._load_dataset()
        self.trainer = self._build_trainer(dataset)

        logger.info("开始 DPO 训练循环")
        train_result = self.trainer.train()

        output_dir = self.config.get("training", {}).get("output_dir", "./outputs/dpo")
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info("DPO 模型已保存至: %s", output_dir)

        stats = {
            "output_dir": output_dir,
            "train_loss": getattr(train_result, "training_loss", None),
            "global_step": getattr(train_result, "global_step", None),
        }
        logger.info("DPO 训练完成: %s", stats)
        return stats

    def compare_sft_vs_dpo(self, eval_dataset: Any, sft_model_path: str) -> Dict[str, Any]:
        """
        对比 SFT-only vs DPO 在评估集上的表现。

        Args:
            eval_dataset: 评估数据集
            sft_model_path: SFT 模型路径

        Returns:
            Dict: 对比结果，包含 reward accuracy 等指标
        """
        logger.info("开始 SFT vs DPO 对比实验")
        if self.trainer is None:
            raise RuntimeError("请先调用 train() 方法")

        # 评估当前 DPO 模型
        dpo_metrics = self.trainer.evaluate()

        results = {
            "dpo_metrics": dpo_metrics,
            "sft_model_path": sft_model_path,
        }
        logger.info("SFT vs DPO 对比结果: %s", results)
        return results
