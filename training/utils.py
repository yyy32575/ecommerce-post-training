"""
训练工具函数模块

提供早停、Checkpoint 管理、模型加载、W&B 初始化等通用工具。
"""

import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """
    设置全局随机种子，确保实验可复现。

    Args:
        seed: 随机种子值，默认 42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("随机种子已设置: %d", seed)


def print_trainable_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    打印模型中可训练参数的数量。

    Args:
        model: PyTorch 模型

    Returns:
        Dict: 包含总参数量和可训练参数量的字典
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    ratio = 100 * trainable_params / max(all_params, 1)
    logger.info(
        "可训练参数: %s / %s (%.2f%%)",
        f"{trainable_params:,}",
        f"{all_params:,}",
        ratio,
    )
    return {
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_ratio": ratio,
    }


def get_model_and_tokenizer(config: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    根据配置统一加载模型和分词器，支持量化（QLoRA）。

    Args:
        config: 训练配置字典，包含 model、lora、quantization 等字段

    Returns:
        Tuple[model, tokenizer]: 加载好的模型和分词器
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model_cfg = config.get("model", {})
    base_model = model_cfg.get("base_model", "Qwen/Qwen2-7B")
    max_seq_length = model_cfg.get("max_seq_length", 2048)
    trust_remote_code = model_cfg.get("trust_remote_code", True)

    logger.info("加载模型: %s", base_model)

    # 分词器
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=trust_remote_code,
        model_max_length=max_seq_length,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 量化配置（QLoRA）
    quant_cfg = config.get("quantization", {})
    bnb_config = None
    if quant_cfg.get("load_in_4bit", False):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(
                torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
            ),
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
        )
        logger.info("使用 4bit 量化（QLoRA 模式）")

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=trust_remote_code,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if bnb_config is None else None,
        device_map="auto",
    )

    # 4bit 量化后的准备工作
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    # LoRA 配置
    lora_cfg = config.get("lora", {})
    if lora_cfg:
        lora_config = LoraConfig(
            r=lora_cfg.get("r", 64),
            lora_alpha=lora_cfg.get("lora_alpha", 128),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
            bias=lora_cfg.get("bias", "none"),
            task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        )
        model = get_peft_model(model, lora_config)
        logger.info("已应用 LoRA 配置: r=%d, alpha=%d", lora_cfg.get("r", 64), lora_cfg.get("lora_alpha", 128))

    print_trainable_parameters(model)
    return model, tokenizer


def setup_wandb(config: Dict[str, Any]) -> Optional[Any]:
    """
    初始化 W&B 日志记录。

    Args:
        config: 包含 wandb 配置字段的字典

    Returns:
        wandb.run 对象或 None（若未配置）
    """
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg:
        logger.info("未配置 W&B，跳过初始化")
        return None

    try:
        import wandb

        run = wandb.init(
            project=wandb_cfg.get("project", "ecommerce-post-training"),
            name=wandb_cfg.get("run_name"),
            tags=wandb_cfg.get("tags", []),
            config=config,
        )
        logger.info("W&B 初始化成功: project=%s, run=%s", wandb_cfg.get("project"), wandb_cfg.get("run_name"))
        return run
    except ImportError:
        logger.warning("未安装 wandb，跳过 W&B 初始化")
        return None
    except Exception as e:
        logger.warning("W&B 初始化失败: %s", e)
        return None


class EarlyStopping:
    """
    早停机制。

    监控验证指标，当指标在 patience 轮内不再改善时停止训练。

    Args:
        patience: 容忍轮次，默认 3
        min_delta: 最小改善幅度，默认 1e-4
        mode: "min" 或 "max"，默认 "min"（监控 loss）
        verbose: 是否打印详细信息，默认 True
    """

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 1e-4,
        mode: str = "min",
        verbose: bool = True,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_value: Optional[float] = None
        self.should_stop = False
        logger.info("初始化 EarlyStopping: patience=%d, mode=%s", patience, mode)

    def __call__(self, metric_value: float) -> bool:
        """
        更新早停状态。

        Args:
            metric_value: 当前轮次的指标值

        Returns:
            bool: 是否应该停止训练
        """
        if self.best_value is None:
            self.best_value = metric_value
            return False

        if self.mode == "min":
            improved = metric_value < self.best_value - self.min_delta
        else:
            improved = metric_value > self.best_value + self.min_delta

        if improved:
            self.best_value = metric_value
            self.counter = 0
            if self.verbose:
                logger.info("指标改善至 %.6f，早停计数器重置", metric_value)
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    "指标未改善（当前=%.6f, 最佳=%.6f），早停计数器: %d/%d",
                    metric_value,
                    self.best_value,
                    self.counter,
                    self.patience,
                )
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info("触发早停！已连续 %d 轮未改善", self.counter)

        return self.should_stop

    def reset(self) -> None:
        """重置早停状态。"""
        self.counter = 0
        self.best_value = None
        self.should_stop = False


class CheckpointManager:
    """
    Checkpoint 管理器。

    管理训练过程中的模型 checkpoint，支持按最优指标保存和限制最大保存数量。

    Args:
        output_dir: Checkpoint 保存目录
        save_total_limit: 最多保存 checkpoint 数量，默认 3
        mode: 指标模式 "min" 或 "max"，默认 "min"
    """

    def __init__(
        self,
        output_dir: str,
        save_total_limit: int = 3,
        mode: str = "min",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.save_total_limit = save_total_limit
        self.mode = mode
        self.checkpoints: List[Tuple[float, str]] = []  # (metric, path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "初始化 CheckpointManager: output_dir=%s, save_total_limit=%d",
            output_dir,
            save_total_limit,
        )

    def save(self, model: torch.nn.Module, metric_value: float, step: int) -> str:
        """
        保存 checkpoint。

        Args:
            model: 要保存的模型
            metric_value: 当前指标值
            step: 当前训练步数

        Returns:
            str: 保存的 checkpoint 路径
        """
        ckpt_path = self.output_dir / f"checkpoint-{step}"
        ckpt_path.mkdir(parents=True, exist_ok=True)

        # 保存模型
        model.save_pretrained(str(ckpt_path))
        logger.info("Checkpoint 保存: %s (metric=%.6f)", ckpt_path, metric_value)

        self.checkpoints.append((metric_value, str(ckpt_path)))

        # 按指标排序，删除最差的
        self.checkpoints.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
        while len(self.checkpoints) > self.save_total_limit:
            worst_metric, worst_path = self.checkpoints.pop()
            if os.path.exists(worst_path):
                shutil.rmtree(worst_path)
                logger.info("删除旧 checkpoint: %s (metric=%.6f)", worst_path, worst_metric)

        return str(ckpt_path)

    def get_best_checkpoint(self) -> Optional[str]:
        """
        获取最优 checkpoint 路径。

        Returns:
            str 或 None: 最优 checkpoint 路径
        """
        if not self.checkpoints:
            return None
        # 已按指标排序，最优在首位
        return self.checkpoints[0][1]
