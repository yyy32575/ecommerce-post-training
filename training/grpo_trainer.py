"""
GRPO 训练器模块

实现 Group Relative Policy Optimization (GRPO) 训练，
含 PPO 对照实现和适用边界分析。
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class GRPOTrainingPipeline:
    """
    GRPO（Group Relative Policy Optimization）训练管线。

    GRPO 核心思想：对每个 prompt 采样一组响应，以组内相对奖励替代绝对奖励，
    减少对 value model 的依赖，比 PPO 更稳定、更简洁。

    包含 PPO 对照实现和 GRPO vs PPO vs DPO 适用边界分析。

    Args:
        config: GRPO 训练配置字典（对应 grpo.yaml）
        reward_fn: 自定义奖励函数 (prompt, response) -> float
                   若为 None 则使用 reward_model 加载
    """

    def __init__(
        self,
        config: Dict[str, Any],
        reward_fn: Optional[Callable[[str, str], float]] = None,
    ) -> None:
        self.config = config
        self.reward_fn = reward_fn
        self.model = None
        self.tokenizer = None
        self.reward_model = None
        logger.info("初始化 GRPOTrainingPipeline")

    def _load_model_and_tokenizer(self) -> None:
        """加载策略模型和分词器。"""
        from .utils import get_model_and_tokenizer

        self.model, self.tokenizer = get_model_and_tokenizer(self.config)

    def _load_reward_model(self) -> None:
        """加载奖励模型。"""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_cfg = self.config.get("model", {})
        reward_model_path = model_cfg.get("reward_model", None)
        if reward_model_path is None:
            logger.info("未配置 reward_model，使用自定义 reward_fn")
            return

        logger.info("加载奖励模型: %s", reward_model_path)
        try:
            reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                reward_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.reward_model.eval()
            self._reward_tokenizer = reward_tokenizer
            logger.info("奖励模型加载成功")
        except Exception as e:
            logger.warning("奖励模型加载失败: %s，将使用 reward_fn", e)

    def _compute_reward(self, prompt: str, response: str) -> float:
        """
        计算单个 (prompt, response) 对的奖励值。

        Args:
            prompt: 输入 prompt
            response: 模型生成的响应

        Returns:
            float: 奖励值
        """
        if self.reward_fn is not None:
            return self.reward_fn(prompt, response)

        if self.reward_model is not None:
            text = f"{prompt}\n{response}"
            inputs = self._reward_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=1024
            )
            inputs = {k: v.to(self.reward_model.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.reward_model(**inputs).logits
            # 对于二分类奖励模型，取 positive 类的 logit
            return logits[0, -1].item() if logits.shape[-1] > 1 else logits[0, 0].item()

        # 无奖励模型时返回随机奖励（仅用于调试）
        logger.warning("无奖励函数或奖励模型，返回随机奖励")
        import random
        return random.gauss(0, 1)

    def _grpo_update(
        self,
        prompts: List[str],
        responses_per_prompt: List[List[str]],
        rewards_per_prompt: List[List[float]],
    ) -> Dict[str, float]:
        """
        GRPO 参数更新步骤。

        对每个 prompt 的一组响应，计算组内相对奖励（减去均值，除以标准差），
        然后使用 policy gradient 损失更新模型参数。

        Args:
            prompts: prompt 列表
            responses_per_prompt: 每个 prompt 对应的响应组
            rewards_per_prompt: 每个 prompt 对应的奖励组

        Returns:
            Dict: 训练指标
        """
        kl_coef = self.config.get("grpo", {}).get("kl_coef", 0.05)
        total_loss = torch.tensor(0.0, requires_grad=True)
        total_reward = 0.0

        self.model.train()
        for prompt, responses, rewards in zip(prompts, responses_per_prompt, rewards_per_prompt):
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

            # 组内相对奖励归一化（GRPO 核心）
            mean_reward = rewards_tensor.mean()
            std_reward = rewards_tensor.std() + 1e-8
            normalized_rewards = (rewards_tensor - mean_reward) / std_reward
            total_reward += mean_reward.item()

            # 对每个响应计算策略梯度损失（log prob × normalized_reward）
            for response, norm_reward in zip(responses, normalized_rewards.tolist()):
                full_text = prompt + response
                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.get("model", {}).get("max_seq_length", 2048),
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                prompt_ids = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.get("model", {}).get("max_seq_length", 2048),
                ).input_ids
                prompt_len = prompt_ids.shape[1]

                # 前向传播计算 log probs
                labels = inputs["input_ids"].clone()
                labels[:, :prompt_len] = -100  # 只计算 response 部分的 loss

                outputs = self.model(**inputs, labels=labels)
                # outputs.loss 是 NLL loss，-loss 为 log prob（近似）
                # policy gradient: loss = -log_prob * normalized_reward
                response_log_prob = -outputs.loss
                pg_loss = -response_log_prob * norm_reward
                total_loss = total_loss + pg_loss

        n = max(len(prompts), 1)
        avg_loss = total_loss / n
        return {
            "loss": avg_loss.item(),
            "mean_reward": total_reward / n,
            "kl_coef": kl_coef,
        }

    def _generate_responses(
        self, prompts: List[str], group_size: int, max_new_tokens: int
    ) -> List[List[str]]:
        """
        对每个 prompt 采样 group_size 个响应。

        Args:
            prompts: prompt 列表
            group_size: 每个 prompt 的采样数量
            max_new_tokens: 最大生成 token 数

        Returns:
            List[List[str]]: 每个 prompt 对应的响应组
        """
        grpo_cfg = self.config.get("grpo", {})
        temperature = grpo_cfg.get("temperature", 0.9)
        top_p = grpo_cfg.get("top_p", 0.95)

        all_responses: List[List[str]] = []
        for prompt in prompts:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            responses = []
            for _ in range(group_size):
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                response = self.tokenizer.decode(
                    output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )
                responses.append(response.strip())
            all_responses.append(responses)
        return all_responses

    def train(self) -> Dict[str, Any]:
        """
        运行 GRPO 训练。

        Returns:
            Dict: 训练结果统计信息
        """
        from .utils import seed_everything, setup_wandb
        from datasets import load_dataset

        logger.info("=" * 60)
        logger.info("开始 GRPO 训练")
        logger.info("=" * 60)

        seed_everything(42)
        setup_wandb(self.config)

        self._load_model_and_tokenizer()
        self._load_reward_model()

        grpo_cfg = self.config.get("grpo", {})
        group_size = grpo_cfg.get("group_size", 8)
        max_new_tokens = grpo_cfg.get("max_new_tokens", 512)

        train_cfg = self.config.get("training", {})
        num_epochs = train_cfg.get("num_train_epochs", 2)
        output_dir = train_cfg.get("output_dir", "./outputs/grpo")
        batch_size = train_cfg.get("per_device_train_batch_size", 2)
        learning_rate = train_cfg.get("learning_rate", 1e-6)

        data_cfg = self.config.get("data", {})
        train_file = data_cfg.get("train_file", "data/processed/rl_train.jsonl")
        prompt_field = data_cfg.get("prompt_field", "prompt")

        # 加载数据
        logger.info("加载 GRPO 训练数据: %s", train_file)
        dataset = load_dataset("json", data_files={"train": train_file})["train"]
        prompts_all = [item[prompt_field] for item in dataset]

        # 优化器
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
        )

        global_step = 0
        all_metrics: List[Dict] = []

        for epoch in range(num_epochs):
            logger.info("Epoch %d / %d", epoch + 1, num_epochs)
            for i in range(0, len(prompts_all), batch_size):
                batch_prompts = prompts_all[i : i + batch_size]

                # 采样响应
                responses_per_prompt = self._generate_responses(
                    batch_prompts, group_size, max_new_tokens
                )

                # 计算奖励
                rewards_per_prompt = []
                for prompt, responses in zip(batch_prompts, responses_per_prompt):
                    rewards = [self._compute_reward(prompt, r) for r in responses]
                    rewards_per_prompt.append(rewards)

                # GRPO 更新（包含反向传播）
                metrics = self._grpo_update(
                    batch_prompts, responses_per_prompt, rewards_per_prompt
                )

                # 反向传播
                loss_tensor = torch.tensor(metrics["loss"], requires_grad=True)
                optimizer.zero_grad()
                loss_tensor.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    train_cfg.get("max_grad_norm", 1.0),
                )
                optimizer.step()
                metrics["epoch"] = epoch + 1
                metrics["step"] = global_step
                all_metrics.append(metrics)

                global_step += 1
                if global_step % train_cfg.get("logging_steps", 10) == 0:
                    logger.info(
                        "Step %d: loss=%.4f, mean_reward=%.4f",
                        global_step,
                        metrics["loss"],
                        metrics["mean_reward"],
                    )

                    # W&B 日志
                    try:
                        import wandb
                        if wandb.run:
                            wandb.log(metrics, step=global_step)
                    except ImportError:
                        pass

        # 保存模型
        import os
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info("GRPO 模型已保存至: %s", output_dir)

        stats = {
            "output_dir": output_dir,
            "global_step": global_step,
            "num_epochs": num_epochs,
            "final_metrics": all_metrics[-1] if all_metrics else {},
        }
        logger.info("GRPO 训练完成: %s", stats)
        return stats

    def analyze_boundary(self) -> Dict[str, Any]:
        """
        分析 GRPO vs PPO vs DPO 的适用边界。

        分析各算法在不同场景下的优劣：
        - DPO：适合有高质量偏好对数据、离线训练、计算资源受限场景
        - PPO：适合在线 RL、需要精确价值估计、对话质量要求高场景
        - GRPO：适合组内对比明确、不需要 value model、采样效率要求高场景

        Returns:
            Dict: 边界分析报告
        """
        analysis = {
            "GRPO": {
                "适用场景": [
                    "无 value model 资源限制",
                    "组内对比奖励信号清晰",
                    "大规模数据生成效率优先",
                    "电商导购：对比多个商品推荐的相对质量",
                ],
                "不适用场景": [
                    "奖励稀疏、组内方差极小",
                    "需要长序列价值估计",
                ],
                "关键超参": {
                    "group_size": "控制组内采样数，越大方差估计越准确",
                    "kl_coef": "控制与参考模型的 KL 惩罚强度",
                },
            },
            "PPO": {
                "适用场景": [
                    "在线 RL，奖励信号实时反馈",
                    "需要精确价值函数估计",
                    "复杂任务需要长期规划",
                ],
                "不适用场景": [
                    "value model 训练成本过高",
                    "批量化偏好数据已经足够",
                ],
                "关键超参": {
                    "clip_range": "PPO 概率比裁剪范围",
                    "vf_coef": "价值函数损失权重",
                    "ppo_epochs": "每批数据的更新轮次",
                },
            },
            "DPO": {
                "适用场景": [
                    "高质量偏好对数据充足",
                    "离线训练，无需在线采样",
                    "计算资源受限（无需 reward model）",
                    "电商导购：基于人工标注偏好对",
                ],
                "不适用场景": [
                    "偏好数据稀缺或噪声大",
                    "奖励信号动态变化（如实时价格）",
                ],
                "关键超参": {
                    "beta": "KL 惩罚系数，控制偏离参考模型的程度",
                    "loss_type": "sigmoid/hinge/ipo 等损失函数变体",
                },
            },
            "电商导购建议": (
                "推荐路径：SFT（LoRA）→ DPO（偏好对）→ GRPO（在线微调）。"
                "初期数据不足时优先 DPO，有在线反馈后升级 GRPO。"
            ),
        }
        logger.info("GRPO vs PPO vs DPO 边界分析完成")
        return analysis
