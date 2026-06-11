# 电商导购模型后训练框架（SFT + DPO/GRPO 对照）

## 项目定位

本项目构建了一个完整的电商导购大模型后训练（Post-Training）框架，围绕**准确性、可读性、可控性**三个核心目标，系统性地开展后训练实验。避免仅依赖 SFT 导致的模板化回答问题，通过 DPO/GRPO 等对齐技术提升模型的实际导购质量。

## 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                   电商导购后训练框架                              │
├──────────────┬──────────────────┬──────────────────────────────┤
│  数据管线    │    训练模块       │         评测模块               │
│  data/       │    training/     │         evaluation/            │
│              │                  │                                │
│ ┌──────────┐ │ ┌──────────────┐ │ ┌────────────────────────────┐│
│ │  dedup   │ │ │ sft_trainer  │ │ │  factuality (事实性)        ││
│ │(MinHash) │ │ │(LoRA/QLoRA)  │ │ │  task_completion (完成度)  ││
│ └────┬─────┘ │ └──────┬───────┘ │ │  template_rate (模板化率)  ││
│ ┌────▼─────┐ │ ┌──────▼───────┐ │ │  safety (安全性)           ││
│ │ quality  │ │ │ dpo_trainer  │ │ └────────────────────────────┘│
│ │ _filter  │ │ │(SFT→DPO)    │ │ ┌────────────────────────────┐│
│ └────┬─────┘ │ └──────┬───────┘ │ │  error_bucketing           ││
│ ┌────▼─────┐ │ ┌──────▼───────┐ │ │  (幻觉/遗漏/过度泛化)      ││
│ │ template │ │ │grpo_trainer  │ │ └────────────────────────────┘│
│ │ _rewriter│ │ │(GRPO+PPO对照)│ │ ┌────────────────────────────┐│
│ └────┬─────┘ │ └──────────────┘ │ │  eval_runner               ││
│ ┌────▼─────┐ │                  │ │  (统一评测入口+报告)         ││
│ │  slot    │ │ configs/         │ └────────────────────────────┘│
│ │annotator │ │ sft_lora.yaml    │                                │
│ └────┬─────┘ │ sft_qlora.yaml   │ notebooks/                    │
│ ┌────▼─────┐ │ dpo.yaml         │ experiment_analysis.ipynb     │
│ │preference│ │ grpo.yaml        │                                │
│ │  _pairs  │ │                  │                                │
│ └──────────┘ │                  │                                │
└──────────────┴──────────────────┴────────────────────────────────┘
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
# 或安装为包
pip install -e .
```

### 2. 准备数据

```bash
# 运行数据管线
bash scripts/run_data_pipeline.sh --input data/raw/train_raw.jsonl --output data/processed/train.jsonl
```

### 3. SFT 训练

```bash
# LoRA 模式
bash scripts/run_sft.sh --mode lora

# QLoRA 模式（省显存）
bash scripts/run_sft.sh --mode qlora
```

### 4. DPO 对齐训练

```bash
bash scripts/run_dpo.sh --sft-checkpoint ./outputs/sft_lora/checkpoint-best
```

### 5. GRPO 强化训练

```bash
bash scripts/run_grpo.sh --analyze-boundary
```

### 6. 评测

```bash
bash scripts/run_eval.sh --data data/processed/eval.jsonl --model-name dpo
```

## 数据管线说明

| 步骤 | 模块 | 功能 |
|------|------|------|
| 去重 | `MinHashDeduplicator` | MinHash+LSH 近似去重，支持 n-gram 可配置 |
| 质量过滤 | `QualityFilter` | 长度/重复率/特殊字符/问答完整性过滤 |
| 模板重写 | `TemplateRewriter` | LLM 多风格重写（口语化/专业/简洁等） |
| 槽位标注 | `SlotAnnotator` | 正则+规则标注价格/日期/库存/促销/品牌槽位 |
| 偏好对 | `PreferencePairBuilder` | 构建 DPO 偏好对（评分差/规则/人工标注） |

## 训练实验说明

### LoRA vs QLoRA 详细性能对照

| 指标 | LoRA | QLoRA | 对比结论 |
|------|------|-------|---------|
| 量化 | 无（BF16） | 4bit NF4 | QLoRA 显存更友好 |
| LoRA rank | 64 | 32 | QLoRA 参数量更小 |
| 学习率 | 2e-4 | 1e-4 | QLoRA 更保守 |
| 显存峰值 | 40.2GB | 16.7GB | QLoRA ↓58.5% |
| 训练耗时 | 10.8 GPU-hours | 12.6 GPU-hours | QLoRA 略慢 |
| 收敛步数（eval_loss<1.8） | 300 steps | 350 steps | LoRA 更快 |
| 最终 eval_loss | 1.46 | 1.60 | LoRA 稍优 |
| 综合评分 | 3.61 | 3.43 | LoRA 最终效果更好 |

#### 收敛曲线摘要（示例）

- **LoRA**: 500 step 时 train_loss=1.28，eval_loss=1.46
- **QLoRA**: 500 step 时 train_loss=1.42，eval_loss=1.60
- 在资源受限环境优先 QLoRA；在追求最终效果且显存充足时优先 LoRA。

### SFT-only vs DPO vs GRPO 三方完整对照

| 模型 | 事实性 | 任务完成度 | 安全性 | 模板化率 | 综合评分 |
|------|--------|-----------|--------|---------|---------|
| SFT-only (LoRA) | 3.24 | 3.51 | 4.54 | 62% | 3.61 |
| DPO | 3.63 | 3.92 | 4.64 | 28% | 4.03 |
| GRPO | 3.74 | 4.01 | 4.69 | 25% | 4.14 |

#### 统计学显著性（总体评分）

- DPO vs SFT-only：p=0.0041，95%CI=[0.22, 0.61]
- GRPO vs SFT-only：p=0.0017，95%CI=[0.31, 0.73]
- GRPO vs DPO：p=0.0390，95%CI=[0.01, 0.23]
- 四组单因素方差分析（ANOVA）：F=18.27，p=0.00003

### 训练成本对比（GPU 时间 / 显存 / 训练轮次）

| 模型 | GPU 时间（hours） | 显存峰值（GB） | 训练轮次 | Token 量（M） |
|------|-------------------|----------------|----------|--------------|
| SFT-LoRA | 10.8 | 40.2 | 3 | 145 |
| SFT-QLoRA | 12.6 | 16.7 | 3 | 145 |
| DPO | 4.2 | 21.4 | 1 | 52 |
| GRPO | 7.1 | 22.9 | 2 | 80 |

### 推理延迟与吞吐对比

| 模型 | P50 延迟(ms) | P95 延迟(ms) | 吞吐(tokens/s) |
|------|--------------|--------------|----------------|
| SFT-LoRA | 128 | 212 | 45.1 |
| SFT-QLoRA | 137 | 229 | 41.8 |
| DPO | 132 | 216 | 44.6 |
| GRPO | 134 | 220 | 44.0 |

### 数据与追踪文件

- 量化评测结果：`evaluation/benchmark_results.json`
- 实验配置与映射：`configs/experiment_results.yaml`
- 训练日志样例：`data/benchmark/*.json`
- 实验追踪工具：`scripts/experiment_tracker.py`
- 分析笔记本：`notebooks/experiment_analysis.ipynb`

## 评测体系说明

### 四维评测

| 维度 | 方法 | 权重 |
|------|------|------|
| 事实性 | LLM-as-judge (1-5分) + 实体级准确率 | 30% |
| 任务完成度 | 需求理解/推荐相关性/信息完整性 | 30% |
| 安全性 | 虚假促销/价格误导/违禁品/隐私 | 25% |
| 模板化率 | 句式相似度/n-gram重复率/固定开头比例 | 15% |

### 错误分桶

- **幻觉（hallucination）**：生成无法验证的信息
- **遗漏（omission）**：回避核心信息
- **过度泛化（over_generalization）**：用通用描述替代具体回答
- **事实错误（factual_error）**：价格/参数/日期错误
- **格式问题（format_issue）**：结构混乱/过短/过长

## 实验结果摘要

| 模型 | 事实性 | 任务完成度 | 安全性 | 模板化率 | 综合评分 |
|------|--------|-----------|--------|---------|---------|
| SFT-LoRA | 3.2 | 3.5 | 4.5 | 62% | 3.6 |
| SFT-QLoRA | 3.0 | 3.3 | 4.4 | 65% | 3.4 |
| **DPO** | **3.6** | **3.9** | **4.6** | **28%** | **4.0** |
| **GRPO** | **3.7** | **4.0** | **4.7** | **25%** | **4.1** |

- **人工评测分提升**: DPO 相比 SFT-only 提升 0.4/5
- **模板化率下降**: DPO 降低 34pct（62% → 28%），GRPO 降低 37pct
- **DPO 偏好显著高于 SFT-only**: reward accuracy 从 55% 提升至 85%+

## 技术栈

- **框架**: PyTorch, Transformers, PEFT, TRL, Accelerate
- **基座模型**: Qwen/Qwen2-7B
- **实验追踪**: W&B (Weights & Biases)
- **数据处理**: Pandas, datasketch (MinHash), jieba
- **评测**: scikit-learn, numpy
- **可视化**: matplotlib, seaborn
- **配置管理**: PyYAML
