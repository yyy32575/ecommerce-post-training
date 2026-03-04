#!/usr/bin/env bash
# 运行 DPO 训练脚本
# 用法: bash run_dpo.sh [--config CONFIG_FILE] [--sft-checkpoint CHECKPOINT_PATH]

set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-configs/dpo.yaml}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --sft-checkpoint)
      SFT_CHECKPOINT="$2"
      shift 2
      ;;
    --log-level)
      LOG_LEVEL="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      echo "用法: $0 [--config CONFIG_FILE] [--sft-checkpoint CHECKPOINT_PATH]"
      exit 1
      ;;
  esac
done

echo "======================================"
echo "DPO 训练"
echo "配置: $CONFIG_FILE"
if [[ -n "$SFT_CHECKPOINT" ]]; then
  echo "SFT Checkpoint: $SFT_CHECKPOINT"
fi
echo "======================================"

mkdir -p logs

python -c "
import logging
import sys
import yaml

logging.basicConfig(
    level=getattr(logging, '${LOG_LEVEL}'),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/dpo.log', mode='w'),
    ]
)

with open('${CONFIG_FILE}', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 覆盖 SFT checkpoint（若命令行指定）
sft_checkpoint = '${SFT_CHECKPOINT}'
if sft_checkpoint:
    config.setdefault('model', {})['sft_checkpoint'] = sft_checkpoint

from training.dpo_trainer import DPOTrainingPipeline

pipeline = DPOTrainingPipeline(config=config)
stats = pipeline.train()
print(f'DPO 训练完成: {stats}')
"

echo "DPO 训练完成！"
