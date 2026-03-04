#!/usr/bin/env bash
# 运行 SFT 训练脚本
# 用法: bash run_sft.sh [--mode lora|qlora] [--config CONFIG_FILE]

set -euo pipefail

# 默认参数
MODE="${MODE:-lora}"
CONFIG_FILE=""
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --log-level)
      LOG_LEVEL="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      echo "用法: $0 [--mode lora|qlora] [--config CONFIG_FILE]"
      exit 1
      ;;
  esac
done

# 根据模式选择默认配置
if [[ -z "$CONFIG_FILE" ]]; then
  if [[ "$MODE" == "qlora" ]]; then
    CONFIG_FILE="configs/sft_qlora.yaml"
  else
    CONFIG_FILE="configs/sft_lora.yaml"
  fi
fi

echo "======================================"
echo "SFT 训练"
echo "模式: $MODE"
echo "配置: $CONFIG_FILE"
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
        logging.FileHandler('logs/sft_${MODE}.log', mode='w'),
    ]
)

with open('${CONFIG_FILE}', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

from training.sft_trainer import SFTTrainingPipeline

pipeline = SFTTrainingPipeline(config=config, mode='${MODE}')
stats = pipeline.train()
print(f'SFT 训练完成: {stats}')
"

echo "SFT 训练完成！模式: $MODE"
