#!/usr/bin/env bash
# 运行 GRPO 训练脚本
# 用法: bash run_grpo.sh [--config CONFIG_FILE]

set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-configs/grpo.yaml}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
ANALYZE_BOUNDARY="${ANALYZE_BOUNDARY:-false}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --log-level)
      LOG_LEVEL="$2"
      shift 2
      ;;
    --analyze-boundary)
      ANALYZE_BOUNDARY="true"
      shift
      ;;
    *)
      echo "未知参数: $1"
      echo "用法: $0 [--config CONFIG_FILE] [--analyze-boundary]"
      exit 1
      ;;
  esac
done

echo "======================================"
echo "GRPO 训练"
echo "配置: $CONFIG_FILE"
echo "边界分析: $ANALYZE_BOUNDARY"
echo "======================================"

mkdir -p logs

python -c "
import logging
import sys
import yaml
import json

logging.basicConfig(
    level=getattr(logging, '${LOG_LEVEL}'),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/grpo.log', mode='w'),
    ]
)

with open('${CONFIG_FILE}', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

from training.grpo_trainer import GRPOTrainingPipeline

pipeline = GRPOTrainingPipeline(config=config)
stats = pipeline.train()
print(f'GRPO 训练完成: {stats}')

if '${ANALYZE_BOUNDARY}' == 'true':
    analysis = pipeline.analyze_boundary()
    print('\\n=== GRPO vs PPO vs DPO 边界分析 ===')
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
"

echo "GRPO 训练完成！"
