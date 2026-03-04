#!/usr/bin/env bash
# 运行数据管线脚本
# 用法: bash run_data_pipeline.sh [--input INPUT_FILE] [--output OUTPUT_FILE]

set -euo pipefail

# 默认参数
INPUT_FILE="${INPUT_FILE:-data/raw/train_raw.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-data/processed/train.jsonl}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT_FILE="$2"
      shift 2
      ;;
    --output)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    --log-level)
      LOG_LEVEL="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      echo "用法: $0 [--input INPUT_FILE] [--output OUTPUT_FILE] [--log-level LOG_LEVEL]"
      exit 1
      ;;
  esac
done

echo "======================================"
echo "电商导购数据管线"
echo "输入: $INPUT_FILE"
echo "输出: $OUTPUT_FILE"
echo "======================================"

python -c "
import logging
import sys

logging.basicConfig(
    level=getattr(logging, '${LOG_LEVEL}'),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/data_pipeline.log', mode='w'),
    ]
)

from data.data_pipeline import DataPipeline

config = {
    'dedup': {'enabled': True, 'num_perm': 128, 'threshold': 0.85, 'ngram_size': 3},
    'quality_filter': {
        'enabled': True,
        'min_length': 20,
        'max_length': 8192,
        'max_repetition_ratio': 0.3,
        'max_special_char_ratio': 0.2,
        'require_qa_pair': True,
    },
    'template_rewrite': {'enabled': False},
    'slot_annotate': {
        'enabled': True,
        'slot_types': ['price', 'date', 'stock', 'promotion', 'brand'],
    },
    'preference_pairs': {'enabled': False},
}

pipeline = DataPipeline(config=config)
stats = pipeline.run('${INPUT_FILE}', '${OUTPUT_FILE}')
print(f'管线完成: 输入={stats[\"input_count\"]}, 输出={stats[\"output_count\"]}')
"

echo "数据管线运行完成！输出: $OUTPUT_FILE"
