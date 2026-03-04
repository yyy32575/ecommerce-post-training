#!/usr/bin/env bash
# 运行评测脚本
# 用法: bash run_eval.sh [--data EVAL_DATA] [--model-name MODEL_NAME] [--output-dir OUTPUT_DIR]

set -euo pipefail

EVAL_DATA="${EVAL_DATA:-data/processed/eval.jsonl}"
MODEL_NAME="${MODEL_NAME:-sft_lora}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
COMPARE_MODELS="${COMPARE_MODELS:-false}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)
      EVAL_DATA="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --log-level)
      LOG_LEVEL="$2"
      shift 2
      ;;
    --compare-models)
      COMPARE_MODELS="true"
      shift
      ;;
    *)
      echo "未知参数: $1"
      echo "用法: $0 [--data EVAL_DATA] [--model-name MODEL_NAME] [--output-dir OUTPUT_DIR] [--compare-models]"
      exit 1
      ;;
  esac
done

echo "======================================"
echo "模型评测"
echo "数据: $EVAL_DATA"
echo "模型: $MODEL_NAME"
echo "输出: $OUTPUT_DIR"
echo "多模型对比: $COMPARE_MODELS"
echo "======================================"

mkdir -p logs "$OUTPUT_DIR"

python -c "
import logging
import sys
import json

logging.basicConfig(
    level=getattr(logging, '${LOG_LEVEL}'),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/eval_${MODEL_NAME}.log', mode='w'),
    ]
)

# 加载评测数据
eval_data_path = '${EVAL_DATA}'
with open(eval_data_path, 'r', encoding='utf-8') as f:
    if eval_data_path.endswith('.jsonl'):
        dataset = [json.loads(line) for line in f if line.strip()]
    else:
        dataset = json.load(f)

logging.getLogger(__name__).info('加载评测数据: %d 条', len(dataset))

from evaluation.eval_runner import EvalRunner

runner = EvalRunner(
    output_dir='${OUTPUT_DIR}',
    enable_factuality=True,
    enable_task_completion=True,
    enable_template_rate=True,
    enable_safety=True,
    enable_error_bucketing=True,
)

if '${COMPARE_MODELS}' == 'true':
    # 多模型对比（示例：从不同子目录加载数据）
    model_datasets = {'${MODEL_NAME}': dataset}
    results = runner.compare_models(model_datasets)
    report_paths = runner.generate_report(results, 'comparison')
else:
    results = runner.run(dataset, model_name='${MODEL_NAME}', save_report=True)
    report_paths = results.get('report_paths', {})

print(f'\\n评测完成！报告路径:')
for fmt, path in report_paths.items():
    print(f'  {fmt}: {path}')
print(f'综合评分: {results.get(\"overall_score\", \"N/A\")}')
"

echo "评测完成！结果保存于: $OUTPUT_DIR"
