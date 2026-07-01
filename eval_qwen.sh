#!/bin/bash
set -euo pipefail

export SEED=${SEED:-1}
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}
export MAX_PIXELS=${MAX_PIXELS:-200704}
export MIN_PIXELS=${MIN_PIXELS:-100352}
export USE_CACHE=${USE_CACHE:-false}
export TASK=${TASK:-textvqa_val_ocr_qaware16}
export OUTPUT_PATH=${OUTPUT_PATH:-./results/textvqa}
export GENERATED_TASK_DIR=${GENERATED_TASK_DIR:-./outputs/generated_tasks}
EXTRA_EVAL_ARGS=()

if [ -n "${TEXTVQA_VAL_DATA_PATH:-}" ] && [ "${TASK}" = "textvqa_val_ocr_qaware16" ]; then
  LOCAL_TASK_DIR="${GENERATED_TASK_DIR}/textvqa_val_ocr_qaware16_local"
  mkdir -p "${LOCAL_TASK_DIR}"
  LOCAL_TASK_YAML="${LOCAL_TASK_DIR}/textvqa_val_ocr_qaware16_local.yaml"
  ${PYTHON:-python3} scripts/create_textvqa_eval_task.py \
    --data-path "${TEXTVQA_VAL_DATA_PATH}" \
    --output "${LOCAL_TASK_YAML}"
  EXTRA_EVAL_ARGS+=(--include_path "${LOCAL_TASK_DIR}")
  export TASK="textvqa_val_ocr_qaware16_local"
fi

${PYTHON:-python3} -c "import torch; print('CUDA available:', torch.cuda.is_available()); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

${PYTHON:-python3} -m lmms_eval \
    --model qwen3_vl \
    --force_simple \
    --model_args pretrained=${MODEL_PATH},attn_implementation=eager,device=cuda,max_pixels=${MAX_PIXELS},min_pixels=${MIN_PIXELS},use_cache=${USE_CACHE},device_map=cuda \
    "${EXTRA_EVAL_ARGS[@]}" \
    --tasks "${TASK}" \
    --batch_size 1 \
    --output_path "${OUTPUT_PATH}"
