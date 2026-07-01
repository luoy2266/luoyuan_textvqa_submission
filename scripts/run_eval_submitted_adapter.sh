#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
export ADAPTER="${ADAPTER:-weights/final_r64_svd_r16_seed3}"
export MERGED_MODEL="${MERGED_MODEL:-outputs/submission_final_r64_svd_r16_seed3_merged}"
export OUTPUT_PATH="${OUTPUT_PATH:-results/eval_submitted_adapter}"
export TASK="${TASK:-textvqa_val_ocr_qaware16}"
export MAX_PIXELS="${MAX_PIXELS:-200704}"
export MIN_PIXELS="${MIN_PIXELS:-100352}"

mkdir -p outputs results

if [ ! -d "${MERGED_MODEL}" ] || [ "${REMERGE:-0}" = "1" ]; then
  echo "[INFO] Merging LoRA adapter into ${MERGED_MODEL}"
  BASE_MODEL="${BASE_MODEL}" ADAPTER="${ADAPTER}" MERGED_MODEL="${MERGED_MODEL}" MERGE_DTYPE=float16 bash run_merge_lora.sh
else
  echo "[INFO] Reusing existing merged model: ${MERGED_MODEL}"
fi

echo "[INFO] Evaluating ${MERGED_MODEL} on ${TASK}"
MODEL_PATH="${MERGED_MODEL}" OUTPUT_PATH="${OUTPUT_PATH}" TASK="${TASK}" MAX_PIXELS="${MAX_PIXELS}" MIN_PIXELS="${MIN_PIXELS}" bash eval_qwen.sh
