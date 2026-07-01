#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export SEED="${SEED:-1}"
export TEACHER_ADAPTER="${TEACHER_ADAPTER:-outputs/textvqa_qwen3vl_final_r64_teacher_seed${SEED}/final}"
export OUTPUT_ADAPTER="${OUTPUT_ADAPTER:-outputs/textvqa_qwen3vl_final_r64_svd_r16_seed${SEED}/final}"
export SVD_RANK="${SVD_RANK:-16}"
export SVD_ALPHA="${SVD_ALPHA:-16}"

python3 compress_lora_svd.py \
  --input-adapter "${TEACHER_ADAPTER}" \
  --output-adapter "${OUTPUT_ADAPTER}" \
  --rank "${SVD_RANK}" \
  --alpha "${SVD_ALPHA}" \
  --use-rslora

echo "[INFO] Compressed adapter saved to ${OUTPUT_ADAPTER}"
