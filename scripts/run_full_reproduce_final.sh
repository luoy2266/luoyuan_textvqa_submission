#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export SEED="${SEED:-1}"
export CONFIG="${CONFIG:-configs/final_train_r64_teacher.yaml}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

bash scripts/run_final_train_teacher.sh
bash scripts/run_final_compress_teacher.sh

export ADAPTER="${OUTPUT_ADAPTER:-outputs/textvqa_qwen3vl_final_r64_svd_r16_seed${SEED}/final}"
export MERGED_MODEL="${MERGED_MODEL:-outputs/textvqa_qwen3vl_final_r64_svd_r16_seed${SEED}/merged}"
export OUTPUT_PATH="${OUTPUT_PATH:-results/final_reproduce_seed${SEED}}"
bash scripts/run_eval_submitted_adapter.sh
