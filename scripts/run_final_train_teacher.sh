#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export CONFIG="${CONFIG:-configs/final_train_r64_teacher.yaml}"
export SEED="${SEED:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
export WANDB_DISABLED=true

echo "[INFO] Preparing TextVQA train split with ${CONFIG}, seed=${SEED}"
bash run_prepare.sh

echo "[INFO] Training r64 rsLoRA teacher with CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
bash run_train.sh
