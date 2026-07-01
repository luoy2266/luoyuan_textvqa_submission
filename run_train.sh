#!/bin/bash
set -euo pipefail

export WANDB_DISABLED=true
export CONFIG=${CONFIG:-configs/vlm_textvqa_lora.yaml}
export SEED=${SEED:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29500}
IFS=',' read -r -a CUDA_DEVICE_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
export NUM_PROCESSES=${NUM_PROCESSES:-${#CUDA_DEVICE_ARRAY[@]}}

${PYTHON:-python3} - <<'PY'
import os
import sys
import yaml

config_path = os.environ["CONFIG"]
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
seed = int(os.environ.get("SEED", cfg.get("seed", 1)))
prepared = os.environ.get("PREPARED_DATA_DIR", cfg["prepared_data_dir"]).format(seed=seed)
if not os.path.isdir(prepared):
    print(f"[ERROR] Prepared dataset not found: {prepared}", file=sys.stderr)
    print(f"[ERROR] Run first: SEED={seed} CONFIG={config_path} bash run_prepare.sh", file=sys.stderr)
    sys.exit(1)
PY

if [ "${NUM_PROCESSES}" -gt 1 ]; then
  accelerate launch \
    --num_processes "${NUM_PROCESSES}" \
    --multi_gpu \
    --mixed_precision fp16 \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    train_textvqa_qwen3vl.py --config "${CONFIG}"
else
  accelerate launch \
    --num_processes 1 \
    --mixed_precision fp16 \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    train_textvqa_qwen3vl.py --config "${CONFIG}"
fi
