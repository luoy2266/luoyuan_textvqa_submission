#!/bin/bash
set -euo pipefail

export SEED=${SEED:-1}
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-VL-2B-Instruct}
export ADAPTER=${ADAPTER:-./outputs/textvqa_qwen3vl_lora_seed${SEED}/final}
export MERGED_MODEL=${MERGED_MODEL:-./outputs/textvqa_qwen3vl_lora_seed${SEED}/merged}
export MERGE_DTYPE=${MERGE_DTYPE:-float16}

${PYTHON:-python3} merge_lora.py \
  --base_model "${BASE_MODEL}" \
  --adapter "${ADAPTER}" \
  --output "${MERGED_MODEL}" \
  --dtype "${MERGE_DTYPE}"
