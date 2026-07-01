#!/bin/bash
set -euo pipefail

export CONFIG=${CONFIG:-configs/vlm_textvqa_lora.yaml}
export SEED=${SEED:-1}

${PYTHON:-python3} prepare_textvqa.py --config "${CONFIG}"
