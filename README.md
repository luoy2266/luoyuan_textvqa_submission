# Luoyuan TextVQA Submission

This folder contains the final reproducible submission for the Parameter Golf TextVQA task.

Final method:

- Base model: `Qwen/Qwen3-VL-2B-Instruct`
- Data/task: TextVQA validation with question-aware top-16 OCR tokens
- Training recipe: full language-model rsLoRA teacher, rank 64, alpha 24, 1024 steps
- Submitted weight: post-training SVD-compressed LoRA, rank 16, alpha 16
- Submitted adapter: `weights/final_r64_svd_r16_seed3`
- Local TextVQA validation exact match: `0.74414` for the submitted seed-3 adapter

No full Qwen model weights are included. The repository only includes source code, evaluation code, lightweight PEFT weights, local result files, and documentation.

## Directory Layout

- `prepare_textvqa.py`, `train_textvqa_qwen3vl.py`, `merge_lora.py`, `compress_lora_svd.py`: training and LoRA compression code.
- `eval_qwen.sh`, `lmms-eval/`: evaluation code, including the TextVQA OCR task variants used locally.
- `configs/final_train_r64_teacher.yaml`: final training config for the r64 teacher.
- `weights/final_r64_svd_r16_seed3/`: final submitted rank-16 LoRA adapter.
- `scripts/run_eval_submitted_adapter.sh`: merge and evaluate the submitted adapter.
- `scripts/run_full_reproduce_final.sh`: train r64 teacher, compress to r16, then evaluate.
- `results/local_eval/`: final local TextVQA submission JSON; aggregate metrics are in `results/local_results_summary.*`.
- `docs/report.md`: final report with the full experiment story.
- `logs_reference/`: key local timing logs for the final method.

## Environment

The local experiments used Python 3.13.5, CUDA 12.4, PyTorch 2.6.0, Transformers 4.57.0, PEFT 0.15.2, Accelerate 1.7.0.

Example setup:

```bash
micromamba create -p ./env python=3.13 -y
micromamba activate ./env
pip install -r requirements.txt
```

`requirements.txt` includes the PyTorch CUDA 12.4 wheel index and installs the bundled `lmms-eval/` package in editable mode.

## Required Paths

Set these paths on the evaluation server:

```bash
export BASE_MODEL=/path/to/Qwen3-VL-2B-Instruct
export TEXTVQA_DATA_PATH='/path/to/textvqa/train-*.parquet'
export TEXTVQA_VAL_DATA_PATH='/path/to/textvqa/validation-*.parquet'
export HF_HOME=/path/to/hf_cache
export HF_DATASETS_CACHE=/path/to/hf_cache/datasets
export CUDA_VISIBLE_DEVICES=0
```

`BASE_MODEL` may also be `Qwen/Qwen3-VL-2B-Instruct` if the model can be downloaded.

## Evaluate Submitted Adapter

This is the fastest way to verify the submitted weight:

```bash
bash scripts/run_eval_submitted_adapter.sh
```

The script merges `weights/final_r64_svd_r16_seed3` into the base model under `outputs/` and runs `lmms-eval` on `textvqa_val_ocr_qaware16`.

## Full Reproduction

To reproduce the full training path:

```bash
export SEED=1
export CUDA_VISIBLE_DEVICES=0,1
bash scripts/run_full_reproduce_final.sh
```

For the submitted local best seed:

```bash
export SEED=3
export CUDA_VISIBLE_DEVICES=0,1
bash scripts/run_full_reproduce_final.sh
```

The full reproduction trains the r64 teacher first, then compresses it to r16 with `compress_lora_svd.py`.

## Local Results

| Method | Seeds | TextVQA val EM |
| --- | ---: | ---: |
| Original baseline LoRA r16, no OCR | 1/2/3 mean | 0.71255 |
| OCR qaware16 + LoRA r16 | 1 | 0.73248 |
| OCR qaware16 + rsLoRA r64 teacher | 1/2/3 mean | 0.74211 |
| OCR qaware16 + r64 teacher -> SVD r16 | 1/2/3 mean | 0.74166 |
| Submitted adapter, seed 3 | 3 | 0.74414 |

See `results/local_results_summary.json` and `docs/report.md` for the full table and ablations.
