# Submission Manifest

Included:

- Training code: `prepare_textvqa.py`, `train_textvqa_qwen3vl.py`, `run_prepare.sh`, `run_train.sh`
- Test/evaluation code: `eval_qwen.sh`, `merge_lora.py`, `run_merge_lora.sh`, `lmms-eval/`
- Final method scripts: `scripts/run_full_reproduce_final.sh`, `scripts/run_final_train_teacher.sh`, `scripts/run_final_compress_teacher.sh`, `scripts/run_eval_submitted_adapter.sh`
- Compression code: `compress_lora_svd.py`
- Configs: `configs/final_train_r64_teacher.yaml`, `configs/baseline_qaware16_r16.yaml`, `configs/vlm_textvqa_lora.yaml`
- Final lightweight weight: `weights/final_r64_svd_r16_seed3/adapter_model.safetensors`
- Local test results: `results/local_results_summary.json`, `results/local_results_summary.csv`, `results/local_eval/`
- Report and reference logs: `docs/report.md`, `logs_reference/`

Not included:

- Full Qwen model weights
- TextVQA parquet/image data
- Hugging Face cache
- Merged full model directories
- Intermediate experiment output folders

Final adapter metadata:

- PEFT type: LoRA / rsLoRA
- Rank: 16
- Alpha: 16
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Base model: `Qwen/Qwen3-VL-2B-Instruct`
- Local TextVQA val task: `textvqa_val_ocr_qaware16`
- Local TextVQA val exact match: `0.74414`

Primary commands:

```bash
# Evaluate the submitted adapter
bash scripts/run_eval_submitted_adapter.sh

# Full reproduction: train r64 teacher -> SVD r16 -> eval
SEED=3 CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_full_reproduce_final.sh
```
