---
base_model: Qwen/Qwen3-VL-2B-Instruct
library_name: peft
---

# TextVQA PEFT Adapter

This is the submitted PEFT adapter for the Parameter Golf TextVQA task.

- Base model: `Qwen/Qwen3-VL-2B-Instruct`
- Method: full-LM rsLoRA r64 alpha24 teacher, compressed to r16 alpha16 by post-training SVD
- OCR prompt: question-aware top-16 OCR tokens from TextVQA metadata
- Submitted adapter: rank-16 LoRA / rsLoRA
- Local TextVQA validation EM: 0.74414 for seed 3
- Multi-seed mean for r64 -> SVD r16: 0.74166

This folder contains only lightweight PEFT adapter/tokenizer/processor files. It does not contain full Qwen base model weights.