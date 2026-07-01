#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--strategy", default="question_aware")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output:
        output = Path(args.output)
    else:
        task_dir = Path(__file__).resolve().parents[1] / "lmms-eval" / "lmms_eval" / "tasks" / "textvqa"
        output = task_dir / "textvqa_val_ocr_qaware16_local.yaml"
    text = f"""task: textvqa_val_ocr_qaware16_local
test_split: validation
dataset_path: parquet
dataset_kwargs:
  data_files:
    validation: {args.data_path}
output_type: generate_until
doc_to_visual: !function utils.textvqa_doc_to_visual
doc_to_text: !function utils.textvqa_doc_to_text
doc_to_target: "answer"
generation_kwargs:
  until:
    - "ASSISTANT:"
process_results: !function utils.textvqa_process_results
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    ignore_punctuation: true
  - metric: submission
    aggregation: !function utils.textvqa_aggregate_submissions
    higher_is_better: true
lmms_eval_specific_kwargs:
  default:
    pre_prompt: ""
    post_prompt: "\\nAnswer the question using a single word or phrase."
    ocr: true
    ocr_max_tokens: {args.max_tokens}
    ocr_strategy: {args.strategy}
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    source_utils = Path(__file__).resolve().parents[1] / "lmms-eval" / "lmms_eval" / "tasks" / "textvqa" / "utils.py"
    if output.parent != source_utils.parent:
        shutil.copy2(source_utils, output.parent / "utils.py")
    print(f"[INFO] Wrote {output}")


if __name__ == "__main__":
    main()
