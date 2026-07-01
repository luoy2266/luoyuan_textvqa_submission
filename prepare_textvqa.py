#!/usr/bin/env python
import argparse
import json
import os
from collections import Counter
from glob import glob

import yaml
from datasets import Dataset

from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor
from lmms_eval.tasks.textvqa.ocr_selection import select_ocr_tokens


DEFAULT_CONFIG = "configs/vlm_textvqa_lora.yaml"
EVAL_ANSWER_PROCESSOR = EvalAIAnswerProcessor()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = int(os.getenv("SEED", cfg.get("seed", 1)))
    cfg["seed"] = seed
    cfg["data_path"] = os.getenv("TEXTVQA_DATA_PATH", cfg["data_path"])
    cfg["prepared_data_dir"] = os.getenv("PREPARED_DATA_DIR", cfg["prepared_data_dir"]).format(seed=seed)
    return cfg


def normalize_answer(answer):
    return EVAL_ANSWER_PROCESSOR(answer)


def textvqa_accuracy(candidate, references):
    scores = []
    for i in range(len(references)):
        other_answers = [references[j] for j in range(len(references)) if i != j]
        scores.append(min(1.0, other_answers.count(candidate) / 3.0))
    return sum(scores) / len(scores) if scores else 0.0


def choose_answer(answers):
    if not isinstance(answers, list):
        return normalize_answer(answers)
    normalized = [normalize_answer(ans) for ans in answers if str(ans).strip()]
    if not normalized:
        return ""
    counts = Counter(normalized)
    return max(counts, key=lambda ans: (textvqa_accuracy(ans, normalized), counts[ans], -normalized.index(ans)))


def build_question(item, use_ocr, max_ocr_tokens, ocr_strategy):
    question = item["question"].strip().capitalize()
    if use_ocr:
        ocr_tokens = select_ocr_tokens(
            question=item["question"],
            ocr_tokens=item.get("ocr_tokens", []),
            strategy=ocr_strategy,
            max_tokens=max_ocr_tokens,
        )
        if ocr_tokens:
            question += "\nReference OCR token: " + ", ".join(ocr_tokens)
    return question + "\nAnswer the question using a single word or phrase."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.config)

    files = sorted(glob(cfg["data_path"]))
    if not files:
        data_path = cfg["data_path"]
        raise FileNotFoundError("No parquet files matched data_path={}".format(data_path))

    ds = Dataset.from_parquet(files)
    ds = ds.shuffle(seed=cfg["seed"])
    max_samples = int(cfg.get("max_train_samples", 0))
    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    use_ocr = bool(cfg.get("use_ocr_tokens", True))
    max_ocr_tokens = int(cfg.get("max_ocr_tokens", 16))
    ocr_strategy = cfg.get("ocr_strategy", "first")

    compact_rows = {"target_answer": [], "user_text": []}
    for item in ds.remove_columns(["image"]):
        compact_rows["target_answer"].append(choose_answer(item["answers"]))
        compact_rows["user_text"].append(build_question(item, use_ocr, max_ocr_tokens, ocr_strategy))

    ds = Dataset.from_dict(compact_rows)
    keep_columns = list(ds.column_names)

    os.makedirs(os.path.dirname(cfg["prepared_data_dir"]), exist_ok=True)
    ds.save_to_disk(cfg["prepared_data_dir"])

    manifest = {
        "source_data_path": cfg["data_path"],
        "prepared_data_dir": cfg["prepared_data_dir"],
        "seed": cfg["seed"],
        "num_samples": len(ds),
        "use_ocr_tokens": use_ocr,
        "max_ocr_tokens": max_ocr_tokens,
        "ocr_strategy": ocr_strategy,
        "saved_columns": keep_columns,
    }
    with open(os.path.join(cfg["prepared_data_dir"], "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print("[INFO] Prepared {} samples at {}".format(len(ds), cfg["prepared_data_dir"]))


if __name__ == "__main__":
    main()
