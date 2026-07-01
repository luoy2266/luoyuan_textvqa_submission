#!/usr/bin/env python
import argparse
import json
import os
import re
import time
from glob import glob

import torch
import yaml
from datasets import Dataset
from datasets import load_from_disk
from peft import AdaLoraConfig, LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForVision2Seq, AutoProcessor, Trainer, TrainerCallback, TrainingArguments, set_seed


DEFAULT_CONFIG = "configs/vlm_textvqa_lora.yaml"
SYSTEM_PROMPT = "You are a helpful assistant."
LM_ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
LM_MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]
VISION_MODULES = ["qkv", "proj", "linear_fc1", "linear_fc2"]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = int(os.getenv("SEED", cfg.get("seed", 1)))
    cfg["seed"] = seed
    cfg["model_path"] = os.getenv("BASE_MODEL", os.getenv("MODEL_PATH", cfg["model_path"]))
    cfg["data_path"] = os.getenv("TEXTVQA_DATA_PATH", cfg["data_path"])
    cfg["output_dir"] = os.getenv("OUTPUT_DIR", cfg["output_dir"]).format(seed=seed)
    cfg["prepared_data_dir"] = os.getenv("PREPARED_DATA_DIR", cfg["prepared_data_dir"]).format(seed=seed)
    return cfg


def load_prepared_dataset(cfg):
    if not os.path.isdir(cfg["prepared_data_dir"]):
        prepared_data_dir = cfg["prepared_data_dir"]
        raise FileNotFoundError(
            "Prepared dataset not found: {}. Run `python prepare_textvqa.py --config {}` first.".format(
                prepared_data_dir, DEFAULT_CONFIG
            )
        )
    ds = load_from_disk(cfg["prepared_data_dir"])
    print("[INFO] Loaded {} prepared TextVQA samples from {}".format(len(ds), cfg["prepared_data_dir"]))
    return ds


def load_raw_dataset(cfg):
    files = sorted(glob(cfg["data_path"]))
    if not files:
        data_path = cfg["data_path"]
        raise FileNotFoundError("No parquet files matched data_path={}".format(data_path))
    ds = Dataset.from_parquet(files)
    ds = ds.shuffle(seed=cfg["seed"])
    max_samples = int(cfg.get("max_train_samples", 0))
    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))
    print("[INFO] Loaded {} raw TextVQA samples from {}".format(len(ds), cfg["data_path"]))
    return ds


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _module_suffix(name):
    return name.rsplit(".", 1)[-1]


def _language_layer_id(name):
    match = re.search(r"(?:^|\.)language_model\.layers\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def _vision_block_id(name):
    match = re.search(r"(?:^|\.)visual\.blocks\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def _last_k_layers(total_layers, last_k):
    last_k = int(last_k)
    if last_k <= 0 or last_k >= total_layers:
        return set(range(total_layers))
    return set(range(total_layers - last_k, total_layers))


def resolve_lora_target_modules(model, cfg):
    """Resolve exact module names so PEFT experiments are reproducible."""
    preset = cfg.get("target_preset", "custom")
    if preset == "custom":
        return list(cfg["target_modules"])

    text_config = getattr(model.config, "text_config", None)
    vision_config = getattr(model.config, "vision_config", None)
    num_lm_layers = int(getattr(text_config, "num_hidden_layers", cfg.get("num_lm_layers", 28)))
    num_vision_layers = int(getattr(vision_config, "depth", cfg.get("num_vision_layers", 24)))

    if "language_target_modules" in cfg:
        lm_suffixes = list(cfg["language_target_modules"])
    elif preset == "lm_attn_only":
        lm_suffixes = LM_ATTN_MODULES
    elif preset == "lm_mlp_only":
        lm_suffixes = LM_MLP_MODULES
    else:
        lm_suffixes = LM_ATTN_MODULES + LM_MLP_MODULES

    lm_layers = set(range(num_lm_layers))
    if "language_layers" in cfg:
        lm_layers = {int(layer) for layer in cfg["language_layers"]}
    elif "language_last_k" in cfg:
        lm_layers = _last_k_layers(num_lm_layers, int(cfg["language_last_k"]))

    if "include_projector_lora" in cfg:
        include_projector = _as_bool(cfg.get("include_projector_lora"), False)
    else:
        include_projector = preset == "lm_projector"
    vision_last_k = int(cfg.get("vision_last_k", 0))
    include_vision = vision_last_k > 0 or preset == "lm_vision_last_k"
    if include_vision and vision_last_k <= 0:
        vision_last_k = int(cfg.get("default_vision_last_k", 4))
    vision_layers = _last_k_layers(num_vision_layers, vision_last_k) if include_vision else set()
    vision_suffixes = list(cfg.get("vision_target_modules", VISION_MODULES))

    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        lm_layer = _language_layer_id(name)
        if lm_layer is not None and lm_layer in lm_layers and _module_suffix(name) in lm_suffixes:
            targets.append(name)
            continue

        if include_projector and name in {"model.visual.merger.linear_fc1", "model.visual.merger.linear_fc2"}:
            targets.append(name)
            continue

        vision_layer = _vision_block_id(name)
        if include_vision and vision_layer in vision_layers and _module_suffix(name) in vision_suffixes:
            targets.append(name)

    if not targets:
        raise ValueError("No LoRA target modules matched target_preset={!r}".format(preset))
    return targets


def build_peft_config(cfg, lora_targets):
    rank_pattern = {str(k): int(v) for k, v in cfg.get("rank_pattern", {}).items()}
    alpha_pattern = {str(k): int(v) for k, v in cfg.get("alpha_pattern", {}).items()}
    peft_type = str(cfg.get("peft_type", "lora")).strip().lower()

    if rank_pattern:
        print("[INFO] LoRA rank_pattern={}".format(rank_pattern), flush=True)
    if alpha_pattern:
        print("[INFO] LoRA alpha_pattern={}".format(alpha_pattern), flush=True)

    common_kwargs = dict(
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=lora_targets,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=_as_bool(cfg.get("use_rslora"), False),
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
    )

    if peft_type == "adalora":
        if _as_bool(cfg.get("use_dora"), False):
            raise ValueError("AdaLoRA does not support DoRA; set use_dora: false.")
        return AdaLoraConfig(
            **common_kwargs,
            target_r=int(cfg.get("adalora_target_r", cfg.get("target_r", 16))),
            init_r=int(cfg.get("adalora_init_r", cfg.get("init_r", 32))),
            tinit=int(cfg.get("adalora_tinit", cfg.get("tinit", 128))),
            tfinal=int(cfg.get("adalora_tfinal", cfg.get("tfinal", 128))),
            deltaT=int(cfg.get("adalora_deltaT", cfg.get("deltaT", 8))),
            beta1=float(cfg.get("adalora_beta1", cfg.get("beta1", 0.85))),
            beta2=float(cfg.get("adalora_beta2", cfg.get("beta2", 0.85))),
            orth_reg_weight=float(cfg.get("adalora_orth_reg_weight", cfg.get("orth_reg_weight", 0.5))),
            total_step=int(cfg.get("adalora_total_step", cfg["max_steps"])),
        )

    if peft_type != "lora":
        raise ValueError("Unsupported peft_type={!r}; expected 'lora' or 'adalora'.".format(peft_type))

    return LoraConfig(
        **common_kwargs,
        r=int(cfg["lora_r"]),
        use_dora=_as_bool(cfg.get("use_dora"), False),
    )


def get_adalora_rank_pattern(model):
    model_to_read = model.module if hasattr(model, "module") else model
    peft_config = getattr(model_to_read, "peft_config", {})
    if isinstance(peft_config, dict):
        config = peft_config.get("default")
    else:
        config = None
    rank_pattern = getattr(config, "rank_pattern", None)
    if not rank_pattern:
        return {}
    return {str(key): _json_safe_value(value) for key, value in rank_pattern.items()}


def _json_safe_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


class TimeLimitCallback(TrainerCallback):
    def __init__(self, max_seconds):
        self.max_seconds = max_seconds
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.max_seconds > 0 and time.time() - self.start_time > self.max_seconds:
            print("[TIMEOUT] Reached {:.1f} minute training budget".format(self.max_seconds / 60))
            control.should_training_stop = True
        return control


class SavePeftStepCallback(TrainerCallback):
    def __init__(self, cfg):
        self.cfg = cfg
        self.save_steps = set(int(step) for step in cfg.get("save_lora_steps", []) if int(step) > 0)
        self.saved_steps = set()

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step not in self.save_steps or step in self.saved_steps:
            return control
        if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero:
            return control
        if hasattr(args, "process_index") and args.process_index != 0:
            return control

        model = kwargs.get("model")
        if model is None:
            return control
        model_to_save = model.module if hasattr(model, "module") else model

        checkpoint_dir = os.path.join(self.cfg["output_dir"], "step{}".format(step))
        os.makedirs(checkpoint_dir, exist_ok=True)
        print("[INFO] Saving LoRA checkpoint at step {} to {}".format(step, checkpoint_dir), flush=True)
        model_to_save.save_pretrained(checkpoint_dir)
        with open(os.path.join(checkpoint_dir, "training_config.json"), "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, indent=2, sort_keys=True)
        with open(os.path.join(checkpoint_dir, "_SUCCESS"), "w", encoding="utf-8") as f:
            f.write("step={}\n".format(step))
        self.saved_steps.add(step)
        return control


class AdaLoraUpdateCallback(TrainerCallback):
    def __init__(self, enabled):
        self.enabled = enabled

    def on_optimizer_step(self, args, state, control, **kwargs):
        if not self.enabled:
            return control

        model = kwargs.get("model")
        if model is None:
            return control
        model_to_update = model.module if hasattr(model, "module") else model
        base_model = getattr(model_to_update, "base_model", None)
        if base_model is None or not hasattr(base_model, "update_and_allocate"):
            return control

        # Transformers calls this after optimizer.step() and before gradients are cleared,
        # which is the timing AdaLoRA's rank allocator expects.
        base_model.update_and_allocate(int(state.global_step) + 1)
        return control


class TextVQADataset(torch.utils.data.Dataset):
    def __init__(self, raw_ds, prepared_ds, processor, cfg):
        if len(raw_ds) != len(prepared_ds):
            raise ValueError(
                "Raw dataset length {} does not match prepared dataset length {}".format(
                    len(raw_ds), len(prepared_ds)
                )
            )
        self.raw_ds = raw_ds
        self.prepared_ds = prepared_ds
        self.processor = processor
        self.cfg = cfg

    def __len__(self):
        return len(self.prepared_ds)

    def __getitem__(self, idx):
        raw_item = self.raw_ds[idx]
        item = self.prepared_ds[idx]
        image = raw_item["image"].convert("RGB")
        answer = item["target_answer"]
        user_text = item["user_text"]

        prompt_conv = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_text}]},
        ]
        full_conv = prompt_conv + [{"role": "assistant", "content": answer}]

        prompt_text = self.processor.apply_chat_template(prompt_conv, tokenize=False, add_generation_prompt=True)
        full_text = self.processor.apply_chat_template(full_conv, tokenize=False, add_generation_prompt=False)

        common_kwargs = dict(
            images=[image],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=int(self.cfg.get("max_seq_length", 1024)),
        )
        prompt_batch = self.processor(text=prompt_text, **common_kwargs)
        full_batch = self.processor(text=full_text, **common_kwargs)

        input_ids = full_batch["input_ids"][0]
        attention_mask = full_batch["attention_mask"][0]
        labels = input_ids.clone()
        labels[: min(prompt_batch["input_ids"].shape[1], labels.shape[0])] = -100

        result = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        if "pixel_values" in full_batch:
            result["pixel_values"] = full_batch["pixel_values"]
        if "image_grid_thw" in full_batch:
            result["image_grid_thw"] = full_batch["image_grid_thw"]
        return result


def collate_fn(examples, processor):
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id

    batch = {
        "input_ids": pad_sequence([ex["input_ids"] for ex in examples], batch_first=True, padding_value=pad_id),
        "attention_mask": pad_sequence([ex["attention_mask"] for ex in examples], batch_first=True, padding_value=0),
        "labels": pad_sequence([ex["labels"] for ex in examples], batch_first=True, padding_value=-100),
    }

    if "pixel_values" in examples[0]:
        batch["pixel_values"] = torch.cat([ex["pixel_values"] for ex in examples], dim=0)
    if "image_grid_thw" in examples[0]:
        batch["image_grid_thw"] = torch.cat([ex["image_grid_thw"] for ex in examples], dim=0)
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    processor = AutoProcessor.from_pretrained(
        cfg["model_path"],
        trust_remote_code=True,
        max_pixels=int(cfg["max_pixels"]),
        min_pixels=int(cfg["min_pixels"]),
    )
    model = AutoModelForVision2Seq.from_pretrained(
        cfg["model_path"],
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation=cfg.get("attn_implementation", "eager"),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    if hasattr(model, "visual"):
        for param in model.visual.parameters():
            param.requires_grad = False

    lora_targets = resolve_lora_target_modules(model, cfg)
    cfg["resolved_target_modules"] = lora_targets
    print(
        "[INFO] LoRA target_preset={} matched {} modules".format(
            cfg.get("target_preset", "custom"), len(lora_targets)
        ),
        flush=True,
    )
    for target in lora_targets[:12]:
        print("[INFO]   target: {}".format(target), flush=True)
    if len(lora_targets) > 12:
        print("[INFO]   ... {} more targets".format(len(lora_targets) - 12), flush=True)

    lora_config = build_peft_config(cfg, lora_targets)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    prepared_ds = load_prepared_dataset(cfg)
    raw_ds = load_raw_dataset(cfg)
    train_ds = TextVQADataset(raw_ds, prepared_ds, processor, cfg)

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        max_steps=int(cfg["max_steps"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        weight_decay=float(cfg["weight_decay"]),
        logging_steps=int(cfg["logging_steps"]),
        save_strategy="no",
        fp16=True,
        bf16=False,
        dataloader_num_workers=int(cfg["dataloader_num_workers"]),
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=lambda examples: collate_fn(examples, processor),
        callbacks=[
            TimeLimitCallback(int(cfg.get("max_train_seconds", 0))),
            SavePeftStepCallback(cfg),
            AdaLoraUpdateCallback(str(cfg.get("peft_type", "lora")).strip().lower() == "adalora"),
        ],
    )
    trainer.train()

    final_dir = os.path.join(cfg["output_dir"], "final")
    os.makedirs(final_dir, exist_ok=True)
    is_main_process = trainer.is_world_process_zero()
    if str(cfg.get("peft_type", "lora")).strip().lower() == "adalora":
        cfg["adalora_rank_pattern"] = get_adalora_rank_pattern(model)
        if is_main_process:
            with open(os.path.join(final_dir, "adalora_rank_pattern.json"), "w", encoding="utf-8") as f:
                json.dump(cfg["adalora_rank_pattern"], f, indent=2, sort_keys=True)
    trainer.save_model(final_dir)
    if is_main_process:
        processor.save_pretrained(final_dir)
        with open(os.path.join(final_dir, "training_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
