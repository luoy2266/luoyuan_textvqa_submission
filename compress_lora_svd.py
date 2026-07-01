#!/usr/bin/env python
import argparse
import json
import math
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def lora_scaling(alpha, rank, use_rslora):
    rank = int(rank)
    if use_rslora:
        return float(alpha) / math.sqrt(rank)
    return float(alpha) / rank


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def compress_pair(a_weight, b_weight, old_scale, new_scale, new_rank):
    """Best rank-k approximation of old_scale * B @ A, returned as new A/B."""
    a = a_weight.float()
    b = b_weight.float()
    old_rank = a.shape[0]
    if new_rank > old_rank:
        raise ValueError(f"new rank {new_rank} cannot exceed old rank {old_rank}")

    # D = B @ A has rank at most old_rank.  QR + tiny SVD gives the exact SVD
    # of D without materializing the full output-by-input delta matrix.
    qb, rb = torch.linalg.qr(b, mode="reduced")
    qa, ra = torch.linalg.qr(a.T, mode="reduced")
    small = rb @ ra.T
    u_small, singular_values, vh_small = torch.linalg.svd(small, full_matrices=False)

    k = int(new_rank)
    u = qb @ u_small[:, :k]
    vh = vh_small[:k, :] @ qa.T
    scaled_s = singular_values[:k] * (float(old_scale) / float(new_scale))
    roots = torch.sqrt(torch.clamp(scaled_s, min=0.0))

    b_new = u * roots.unsqueeze(0)
    a_new = roots.unsqueeze(1) * vh
    return a_new.to(dtype=a_weight.dtype), b_new.to(dtype=b_weight.dtype), singular_values


def copy_sidecar_files(input_dir, output_dir):
    skip = {"adapter_model.safetensors", "adapter_config.json"}
    for path in input_dir.iterdir():
        if not path.is_file() or path.name in skip:
            continue
        shutil.copy2(path, output_dir / path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-adapter", required=True)
    parser.add_argument("--output-adapter", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--use-rslora", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_adapter)
    output_dir = Path(args.output_adapter)
    if not (input_dir / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Missing adapter_model.safetensors under {input_dir}")
    if not (input_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing adapter_config.json under {input_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output adapter dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with (input_dir / "adapter_config.json").open("r", encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    old_rank = int(adapter_cfg["r"])
    old_alpha = float(adapter_cfg["lora_alpha"])
    old_use_rslora = _as_bool(adapter_cfg.get("use_rslora", False))
    new_use_rslora = bool(args.use_rslora)
    old_scale = lora_scaling(old_alpha, old_rank, old_use_rslora)
    new_scale = lora_scaling(args.alpha, args.rank, new_use_rslora)

    state = load_file(str(input_dir / "adapter_model.safetensors"), device="cpu")
    new_state = {}
    total_energy = 0.0
    kept_energy = 0.0
    compressed_pairs = 0

    for key in sorted(state):
        if not key.endswith(".lora_A.weight"):
            continue
        b_key = key.replace(".lora_A.weight", ".lora_B.weight")
        if b_key not in state:
            raise KeyError(f"Missing matching B weight for {key}")

        a_new, b_new, singular_values = compress_pair(
            state[key], state[b_key], old_scale=old_scale, new_scale=new_scale, new_rank=args.rank
        )
        new_state[key] = a_new
        new_state[b_key] = b_new
        compressed_pairs += 1
        sq = singular_values.square()
        total_energy += float(sq.sum())
        kept_energy += float(sq[: args.rank].sum())

    for key, tensor in state.items():
        if key not in new_state:
            new_state[key] = tensor

    adapter_cfg["r"] = int(args.rank)
    adapter_cfg["lora_alpha"] = int(args.alpha) if float(args.alpha).is_integer() else float(args.alpha)
    adapter_cfg["use_rslora"] = new_use_rslora
    adapter_cfg["rank_pattern"] = {}
    adapter_cfg["alpha_pattern"] = {}
    adapter_cfg["inference_mode"] = True

    save_file(new_state, str(output_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    with (output_dir / "adapter_config.json").open("w", encoding="utf-8") as f:
        json.dump(adapter_cfg, f, indent=2, sort_keys=True)
    copy_sidecar_files(input_dir, output_dir)

    compression_meta = {
        "method": "svd_low_rank_product",
        "input_adapter": str(input_dir),
        "output_adapter": str(output_dir),
        "old_rank": old_rank,
        "old_alpha": old_alpha,
        "old_use_rslora": old_use_rslora,
        "old_scaling": old_scale,
        "new_rank": int(args.rank),
        "new_alpha": adapter_cfg["lora_alpha"],
        "new_use_rslora": new_use_rslora,
        "new_scaling": new_scale,
        "compressed_pairs": compressed_pairs,
        "energy_kept_ratio": kept_energy / total_energy if total_energy else None,
    }
    with (output_dir / "compression_config.json").open("w", encoding="utf-8") as f:
        json.dump(compression_meta, f, indent=2, sort_keys=True)

    training_config_path = output_dir / "training_config.json"
    if training_config_path.is_file():
        with training_config_path.open("r", encoding="utf-8") as f:
            training_cfg = json.load(f)
        training_cfg["lora_r"] = int(args.rank)
        training_cfg["lora_alpha"] = adapter_cfg["lora_alpha"]
        training_cfg["use_rslora"] = new_use_rslora
        training_cfg["compression"] = compression_meta
        with training_config_path.open("w", encoding="utf-8") as f:
            json.dump(training_cfg, f, indent=2, sort_keys=True)

    with (output_dir / "README.md").open("a", encoding="utf-8") as f:
        f.write("\n\nSVD compression\n")
        f.write("================\n")
        f.write(
            f"Compressed from rank {old_rank} to rank {args.rank} with "
            f"alpha={adapter_cfg['lora_alpha']} use_rslora={new_use_rslora}.\n"
        )
        if compression_meta["energy_kept_ratio"] is not None:
            f.write(f"Global squared singular-value energy kept: {compression_meta['energy_kept_ratio']:.6f}\n")

    print(json.dumps(compression_meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
