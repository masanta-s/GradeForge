"""Merge a LoRA adapter into the original weights, one safetensors shard at a time.

Ollama 0.34 can only apply LoRA adapters to llama and gemma2 models (convert.ConvertAdapter), so
for Qwen 3.5 and Gemma 4 the adapter is merged into a full copy of the weights, which Ollama then
imports and quantises itself (its converter lists Qwen3_5ForConditionalGeneration and
Gemma4ForConditionalGeneration; verified with a tiny Qwen3.5 model, 2026-09-11).

Memory stays around one shard (~5 GB for Qwen3.5-9B), so a 32 GB laptop can merge a 19 GB model.
"""
from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import torch

_COPY = ("*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken")


def _adapter(adapter_dir: Path) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], float]:
    from safetensors.torch import load_file

    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    rank, alpha = config["r"], config["lora_alpha"]
    scale = alpha / (rank ** 0.5) if config.get("use_rslora") else alpha / rank
    state = load_file(adapter_dir / "adapter_model.safetensors")
    pairs: dict[str, list] = {}
    for key, tensor in state.items():
        for part in ("lora_A", "lora_B"):
            marker = f".{part}."
            if marker in key:
                module = key.removeprefix("base_model.model.").split(marker)[0]
                pairs.setdefault(module, [None, None])["AB".index(part[-1])] = tensor
    missing = [m for m, (a, b) in pairs.items() if a is None or b is None]
    if missing:
        raise ValueError(f"adapter is incomplete: {missing[:3]}")
    return {m: (a, b) for m, (a, b) in pairs.items()}, scale


def merge_adapter(weights_dir: Path, adapter_dir: Path, out_dir: Path,
                  progress: Callable[[float, str], None] | None = None) -> Path:
    from safetensors import safe_open
    from safetensors.torch import save_file

    report = progress or (lambda p, m: None)
    weights_dir, adapter_dir, out_dir = Path(weights_dir), Path(adapter_dir), Path(out_dir)
    pairs, scale = _adapter(adapter_dir)
    shards = sorted(weights_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors weights in {weights_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = set()
    for i, shard in enumerate(shards):
        report(i / len(shards), f"Merging {shard.name} ({i + 1} of {len(shards)})")
        tensors = {}
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            metadata = f.metadata() or {"format": "pt"}
            for key in f.keys():
                tensor = f.get_tensor(key)
                module = key.removesuffix(".weight")
                if key.endswith(".weight") and module in pairs:
                    a, b = pairs[module]
                    tensor = (tensor.float() + scale * (b.float() @ a.float())).to(tensor.dtype)
                    merged.add(module)
                tensors[key] = tensor
        save_file(tensors, out_dir / shard.name, metadata=metadata)
        del tensors
    unmatched = set(pairs) - merged
    if unmatched:
        raise ValueError(f"{len(unmatched)} adapter layers matched no weight, e.g. {sorted(unmatched)[:2]}")
    for pattern in _COPY:
        for file in weights_dir.glob(pattern):
            shutil.copy2(file, out_dir / file.name)
    report(1.0, f"Merged {len(merged)} layers")
    return out_dir
