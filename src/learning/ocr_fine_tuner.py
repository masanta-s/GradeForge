"""Mechanism 3: LoRA fine-tuning of TrOCR on a teacher's OCR corrections.

TrOCR is a fixed component, so this always works. A checkpoint is only promoted if its CER
(character error rate) on held-out corrected lines beats the current model: otherwise the
"it learns!" demo could quietly make recognition worse. Base vs tuned CER is logged either way.

Details that matter (measured on transformers 5.17, 2026-09-10):
- Labels must match how TrOCR was pretrained: generation starts with </s> (id 2) then " text"
  WITH a leading space and no <s>, e.g. [2, 7841 (" Cell"), ...]. The tokenizer's default
  ([0 <s>, 40216 "Cell", ...]) would spend a tiny dataset re-learning the output format.
- VisionEncoderDecoderModel.forward() builds decoder inputs from config.decoder_start_token_id
  and config.pad_token_id, which transformers 5 no longer puts on the config: set them.
- Encoder and decoder attention both use q_proj / v_proj names in transformers 5.
"""
from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src import config

ACTIVE_FILE = "active.json"
MIN_SAMPLES = 20
MIN_IMPROVEMENT = 0.005  # absolute CER; smaller gains are within noise on a few held-out lines


@dataclass(frozen=True)
class OCRTrainingResult:
    version: str
    baseline: str              # "base" or the previously active version
    baseline_cer: float
    tuned_cer: float
    promoted: bool
    num_train: int
    num_heldout: int
    seconds: float
    adapter_dir: str


def should_promote(baseline_cer: float, tuned_cer: float) -> bool:
    """Only a clear improvement replaces the model in use; ties and noise keep the old one."""
    return tuned_cer < baseline_cer - MIN_IMPROVEMENT


def active_adapter(root: Path = config.TROCR_FINETUNED_DIR) -> Path | None:
    path = Path(root) / ACTIVE_FILE
    if not path.exists():
        return None
    adapter = Path(root) / json.loads(path.read_text(encoding="utf-8"))["version"]
    return adapter if adapter.exists() else None


def _next_version(root: Path) -> str:
    existing = [int(p.name[1:]) for p in root.glob("v*") if p.name[1:].isdigit()]
    return f"v{max(existing, default=0) + 1}"


def _augment(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Small scale/rotation/contrast jitter so a few dozen lines don't simply get memorised."""
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-1.5, 1.5), rng.uniform(0.92, 1.08))
    out = cv2.warpAffine(image, matrix, (w, h), borderValue=255)
    alpha, beta = rng.uniform(0.85, 1.15), rng.uniform(-15, 15)
    return np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def _labels(tokenizer, texts: list[str], eos_id: int, pad_id: int):
    import torch

    ids = [tokenizer(" " + t.strip(), add_special_tokens=False).input_ids + [eos_id] for t in texts]
    width = max(map(len, ids))
    labels = torch.full((len(ids), width), -100, dtype=torch.long)  # -100 = ignored by the loss
    for row, seq in enumerate(ids):
        labels[row, :len(seq)] = torch.tensor(seq)
    return labels


def _cer(extractor, samples: list[tuple[np.ndarray, str]]) -> float:
    import jiwer

    readings = extractor.read_lines([img for img, _ in samples])
    return float(jiwer.cer([t for _, t in samples], [r.text for r in readings]))


def train_trocr_lora(
    samples: list[tuple[np.ndarray, str]],
    *,
    heldout: list[tuple[np.ndarray, str]] | None = None,
    root: Path = config.TROCR_FINETUNED_DIR,
    base_dir: Path = config.TROCR_BASE_DIR,
    epochs: int = 12,
    learning_rate: float = 5e-4,
    batch_size: int = 8,
    heldout_fraction: float = 0.25,
    min_heldout: int = 5,
    seed: int = 0,
    progress: Callable[[float, str], None] | None = None,
) -> OCRTrainingResult:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import VisionEncoderDecoderModel

    from src.ocr.text_extractor import TrOCRExtractor, _to_pil, load_trocr_processor

    if len(samples) + len(heldout or []) < MIN_SAMPLES:
        raise ValueError(f"need at least {MIN_SAMPLES} lines to train on, have {len(samples)}")
    report = progress or (lambda p, m: None)
    started = time.time()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    if heldout:
        # Given explicitly when a public dataset is mixed in: accuracy is judged on the teacher's
        # own students' lines, so extra data can never buy a promotion by itself.
        train = list(samples)
    else:
        order = list(range(len(samples)))
        rng.shuffle(order)
        n_held = max(min_heldout, round(len(samples) * heldout_fraction))
        heldout = [samples[i] for i in order[:n_held]]
        train = [samples[i] for i in order[n_held:]]

    # Baseline: whatever is in use now (a previous fine-tune, or the base model).
    current = active_adapter(root)
    report(0.02, "Measuring current accuracy")
    baseline_extractor = TrOCRExtractor(base_dir, adapter_dir=current)
    baseline_cer = _cer(baseline_extractor, heldout)
    baseline_extractor.unload()

    processor = load_trocr_processor(base_dir)
    model = VisionEncoderDecoderModel.from_pretrained(base_dir)
    gen = model.generation_config
    model.config.decoder_start_token_id = gen.decoder_start_token_id
    model.config.pad_token_id = gen.pad_token_id
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"]))
    model.to("cuda").train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)

    steps = epochs * ((len(train) + batch_size - 1) // batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=steps,
                                                    pct_start=0.1)
    step = 0
    for epoch in range(epochs):
        rng.shuffle(train)
        for start in range(0, len(train), batch_size):
            batch = train[start:start + batch_size]
            images = [_to_pil(_augment(img, rng)) for img, _ in batch]
            pixel_values = processor(images=images, return_tensors="pt").pixel_values.to("cuda")
            labels = _labels(processor.tokenizer, [t for _, t in batch], gen.eos_token_id, gen.pad_token_id).to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(pixel_values=pixel_values, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            report(0.05 + 0.8 * step / steps, f"Training on this handwriting (epoch {epoch + 1}/{epochs}, loss {loss.item():.2f})")

    version = _next_version(root)
    adapter_dir = root / version
    model.save_pretrained(adapter_dir)
    del model, optimizer
    torch.cuda.empty_cache()

    report(0.9, "Measuring new accuracy")
    tuned_extractor = TrOCRExtractor(base_dir, adapter_dir=adapter_dir)
    tuned_cer = _cer(tuned_extractor, heldout)
    tuned_extractor.unload()

    promoted = should_promote(baseline_cer, tuned_cer)
    if promoted:
        (root / ACTIVE_FILE).write_text(json.dumps({"version": version, "cer": tuned_cer}), encoding="utf-8")
    return OCRTrainingResult(version, current.name if current else "base", baseline_cer, tuned_cer, promoted,
                             len(train), len(heldout), round(time.time() - started, 1), str(adapter_dir))
