"""TrOCR line recognition with per-line confidence.

Confidence = geometric-mean probability of the generated tokens (greedy decoding), plus the
single least-confident token. Low-confidence lines are queued for the VLM cross-check in the
LLM pass (plan: VRAM Budget / Gap #3). Models load from local directories only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, RobertaTokenizer, TrOCRProcessor, VisionEncoderDecoderModel

from src import config


_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?)\]])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[])\s+")


def clean_text(text: str) -> str:
    """TrOCR's RoBERTa BPE emits 'sunlight , water' and '( a )' — tighten the spacing."""
    return _SPACE_AFTER_OPEN.sub(r"\1", _SPACE_BEFORE_PUNCT.sub(r"\1", text)).strip()


@dataclass(frozen=True)
class LineReading:
    text: str
    confidence: float      # geometric-mean token probability, 0..1
    min_token_prob: float  # weakest single token, 0..1


def load_trocr_processor(model_dir: Path) -> TrOCRProcessor:
    """Build the processor from its parts.

    `TrOCRProcessor.from_pretrained` fails on transformers 5: TrOCR's config maps to no
    tokenizer class, and the checkpoint ships only vocab.json + merges.txt (no tokenizer.json),
    so the generic fallback can't build one. The RoBERTa BPE tokenizer loads fine directly.
    """
    return TrOCRProcessor(
        image_processor=AutoImageProcessor.from_pretrained(model_dir),
        tokenizer=RobertaTokenizer.from_pretrained(model_dir),
    )


def _to_pil(image: np.ndarray | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")
    return Image.fromarray(image[:, :, ::-1])  # OpenCV BGR -> RGB


class TrOCRExtractor:
    def __init__(
        self,
        model_dir: Path = config.TROCR_BASE_DIR,
        adapter_dir: Path | None = None,
        device: str | None = None,
        batch_size: int = 8,
        max_new_tokens: int = 64,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

        self.processor = load_trocr_processor(model_dir)
        model = VisionEncoderDecoderModel.from_pretrained(model_dir, dtype=self.dtype)
        if adapter_dir is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
        self.model = model.to(self.device).eval()

    @torch.inference_mode()
    def read_lines(self, images: list[np.ndarray | Image.Image]) -> list[LineReading]:
        readings: list[LineReading] = []
        for start in range(0, len(images), self.batch_size):
            batch = [_to_pil(img) for img in images[start:start + self.batch_size]]
            pixel_values = self.processor(images=batch, return_tensors="pt").pixel_values
            output = self.model.generate(
                pixel_values.to(self.device, self.dtype),
                max_new_tokens=self.max_new_tokens,
                num_beams=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
            log_probs = self.model.compute_transition_scores(
                output.sequences, output.scores, normalize_logits=True
            ).float().cpu()
            texts = self.processor.batch_decode(output.sequences, skip_special_tokens=True)

            generated = output.sequences[:, 1:].cpu()  # drop decoder start token
            pad_id = self.model.generation_config.pad_token_id
            eos_id = self.model.generation_config.eos_token_id
            for text, token_ids, scores in zip(texts, generated, log_probs):
                keep = token_ids != pad_id
                if eos_id is not None:  # score the EOS itself, not the padding after it
                    keep |= token_ids == eos_id
                valid = scores[keep[: scores.shape[0]]]
                valid = valid[torch.isfinite(valid)]
                if valid.numel() == 0:
                    readings.append(LineReading(clean_text(text), 0.0, 0.0))
                    continue
                readings.append(LineReading(
                    text=clean_text(text),
                    confidence=float(valid.mean().exp()),
                    min_token_prob=float(valid.min().exp()),
                ))
        return readings

    def unload(self) -> None:
        """Free GPU memory before the LLM pass or TrOCR fine-tuning."""
        del self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()
