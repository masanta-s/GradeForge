"""Where (and whether) the grading LLM can be fine-tuned on the teacher's corrections.

Checks, in order: a trainable HF source and supported architecture (layers 2-3), the training
method (QLoRA unless the registry says the family shouldn't be 4-bit trained), training VRAM
(documented figure from the registry, else estimated from the parameter count), and then each
place it could run:
  local   the pinned .venv-train on this GPU: needs the VRAM plus 40-60 GB of transient disk
  colab   free T4, 15 GB
  kaggle  free 2x T4, but they are two separate 15 GB GPUs, not one 30 GB pool
The same model can be refused here and routed locally on a 24 GB GPU with no code change.
"""
from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import asdict, dataclass

from src import config
from src.learning.llm_finetune_export import RECOMMENDED_CORRECTIONS
from src.models.arch_gate import TRAIN_ENV_PYTHON, TrainabilityResult
from src.models.override_registry import Overrides

FREE_TIER_GB = 15
LOCAL_DISK_GB = 60
VRAM_MARGIN_GB = 0.5
STREAM_MIN_VRAM_GB = 6.0      # one streamed layer + a 4k-token chunk peaked at 5.8 GiB (measured)
# Measured on the RTX 4060 Laptop: ~0.38 s per Qwen3.5-9B layer per 4096 tokens, forward+backward
# with recomputation, i.e. ~0.31 ms per token per billion parameters; +20% for loading layers.
_STREAM_MS_PER_TOKEN_PER_B = 0.31 * 1.2
_TOKENS_PER_ANSWER, _EPOCHS = 600, 3
METHOD_LABEL = {"qlora": "QLoRA (4-bit)", "lora": "LoRA (16-bit)"}


def stream_minutes(parameter_count: int | None, answers: int) -> float:
    billions = (parameter_count or 0) / 1e9
    return answers * _TOKENS_PER_ANSWER * _EPOCHS * _STREAM_MS_PER_TOKEN_PER_B * billions / 1000 / 60


@dataclass(frozen=True)
class Hardware:
    gpu: str | None
    vram_gb: float
    disk_free_gb: float
    train_env: bool


def detect_hardware(disk_path=config.PROJECT_ROOT) -> Hardware:
    gpu, vram = None, 0.0
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        name, mib = out.splitlines()[0].rsplit(",", 1)
        gpu, vram = name.strip(), round(float(mib) / 1024, 1)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    free = shutil.disk_usage(disk_path).free / 2**30
    return Hardware(gpu, vram, round(free, 1), TRAIN_ENV_PYTHON.exists())


@dataclass(frozen=True)
class TrainingEstimate:
    method: str        # qlora | lora
    vram_gb: float
    source: str
    note: str = ""     # why QLoRA isn't used, when it isn't

    @property
    def label(self) -> str:
        return METHOD_LABEL[self.method]


def estimate_training(repo: str | None, parameter_count: int | None, family: str,
                      overrides: Overrides) -> TrainingEstimate | None:
    """Documented figures win; otherwise a rough fit to Unsloth's published numbers
    (QLoRA ~0.55 GB per billion + 5 GB for activations/large vocabularies; LoRA ~2.1 GB/B + 2 GB)."""
    documented = overrides.training_vram(repo)
    rules = overrides.family(family)
    qlora_ok = rules.get("qlora", True)
    method = "qlora" if qlora_ok else "lora"
    note = "" if qlora_ok else rules.get("qlora_note", "4-bit QLoRA isn't advised for this model family")
    if documented.get(method):
        return TrainingEstimate(method, float(documented[method]), documented.get("source", "override registry"), note)
    if not parameter_count:
        return None
    billions = parameter_count / 1e9
    gb = 0.55 * billions + 5 if method == "qlora" else 2.1 * billions + 2
    return TrainingEstimate(method, float(math.ceil(gb)), f"estimated from {billions:.1f}B parameters", note)


@dataclass(frozen=True)
class RouteOption:
    target: str        # local | colab | kaggle
    label: str
    available: bool
    reason: str


@dataclass(frozen=True)
class TrainingPlan:
    model: str
    repo: str | None
    estimate: TrainingEstimate | None
    options: tuple[RouteOption, ...]
    recommended: str | None
    corrections: int
    enough_corrections: bool
    blocked_reason: str | None

    @property
    def where(self) -> str | None:
        return {"local": "local", "stream": "this computer, streamed", "colab": "Colab/Kaggle export",
                "kaggle": "Colab/Kaggle export"}.get(self.recommended or "")

    def to_dict(self) -> dict:
        data = asdict(self)
        if self.estimate:
            data["estimate"]["label"] = self.estimate.label
        return data | {"where": self.where}


class TrainingRouter:
    def __init__(self, hardware: Hardware, overrides: Overrides):
        self.hardware = hardware
        self.overrides = overrides

    def _local(self, est: TrainingEstimate) -> RouteOption:
        hw = self.hardware
        if est.vram_gb > hw.vram_gb - VRAM_MARGIN_GB:
            reason = f"{est.label} needs ~{est.vram_gb:g} GB; this GPU has {hw.vram_gb:g} GB"
        elif hw.disk_free_gb < LOCAL_DISK_GB:
            reason = f"needs ~{LOCAL_DISK_GB} GB of free disk for weights and conversion; {hw.disk_free_gb:g} GB free"
        elif not hw.train_env:
            reason = "set up the training environment first (.venv-train from requirements-train.txt)"
        else:
            return RouteOption("local", "This computer", True, f"{est.label}, ~{est.vram_gb:g} GB of {hw.vram_gb:g} GB")
        return RouteOption("local", "This computer", False, reason)

    def _stream(self, est: TrainingEstimate, model_type: str | None, parameter_count: int | None) -> RouteOption:
        """Layers streamed through this GPU one at a time: slow, but private, and any model size."""
        from src.learning.lora_stream_trainer import SUPPORTED_MODEL_TYPES

        label = "This computer (layers streamed through the GPU)"
        hw = self.hardware
        weights_gb = (parameter_count or 0) * 2 / 2**30
        if model_type not in SUPPORTED_MODEL_TYPES:
            return RouteOption("stream", label, False, f"streamed training isn't verified for {model_type} yet")
        if hw.vram_gb < STREAM_MIN_VRAM_GB:
            return RouteOption("stream", label, False, f"needs a GPU with {STREAM_MIN_VRAM_GB:g} GB; this one has {hw.vram_gb:g}")
        needed = 2 * weights_gb + 5   # original weights + one merged copy, before Ollama imports it
        if hw.disk_free_gb < needed:
            return RouteOption("stream", label, False, f"needs ~{needed:.0f} GB free disk; {hw.disk_free_gb:g} GB free")
        minutes = stream_minutes(parameter_count, answers=200)
        return RouteOption("stream", label, True, f"LoRA (16-bit), private, ~{minutes:.0f} min per 200 checked answers "
                                                  f"(estimate); first needs the {weights_gb:.0f} GB original weights")

    def _free_tier(self, est: TrainingEstimate, target: str) -> RouteOption:
        label = {"colab": "Google Colab (free T4, 15 GB)", "kaggle": "Kaggle (free 2x T4, 15 GB each)"}[target]
        if est.vram_gb <= FREE_TIER_GB:
            detail = ("free, needs a Google login, sessions up to ~12 h" if target == "colab"
                      else "free, 30 h a week")
            return RouteOption(target, label, True, f"{est.label} ~{est.vram_gb:g} GB: {detail}")
        if target == "kaggle" and est.method == "lora" and est.vram_gb <= 2 * FREE_TIER_GB - 2:
            # Kaggle's two T4s are separate GPUs, but the model's layers can be split across them.
            return RouteOption(target, label, True, f"{est.label} ~{est.vram_gb:g} GB split across both T4s "
                                                    "(fp16; slower than one big GPU); free, 30 h a week")
        where = "one GPU; free T4s have 15 GB" if target == "colab" else "two 15 GB T4s even when split"
        return RouteOption(target, label, False, f"needs ~{est.vram_gb:g} GB: more than {where}")

    def plan(self, model: str, family: str, trainability: TrainabilityResult | None,
             parameter_count: int | None, corrections: int) -> TrainingPlan:
        enough = corrections >= RECOMMENDED_CORRECTIONS
        repo = trainability.repo if trainability else None

        if trainability is None or not trainability.supported:
            blocker = trainability.blocker if trainability else None
            if blocker is None or blocker.component == "HuggingFace source":
                reason = "LLM weight fine-tuning unavailable: no trainable HF source found."
            else:
                reason = (f"LLM weight fine-tuning unavailable: {trainability.model_type or 'this architecture'} "
                          f"is not supported by {blocker.component} yet.")
            return TrainingPlan(model, repo, None, (), None, corrections, enough, reason)

        estimate = estimate_training(repo, parameter_count, family, self.overrides)
        if estimate is None:
            return TrainingPlan(model, repo, None, (), None, corrections, enough,
                                "LLM weight fine-tuning unavailable: the model's size is unknown.")

        # Private routes first: student data only leaves the computer when nothing local works.
        options = (self._local(estimate), self._stream(estimate, trainability.model_type, parameter_count),
                   self._free_tier(estimate, "colab"), self._free_tier(estimate, "kaggle"))
        recommended = next((o.target for o in options if o.available), None)
        blocked = None
        if recommended is None:
            blocked = (f"LLM weight fine-tuning needs ~{estimate.vram_gb:g} GB GPU memory: more than this machine "
                       f"({self.hardware.vram_gb:g} GB) or free Colab/Kaggle ({FREE_TIER_GB} GB per GPU).")
            if estimate.note:
                blocked += f" {estimate.note}"
        return TrainingPlan(model, repo, estimate, options, recommended, corrections, enough, blocked)
