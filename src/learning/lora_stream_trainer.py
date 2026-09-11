"""Mechanism 4 on this computer: LoRA fine-tuning of a model too large for the GPU, by streaming
its layers through the GPU one at a time.

The frozen weights stay in the safetensors files (memory-mapped; the OS caches them in RAM when
it can). Before a decoder layer runs, its big weights are copied to the GPU; afterwards they are
dropped. With per-layer gradient checkpointing, the backward pass re-runs each layer's forward,
which loads that layer again, so at most about one layer's weights are on the GPU at a time.
Only the LoRA adapters (tens of MB) and small tensors (norms, gates) stay resident, and only the
LoRA adapters learn. The embedding table is looked up on the CPU, and lm_head is loaded only to
score the target tokens.

The result is exactly the same gradient as normal training (same maths, different schedule):
tests/test_lora_stream_gpu.py checks it against a fully loaded model. Measured on the RTX 4060
Laptop (2026-09-11): one Qwen3.5-9B layer loads in ~35 ms (11.5 GiB/s) and trains on a 4k-token
chunk in ~380 ms including recomputation, so compute, not PCIe, sets the pace.

Architectures are added here only after passing that equivalence test: the trainer relies on
the model's module layout (text decoder layers, embed_tokens, lm_head).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

SUPPORTED_MODEL_TYPES = {"qwen3_5"}
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",          # full-attention layers
                  "in_proj_qkv", "in_proj_z", "out_proj",          # Gated DeltaNet layers
                  "gate_proj", "up_proj", "down_proj"]             # MLPs (vision tower names differ)
STREAM_MIN_NUMEL = 1_000_000   # smaller tensors (norms, gates, conv) stay on the GPU
_TEXT = "base_model.model.model.language_model."
_EMBED_KEY = "model.language_model.embed_tokens.weight"

Progress = Callable[[float, str], None]


class UnsupportedArchitecture(ValueError):
    pass


@dataclass(frozen=True)
class LoraSettings:
    rank: int = 16
    alpha: int = 16
    dropout: float = 0.0


@dataclass(frozen=True)
class TrainSettings:
    learning_rate: float = 1e-4
    epochs: int = 4               # an upper bound: early stopping usually ends sooner
    examples_per_step: int = 8
    batch_tokens: int = 4096      # 8k tokens overflow 8 GB (measured: 10.8 GiB)
    max_length: int = 2048
    patience: int = 2             # validation checks without improvement before stopping
    warmup_fraction: float = 0.05
    checkpoint_every: int = 10    # optimizer steps between resumable checkpoints
    seed: int = 0


@dataclass(frozen=True)
class Tokenized:
    input_ids: list[int]
    labels: list[int]             # -100 on the prompt: only the target is learned

    @property
    def targets(self) -> int:
        return sum(label != -100 for label in self.labels[1:])


def tokenize(tokenizer, examples: Sequence, max_length: int) -> tuple[list[Tokenized], int]:
    """Prompt exactly as Ollama renders it for grading (thinking off), then the target."""
    out, skipped = [], 0
    for ex in examples:
        prompt = tokenizer.apply_chat_template(ex.messages, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
        p = tokenizer(prompt, add_special_tokens=False).input_ids
        t = tokenizer(ex.target, add_special_tokens=False).input_ids
        if len(p) + len(t) > max_length:
            skipped += 1
            continue
        out.append(Tokenized(p + t, [-100] * len(p) + t))
    return out, skipped


def pack(examples: Sequence[Tokenized], batch_tokens: int) -> list[list[Tokenized]]:
    """Micro-batches of similar length whose padded size stays within the token budget."""
    batches, current = [], []
    for ex in sorted(examples, key=lambda e: len(e.input_ids)):
        longest = max([len(e.input_ids) for e in current] + [len(ex.input_ids)])
        if current and longest * (len(current) + 1) > batch_tokens:
            batches.append(current)
            current = []
        current.append(ex)
    if current:
        batches.append(current)
    return batches


def collate(batch: Sequence[Tokenized], pad_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(e.input_ids) for e in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    for row, e in enumerate(batch):
        n = len(e.input_ids)
        ids[row, :n] = torch.tensor(e.input_ids)
        mask[row, :n] = 1
        labels[row, :n] = torch.tensor(e.labels)
    return ids, mask, labels


class WeightSource:
    """Tensors by checkpoint name, read on demand from memory-mapped safetensors shards."""

    def __init__(self, weights_dir: Path):
        from safetensors import safe_open

        weights_dir = Path(weights_dir)
        files = sorted(weights_dir.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no .safetensors weights in {weights_dir}")
        self._handles = [safe_open(str(f), framework="pt", device="cpu") for f in files]
        self._where = {key: h for h in self._handles for key in h.keys()}

    def __contains__(self, key: str) -> bool:
        return key in self._where

    def get(self, key: str) -> torch.Tensor:
        return self._where[key].get_tensor(key)


def _checkpoint_key(param_name: str) -> str:
    return param_name.removeprefix("base_model.model.").replace(".base_layer.", ".")


class StreamedLoRAModel:
    def __init__(self, weights_dir: Path, lora: LoraSettings = LoraSettings(), *, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16, stream_min_numel: int = STREAM_MIN_NUMEL,
                 init_adapter: Path | None = None, seed: int = 0):
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        from peft import LoraConfig, get_peft_model
        from transformers import AutoConfig, AutoModelForImageTextToText

        self._set = set_module_tensor_to_device
        self.device, self.dtype = torch.device(device), dtype
        config = AutoConfig.from_pretrained(weights_dir)
        if config.model_type not in SUPPORTED_MODEL_TYPES:
            raise UnsupportedArchitecture(f"streamed training isn't verified for {config.model_type} yet")
        with init_empty_weights():
            base = AutoModelForImageTextToText.from_config(config, dtype=dtype)
        self.lora_config = LoraConfig(r=lora.rank, lora_alpha=lora.alpha, lora_dropout=lora.dropout,
                                      target_modules=TARGET_MODULES, task_type="CAUSAL_LM")
        self.peft = get_peft_model(base, self.lora_config)
        self.text = self.peft.base_model.model.model.language_model
        self.source = WeightSource(weights_dir)
        text_config = config.get_text_config()
        self.pad_id = getattr(text_config, "pad_token_id", None) or 0
        self._lm_key = "lm_head.weight" if "lm_head.weight" in self.source else _EMBED_KEY
        self.embed = self.source.get(_EMBED_KEY).to(dtype)            # looked up on the CPU

        generator = torch.Generator().manual_seed(seed)
        self.streamed: dict[int, list[tuple[str, str]]] = {}
        layer_index = {id(layer): i for i, layer in enumerate(self.text.layers)}
        for name, param in list(self.peft.named_parameters()):
            if not name.startswith(_TEXT) or name.endswith("embed_tokens.weight"):
                continue                                               # vision tower stays unloaded
            if "lora_" in name:
                value = torch.zeros(param.shape) if "lora_B" in name else torch.empty(param.shape)
                if "lora_A" in name:
                    torch.nn.init.kaiming_uniform_(value, a=math.sqrt(5), generator=generator)
                self._set(self.peft, name, self.device, value=value.float())
                continue
            key = _checkpoint_key(name)
            layer = self._layer_of(name)
            if layer is not None and param.numel() >= stream_min_numel:
                self.streamed.setdefault(id(layer), []).append((name, key))
                continue
            self._set(self.peft, name, self.device, value=self.source.get(key).to(dtype))
        for name, buffer in list(self.peft.named_buffers()):
            if name.startswith(_TEXT):
                self._set(self.peft, name, self.device, value=buffer)
        self.lora_params = [p for n, p in self.peft.named_parameters() if "lora_" in n]
        for p in self.lora_params:
            p.requires_grad_(True)
        if init_adapter is not None:
            self.load_adapter(init_adapter)

        self._loaded: list[int] = []
        for layer in self.text.layers:
            if id(layer) in self.streamed:
                layer.register_forward_pre_hook(lambda module, args: self._load(module))
                layer.register_forward_hook(lambda module, args, output: self._unload(id(module)))
        self.peft.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.layer_count = len(layer_index)

    def _layer_of(self, name: str):
        marker = _TEXT + "layers."
        if not name.startswith(marker):
            return None
        return self.text.layers[int(name[len(marker):].split(".", 1)[0])]

    # --- streaming ------------------------------------------------------------------------
    def _load(self, layer) -> None:
        for other in [i for i in self._loaded if i != id(layer)]:
            self._unload(other)   # one layer resident at a time, even when a recompute skipped its unload hook
        if id(layer) in self._loaded:
            return
        for name, key in self.streamed[id(layer)]:
            self._set(self.peft, name, self.device, value=self.source.get(key).to(self.dtype))
        self._loaded.append(id(layer))

    def _unload(self, layer_id: int) -> None:
        if layer_id not in self._loaded:
            return
        for name, _ in self.streamed[layer_id]:
            self._set(self.peft, name, "meta")   # tensors still used by autograd stay alive until used
        self._loaded.remove(layer_id)

    # --- loss -----------------------------------------------------------------------------
    def _hidden(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embeds = F.embedding(ids, self.embed).to(self.device)
        return self.text(inputs_embeds=embeds, attention_mask=mask.to(self.device), use_cache=False).last_hidden_state

    def _targets(self, hidden: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shifted = labels[:, 1:].to(self.device)
        keep = shifted != -100
        return hidden[:, :-1][keep], shifted[keep]

    def _lm_loss(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        weight = self.source.get(self._lm_key).to(self.device, self.dtype)
        return F.cross_entropy((h.to(self.dtype) @ weight.T).float(), y, reduction="sum")

    def loss_and_backward(self, batch: Sequence[Tokenized], normaliser: int) -> float:
        """Summed target-token loss of one micro-batch; LoRA gradients accumulate."""
        ids, mask, labels = collate(batch, self.pad_id)
        h, y = self._targets(self._hidden(ids, mask), labels)
        leaf = h.detach().requires_grad_()
        loss = self._lm_loss(leaf, y)
        (loss / normaliser).backward()           # lm_head part: gradient w.r.t. the hidden states
        h.backward(leaf.grad)                    # then back through the streamed layers
        # Checkpoint recomputation stops as soon as it has what backward needs (it skips the rest of
        # the layer, which saves time), so the last layer's unload hook may not have run.
        for layer_id in list(self._loaded):
            self._unload(layer_id)
        return float(loss.detach())

    @torch.no_grad()
    def evaluate(self, examples: Sequence[Tokenized], batch_tokens: int) -> float:
        """Mean loss per target token (lower = closer to the teacher's scores)."""
        self.peft.eval()
        total = count = 0.0
        for batch in pack(examples, batch_tokens):
            ids, mask, labels = collate(batch, self.pad_id)
            h, y = self._targets(self._hidden(ids, mask), labels)
            total += float(self._lm_loss(h, y))
            count += int(y.numel())
        self.peft.train()
        return total / max(1, count)

    # --- adapters -------------------------------------------------------------------------
    def lora_state(self) -> dict[str, torch.Tensor]:
        from peft import get_peft_model_state_dict

        return {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(self.peft).items()}

    def set_lora_state(self, state: dict[str, torch.Tensor]) -> None:
        from peft import set_peft_model_state_dict

        missing = set_peft_model_state_dict(self.peft, {k: v.to(self.device) for k, v in state.items()})
        if getattr(missing, "unexpected_keys", None):
            raise ValueError(f"adapter doesn't fit this model: {missing.unexpected_keys[:3]}")

    def save_adapter(self, out_dir: Path) -> Path:
        from safetensors.torch import save_file

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous() for k, v in self.lora_state().items()}, out_dir / "adapter_model.safetensors")
        self.lora_config.save_pretrained(out_dir)
        return out_dir

    def load_adapter(self, adapter_dir: Path) -> None:
        from safetensors.torch import load_file

        self.set_lora_state(load_file(Path(adapter_dir) / "adapter_model.safetensors"))


# --- training loop ----------------------------------------------------------------------------

@dataclass
class TrainResult:
    steps: int
    epochs_run: float
    best_validation_loss: float | None
    stopped_early: bool
    resumed: bool
    seconds: float
    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _fingerprint(train: Sequence[Tokenized], settings: TrainSettings, lora: LoraSettings) -> str:
    digest = hashlib.sha1(json.dumps([asdict(settings), asdict(lora)]).encode())
    for ex in train:
        digest.update(bytes(str(ex.input_ids[-8:]) + str(len(ex.input_ids)), "utf-8"))
    return digest.hexdigest()


def train_lora(model: StreamedLoRAModel, train: Sequence[Tokenized], validation: Sequence[Tokenized],
               settings: TrainSettings, work_dir: Path, *, lora: LoraSettings = LoraSettings(),
               progress: Progress | None = None, should_stop: Callable[[], bool] | None = None) -> TrainResult:
    """AdamW on the LoRA weights with warmup + cosine decay. Validation loss is checked twice per
    epoch; training stops when it stops improving (`patience` checks) and the best weights are
    kept. A checkpoint every few steps lets an interrupted run resume where it stopped."""
    report = progress or (lambda p, m: None)
    if not train:
        raise ValueError("nothing to train on")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    steps_per_epoch = math.ceil(len(train) / settings.examples_per_step)
    total_steps = steps_per_epoch * settings.epochs
    eval_every = max(1, steps_per_epoch // 2)
    warmup = max(1, round(total_steps * settings.warmup_fraction))

    optimizer = torch.optim.AdamW(model.lora_params, lr=settings.learning_rate, weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, max(0, s - warmup) / max(1, total_steps - warmup)))))

    fingerprint = _fingerprint(train, settings, lora)
    checkpoint_path = work_dir / "checkpoint.pt"
    state = {"step": 0, "best": None, "best_state": None, "bad": 0, "train_loss": [], "validation_loss": []}
    resumed = False
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("fingerprint") == fingerprint:
            model.set_lora_state(saved["lora"])
            optimizer.load_state_dict(saved["optimizer"])
            schedule.load_state_dict(saved["schedule"])
            state = saved["state"]
            resumed = True

    def save_checkpoint() -> None:
        tmp = checkpoint_path.with_suffix(".tmp")
        torch.save({"fingerprint": fingerprint, "lora": model.lora_state(), "optimizer": optimizer.state_dict(),
                    "schedule": schedule.state_dict(), "state": state}, tmp)
        tmp.replace(checkpoint_path)

    stopped_early = False
    model.peft.train()
    while state["step"] < total_steps:
        step = state["step"]
        epoch, index = divmod(step, steps_per_epoch)
        order = list(range(len(train)))
        random.Random(settings.seed + epoch).shuffle(order)
        chosen = [train[i] for i in order[index * settings.examples_per_step:(index + 1) * settings.examples_per_step]]
        normaliser = max(1, sum(e.targets for e in chosen))
        optimizer.zero_grad(set_to_none=True)
        loss = sum(model.loss_and_backward(batch, normaliser) for batch in pack(chosen, settings.batch_tokens))
        torch.nn.utils.clip_grad_norm_(model.lora_params, 1.0)
        optimizer.step()
        schedule.step()
        state["step"] = step + 1
        state["train_loss"].append(loss / normaliser)
        report(state["step"] / total_steps * 0.97,
               f"Training step {state['step']} of up to {total_steps} (loss {loss / normaliser:.3f})")

        if validation and state["step"] % eval_every == 0:
            report(state["step"] / total_steps * 0.97, "Checking held-back validation answers")
            value = model.evaluate(validation, settings.batch_tokens)
            state["validation_loss"].append(value)
            if state["best"] is None or value < state["best"] - 1e-4:
                state.update(best=value, best_state=model.lora_state(), bad=0)
            else:
                state["bad"] += 1
            if state["bad"] >= settings.patience:
                stopped_early = True
                save_checkpoint()
                break
        if state["step"] % settings.checkpoint_every == 0:
            save_checkpoint()
        if should_stop and should_stop():
            save_checkpoint()
            raise InterruptedError("training paused; it resumes from the last checkpoint")

    if state["best_state"] is not None:
        model.set_lora_state(state["best_state"])   # the version that matched the validation answers best
    checkpoint_path.unlink(missing_ok=True)
    return TrainResult(state["step"], round(state["step"] / steps_per_epoch, 2), state["best"], stopped_early,
                       resumed, round(time.time() - started, 1), state["train_loss"], state["validation_loss"])
