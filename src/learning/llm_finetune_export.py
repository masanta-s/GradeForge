"""Mechanism 4 in the cloud: a Colab/Kaggle notebook that trains the same LoRA adapter as the local
streamed trainer, for when this computer can't (or the teacher prefers a cloud GPU).

It uses plain transformers + PEFT (no Unsloth), spreads the model over both of Kaggle's T4s when
it needs more than one (device_map="auto"), and trains exactly like the local trainer: the
grader's own prompts, loss only on the teacher's score (see llm_dataset), validation-based early
stopping, best weights kept. The output is the adapter (tens of MB), not a GGUF: GradeForge
merges it into the original weights and lets Ollama import it (llama.cpp-made Qwen 3.5 GGUFs don't
load in Ollama; Ollama's own converter does).

Privacy: the notebook carries students' answers off this computer, so the API requires explicit
consent. Only prompts and teacher scores are included, never names, sheet ids or exam names.

The code cells are written to run unchanged on a tiny local model, which is how they are tested
(tests/test_notebook_gpu.py): only the pip cell and the configuration cell differ.
"""
from __future__ import annotations

import json
from collections.abc import Sequence

from src.learning.llm_dataset import TrainingExample

RECOMMENDED_CORRECTIONS = 200
CONFIG_MARKER = "# GRADEFORGE-CONFIG"

LOAD = r'''
import json, math, os, random, shutil, time
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForImageTextToText, AutoTokenizer

# LoRA on the language model only (not the vision/audio towers). Same layers as GradeForge's local trainer.
TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj|gate_proj|up_proj|down_proj)"

gpus = torch.cuda.device_count()
assert gpus, "Turn on a GPU accelerator first (Kaggle: Settings -> Accelerator; Colab: Runtime -> Change runtime type)."
# T4s (Turing) report bf16 "supported" but only emulate it, which is slow: use fp16 there and
# bf16 only on Ampere or newer (compute capability 8.0+).
native_bf16 = min(torch.cuda.get_device_capability(i)[0] for i in range(gpus)) >= 8
dtype = torch.bfloat16 if native_bf16 else torch.float16
load = {"dtype": dtype}
if SPLIT_ACROSS_GPUS and gpus > 1:
    # Kaggle's 2x T4 are separate 15 GB GPUs: put about half of the layers on each.
    load["device_map"] = "auto"
    load["max_memory"] = {i: f"{int(torch.cuda.get_device_properties(i).total_memory / 2**30 * 0.85)}GiB"
                          for i in range(gpus)}
else:
    load["device_map"] = {"": 0}
if LOAD_IN_4BIT:
    from transformers import BitsAndBytesConfig
    load["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                     bnb_4bit_compute_dtype=dtype)
try:
    model = AutoModelForImageTextToText.from_pretrained(BASE_MODEL, **load)
except Exception as e:
    raise SystemExit(f"Couldn't load {BASE_MODEL}: {e}\nUpdate transformers (first cell), or fine-tune another model.")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if LOAD_IN_4BIT:
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
else:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model = get_peft_model(model, LoraConfig(r=SETTINGS["rank"], lora_alpha=SETTINGS["alpha"], lora_dropout=0.0,
                                         target_modules=TARGETS, task_type="CAUSAL_LM"))
model.print_trainable_parameters()
for i in range(gpus):
    print(f"GPU {i}: {torch.cuda.memory_allocated(i) / 2**30:.1f} GiB loaded")
'''

TOKENIZE = r'''
def encode(example):
    """The prompt exactly as Ollama renders it for grading (thinking off), then the teacher's score."""
    prompt = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    target_ids = tokenizer(example["target"], add_special_tokens=False).input_ids
    return prompt_ids + target_ids, len(target_ids)

train = [x for x in map(encode, EXAMPLES) if len(x[0]) <= SETTINGS["max_length"]]
validation = [x for x in map(encode, VALIDATION) if len(x[0]) <= SETTINGS["max_length"]]
print(len(train), "training answers,", len(validation), "validation answers")
'''

TRAIN = r'''
first_device = model.get_input_embeddings().weight.device

def loss_of(ids, n_target):
    """Cross-entropy on the target tokens only; logits are computed just for those positions."""
    input_ids = torch.tensor([ids], device=first_device)
    logits = model(input_ids=input_ids, logits_to_keep=n_target + 1).logits[0, :-1].float()
    return torch.nn.functional.cross_entropy(logits, input_ids[0, -n_target:].to(logits.device))

def validate():
    model.eval()
    with torch.no_grad():
        values = [float(loss_of(ids, n)) for ids, n in validation]
    model.train()
    return sum(values) / max(1, len(values))

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=SETTINGS["learning_rate"])
accumulate = SETTINGS["accumulate"]
steps_per_epoch = math.ceil(len(train) / accumulate)
total_steps = steps_per_epoch * SETTINGS["epochs"]
warmup = max(1, total_steps // 20)
schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (
    1 + math.cos(math.pi * min(1.0, max(0, s - warmup) / max(1, total_steps - warmup)))))
scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

best, best_state, bad, step, history, val_history = None, None, 0, 0, [], []
started = time.time()
model.train()
for epoch in range(SETTINGS["epochs"]):
    order = list(range(len(train)))
    random.Random(epoch).shuffle(order)
    for s in range(steps_per_epoch):
        chunk = order[s * accumulate:(s + 1) * accumulate]
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for i in chunk:
            loss = loss_of(*train[i]) / len(chunk)
            scaler.scale(loss).backward()
            total += float(loss.detach())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)   # skips the step if fp16 overflowed
        scaler.update()
        schedule.step()
        step += 1
        history.append(total)
        if step <= 3 or step % 10 == 0:
            memory = ", ".join(f"GPU {i} {torch.cuda.max_memory_allocated(i) / 2**30:.1f} GiB" for i in range(gpus))
            print(f"step {step}/{total_steps}  loss {total:.3f}  {(time.time() - started) / step:.1f} s/step  {memory}")
        if not math.isfinite(total):
            raise SystemExit("The loss overflowed (fp16 on T4). Try a smaller SETTINGS['learning_rate'].")
        if validation and step % max(1, steps_per_epoch // 2) == 0:
            value = validate()
            val_history.append(value)
            print(f"  validation loss {value:.3f}")
            if best is None or value < best - 1e-4:
                best, bad = value, 0
                best_state = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
            else:
                bad += 1
        if bad >= SETTINGS["patience"]:
            break
    if bad >= SETTINGS["patience"]:
        print("Stopped early: validation answers stopped improving (this prevents overfitting).")
        break
if best_state is not None:
    set_peft_model_state_dict(model, best_state)
print(f"Done: {step} steps in {(time.time() - started) / 60:.1f} min; best validation loss {best}")
'''

SAVE = r'''
model.save_pretrained(OUTPUT_DIR)   # the LoRA adapter only: adapter_model.safetensors + adapter_config.json
with open(os.path.join(OUTPUT_DIR, "gradeforge.json"), "w") as f:
    json.dump({"base_model": BASE_MODEL, "examples": len(train), "trained_until": TRAINED_UNTIL,
               "best_validation_loss": best, "steps": step, "validation_loss": val_history,
               "gpus": gpus, "dtype": str(dtype)}, f, indent=2)
archive = shutil.make_archive(OUTPUT_DIR, "zip", OUTPUT_DIR)
print("Download:", archive)
try:
    from google.colab import files
    files.download(archive)
except ImportError:
    pass  # Kaggle: download it from the Output panel
'''


def _cell(kind: str, source: str) -> dict:
    cell = {"cell_type": kind, "metadata": {}, "source": source.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        cell.update(outputs=[], execution_count=None)
    return cell


def config_source(*, base_model: str, load_in_4bit: bool, split: bool, trained_until: str, examples: list[dict],
                  validation: list[dict], settings: dict, output_dir: str = "gradeforge_adapter") -> str:
    return "\n".join([
        CONFIG_MARKER,
        "import json",
        f"BASE_MODEL = {base_model!r}",
        f"LOAD_IN_4BIT = {load_in_4bit}   # QLoRA; False = 16-bit LoRA",
        f"SPLIT_ACROSS_GPUS = {split}",
        f"TRAINED_UNTIL = {trained_until!r}",
        f"OUTPUT_DIR = {output_dir!r}",
        f"SETTINGS = {settings!r}",
        f"EXAMPLES = json.loads({json.dumps(examples, ensure_ascii=False)!r})",
        f"VALIDATION = json.loads({json.dumps(validation, ensure_ascii=False)!r})",
    ])


def build_notebook(train: Sequence[TrainingExample], validation: Sequence[TrainingExample], *, base_model: str,
                   ollama_base: str, load_in_4bit: bool, split: bool, trained_until: str) -> dict:
    if not train:
        raise ValueError("no written-answer corrections to train on yet")
    settings = {"learning_rate": 1e-4, "epochs": 4, "accumulate": 8, "patience": 2, "rank": 16, "alpha": 16,
                "max_length": 2048}
    method = "QLoRA (4-bit)" if load_in_4bit else "LoRA (16-bit)"
    where = "both of Kaggle's T4s (the model is split across them)" if split else "one GPU"
    warning = ("" if len(train) >= RECOMMENDED_CORRECTIONS else
               f"\n\n> **Only {len(train)} examples.** Fine-tuning is recommended from "
               f"{RECOMMENDED_CORRECTIONS}; until then, few-shot examples and calibration do the learning.")
    to_rows = lambda examples: [{"messages": e.messages, "target": e.target} for e in examples]  # noqa: E731
    cells = [
        _cell("markdown", f"""
# GradeForge: fine-tune the grading model on your checked answers

Trains a {method} adapter for `{base_model}` (the original weights of `{ollama_base}`) on {where}.
Choose **GPU T4 x2** on Kaggle (Settings → Accelerator) and turn internet on, then run all cells.

* **{len(train)} training answers** and {len(validation)} validation answers: the grader's prompt → your
  score. No student names, sheet ids or exam names are included. Held-out answers GradeForge uses to
  judge the result are not in this notebook.
* Training stops early when the validation answers stop improving, so it can't overfit them.
* The result is a small adapter (`gradeforge_adapter.zip`). In GradeForge: **Learning → Import trained
  adapter**. GradeForge merges it, imports it into Ollama, and only switches to it if it matches your
  marks better than the current model.
* This notebook was generated by GradeForge and hasn't been run on Kaggle by its authors; the first
  steps print speed and GPU memory so problems show up within minutes.{warning}
"""),
        _cell("code", "!pip install -q -U transformers peft accelerate" + (" bitsandbytes" if load_in_4bit else "")),
        _cell("code", config_source(base_model=base_model, load_in_4bit=load_in_4bit, split=split,
                                    trained_until=trained_until, examples=to_rows(train),
                                    validation=to_rows(validation), settings=settings)),
        _cell("code", LOAD),
        _cell("code", TOKENIZE),
        _cell("code", TRAIN),
        _cell("code", SAVE),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                         "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
