"""Fine-tune the grading model on the teacher's checked answers, then keep it only if it's better.

    1. training data   the grader's own prompts for the training split (held-out answers excluded),
                       incremental: answers since the version in use + a replay of older ones
    2. train           LoRA, layers streamed through this GPU (or: an adapter from the Kaggle notebook)
    3. merge           adapter + original weights, one shard at a time
    4. import          Ollama converts and quantises the merged model; the merged copy is then deleted
    5. check           the capability probe (critical checks must pass)
    6. compare         current setup vs new setup on the same held-out answers
    7. promote         only on a clear gain; either way the numbers are logged for the teacher

Every version keeps its adapter (tens of MB) in models/llm_checkpoints/<name>/v<N>/, so any
version can be rebuilt; checkpoint clean-up removes old versions from Ollama.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

from src import config

RUN_DIR = config.CACHE_DIR / "train"


def versions(name: str, root: Path | None = None) -> list[dict]:
    folder = Path(root or config.LLM_CHECKPOINTS_DIR) / name
    out = []
    for meta_file in sorted(folder.glob("v*/meta.json"), key=lambda p: int(p.parent.name[1:])):
        out.append(json.loads(meta_file.read_text(encoding="utf-8")) | {"path": str(meta_file.parent)})
    return out


def promoted_version(name: str, root: Path | None = None) -> dict | None:
    return next((v for v in reversed(versions(name, root)) if v.get("promoted")), None)


def training_data(services, *, since: str | None):
    """(training examples for this round, validation examples) as the grader would prompt them."""
    from server.measure import question_lookup
    from src.learning.evaluation import holdout_ids
    from src.learning.llm_dataset import build_examples, select_round

    corrections = services.corrections.grade_corrections()
    held_out = holdout_ids(corrections)
    question_for = question_lookup(services)

    def examples_for(c):
        question = question_for(c)
        return services.retriever.retrieve(question.text, c.student_answer, subject=c.subject, k=4,
                                           exclude_ids=held_out | {c.id})

    common = dict(question_for=question_for, examples_for=examples_for)
    train = select_round(build_examples(corrections, splits=("train",), **common), since)
    validation = build_examples(corrections, splits=("validation",), **common)
    return train, validation


def free_gpu(services) -> None:
    from src.models.ollama_probe import OllamaModelProbe

    services.reload_ocr()
    try:
        probe = OllamaModelProbe(timeout=10)
        for name in probe.loaded_models():
            probe.unload(name)
    except Exception:
        pass  # Ollama not running: nothing to unload
    import torch

    torch.cuda.empty_cache()


def train_locally(services, job, *, base_model: str, repo: str, weights_dir: Path, name: str, version: str,
                  parent: dict | None, lora=None, settings=None) -> tuple[Path, dict, int]:
    import torch
    from transformers import AutoTokenizer

    from src.learning.lora_stream_trainer import LoraSettings, StreamedLoRAModel, TrainSettings, tokenize, train_lora

    lora, settings = lora or LoraSettings(), settings or TrainSettings()
    job.report(0.01, "Collecting your checked answers")
    train, validation = training_data(services, since=parent.get("trained_until") if parent else None)
    if not train:
        raise ValueError("nothing new to learn: check or approve more graded answers first")
    tokenizer = AutoTokenizer.from_pretrained(weights_dir)
    train_tok, skipped = tokenize(tokenizer, train, settings.max_length)
    validation_tok, _ = tokenize(tokenizer, validation, settings.max_length)
    job.report(0.03, "Freeing the GPU")
    free_gpu(services)
    parent_adapter = Path(parent["path"]) / "adapter" if parent else None
    model = StreamedLoRAModel(weights_dir, lora, init_adapter=parent_adapter)
    result = train_lora(model, train_tok, validation_tok, settings, RUN_DIR / f"{name}-{version}", lora=lora,
                        progress=lambda p, m: job.report(0.05 + 0.55 * p, m),
                        should_stop=lambda: services.pause_requested)
    adapter = model.save_adapter(Path(config.LLM_CHECKPOINTS_DIR) / name / version / "adapter")
    del model
    torch.cuda.empty_cache()
    info = {"route": "stream", "examples": len(train_tok), "skipped_too_long": skipped,
            "validation_examples": len(validation_tok), "parent": parent["ollama_name"] if parent else None,
            "train": result.to_dict(), "lora": asdict(lora), "settings": asdict(settings)}
    return adapter, info, len(train_tok)


def finish(services, job, *, adapter: Path, base_model: str, repo: str, weights_dir: Path, name: str,
           version: str, info: dict, trained_until: str) -> dict:
    """Steps 3-7, shared by local training and an adapter imported from the Kaggle notebook."""
    from server.measure import grade_fn
    from src.learning.evaluation import holdout_ids, holdout_items, is_better, measure
    from src.learning.lora_merge import merge_adapter
    from src.models.model_manager import import_safetensors

    run_dir = RUN_DIR / f"{name}-{version}"
    merged = merge_adapter(weights_dir, adapter, run_dir / "merged",
                           progress=lambda p, m: job.report(0.60 + 0.10 * p, m))
    job.report(0.72, "Importing into Ollama (it converts and compresses the model)")
    try:
        record = import_safetensors(merged, name, base_model, version, probe=services.models.probe,
                                    meta={"hf_repo": repo, "trained_until": trained_until, **info})
    finally:
        shutil.rmtree(merged, ignore_errors=True)   # ~19 GB, no longer needed once Ollama has it
    tag = record["ollama_name"]

    job.report(0.80, f"Checking {tag}")
    report = services.models.run_probe(tag, services.make_llm(tag))
    corrections = services.corrections.grade_corrections()
    items = holdout_items(corrections)
    held_out = holdout_ids(corrections)
    current_model = services.model_name
    before = after = None
    if report.critical_passed and items:
        before, _ = measure(items, grade_fn(services, services.llm, current_model, learned=True, holdout=held_out),
                            progress=lambda p, m: job.report(0.82 + 0.08 * p, f"Current model: {m}"))
        after, _ = measure(items, grade_fn(services, services.make_llm(tag), tag, learned=True, holdout=held_out),
                           progress=lambda p, m: job.report(0.90 + 0.09 * p, f"New version: {m}"))
        total = len(holdout_items(corrections, limit=None))
        for model, result in ((current_model, before), (tag, after)):
            services.corrections.add_benchmark_run(label=f"{model}, with your examples and calibration", model=model,
                                                   setup="learned", holdout_total=total,
                                                   trigger=f"retrain {version}", **result.to_dict())
    promoted = bool(before and after and is_better(after, before))
    if promoted:
        services.update_settings({"model": tag, "provider": "ollama"})
    verdict = ("no held-out answers to compare on yet" if not items else
               "failed its capability check" if not report.critical_passed else
               "better on held-out answers" if promoted else "not clearly better on held-out answers")
    meta_file = Path(record["path"]) / "meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) | {
        "promoted": promoted, "verdict": verdict, "before": before.to_dict() if before else None,
        "after": after.to_dict() if after else None, "probe_passed": report.critical_passed}
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    services.corrections.log_training(
        mechanism="llm_lora", model=base_model, version=version, metric_name="mae_pct",
        metric_before=before.mae_pct if before else None, metric_after=after.mae_pct if after else None,
        num_samples=info.get("examples", 0), promoted=promoted, notes=f"{tag}: {verdict}")
    job.report(1.0, f"{tag}: {verdict}")
    return meta


def retrain(services, job, *, base_model: str, from_scratch: bool = False, lora=None, settings=None) -> dict:
    from datetime import datetime, timezone

    from src.models.model_manager import next_version, tuned_name
    from src.models.weights_download import is_complete, weights_path

    base_model = base_of(base_model)
    repo = _repo(services, base_model)
    if not is_complete(repo):
        raise ValueError(f"download the original weights of {base_model} ({repo}) first")
    name = tuned_name(base_model)
    version = next_version(Path(config.LLM_CHECKPOINTS_DIR) / name)
    parent = None if from_scratch else promoted_version(name)
    trained_until = datetime.now(timezone.utc).isoformat(timespec="seconds")
    services.pause_requested = False
    started = time.time()
    adapter, info, _ = train_locally(services, job, base_model=base_model, repo=repo, weights_dir=weights_path(repo),
                                     name=name, version=version, parent=parent, lora=lora, settings=settings)
    info["seconds_training"] = round(time.time() - started)
    return finish(services, job, adapter=adapter, base_model=base_model, repo=repo, weights_dir=weights_path(repo),
                  name=name, version=version, info=info, trained_until=trained_until)


def import_adapter(services, job, *, base_model: str, adapter_dir: Path) -> dict:
    """An adapter trained by the exported Kaggle/Colab notebook: merge, import, check, compare."""
    from src.models.model_manager import next_version, tuned_name
    from src.models.weights_download import is_complete, weights_path

    adapter_dir = Path(adapter_dir)
    if not (adapter_dir / "adapter_model.safetensors").exists():
        raise ValueError(f"{adapter_dir} has no adapter_model.safetensors (unzip the notebook's output first)")
    base_model = base_of(base_model)
    repo = _repo(services, base_model)
    if not is_complete(repo):
        raise ValueError(f"download the original weights of {base_model} ({repo}) first: the adapter is merged into them")
    name = tuned_name(base_model)
    version = next_version(Path(config.LLM_CHECKPOINTS_DIR) / name)
    target = Path(config.LLM_CHECKPOINTS_DIR) / name / version / "adapter"
    shutil.copytree(adapter_dir, target)
    exported = {}
    if (adapter_dir / "gradeforge.json").exists():
        exported = json.loads((adapter_dir / "gradeforge.json").read_text(encoding="utf-8"))
    free_gpu(services)
    info = {"route": "notebook", "examples": exported.get("examples", 0), "notebook": exported}
    return finish(services, job, adapter=target, base_model=base_model, repo=repo, weights_dir=weights_path(repo),
                  name=name, version=version, info=info, trained_until=exported.get("trained_until", ""))


def base_of(model: str, root: Path | None = None) -> str:
    """A fine-tune's base model (gradeforge-qwen3-5-9b:v2 -> qwen3.5:9b); other names unchanged."""
    name = model.split(":")[0]
    for meta in versions(name, root):
        if meta.get("ollama_name") == model or not model.count(":"):
            return meta["base_model"]
    return model


def _repo(services, base_model: str) -> str:
    resolution = services.models.resolution(base_model) or services.models.resolve(base_model)
    if not resolution.resolved:
        raise ValueError(f"no trainable HuggingFace source found for {base_model}")
    return resolution.repo
