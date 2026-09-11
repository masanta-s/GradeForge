"""Clean-up of old fine-tuning output.

    models/llm_checkpoints/<name>/v<N>/   imported GGUF fine-tunes (~5 GB each) + Modelfile
    models/trocr_finetuned/v<N>/          TrOCR LoRA adapters (a few MB each)
    .cache/train/                         transient training files (HF weights, merged models,
                                          GGUF conversions: 40-60 GB per local run)

Keeps the newest N versions of each model, never the version in use (the active TrOCR adapter,
or an Ollama model the teacher has selected), and empties the transient folder. Nothing is
deleted without a plan the teacher has seen: `plan()` lists it first, `collect()` deletes.
Corrections are never touched: they live in data/, not here.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from src import config

_VERSION = re.compile(r"^v(\d+)$")


@dataclass(frozen=True)
class GCItem:
    path: str
    kind: str              # llm_checkpoint | trocr_adapter | transient
    size_bytes: int
    reason: str
    ollama_name: str | None = None


@dataclass(frozen=True)
class GCResult:
    removed: list[GCItem] = field(default_factory=list)
    freed_bytes: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DiskReport:
    drive: str
    llm_checkpoints_bytes: int
    trocr_adapters_bytes: int
    transient_bytes: int
    free_bytes: int
    total_bytes: int


def folder_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _versions(folder: Path) -> list[tuple[int, Path]]:
    if not folder.is_dir():
        return []
    found = [(int(m.group(1)), p) for p in folder.iterdir() if p.is_dir() and (m := _VERSION.match(p.name))]
    return sorted(found, reverse=True)  # newest first


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _make_writable(func, path, _exc) -> None:
    os.chmod(path, stat.S_IWRITE)  # Windows: read-only files (e.g. from git or HF) block rmtree
    func(path)


class CheckpointGC:
    def __init__(self, llm_root: Path = config.LLM_CHECKPOINTS_DIR, trocr_root: Path = config.TROCR_FINETUNED_DIR,
                 transient_root: Path = config.CACHE_DIR / "train",
                 remove_ollama: Callable[[str], None] | None = None):
        self.llm_root = Path(llm_root)
        self.trocr_root = Path(trocr_root)
        self.transient_root = Path(transient_root)
        self.remove_ollama = remove_ollama

    def plan(self, keep_latest_n: int = 2, protect: set[str] = frozenset()) -> list[GCItem]:
        """`protect`: Ollama model names in use; their checkpoint folders are always kept."""
        if keep_latest_n < 1:
            raise ValueError("keep at least one version")
        items = []
        if self.llm_root.is_dir():
            for model_dir in sorted(p for p in self.llm_root.iterdir() if p.is_dir()):
                for rank, (_, path) in enumerate(_versions(model_dir)):
                    ollama_name = _read_json(path / "meta.json").get("ollama_name")
                    if rank < keep_latest_n or ollama_name in protect:
                        continue
                    items.append(GCItem(str(path), "llm_checkpoint", folder_size(path),
                                        f"older than the newest {keep_latest_n} of {model_dir.name}", ollama_name))

        active = _read_json(self.trocr_root / "active.json").get("version")
        for rank, (_, path) in enumerate(_versions(self.trocr_root)):
            if rank < keep_latest_n or path.name == active:
                continue
            items.append(GCItem(str(path), "trocr_adapter", folder_size(path),
                                f"handwriting adapter older than the newest {keep_latest_n}, not in use"))

        if self.transient_root.is_dir():
            for path in sorted(self.transient_root.iterdir()):
                items.append(GCItem(str(path), "transient", folder_size(path), "temporary training file"))
        return items

    def _inside_roots(self, path: Path) -> bool:
        resolved = path.resolve()
        return any(resolved != root.resolve() and resolved.is_relative_to(root.resolve())
                   for root in (self.llm_root, self.trocr_root, self.transient_root))

    def collect(self, keep_latest_n: int = 2, protect: set[str] = frozenset()) -> GCResult:
        removed, errors = [], []
        for item in self.plan(keep_latest_n, protect):
            path = Path(item.path)
            if not self._inside_roots(path):  # defence in depth: never delete outside the three roots
                errors.append(f"refused to delete {path}: outside the checkpoint folders")
                continue
            try:
                if item.ollama_name and self.remove_ollama:
                    self.remove_ollama(item.ollama_name)
                if path.is_dir():
                    shutil.rmtree(path, onexc=_make_writable)
                else:
                    path.unlink()
                removed.append(item)
            except Exception as e:
                errors.append(f"{path.name}: {type(e).__name__}: {e}")
        return GCResult(removed, sum(i.size_bytes for i in removed), errors)

    def get_disk_report(self) -> DiskReport:
        usage = shutil.disk_usage(self.llm_root if self.llm_root.exists() else config.PROJECT_ROOT)
        size = lambda p: folder_size(p) if p.exists() else 0  # noqa: E731
        return DiskReport(Path(self.llm_root).resolve().anchor, size(self.llm_root), size(self.trocr_root),
                          size(self.transient_root), usage.free, usage.total)
