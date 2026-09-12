"""Run the fine-tuning notebook on Kaggle's free GPUs and bring the adapter back.

Measured with a probe notebook (2026-09-12): pushing with `machine_shape = "NvidiaTeslaT4"` gives
**two** Tesla T4s (14.6 GiB each), internet inside the kernel, 31 GiB RAM and ~1 TB of scratch in
/tmp, so Qwen 3.5 9B's ~22 GB of 16-bit LoRA training fits when the model is split across both.

The notebook is pushed private, GradeForge polls its status, and when it finishes the output
(gradeforge_adapter.zip) is downloaded and unpacked. Everything after that — merging, importing
into Ollama, the capability check and the held-out comparison — is the same as for a local run.
"""
from __future__ import annotations

import json
import shutil
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from src import config
from src.models import kaggle_auth

SLUG = "gradeforge-finetune"
TITLE = "GradeForge fine-tune"
ACCELERATOR = "NvidiaTeslaT4"    # two T4s, confirmed by the probe notebook
RUN_FILE = config.DATA_DIR / "kaggle_run.json"
DONE = ("COMPLETE", "ERROR", "CANCEL")


@dataclass
class Run:
    kernel: str
    url: str
    version: int | None
    pushed_at: str
    model: str
    examples: int
    status: str = "QUEUED"
    message: str = ""
    imported: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def load_run(path: Path | None = None) -> Run | None:
    path = Path(path or RUN_FILE)
    if not path.exists():
        return None
    try:
        return Run(**json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, TypeError):
        return None


def save_run(run: Run, path: Path | None = None) -> None:
    path = Path(path or RUN_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")


def _state(status) -> str:
    """Kaggle returns an enum like KernelWorkerStatus.RUNNING; keep the bare name."""
    return str(getattr(status, "status", status)).rsplit(".", 1)[-1].upper()


def push(notebook: dict, *, model: str, examples: int, folder: Path, client=None, slug: str = SLUG,
         accelerator: str = ACCELERATOR) -> Run:
    """Upload the notebook as a private kernel and start it on Kaggle's GPUs."""
    from datetime import datetime, timezone

    client = client or kaggle_auth.api()
    user = client.config_values["username"]
    kernel = f"{user}/{slug}"
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "finetune.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel, "title": TITLE, "code_file": "finetune.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
        "enable_internet": "true", "machine_shape": accelerator,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }, indent=2), encoding="utf-8")

    response = client.kernels_push(str(folder))
    error = getattr(response, "error", None)
    if error:
        raise RuntimeError(f"Kaggle refused the notebook: {error}")
    return Run(kernel=kernel, url=getattr(response, "url", f"https://www.kaggle.com/code/{kernel}"),
               version=getattr(response, "version_number", None), model=model, examples=examples,
               pushed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def poll(run: Run, client=None) -> Run:
    client = client or kaggle_auth.api()
    status = client.kernels_status(run.kernel)
    run.status = _state(status)
    run.message = str(getattr(status, "failure_message", "") or "")
    return run


def finished(run: Run) -> bool:
    return any(state in run.status for state in DONE)


def tail_log(run: Run, dest: Path, client=None, lines: int = 12) -> list[str]:
    """The notebook's own output (speed, GPU memory, validation loss), for the progress panel."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    client = client or kaggle_auth.api()
    client.kernels_output(run.kernel, str(dest))
    log = next(iter(sorted(dest.glob("*.log"))), None)
    if log is None:
        return []
    try:
        entries = json.loads(log.read_text(encoding="utf-8", errors="replace"))
    except ValueError:
        return []
    text = [str(e.get("data", "")).rstrip() for e in entries if str(e.get("data", "")).strip()]
    return [line for line in text if not line.startswith(("/usr/", "0.00s -", "[NbConvert"))][-lines:]


def fetch_adapter(run: Run, dest: Path, client=None,
                  progress: Callable[[float, str], None] | None = None) -> Path:
    """Download the finished kernel's output and unpack the LoRA adapter."""
    report = progress or (lambda p, m: None)
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    report(0.2, "Downloading the trained adapter from Kaggle")
    (client or kaggle_auth.api()).kernels_output(run.kernel, str(dest))
    archive = next(iter(sorted(dest.glob("*adapter*.zip"))), None)
    if archive is not None:
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(dest / "adapter")
        return dest / "adapter"
    found = next(iter(sorted(dest.rglob("adapter_model.safetensors"))), None)
    if found is None:
        raise FileNotFoundError("the Kaggle run produced no adapter: open the notebook's log to see why")
    return found.parent
