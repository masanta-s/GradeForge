"""Download a model's original HuggingFace weights for local fine-tuning (e.g. ~19 GB for
Qwen3.5-9B), into models/hf_weights/ on the project drive.

Only on the teacher's explicit request: it is the one large download GradeForge makes after setup.
Files stream to `<name>.part` and resume with an HTTP Range request if interrupted; a
`.complete` marker is written only when every file has its full size.
"""
from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import httpx

from src import config

HF_URL = "https://huggingface.co"
WEIGHTS_DIR = config.MODELS_DIR / "hf_weights"
_WANTED = (".safetensors", ".json", ".jinja", ".txt", ".model", ".tiktoken")
_MARKER = ".complete"
DISK_MARGIN = 5 * 2**30   # beyond the weights themselves


def weights_path(repo: str, root: Path | None = None) -> Path:
    return Path(root or WEIGHTS_DIR) / repo.replace("/", "--")


def is_complete(repo: str, root: Path | None = None) -> bool:
    return (weights_path(repo, root) / _MARKER).exists()


def list_files(repo: str, http: httpx.Client) -> list[tuple[str, int]]:
    response = http.get(f"/api/models/{repo}", params={"blobs": "true"})
    response.raise_for_status()
    files = [(s["rfilename"], s.get("size") or 0) for s in response.json().get("siblings", [])]
    return [(name, size) for name, size in files if name.endswith(_WANTED) and "/" not in name]


def status(repo: str, http: httpx.Client | None = None, root: Path | None = None) -> dict:
    folder = weights_path(repo, root)
    have = sum(f.stat().st_size for f in folder.glob("*") if f.is_file()) if folder.exists() else 0
    result = {"repo": repo, "path": str(folder), "complete": is_complete(repo, root), "downloaded_bytes": have}
    if http is not None and not result["complete"]:
        result["total_bytes"] = sum(size for _, size in list_files(repo, http))
    return result


def download(repo: str, *, token: str | None = None, root: Path | None = None, http: httpx.Client | None = None,
             progress: Callable[[float, str], None] | None = None) -> Path:
    report = progress or (lambda p, m: None)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    client = http or httpx.Client(base_url=HF_URL, timeout=httpx.Timeout(60, read=300), follow_redirects=True,
                                  headers=headers)
    folder = weights_path(repo, root)
    folder.mkdir(parents=True, exist_ok=True)
    files = list_files(repo, client)
    if not any(name.endswith(".safetensors") for name, _ in files):
        raise ValueError(f"{repo} has no safetensors weights")
    total = sum(size for _, size in files)
    have = sum((folder / name).stat().st_size for name, _ in files if (folder / name).exists())
    free = shutil.disk_usage(folder).free
    if total - have + DISK_MARGIN > free:
        raise OSError(f"needs {(total - have + DISK_MARGIN) / 2**30:.0f} GB free on {folder.anchor}, "
                      f"{free / 2**30:.0f} GB available")

    done = have + sum((folder / (name + ".part")).stat().st_size for name, _ in files
                      if (folder / (name + ".part")).exists())
    for name, size in files:
        target = folder / name
        if target.exists() and (size == 0 or target.stat().st_size == size):
            continue
        part = target.with_name(target.name + ".part")
        start = part.stat().st_size if part.exists() else 0
        request_headers = {"Range": f"bytes={start}-"} if start else {}
        with client.stream("GET", f"/{repo}/resolve/main/{name}", headers=request_headers) as response:
            if response.status_code == 200 and start:
                start = 0  # server ignored the range: start over
            elif response.status_code not in (200, 206):
                response.raise_for_status()
            with part.open("ab" if start else "wb") as f:
                for chunk in response.iter_bytes(8 * 2**20):
                    f.write(chunk)
                    done += len(chunk)
                    report(done / max(1, total), f"Downloading {name} ({done / 2**30:.1f} of {total / 2**30:.1f} GB)")
        if size and part.stat().st_size != size:
            raise OSError(f"{name}: got {part.stat().st_size} bytes, expected {size}; run the download again to resume")
        part.replace(target)
    (folder / _MARKER).write_text(json.dumps({"repo": repo, "files": [n for n, _ in files]}), encoding="utf-8")
    report(1.0, "Download complete")
    return folder
