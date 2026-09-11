"""Layer 1 of model resolution: identify models from Ollama's own metadata.

No static registry — architecture, size, quantisation, context length and capabilities all
come from /api/show, and VRAM is *measured* via /api/ps after a real load. Disk size is not
a VRAM estimate: gemma4:e4b is 9.6 GB on disk but loads as ~3.2 GB because its per-layer
embeddings stay in system RAM.
"""
from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import httpx

from src import config


@dataclass(frozen=True)
class ModelIdentity:
    name: str
    family: str
    architecture: str
    parameter_size: str
    parameter_count: int | None
    quantization: str
    context_length: int | None
    capabilities: tuple[str, ...]

    @property
    def has_vision(self) -> bool:
        return "vision" in self.capabilities

    @property
    def can_think(self) -> bool:
        return "thinking" in self.capabilities


@dataclass(frozen=True)
class VRAMMeasurement:
    model: str
    num_ctx: int
    size_bytes: int
    size_vram_bytes: int
    measured_at: float

    @property
    def gpu_fraction(self) -> float:
        return self.size_vram_bytes / self.size_bytes if self.size_bytes else 0.0

    @property
    def fully_on_gpu(self) -> bool:
        return self.size_bytes > 0 and self.size_vram_bytes >= self.size_bytes


class OllamaUnavailable(RuntimeError):
    pass


class OllamaModelProbe:
    def __init__(self, base_url: str = config.OLLAMA_URL, timeout: float = 600.0):
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.ConnectError as e:
            raise OllamaUnavailable(f"Ollama is not reachable at {self._client.base_url}") from e
        response.raise_for_status()
        return response.json()

    def version(self) -> str:
        return self._request("GET", "/api/version")["version"]

    def list_models(self) -> list[str]:
        return [m["name"] for m in self._request("GET", "/api/tags").get("models", [])]

    def digests(self) -> dict[str, str]:
        """Model name -> content digest; changes when a model is re-pulled or re-created."""
        return {m["name"]: m.get("digest", "") for m in self._request("GET", "/api/tags").get("models", [])}

    def show(self, model_name: str) -> dict:
        return self._request("POST", "/api/show", json={"model": model_name})

    def remove(self, model_name: str) -> None:
        response = self._client.request("DELETE", "/api/delete", json={"model": model_name})
        if response.status_code != 404:  # already gone is fine
            response.raise_for_status()

    def probe(self, model_name: str) -> ModelIdentity:
        data = self.show(model_name)
        details = data.get("details", {})
        info = data.get("model_info", {})
        arch = info.get("general.architecture") or details.get("family", "")
        return ModelIdentity(
            name=model_name,
            family=details.get("family", ""),
            architecture=arch,
            parameter_size=details.get("parameter_size", ""),
            parameter_count=info.get("general.parameter_count"),
            quantization=details.get("quantization_level", ""),
            context_length=info.get(f"{arch}.context_length"),
            capabilities=tuple(data.get("capabilities", [])),
        )

    def pull(self, model_name: str,
             on_progress: Callable[[str, int, int], None] | None = None) -> None:
        with self._client.stream("POST", "/api/pull", json={"model": model_name}) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if "error" in event:
                    raise RuntimeError(f"ollama pull {model_name}: {event['error']}")
                if on_progress:
                    on_progress(event.get("status", ""), event.get("completed", 0), event.get("total", 0))

    def measure_vram(self, model_name: str, num_ctx: int = 8192) -> VRAMMeasurement:
        """Load the model with a tiny prompt at the given context size and read /api/ps."""
        payload = {
            "model": model_name,
            "prompt": "Reply with OK.",
            "stream": False,
            "keep_alive": "2m",
            "options": {"num_ctx": num_ctx, "num_predict": 4},
        }
        if self.probe(model_name).can_think:
            payload["think"] = False
        self._request("POST", "/api/generate", json=payload)

        for entry in self._request("GET", "/api/ps").get("models", []):
            if entry.get("name") == model_name or entry.get("model") == model_name:
                return VRAMMeasurement(
                    model=model_name,
                    num_ctx=num_ctx,
                    size_bytes=entry.get("size", 0),
                    size_vram_bytes=entry.get("size_vram", 0),
                    measured_at=time.time(),
                )
        raise RuntimeError(f"{model_name} did not appear in /api/ps after loading")

    def unload(self, model_name: str) -> None:
        self._request("POST", "/api/generate", json={"model": model_name, "keep_alive": 0})


# --- VRAM measurement cache (data/resolution_cache.db) ---------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vram_measurements (
    model           TEXT    NOT NULL,
    num_ctx         INTEGER NOT NULL,
    size_bytes      INTEGER NOT NULL,
    size_vram_bytes INTEGER NOT NULL,
    measured_at     REAL    NOT NULL,
    PRIMARY KEY (model, num_ctx)
)
"""


def _connect(path: Path | None = None) -> sqlite3.Connection:
    path = Path(path or config.RESOLUTION_CACHE_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(_SCHEMA)
    return conn


def save_measurement(m: VRAMMeasurement, path: Path | None = None) -> None:
    with closing(_connect(path)) as conn, conn:
        conn.execute(
            "INSERT OR REPLACE INTO vram_measurements VALUES (?, ?, ?, ?, ?)",
            (m.model, m.num_ctx, m.size_bytes, m.size_vram_bytes, m.measured_at),
        )


def load_measurement(model: str, num_ctx: int, path: Path | None = None) -> VRAMMeasurement | None:
    with closing(_connect(path)) as conn:
        row = conn.execute(
            "SELECT model, num_ctx, size_bytes, size_vram_bytes, measured_at "
            "FROM vram_measurements WHERE model = ? AND num_ctx = ?",
            (model, num_ctx),
        ).fetchone()
    return VRAMMeasurement(*row) if row else None
