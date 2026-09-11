"""Override registry: a small JSON of patches, not a catalogue of models.

The bundled copy (registry/overrides.json) always works offline. A newer copy is fetched from
the project's GitHub repository at most once a day, and only when the teacher asks GradeForge
to check a model (never at startup, so the app stays silent on the network otherwise).

Sections:
  families          per Ollama family: extra HF search terms, whether QLoRA is advisable
  models            per Ollama model name: a fixed HF repo, for names auto-resolution gets wrong
  training_vram_gb  per HF repo: documented training VRAM that beats the generic estimate
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from src import config
from src.models.cache import ModelCache

BUNDLED = config.PROJECT_ROOT / "registry" / "overrides.json"
REMOTE_URL = "https://raw.githubusercontent.com/masanta-s/GradeForge/main/registry/overrides.json"
SCHEMA = 1
REFRESH_EVERY = 24 * 3600
_SECTIONS = ("families", "models", "training_vram_gb")


@dataclass(frozen=True)
class Overrides:
    updated: str = ""
    families: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)
    training_vram_gb: dict = field(default_factory=dict)
    source: str = "bundled"

    def family(self, family: str) -> dict:
        return self.families.get(family.lower(), {})

    def model(self, name: str) -> dict:
        return self.models.get(name, {})

    def training_vram(self, repo: str | None) -> dict:
        if not repo:
            return {}
        lowered = {k.lower(): v for k, v in self.training_vram_gb.items()}
        return lowered.get(repo.lower(), {})


def _validate(data) -> dict:
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError("unrecognised override registry format")
    for section in _SECTIONS:
        if not isinstance(data.get(section, {}), dict):
            raise ValueError(f"registry section {section!r} must be an object")
    return data


def _from_dict(data: dict, source: str) -> Overrides:
    return Overrides(updated=str(data.get("updated", "")),
                     families={k.lower(): v for k, v in data.get("families", {}).items()},
                     models=dict(data.get("models", {})),
                     training_vram_gb=dict(data.get("training_vram_gb", {})),
                     source=source)


def load_overrides(cache: ModelCache | None = None, bundled: Path = BUNDLED) -> Overrides:
    """The newest of the bundled file and the last successfully fetched remote copy."""
    local = _validate(json.loads(Path(bundled).read_text(encoding="utf-8")))
    remote = (cache or ModelCache()).get("registry", REMOTE_URL)
    if remote and remote.get("updated", "") > local.get("updated", ""):
        return _from_dict(remote, "remote")
    return _from_dict(local, "bundled")


def refresh_overrides(cache: ModelCache | None = None, http: httpx.Client | None = None,
                      force: bool = False) -> bool:
    """Fetch the remote registry if the cached copy is over a day old. Returns True if fetched.
    Failures are silent: the bundled copy is always a valid fallback."""
    cache = cache or ModelCache()
    stored = cache.stored_at("registry", REMOTE_URL)
    if not force and stored is not None and time.time() - stored < REFRESH_EVERY:
        return False
    client = http or httpx.Client(timeout=5, follow_redirects=True)
    try:
        response = client.get(REMOTE_URL)
        response.raise_for_status()
        cache.put("registry", REMOTE_URL, _validate(response.json()))
        return True
    except (httpx.HTTPError, ValueError):
        return False
