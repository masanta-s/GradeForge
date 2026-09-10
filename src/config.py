"""Project-wide paths, model defaults and environment setup.

Import this module before transformers / huggingface_hub / sentence-transformers / litellm:
it redirects every cache and temp directory into the project (nothing is written to C:)
and turns HuggingFace offline mode on unless PAPERMIND_ALLOW_DOWNLOADS=1.
"""
import os
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / ".cache"

TROCR_BASE_DIR = MODELS_DIR / "trocr_base"
TROCR_FINETUNED_DIR = MODELS_DIR / "trocr_finetuned"
EMBEDDERS_DIR = MODELS_DIR / "embedders"
LLM_CHECKPOINTS_DIR = MODELS_DIR / "llm_checkpoints"

CORRECTIONS_DB = DATA_DIR / "corrections.db"
DISPUTES_DB = DATA_DIR / "disputes.db"
RESOLUTION_CACHE_DB = DATA_DIR / "resolution_cache.db"

TROCR_REPO = "microsoft/trocr-base-handwritten"
EMBEDDER_REPOS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
}
DEFAULT_EMBEDDER = "all-MiniLM-L6-v2"

# Defaults, not a whitelist — any model Ollama serves can be selected.
DEFAULT_LLM = "qwen3.5:9b"
SECONDARY_LLM = "gemma4:e4b"
# Measured on the 8 GB RTX 4060: qwen3.5:9b is 100% on GPU at 8192 but spills (84%) at 16384.
LLM_NUM_CTX = 8192


def _normalise_ollama_url(host: str) -> str:
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    if host.count(":") == 1:  # scheme only, no port
        host = f"{host}:11434"
    return host.replace("0.0.0.0", "127.0.0.1")


OLLAMA_URL = _normalise_ollama_url(os.environ.get("OLLAMA_HOST", "127.0.0.1:11434"))

ALLOW_DOWNLOADS = os.environ.get("PAPERMIND_ALLOW_DOWNLOADS") == "1"

_TMP_DIR = CACHE_DIR / "tmp"
_CACHE_ENV = {
    "HF_HOME": MODELS_DIR / "hf",
    "PIP_CACHE_DIR": CACHE_DIR / "pip",
    "TORCH_HOME": CACHE_DIR / "torch",
    "TRITON_CACHE_DIR": CACHE_DIR / "triton",
    "TORCHINDUCTOR_CACHE_DIR": CACHE_DIR / "inductor",
    "XDG_CACHE_HOME": CACHE_DIR / "xdg",
}


def _apply_environment() -> None:
    for name, path in _CACHE_ENV.items():
        os.environ.setdefault(name, str(path))

    # Windows always defines TEMP (on C:), so override rather than setdefault.
    _TMP_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = os.environ["TMP"] = str(_TMP_DIR)
    tempfile.tempdir = str(_TMP_DIR)

    offline = "0" if ALLOW_DOWNLOADS else "1"
    os.environ["HF_HUB_OFFLINE"] = offline
    os.environ["TRANSFORMERS_OFFLINE"] = offline
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # LiteLLM otherwise fetches its model price map from GitHub at import time.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def ensure_dirs() -> None:
    for d in (DATA_DIR, MODELS_DIR, TROCR_FINETUNED_DIR, EMBEDDERS_DIR, LLM_CHECKPOINTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


_apply_environment()
