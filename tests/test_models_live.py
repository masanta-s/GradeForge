"""Live: the real capability probe on qwen3.5:9b and real HuggingFace resolution.
Skipped when Ollama (with the model) or huggingface.co is unavailable."""
import httpx
import pytest

from src import config
from src.models.cache import ModelCache
from tests.model_fakes import GEMMA, QWEN


def _ollama_has(model: str) -> bool:
    try:
        from src.models.ollama_probe import OllamaModelProbe

        return model in OllamaModelProbe(timeout=5).list_models()
    except Exception:
        return False


def _hf_reachable() -> bool:
    try:
        return httpx.get("https://huggingface.co/api/models?limit=1", timeout=5).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.mark.skipif(not _hf_reachable(), reason="huggingface.co not reachable")
def test_real_hf_resolution(tmp_path):
    from src.models.hf_resolver import HFSourceResolver
    from src.models.override_registry import load_overrides

    cache = ModelCache(tmp_path / "cache.db")
    resolver = HFSourceResolver(cache=cache, overrides=load_overrides(cache))
    qwen, gemma = resolver.resolve(QWEN), resolver.resolve(GEMMA)
    assert (qwen.repo, qwen.confidence, qwen.model_type) == ("Qwen/Qwen3.5-9B", "high", "qwen3_5")
    assert (gemma.repo, gemma.confidence, gemma.model_type) == ("google/gemma-4-E4B-it", "high", "gemma4")


@pytest.mark.skipif(not _ollama_has(config.DEFAULT_LLM), reason=f"Ollama with {config.DEFAULT_LLM} not available")
def test_real_capability_probe(tmp_path):
    from src.knowledge.llm_client import LLMClient
    from src.models.capability_prober import CapabilityProber

    report = CapabilityProber(LLMClient(config.DEFAULT_LLM), ModelCache(tmp_path / "cache.db")).probe(
        config.DEFAULT_LLM, has_vision=True)
    failed = [(r.label, r.detail) for r in report.results if r.status != "pass"]
    assert report.critical_passed and report.vision_ok, failed
