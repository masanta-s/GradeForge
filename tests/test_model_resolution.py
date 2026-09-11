"""Three-layer model resolution: override registry, HF source resolver, architecture gate."""
import json

import httpx
import pytest

from src.models import override_registry
from src.models.arch_gate import ArchitectureGate, transformers_check
from src.models.cache import ModelCache
from src.models.hf_resolver import (HFResolution, HFSourceResolver, ResolverUnavailable, search_terms,
                                    size_label)
from src.models.override_registry import Overrides, load_overrides, refresh_overrides
from tests.model_fakes import GEMMA, QWEN, TINY, FakeHF, hf


@pytest.fixture
def cache(tmp_path):
    return ModelCache(tmp_path / "cache.db")


def resolver(cache, fake=None, overrides=None):
    fake = fake or FakeHF()
    return HFSourceResolver(cache=cache, http=fake.client(), overrides=overrides or load_overrides(cache)), fake


# --- override registry ---------------------------------------------------------------------

def test_bundled_registry_is_valid_and_patches_only(cache):
    overrides = load_overrides(cache)
    assert overrides.source == "bundled"
    assert overrides.family("qwen35")["qlora"] is False            # Unsloth: no 4-bit QLoRA for Qwen 3.5
    assert overrides.training_vram("google/gemma-4-e4b-it")["qlora"] == 10  # case-insensitive repo match
    assert overrides.family("gemma4") == {} and overrides.models == {}  # not a model catalogue


def test_newer_remote_registry_wins_and_refresh_is_throttled(cache):
    bundled = json.loads(override_registry.BUNDLED.read_text(encoding="utf-8"))
    newer = {**bundled, "updated": "2099-01-01", "models": {"odd:7b": {"hf_repo": "org/Odd-7B"}}}
    calls = []

    def handler(request):
        calls.append(request.url)
        return httpx.Response(200, json=newer)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    assert refresh_overrides(cache, http) is True
    assert refresh_overrides(cache, http) is False      # fetched less than a day ago
    assert len(calls) == 1
    overrides = load_overrides(cache)
    assert overrides.source == "remote" and overrides.model("odd:7b")["hf_repo"] == "org/Odd-7B"


def test_broken_or_offline_remote_registry_falls_back_to_bundled(cache):
    bad = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"schema": 99})))
    assert refresh_overrides(cache, bad, force=True) is False
    offline = httpx.Client(transport=httpx.MockTransport(
        lambda r: (_ for _ in ()).throw(httpx.ConnectError("no network", request=r))))
    assert refresh_overrides(cache, offline, force=True) is False
    assert load_overrides(cache).source == "bundled"


# --- HF resolver ---------------------------------------------------------------------------

@pytest.mark.parametrize(("name", "label"), [("qwen3.5:9b", "9b"), ("gemma4:e4b", "e4b"),
                                             ("llama3.1:8b-instruct-q4_K_M", "8b"), ("mistral:latest", None),
                                             ("mistral", None)])
def test_size_label(name, label):
    assert size_label(name) == label


def test_search_terms_use_the_ollama_name_and_registry():
    registry = Overrides(families={"qwen35": {"hf_search": ["qwen3.5"]}})
    assert search_terms(QWEN, registry) == ["qwen3.5", "qwen35"]
    assert search_terms(GEMMA, Overrides()) == ["gemma4"]


def test_resolves_qwen_to_the_official_instruct_repo(cache):
    r, fake = resolver(cache)
    result = r.resolve(QWEN)
    assert (result.repo, result.confidence, result.model_type) == ("Qwen/Qwen3.5-9B", "high", "qwen3_5")
    ranked = [c.repo for c in result.candidates]
    # the pretrained base, a 4-bit re-upload and an "uncensored" finetune all rank lower
    for other in ("Qwen/Qwen3.5-9B-Base", "QuantTrio/Qwen3.5-9B-AWQ", "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"):
        assert other in ranked and ranked.index(other) > 0
    assert "unsloth/Qwen3.5-9B-GGUF" not in ranked            # no safetensors: not trainable
    assert any("matches Ollama" in reason for reason in result.reasons)


def test_prefers_the_instruct_model_over_the_pretrained_one(cache):
    # the pretrained google/gemma-4-E4B has MORE downloads here; the -it is still chosen
    result = resolver(cache)[0].resolve(GEMMA)
    assert result.repo == "google/gemma-4-E4B-it" and result.confidence == "high"


def test_resolution_is_cached_for_30_days(cache):
    r, fake = resolver(cache)
    r.resolve(QWEN)
    n = len(fake.requests)
    again = r.resolve(QWEN)
    assert len(fake.requests) == n and again.repo == "Qwen/Qwen3.5-9B"
    assert HFResolution.from_dict(again.to_dict()) == again
    r.resolve(QWEN, refresh=True)
    assert len(fake.requests) > n


def test_unknown_model_resolves_to_none(cache):
    result = resolver(cache)[0].resolve(TINY)
    assert result.repo is None and result.confidence == "none" and not result.resolved


def test_manual_choice_is_verified_and_pinned(cache):
    r, _ = resolver(cache)
    with pytest.raises(ValueError, match="not found"):
        r.set_manual(QWEN, "Nobody/NoSuchRepo")
    pinned = r.set_manual(QWEN, "Qwen/Qwen3.5-9B-Base")
    assert (pinned.repo, pinned.source) == ("Qwen/Qwen3.5-9B-Base", "manual")
    assert r.resolve(QWEN, refresh=True).repo == "Qwen/Qwen3.5-9B-Base"   # stays pinned
    assert r.clear_manual(QWEN).repo == "Qwen/Qwen3.5-9B"


def test_pinning_a_gguf_repo_is_not_trainable(cache):
    result = resolver(cache)[0].set_manual(QWEN, "unsloth/Qwen3.5-9B-GGUF")
    assert result.repo is None and "no safetensors" in result.reasons[0]


def test_registry_can_fix_a_model_auto_resolution_gets_wrong(cache):
    overrides = Overrides(models={"tiny:1b": {"hf_repo": "Qwen/Qwen3.5-4B"}})
    result = resolver(cache, overrides=overrides)[0].resolve(TINY)
    assert (result.repo, result.source) == ("Qwen/Qwen3.5-4B", "override")


def test_offline_is_reported_not_crashed(cache):
    r, _ = resolver(cache, FakeHF(fail=True))
    with pytest.raises(ResolverUnavailable, match="huggingface.co"):
        r.resolve(QWEN)


# --- architecture gate --------------------------------------------------------------------

def test_transformers_gate_uses_the_installed_library():
    assert transformers_check("qwen3_5").status == "pass"
    assert transformers_check("gemma4").status == "pass"
    future = transformers_check("gemma5")
    assert future.status == "fail" and "pip install -U transformers" in future.detail


def test_gate_passes_resolved_models_and_explains_failures(cache, tmp_path):
    gate = ArchitectureGate(train_python=tmp_path / "missing" / "python.exe")
    ok = gate.check(resolver(cache)[0].resolve(QWEN), QWEN)
    assert ok.supported and ok.blocker is None
    unsloth = next(c for c in ok.checks if c.component == "Unsloth")
    assert unsloth.status == "unknown" and "notebook" in unsloth.detail   # checked on Colab, not here

    unresolved = gate.check(resolver(cache)[0].resolve(TINY), TINY)
    assert not unresolved.supported and unresolved.blocker.component == "HuggingFace source"

    future = HFResolution("x:7b", "org/X-7B", "high", "auto", 7, 7, "gemma5", False, ())
    blocked = gate.check(future, QWEN)
    assert not blocked.supported and blocked.blocker.component == "transformers"


def test_gate_checks_the_local_training_env_when_installed(tmp_path):
    python = tmp_path / "python.exe"
    python.write_text("")
    resolution = HFResolution("qwen3.5:9b", "Qwen/Qwen3.5-9B", "high", "auto", 1, 1, "qwen3_5", True, ())

    class Done:
        def __init__(self, code):
            self.returncode = code

    old = ArchitectureGate(python, run=lambda *a, **k: Done(3)).check(resolution, QWEN)
    assert old.blocker.component == "Unsloth" and "requirements-train" in old.blocker.detail
    new = ArchitectureGate(python, run=lambda *a, **k: Done(0)).check(resolution, QWEN)
    assert new.supported
    assert any(c.component == "HuggingFace access" for c in new.checks)   # gated repo: licence needed


def test_hf_candidate_scoring_flags_quantised_derivatives():
    from src.models.hf_resolver import score_candidate

    awq = score_candidate(hf("X/Model-9B-AWQ", 100, "m", ["awq", "base_model:quantized:Org/Model-9B"]), 100, "9b", set())
    original = score_candidate(hf("Org/Model-9B", 100, "m"), 100, "9b", {"Org/Model-9B"})
    assert original.score > awq.score
    assert any("quantised" in r for r in awq.reasons) and any("derivative" in r for r in awq.reasons)
