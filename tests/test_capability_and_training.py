"""Capability probe, model tiers and the training router."""
import pytest

from src.models.arch_gate import ComponentCheck, TrainabilityResult
from src.models.cache import ModelCache
from src.models.capability_prober import CapabilityProber, assign_tier, cached_report
from src.models.override_registry import load_overrides
from src.models.training_router import Hardware, TrainingRouter, estimate_training
from tests.model_fakes import BadLLM, GoodLLM


@pytest.fixture
def cache(tmp_path):
    return ModelCache(tmp_path / "cache.db")


def supported(repo, model_type="qwen3_5"):
    return TrainabilityResult(repo, model_type, (ComponentCheck("transformers", "pass", "ok"),))


LAPTOP = Hardware("RTX 4060 Laptop", 8.0, 180.0, train_env=False)
WORKSTATION = Hardware("RTX 4090", 24.0, 500.0, train_env=True)


# --- capability probe -----------------------------------------------------------------------

def test_a_capable_model_passes_every_probe(cache):
    report = CapabilityProber(GoodLLM(), cache).probe("qwen3.5:9b", "sha-1", has_vision=True)
    assert [r.status for r in report.results] == ["pass"] * 5
    assert report.critical_passed and report.vision_ok and report.native_json
    rubric = next(r for r in report.results if r.name == "rubric")
    assert "full 5.0/5" in rubric.detail and "wrong 0.0/5" in rubric.detail


def test_fenced_json_passes_but_is_not_native(cache):
    report = CapabilityProber(GoodLLM(fenced=True), cache).probe("m", has_vision=True)
    assert report.critical_passed and not report.native_json
    assert "after clean-up/repair" in report.results[0].detail


def test_a_model_that_cannot_see_images_keeps_grading_but_not_diagram_vision(cache):
    # gemma4:e4b on Ollama: reports "vision", answers "I cannot see the image"
    report = CapabilityProber(GoodLLM(sees_images=False), cache).probe("gemma4:e4b", has_vision=True)
    assert report.critical_passed and not report.vision_ok
    assert "labels and shape" in next(r for r in report.results if r.name == "vision").detail


def test_models_without_vision_skip_the_vision_probe(cache):
    llm = GoodLLM()
    report = CapabilityProber(llm, cache).probe("text-only", has_vision=False)
    assert next(r for r in report.results if r.name == "vision").status == "skipped"
    assert report.critical_passed and not report.vision_ok


def test_a_chatty_model_is_incompatible(cache):
    report = CapabilityProber(BadLLM(), cache).probe("chatty:3b", has_vision=False)
    assert not report.critical_passed
    assert {r.name for r in report.failed_critical} == {"json", "instructions", "rubric"}
    tier = assign_tier(report, None)
    assert tier.level == "red" and "not safe to grade" in tier.reason


def test_a_crashing_probe_is_a_failure_not_an_exception(cache):
    def broken(messages, schema=None):
        raise ConnectionError("model crashed")

    report = CapabilityProber(broken, cache).probe("broken", has_vision=True)
    assert all(r.status == "fail" for r in report.results)
    assert "model crashed" in report.results[0].detail


def test_reports_are_cached_per_digest(cache):
    CapabilityProber(GoodLLM(), cache).probe("qwen3.5:9b", "sha-1", has_vision=True)
    assert cached_report("qwen3.5:9b", "sha-1", cache).critical_passed
    assert cached_report("qwen3.5:9b", None, cache) is not None
    assert cached_report("qwen3.5:9b", "sha-2", cache) is None   # re-pulled: check again
    assert cached_report("other", None, cache) is None


# --- training router ------------------------------------------------------------------------

@pytest.fixture
def overrides(cache):
    return load_overrides(cache)


def test_gemma_e4b_is_trainable_via_colab_not_locally(overrides):
    plan = TrainingRouter(LAPTOP, overrides).plan("gemma4:e4b", "gemma4", supported("google/gemma-4-E4B-it", "gemma4"),
                                                  7_996_157_674, corrections=247)
    assert plan.estimate.method == "qlora" and plan.estimate.vram_gb == 10
    local, colab, kaggle = plan.options
    assert not local.available and "QLoRA (4-bit) needs ~10 GB; this GPU has 8 GB" in local.reason
    assert colab.available and kaggle.available
    assert plan.recommended == "colab" and plan.where == "Colab/Kaggle export" and plan.enough_corrections
    assert assign_tier(_passed(), plan).level == "green"


def test_qwen_9b_is_inference_only_on_free_hardware(overrides):
    plan = TrainingRouter(LAPTOP, overrides).plan("qwen3.5:9b", "qwen35", supported("Qwen/Qwen3.5-9B"),
                                                  9_653_104_368, corrections=12)
    assert plan.estimate.method == "lora" and plan.estimate.vram_gb == 22     # QLoRA not advised for Qwen 3.5
    assert plan.recommended is None and not plan.enough_corrections
    assert "~22 GB" in plan.blocked_reason and "15 GB per GPU" in plan.blocked_reason
    assert "QLoRA" in plan.blocked_reason
    tier = assign_tier(_passed(), plan)
    assert tier.level == "yellow" and "3 of 4 learning mechanisms active" in tier.reason


def test_the_same_model_trains_locally_on_a_24gb_gpu(overrides):
    plan = TrainingRouter(WORKSTATION, overrides).plan("qwen3.5:9b", "qwen35", supported("Qwen/Qwen3.5-9B"),
                                                       9_653_104_368, corrections=300)
    assert plan.recommended == "local" and plan.where == "local"


def test_local_training_needs_disk_and_the_training_env(overrides):
    low_disk = Hardware("RTX 4090", 24.0, 20.0, train_env=True)
    no_env = Hardware("RTX 4090", 24.0, 500.0, train_env=False)
    trainability = supported("google/gemma-4-E4B-it", "gemma4")
    local = TrainingRouter(low_disk, overrides).plan("g", "gemma4", trainability, 8e9, 0).options[0]
    assert not local.available and "60 GB" in local.reason
    local = TrainingRouter(no_env, overrides).plan("g", "gemma4", trainability, 8e9, 0).options[0]
    assert not local.available and ".venv-train" in local.reason


def test_unlisted_models_get_an_estimate_from_their_size(overrides):
    estimate = estimate_training("someorg/New-3B", 3_000_000_000, "newfam", overrides)
    assert estimate.method == "qlora" and estimate.vram_gb == 7 and "estimated from 3.0B" in estimate.source
    assert estimate_training("someorg/New", None, "newfam", overrides) is None


def test_untrainable_models_explain_why(overrides):
    router = TrainingRouter(LAPTOP, overrides)
    no_source = TrainabilityResult(None, None, (ComponentCheck("HuggingFace source", "fail", "none"),))
    assert "no trainable HF source" in router.plan("x", "f", no_source, 1, 0).blocked_reason
    future = TrainabilityResult("org/X", "gemma5", (ComponentCheck("transformers", "fail", "unknown arch"),))
    assert "gemma5 is not supported by transformers yet" in router.plan("x", "f", future, 1, 0).blocked_reason


def test_tiers_before_any_check():
    assert assign_tier(None, None).level == "unknown"
    assert "not checked yet" in assign_tier(_passed(), None).reason


def _passed():
    from src.models.capability_prober import ProbeReport, ProbeResult

    return ProbeReport("m", None, (ProbeResult("json", "JSON adherence", "pass", True, ""),), True)
