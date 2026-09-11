"""The whole retraining round on a tiny Qwen3.5-architecture model: the teacher's checked answers ->
streamed LoRA training -> merge -> (stand-in) Ollama import and check -> comparison on held-out
answers -> promoted only when it matches the teacher better."""
import json

import pytest
import torch

from src import config
from src.learning.evaluation import split_of
from tests.test_models_api import failed as failed_job
from tests.test_server import client, wait  # noqa: F401  (fixture + helper)
from tests.tiny_qwen import make_tiny_qwen

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

ANSWERS = {"Plants use sunlight to make glucose": 3.0, "Plants breathe at night": 0.5,
           "Chlorophyll absorbs light energy": 2.0, "It happens in the chloroplast and releases oxygen": 3.5}


class TeacherLikeLLM:
    """A 'fine-tuned' grader that marks like the teacher (vs. the current one's flat 0.6)."""

    def __call__(self, messages, schema=None):
        answer = messages[-1]["content"].split("<student_answer>")[-1]
        marks = next((m for a, m in ANSWERS.items() if a in answer), 2.0)
        return json.dumps({"quality": (marks / 4) ** (1 / 0.866), "feedback": "ok", "missing_points": []})


@pytest.fixture
def rig(client, tmp_path, monkeypatch):
    from server import retrain as retrain_module
    from src.models import model_manager, weights_download
    from src.models.arch_gate import ArchitectureGate
    from src.models.capability_prober import ProbeReport, ProbeResult
    from src.models.hf_resolver import HFSourceResolver
    from src.models.model_manager import ModelManager
    from src.models.override_registry import load_overrides
    from src.models.training_router import Hardware
    from tests.model_fakes import GEMMA, QWEN, FakeHF, FakeProbe

    services = client.app.state.services
    for i in range(60):
        answer = list(ANSWERS)[i % 4]
        services.corrections.add_grade_correction(
            exam_id="e", sheet_id=f"sheet-{i}", question_id="2", subject="Biology", model="qwen3.5:9b",
            question_text="What is photosynthesis?", model_answer="Plants use sunlight to make glucose.",
            student_answer=answer, max_marks=4, ai_marks=2.5, teacher_marks=ANSWERS[answer], ai_quality=0.6,
            strictness=50)
    texts = [c.student_answer for c in services.corrections.grade_corrections()] + ["What is photosynthesis ?"]
    tiny = make_tiny_qwen(tmp_path / "weights", texts)

    services.models = ModelManager(
        cache=services.model_cache, probe=FakeProbe((QWEN, GEMMA)), secrets=services.secrets,
        hardware=Hardware("RTX 4060 Laptop", 8.0, 180.0, False), gate=ArchitectureGate(train_python=tmp_path / "x"),
        resolver=HFSourceResolver(cache=services.model_cache, http=FakeHF().client(),
                                  overrides=load_overrides(services.model_cache)))
    passed = ProbeReport("t", None, (ProbeResult("json", "JSON adherence", "pass", True, ""),), True)
    monkeypatch.setattr(services.models, "run_probe", lambda name, llm, progress=None: passed)
    services.make_llm = lambda name: TeacherLikeLLM()
    monkeypatch.setattr(weights_download, "weights_path", lambda repo, root=None: tiny)
    monkeypatch.setattr(weights_download, "is_complete", lambda repo, root=None: True)
    monkeypatch.setattr(config, "LLM_CHECKPOINTS_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(retrain_module, "RUN_DIR", tmp_path / "runs")
    imported = []

    def fake_import(model_dir, name, base_model, version, *, probe=None, meta=None, **kwargs):
        assert (model_dir / "config.json").exists() and list(model_dir.glob("*.safetensors"))
        imported.append(f"{name}:{version}")
        folder = config.LLM_CHECKPOINTS_DIR / name / version
        record = {"ollama_name": f"{name}:{version}", "base_model": base_model, **(meta or {})}
        (folder / "meta.json").write_text(json.dumps(record))
        return record | {"path": str(folder)}

    monkeypatch.setattr(model_manager, "import_safetensors", fake_import)
    return client, services, imported


def test_a_round_trains_merges_compares_and_promotes(rig, tmp_path):
    client, services, imported = rig
    assert any(split_of(c) == "holdout" for c in services.corrections.grade_corrections())
    job = client.post("/api/learning/llm/retrain", json={"model": "qwen3.5:9b"}).json()
    meta = wait(client, job["job_id"], timeout=600)

    assert imported == ["gradeforge-qwen3-5-9b:v1"]
    assert meta["promoted"] and meta["verdict"] == "better on held-out answers"
    assert meta["after"]["mae"] < meta["before"]["mae"]
    assert services.settings["model"] == "gradeforge-qwen3-5-9b:v1"        # now grading with it
    assert meta["route"] == "stream" and meta["examples"] > 0 and meta["train"]["steps"] > 0
    version = config.LLM_CHECKPOINTS_DIR / "gradeforge-qwen3-5-9b" / "v1"
    assert (version / "adapter" / "adapter_model.safetensors").exists()
    assert not (tmp_path / "runs" / "gradeforge-qwen3-5-9b-v1" / "merged").exists()   # the 19 GB copy is gone

    [history] = services.corrections.training_history("llm_lora")
    assert history["promoted"] and history["metric_after"] < history["metric_before"]
    runs = services.corrections.benchmark_runs()
    assert {r["trigger"] for r in runs} == {"retrain v1"} and len(runs) == 2
    progress = client.get("/api/learning/progress").json()
    assert progress["training"][0]["version"] == "v1"
    [listed] = client.get("/api/learning/llm/versions", params={"model": "qwen3.5:9b"}).json()
    assert listed["promoted"] and listed["ollama_name"] == "gradeforge-qwen3-5-9b:v1"

    # grading with the fine-tune means training it again continues from it: nothing new yet
    again = client.post("/api/learning/llm/retrain", json={}).json()
    assert "nothing new to learn" in failed_job(client, again["job_id"])
