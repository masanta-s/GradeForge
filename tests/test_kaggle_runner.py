"""Pushing the fine-tuning notebook to Kaggle, watching it, and bringing the adapter back."""
import json
import zipfile

import pytest

from src import config
from src.models import kaggle_auth, kaggle_runner
from src.models.kaggle_runner import Run, fetch_adapter, finished, load_run, poll, push, save_run, tail_log
from tests.test_models_api import api  # noqa: F401  (fixture)
from tests.test_server import client  # noqa: F401  (the api fixture builds on it)

LOG = [{"data": "/usr/local/lib/warning about nbformat"}, {"data": "GPUs: 2"},
       {"data": "step 1/24  loss 1.2  3.4 s/step  GPU 0 9.8 GiB, GPU 1 7.4 GiB"},
       {"data": "[NbConvertApp] Converting notebook"}]


class FakeKaggle:
    """Stands in for Kaggle's client: records what was pushed, then reports a finished run."""

    def __init__(self, state="COMPLETE"):
        self.config_values = {"username": "tester"}
        self.state = state
        self.pushed = None

    def kernels_push(self, folder):
        from pathlib import Path

        self.pushed = json.loads((Path(folder) / "kernel-metadata.json").read_text())
        self.notebook = json.loads((Path(folder) / "finetune.ipynb").read_text())
        return type("Pushed", (), {"url": "https://kaggle.com/code/tester/gradeforge-finetune",
                                   "version_number": 3, "error": None})()

    def kernels_status(self, kernel):
        return type("Status", (), {"status": f"KernelWorkerStatus.{self.state}", "failure_message": ""})()

    def kernels_output(self, kernel, path):
        from pathlib import Path

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        (out / "gradeforge-finetune.log").write_text(json.dumps(LOG), encoding="utf-8")
        with zipfile.ZipFile(out / "gradeforge_adapter.zip", "w") as archive:
            archive.writestr("adapter_model.safetensors", b"weights")
            archive.writestr("adapter_config.json", json.dumps({"r": 16, "lora_alpha": 16}))
            archive.writestr("gradeforge.json", json.dumps({"examples": 42, "trained_until": "2026-09-12T00:00:00"}))


def test_push_asks_for_two_t4s_internet_and_privacy(tmp_path):
    fake = FakeKaggle()
    run = push({"cells": []}, model="qwen3.5:9b", examples=42, folder=tmp_path / "push", client=fake)
    assert fake.pushed["machine_shape"] == "NvidiaTeslaT4"      # the probe showed this gives 2x T4
    assert fake.pushed["enable_gpu"] == "true" and fake.pushed["enable_internet"] == "true"
    assert fake.pushed["is_private"] == "true" and fake.pushed["id"] == "tester/gradeforge-finetune"
    assert (run.kernel, run.version, run.examples, run.status) == ("tester/gradeforge-finetune", 3, 42, "QUEUED")
    assert not run.imported and run.url.endswith("gradeforge-finetune")


def test_status_and_log(tmp_path):
    run = Run("tester/k", "url", 1, "now", "qwen3.5:9b", 42)
    assert not finished(run)
    running = poll(run, client=FakeKaggle("RUNNING"))
    assert running.status == "RUNNING" and not finished(running)
    done = poll(run, client=FakeKaggle("COMPLETE"))
    assert done.status == "COMPLETE" and finished(done)
    assert finished(poll(run, client=FakeKaggle("ERROR")))

    lines = tail_log(run, tmp_path / "log", client=FakeKaggle())
    assert lines == ["GPUs: 2", "step 1/24  loss 1.2  3.4 s/step  GPU 0 9.8 GiB, GPU 1 7.4 GiB"]


def test_fetch_adapter_unpacks_the_zip(tmp_path):
    run = Run("tester/k", "url", 1, "now", "qwen3.5:9b", 42)
    adapter = fetch_adapter(run, tmp_path / "out", client=FakeKaggle())
    assert (adapter / "adapter_model.safetensors").read_bytes() == b"weights"
    assert json.loads((adapter / "gradeforge.json").read_text())["examples"] == 42


def test_run_is_remembered_between_restarts(tmp_path):
    path = tmp_path / "kaggle_run.json"
    assert load_run(path) is None
    run = Run("tester/k", "url", 2, "now", "qwen3.5:9b", 10, status="RUNNING")
    save_run(run, path)
    assert load_run(path) == run
    path.write_text("not json", encoding="utf-8")
    assert load_run(path) is None          # a damaged file just means "no run"


@pytest.fixture
def kaggle_api(api, tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle_runner, "RUN_FILE", tmp_path / "kaggle_run.json")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    return api


def test_endpoints_need_consent_and_a_token(kaggle_api, monkeypatch):
    assert kaggle_api.get("/api/learning/llm/kaggle").json() == {"configured": False, "run": None, "log": []}
    assert kaggle_api.post("/api/learning/llm/kaggle/start", json={}).status_code == 403      # no consent
    body = {"model": "qwen3.5:9b", "consent": True}
    assert "KAGGLE_API_TOKEN" in kaggle_api.post("/api/learning/llm/kaggle/start", json=body).json()["detail"]

    monkeypatch.setenv(kaggle_auth.TOKEN_ENV, "kg_test")
    assert kaggle_api.get("/api/learning/llm/kaggle").json()["configured"]
    assert kaggle_api.post("/api/learning/llm/kaggle/import").status_code == 409               # nothing to import


def test_a_kaggle_round_starts_and_reports(kaggle_api, monkeypatch):
    from tests.test_models_api import wait

    services = kaggle_api.app.state.services
    for i in range(12):   # a few, so some land in the training split (1 in 5 is held out)
        services.corrections.add_grade_correction(
            exam_id="e", sheet_id=f"s{i}", question_id="2", subject="Biology", model="qwen3.5:9b",
            question_text="What is photosynthesis?", model_answer="Plants make glucose.",
            student_answer=f"plants make food {i}", max_marks=4, ai_marks=3, teacher_marks=1, ai_quality=0.7,
            strictness=50)
    wait(kaggle_api, kaggle_api.post("/api/models/resolve", json={"name": "qwen3.5:9b"}).json()["job_id"])
    monkeypatch.setenv(kaggle_auth.TOKEN_ENV, "kg_test")
    fake = FakeKaggle("RUNNING")
    monkeypatch.setattr(kaggle_auth, "api", lambda: fake)

    start = kaggle_api.post("/api/learning/llm/kaggle/start", json={"model": "qwen3.5:9b", "consent": True})
    run = wait(kaggle_api, start.json()["job_id"], timeout=180)   # first call imports transformers
    assert run["kernel"] == "tester/gradeforge-finetune" and run["examples"] >= 1
    notebook = json.dumps(fake.notebook)
    assert "SPLIT_ACROSS_GPUS = True" in notebook and "Qwen/Qwen3.5-9B" in notebook   # 22 GB split over 2 T4s

    status = kaggle_api.get("/api/learning/llm/kaggle").json()
    assert status["run"]["status"] == "RUNNING" and not status.get("ready")
    assert status["log"] == ["GPUs: 2", "step 1/24  loss 1.2  3.4 s/step  GPU 0 9.8 GiB, GPU 1 7.4 GiB"]
    assert kaggle_api.post("/api/learning/llm/kaggle/import").json()["detail"].endswith("running")

    fake.state = "COMPLETE"
    assert kaggle_api.get("/api/learning/llm/kaggle").json()["ready"]
