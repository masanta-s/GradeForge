"""Models, cloud and storage endpoints, with stand-in Ollama / HuggingFace / LLMs."""
import json
import time

import pytest

from src import config
from src.models.arch_gate import ArchitectureGate
from src.models.hf_resolver import HFSourceResolver
from src.models.model_manager import ModelManager
from src.models.override_registry import load_overrides
from src.models.training_router import Hardware
from tests.model_fakes import GEMMA, QWEN, TINY, BadLLM, FakeHF, FakeProbe, GoodLLM
from tests.test_server import client, wait  # noqa: F401  (fixture + helper)


@pytest.fixture
def api(client, tmp_path):
    services = client.app.state.services
    resolver = HFSourceResolver(cache=services.model_cache, http=FakeHF().client(),
                                overrides=load_overrides(services.model_cache))
    services.models = ModelManager(cache=services.model_cache, probe=FakeProbe((QWEN, GEMMA, TINY)),
                                   secrets=services.secrets, hardware=Hardware("RTX 4060 Laptop", 8.0, 180.0, False),
                                   gate=ArchitectureGate(train_python=tmp_path / "no-train-env" / "python.exe"),
                                   resolver=resolver)
    llms = {"qwen3.5:9b": GoodLLM(), "gemma4:e4b": GoodLLM(sees_images=False), "tiny:1b": BadLLM()}
    services.make_llm = lambda name: llms[name]
    return client


def failed(client, job_id, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            assert job["status"] == "failed", "expected the job to fail"
            return job["error"]
        time.sleep(0.05)
    raise TimeoutError(job_id)


def by_name(models):
    return {m["name"]: m for m in models}


def test_models_start_unchecked_and_listing_touches_no_network(api):
    models = by_name(api.get("/api/models").json())
    assert set(models) == {"qwen3.5:9b", "gemma4:e4b", "tiny:1b"}
    assert models["qwen3.5:9b"]["tier"]["level"] == "unknown" and models["qwen3.5:9b"]["active"]
    assert not models["gemma4:e4b"]["active"]


def test_probe_resolve_and_card_give_the_plans_tiers(api):
    report = wait(api, api.post("/api/models/probe", json={"name": "gemma4:e4b"}).json()["job_id"])
    assert report["critical_passed"] and not report["vision_ok"]
    assert by_name(api.get("/api/models").json())["gemma4:e4b"]["tier"]["level"] == "yellow"  # source unknown yet

    resolution = wait(api, api.post("/api/models/resolve", json={"name": "gemma4:e4b"}).json()["job_id"])
    assert resolution["repo"] == "google/gemma-4-E4B-it" and resolution["confidence"] == "high"

    card = api.get("/api/models/card", params={"name": "gemma4:e4b"}).json()
    assert card["tier"]["level"] == "green" and card["tier"]["where"] == "Colab/Kaggle export"
    assert card["trainability"]["supported"] and card["training_plan"]["recommended"] == "colab"
    assert card["training_plan"]["estimate"]["label"] == "QLoRA (4-bit)"
    assert card["probe"]["results"][3]["status"] == "fail"       # vision: flag said yes, model couldn't see
    assert not card["has_hf_token"]

    wait(api, api.post("/api/models/probe", json={"name": "qwen3.5:9b"}).json()["job_id"])
    wait(api, api.post("/api/models/resolve", json={"name": "qwen3.5:9b"}).json()["job_id"])
    qwen = api.get("/api/models/card", params={"name": "qwen3.5:9b"}).json()
    # LoRA needs ~22 GB, but its layers can be streamed through this 8 GB GPU
    assert qwen["tier"]["level"] == "green" and qwen["tier"]["where"] == "this computer, streamed"
    assert qwen["training_plan"]["estimate"]["vram_gb"] == 22


def test_not_right_override(api):
    body = {"name": "qwen3.5:9b", "repo": "Qwen/Qwen3.5-9B-Base"}
    pinned = wait(api, api.put("/api/models/resolution", json=body).json()["job_id"])
    assert pinned["source"] == "manual" and pinned["repo"] == "Qwen/Qwen3.5-9B-Base"
    error = failed(api, api.put("/api/models/resolution", json={**body, "repo": "No/Such"}).json()["job_id"])
    assert "not found" in error
    back = wait(api, api.put("/api/models/resolution", json={"name": "qwen3.5:9b"}).json()["job_id"])
    assert back["source"] == "auto" and back["repo"] == "Qwen/Qwen3.5-9B"


def test_switching_checks_first_and_refuses_incompatible_models(api):
    services = api.app.state.services
    error = failed(api, api.post("/api/models/switch", json={"name": "tiny:1b"}).json()["job_id"])
    assert "failed the json adherence" in error.lower() and services.settings["model"] == "qwen3.5:9b"

    done = wait(api, api.post("/api/models/switch", json={"name": "gemma4:e4b"}).json()["job_id"])
    assert done["model"] == "gemma4:e4b" and services.settings["model"] == "gemma4:e4b"
    # gemma failed the vision check, so diagrams are not sent to it
    assert not services.vision_allowed()
    assert failed(api, api.post("/api/models/switch", json={"name": "nope:1b"}).json()["job_id"])


def test_pull_measures_vram(api):
    result = wait(api, api.post("/api/models/pull", json={"name": "tiny:1b"}).json()["job_id"])
    assert result["size_vram_bytes"] == 3 * 2**30
    assert api.app.state.services.models.probe.pulled == ["tiny:1b"]


def test_credentials_are_reported_never_saved_by_the_app(api):
    names = {s["name"]: s for s in api.get("/api/models/secrets").json()}
    assert names["huggingface"]["variable"] == "HF_TOKEN" and not names["huggingface"]["set"]
    assert set(names) == {"kaggle", "huggingface", "openai", "anthropic", "gemini"}

    api.app.state.services.secrets.set("huggingface", "hf_abcdefghijklmnop")   # stands in for .env
    updated = {s["name"]: s for s in api.get("/api/models/secrets").json()}
    assert updated["huggingface"]["set"] and updated["huggingface"]["masked"] == "hf_…mnop"
    assert "abcdefghijkl" not in api.get("/api/models/secrets").text           # only ever masked
    assert api.put("/api/models/hf-token", json={"token": "x"}).status_code == 405   # no way to save one


def test_storage_report_and_clean_up(api, tmp_path, monkeypatch):
    from tests.test_checkpoints_and_cloud import _tree

    llm, trocr, transient = _tree(tmp_path / "store")
    monkeypatch.setattr(config, "LLM_CHECKPOINTS_DIR", llm)
    monkeypatch.setattr(config, "TROCR_FINETUNED_DIR", trocr)
    monkeypatch.setattr(config, "CACHE_DIR", transient.parent)
    services = api.app.state.services
    services.update_settings({"model": "gradeforge-grader:v1"})   # in use: never deleted

    report = api.get("/api/storage").json()
    assert {p["kind"] for p in report["plan"]} == {"llm_checkpoint", "trocr_adapter", "transient"}
    assert report["reclaimable_bytes"] > 0 and report["disk"]["free_bytes"] > 0
    result = api.post("/api/storage/clean", json={"keep": 2}).json()
    assert len(result["removed"]) == 3 and not result["errors"]
    assert services.models.probe.removed == ["gradeforge-grader:v2"]
    assert (llm / "gradeforge-grader" / "v1").exists()


def test_cloud_needs_a_key_and_consent(api):
    services = api.app.state.services
    overview = api.get("/api/cloud").json()
    assert "gpt-4o-mini" in overview["models"]["openai"] and not overview["settings"]["active"]

    body = {"provider": "cloud", "cloud_provider": "openai", "cloud_model": "gpt-4o-mini", "consent": False}
    assert api.put("/api/cloud/settings", json=body).status_code == 403          # no consent
    assert api.put("/api/cloud/settings", json={**body, "consent": True}).status_code == 409  # no key

    services.secrets.set("openai", "sk-proj-0123456789abcdWXYZ")   # stands in for OPENAI_API_KEY in .env
    status = api.get("/api/cloud").json()["providers"]
    openai = next(p for p in status if p["id"] == "openai")
    assert openai["masked_key"] == "sk-…WXYZ" and "0123456789" not in json.dumps(status)
    assert openai["variable"] == "OPENAI_API_KEY"

    settings = api.put("/api/cloud/settings", json={**body, "consent": True}).json()
    assert settings["active"] and services.model_name == "gpt-4o-mini"
    assert api.get("/api/health").json()["provider"] == "cloud"
    services._llm = None
    client_llm = services.llm
    assert (client_llm.provider, client_llm.model, client_llm.api_key) == ("cloud", "gpt-4o-mini",
                                                                          "sk-proj-0123456789abcdWXYZ")

    # removing the key from .env falls back to local grading at the next switch
    services.secrets.delete("openai")
    assert api.put("/api/cloud/settings", json={**body, "consent": True}).status_code == 409


def test_cloud_cost_estimate(api):
    typical = api.post("/api/cloud/estimate", json={"model": "gpt-4o-mini", "papers": 30}).json()
    assert typical["usd"] > 0 and typical["basis"].startswith("a typical")
    assert api.post("/api/cloud/estimate", json={"model": "gpt-4o-mini", "exam_id": "nope"}).status_code == 404


def test_llm_plan_and_notebook_export_follow_the_router(api):
    services = api.app.state.services
    plan = api.get("/api/learning/llm/plan", params={"model": "gemma4:e4b"}).json()
    assert plan["needs_source"] and plan["plan"] is None
    no_data = api.post("/api/learning/llm/export", json={"consent": True, "model": "gemma4:e4b"})
    assert no_data.status_code == 409 and "no written-answer corrections" in no_data.json()["detail"]

    from src.learning.evaluation import split_of

    for i in range(12):
        services.corrections.add_grade_correction(
            exam_id="e", sheet_id=f"riya-sharma-{i}", question_id="2", subject="Biology", model="qwen3.5:9b",
            question_text="What is photosynthesis?", model_answer="Plants make glucose using sunlight.",
            student_answer=f"plants make food {i}", max_marks=4, ai_marks=3, teacher_marks=1, ai_quality=0.7,
            strictness=50, note=f"too vague {i}", teacher_name="Mrs. Iyer")
    splits = {c.student_answer: split_of(c) for c in services.corrections.grade_corrections()}

    response = api.post("/api/learning/llm/export", json={"consent": True, "model": "gemma4:e4b"})
    assert response.status_code == 200 and "attachment" in response.headers["content-disposition"]
    notebook = json.loads(response.text)
    text = json.dumps(notebook)
    assert notebook["nbformat"] == 4 and "BASE_MODEL = 'google/gemma-4-E4B-it'" in text
    assert "LOAD_IN_4BIT = True" in text and "SPLIT_ACROSS_GPUS = False" in text   # QLoRA on one T4
    config_cell = "".join(notebook["cells"][2]["source"])
    data: dict = {}
    exec(config_cell, data)   # plain assignments: the notebook's training and validation data
    contents = "\n".join(m["content"] for e in data["EXAMPLES"] + data["VALIDATION"] for m in e["messages"])
    for answer, split in splits.items():   # held-out answers never leave: they judge the result
        assert (f"{answer}\n" in contents) == (split != "holdout"), (answer, split)
    assert "holdout" in splits.values()
    assert "riya" not in text.lower() and "Mrs. Iyer" not in text  # no student or teacher identity
    assert "Only " in text  # below the recommended 200
    plan = api.get("/api/learning/llm/plan", params={"model": "gemma4:e4b"}).json()
    assert plan["plan"]["recommended"] == "colab" and not plan["needs_source"] and not plan["weights_ready"]

    # qwen3.5:9b (22 GB): trains here by streaming; its notebook splits the model across Kaggle's two T4s
    qwen = api.post("/api/learning/llm/export", json={"consent": True, "model": "qwen3.5:9b"})
    assert qwen.status_code == 200 and "SPLIT_ACROSS_GPUS = True" in qwen.text and "LOAD_IN_4BIT = False" in qwen.text
