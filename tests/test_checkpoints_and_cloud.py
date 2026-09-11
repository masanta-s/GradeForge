"""Checkpoint clean-up, GGUF import, secrets and cloud-provider helpers."""
import json
from datetime import date
from pathlib import Path

import litellm
import pytest

from demo.samples import SAMPLE_KEY
from src.models.checkpoint_gc import CheckpointGC
from src.models.cloud_provider import CloudProvider, chat_models, estimate_cost, litellm_model
from src.models.model_manager import import_gguf, modelfile_directives
from src.models.secret_store import MAX_SECRET_CHARS, MemorySecretStore, SecretStore
from tests.model_fakes import QWEN_MODELFILE, FakeProbe


# --- checkpoint GC --------------------------------------------------------------------------

def _tree(root):
    llm, trocr, transient = root / "llm_checkpoints", root / "trocr_finetuned", root / "train"
    for v in range(1, 5):
        folder = llm / "gradeforge-grader" / f"v{v}"
        folder.mkdir(parents=True)
        (folder / "model.gguf").write_bytes(b"GGUF" + b"x" * 1000 * v)
        (folder / "meta.json").write_text(json.dumps({"ollama_name": f"gradeforge-grader:v{v}"}))
        adapter = trocr / f"v{v}"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"a" * 100)
    (trocr / "active.json").write_text(json.dumps({"version": "v1", "cer": 0.02}))
    transient.mkdir()
    (transient / "merged").mkdir()
    (transient / "merged" / "weights.bin").write_bytes(b"w" * 5000)
    return llm, trocr, transient


def test_gc_keeps_newest_and_in_use_versions(tmp_path):
    llm, trocr, transient = _tree(tmp_path)
    removed_from_ollama = []
    gc = CheckpointGC(llm, trocr, transient, remove_ollama=removed_from_ollama.append)

    plan = gc.plan(keep_latest_n=2, protect={"gradeforge-grader:v1"})
    planned = {(i.kind, Path(i.path).name) for i in plan}
    # v4, v3 are newest; v1 is in use (LLM: selected in Settings, TrOCR: active adapter)
    assert planned == {("llm_checkpoint", "v2"), ("trocr_adapter", "v2"), ("transient", "merged")}
    assert all((llm / "gradeforge-grader" / v).exists() for v in ("v1", "v2", "v3", "v4"))  # plan deletes nothing

    result = gc.collect(keep_latest_n=2, protect={"gradeforge-grader:v1"})
    assert not result.errors and len(result.removed) == 3 and result.freed_bytes == sum(i.size_bytes for i in plan)
    assert removed_from_ollama == ["gradeforge-grader:v2"]
    assert sorted(p.name for p in (llm / "gradeforge-grader").iterdir()) == ["v1", "v3", "v4"]
    assert sorted(p.name for p in trocr.iterdir()) == ["active.json", "v1", "v3", "v4"]
    assert transient.exists() and not any(transient.iterdir())
    assert gc.plan(2, {"gradeforge-grader:v1"}) == []


def test_gc_never_deletes_outside_its_folders(tmp_path, monkeypatch):
    llm, trocr, transient = _tree(tmp_path)
    precious = tmp_path / "corrections.db"
    precious.write_text("teacher intent")
    gc = CheckpointGC(llm, trocr, transient)
    with pytest.raises(ValueError):
        gc.plan(keep_latest_n=0)
    from src.models.checkpoint_gc import GCItem

    monkeypatch.setattr(gc, "plan", lambda *a, **k: [GCItem(str(precious), "transient", 1, "bad plan")])
    result = gc.collect()
    assert precious.exists() and "outside the checkpoint folders" in result.errors[0]


def test_disk_report(tmp_path):
    llm, trocr, transient = _tree(tmp_path)
    report = CheckpointGC(llm, trocr, transient).get_disk_report()
    assert report.llm_checkpoints_bytes == sum(4 + 1000 * v for v in range(1, 5)) + sum(
        len(json.dumps({"ollama_name": f"gradeforge-grader:v{v}"})) for v in range(1, 5))
    assert report.transient_bytes == 5000 and report.free_bytes > 0 and report.drive


# --- GGUF import ----------------------------------------------------------------------------

def test_modelfile_keeps_template_renderer_and_parameters_only():
    lines = modelfile_directives(QWEN_MODELFILE)
    assert lines == ["TEMPLATE {{ .Prompt }}", "RENDERER qwen3.5", "PARSER qwen3.5",
                     "PARAMETER temperature 1", "PARAMETER top_k 20"]


def test_import_gguf_registers_a_versioned_model(tmp_path):
    gguf = tmp_path / "download" / "unsloth.Q4_K_M.gguf"
    gguf.parent.mkdir()
    gguf.write_bytes(b"GGUF" + b"\0" * 64)
    commands = []

    class Done:
        returncode, stdout, stderr = 0, "success", ""

    def run(cmd, **kwargs):
        commands.append((cmd, kwargs["cwd"]))
        return Done()

    root = tmp_path / "llm_checkpoints"
    meta = import_gguf(gguf, "gradeforge-grader", "gemma4:e4b", root=root, probe=FakeProbe(), run=run,
                       hf_repo="google/gemma-4-E4B-it")
    folder = root / "gradeforge-grader" / "v1"
    assert meta["ollama_name"] == "gradeforge-grader:v1" and (folder / "model.gguf").exists()
    modelfile = (folder / "Modelfile").read_text()
    assert modelfile.startswith("FROM ./model.gguf\n") and "RENDERER qwen3.5" in modelfile
    assert "LICENSE" not in modelfile
    cmd, cwd = commands[0]
    assert cmd[1:] == ["create", "gradeforge-grader:v1", "-f", str(folder / "Modelfile")] and cwd == folder
    assert json.loads((folder / "meta.json").read_text())["hf_repo"] == "google/gemma-4-E4B-it"

    meta2 = import_gguf(gguf, "gradeforge-grader", "gemma4:e4b", root=root, probe=FakeProbe(), run=run)
    assert meta2["ollama_name"] == "gradeforge-grader:v2"


def test_import_gguf_rejects_bad_input_and_cleans_up(tmp_path):
    fake = tmp_path / "notes.gguf"
    fake.write_bytes(b"PK\x03\x04")
    root = tmp_path / "llm_checkpoints"
    with pytest.raises(ValueError, match="not a valid GGUF"):
        import_gguf(fake, "grader", "gemma4:e4b", root=root, probe=FakeProbe())
    with pytest.raises(ValueError, match="model name"):
        import_gguf(fake, "Bad Name!", "gemma4:e4b", root=root, probe=FakeProbe())

    good = tmp_path / "m.gguf"
    good.write_bytes(b"GGUF")

    class Failed:
        returncode, stdout, stderr = 1, "", "Error: unsupported architecture"

    with pytest.raises(RuntimeError, match="unsupported architecture"):
        import_gguf(good, "grader", "gemma4:e4b", root=root, probe=FakeProbe(), run=lambda *a, **k: Failed())
    assert not (root / "grader" / "v1").exists()


# --- secrets --------------------------------------------------------------------------------

def test_secret_store_masks_and_limits_keys():
    store = SecretStore(service="gradeforge-test")   # the in-memory keyring from conftest
    store.set("openai", "  sk-proj-abcdefghijklmnopWXYZ  ")
    assert store.get("openai") == "sk-proj-abcdefghijklmnopWXYZ" and store.masked("openai") == "sk-…WXYZ"
    with pytest.raises(ValueError, match="1280"):
        store.set("openai", "x" * (MAX_SECRET_CHARS + 1))
    with pytest.raises(ValueError):
        store.set("openai", "   ")
    store.delete("openai")
    store.delete("openai")   # deleting twice is fine
    assert store.get("openai") is None and store.masked("openai") is None


# --- cloud provider -------------------------------------------------------------------------

def test_chat_models_come_from_litellms_price_list():
    openai = chat_models("openai", today=date(2026, 9, 11))
    assert "gpt-4o-mini" in openai
    assert not any(m.startswith("ft:") or m.endswith("-2024-08-06") for m in openai)
    anthropic = chat_models("anthropic", today=date(2026, 9, 11))
    assert "claude-sonnet-4-5" in anthropic                  # deprecation_date 2026-09-29: still listed
    assert "claude-sonnet-4-5" not in chat_models("anthropic", today=date(2026, 10, 1))
    gemini = chat_models("gemini")
    assert "gemini/gemini-2.5-flash" in gemini and all(m.startswith("gemini/") for m in gemini)
    assert not any("lyria" in m or "container" in m for m in gemini + openai)


def test_litellm_model_names():
    assert litellm_model("openai", "gpt-4o-mini") == "gpt-4o-mini"
    assert litellm_model("gemini", "gemini/gemini-2.5-flash") == "gemini/gemini-2.5-flash"
    assert litellm_model("gemini", "gemini-flash-latest") == "gemini/gemini-flash-latest"  # not Vertex
    assert litellm_model("openai", " brand-new-model ") == "openai/brand-new-model"


def test_cost_estimate_counts_only_llm_graded_questions():
    estimate = estimate_cost("gpt-4o-mini", SAMPLE_KEY.questions, papers=30)
    llm_graded = [q for q in SAMPLE_KEY.questions
                  if q.qtype in ("short", "descriptive") or (q.qtype == "mixed" and q.has_written_part)]
    diagrams = [q for q in SAMPLE_KEY.questions if q.diagram is not None]
    assert estimate.calls_per_paper == len(llm_graded) + len(diagrams)
    assert 0 < estimate.usd < 1 and estimate.per_paper_usd * 30 == pytest.approx(estimate.usd, abs=1e-3)
    assert estimate_cost("gpt-4o", SAMPLE_KEY.questions, 30).usd > estimate.usd
    unknown = estimate_cost("somebody/unknown-model", SAMPLE_KEY.questions, 30)
    assert unknown.usd is None and "no price" in unknown.note


def test_api_key_check_explains_failures():
    def rejects(**kwargs):
        raise litellm.AuthenticationError("bad key", llm_provider="openai", model=kwargs["model"])

    def not_found(**kwargs):
        raise litellm.NotFoundError("no such model", llm_provider="openai", model=kwargs["model"])

    seen = {}

    def works(**kwargs):
        seen.update(kwargs)
        return object()

    secrets = MemorySecretStore()
    assert not CloudProvider(secrets, completion=works).validate_api_key("openai", "gpt-4o-mini").ok
    secrets.set("openai", "sk-test-1234567890")
    assert "rejected" in CloudProvider(secrets, completion=rejects).validate_api_key("openai", "gpt-4o-mini").detail
    assert "not found" in CloudProvider(secrets, completion=not_found).validate_api_key("openai", "gpt-9").detail
    check = CloudProvider(secrets, completion=works).validate_api_key("openai", "gpt-4o-mini")
    assert check.ok and seen["api_key"] == "sk-test-1234567890" and seen["max_tokens"] == 5
    assert "student" not in json.dumps(seen["messages"]).lower()


def test_provider_status_never_returns_the_key():
    secrets = MemorySecretStore()
    provider = CloudProvider(secrets)
    provider.save_key("anthropic", "sk-ant-api03-secretsecretABCD")
    status = {p["id"]: p for p in provider.status()}
    assert status["anthropic"]["has_key"] and status["anthropic"]["masked_key"] == "sk-…ABCD"
    assert "secretsecret" not in json.dumps(status) and not status["openai"]["has_key"]
    with pytest.raises(ValueError, match="unknown provider"):
        provider.save_key("myspace", "k")
