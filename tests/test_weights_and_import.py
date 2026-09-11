"""Downloading original weights (resumable, disk-checked) and registering a merged fine-tune."""
import json

import httpx
import pytest

from src.models import weights_download
from src.models.model_manager import import_safetensors, tuned_name
from tests.model_fakes import QWEN_MODELFILE, FakeProbe

FILES = {"config.json": b'{"model_type": "qwen3_5"}', "model-00001-of-00002.safetensors": b"A" * 3000,
         "model-00002-of-00002.safetensors": b"B" * 2000, "tokenizer.json": b"{}", "README.md": b"# hi",
         "subdir/extra.safetensors": b"x"}


class FakeHub:
    def __init__(self, fail_after: int | None = None):
        self.fail_after = fail_after
        self.ranges = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/api/models/"):
            return httpx.Response(200, json={"siblings": [{"rfilename": n, "size": len(b)} for n, b in FILES.items()]})
        name = request.url.path.split("/resolve/main/")[1]
        body = FILES[name]
        if (rng := request.headers.get("Range")):
            self.ranges.append((name, rng))
            start = int(rng.split("=")[1].rstrip("-"))
            return httpx.Response(206, content=body[start:])
        if self.fail_after is not None and name.endswith(".safetensors") and len(body) > self.fail_after:
            return httpx.Response(200, content=body[:self.fail_after])   # a cut-off download
        return httpx.Response(200, content=body)

    def client(self):
        return httpx.Client(base_url="https://huggingface.co", transport=httpx.MockTransport(self))


def test_download_resumes_and_marks_completion(tmp_path):
    hub = FakeHub(fail_after=1000)
    with pytest.raises(OSError, match="resume"):
        weights_download.download("Qwen/Qwen3.5-9B", root=tmp_path, http=hub.client())
    assert not weights_download.is_complete("Qwen/Qwen3.5-9B", root=tmp_path)
    folder = weights_download.weights_path("Qwen/Qwen3.5-9B", root=tmp_path)
    assert (folder / "model-00001-of-00002.safetensors.part").stat().st_size == 1000

    hub.fail_after = None
    seen = []
    path = weights_download.download("Qwen/Qwen3.5-9B", root=tmp_path, http=hub.client(),
                                     progress=lambda p, m: seen.append(p))
    assert path == folder and weights_download.is_complete("Qwen/Qwen3.5-9B", root=tmp_path)
    assert ("model-00001-of-00002.safetensors", "bytes=1000-") in hub.ranges      # resumed, not restarted
    assert (folder / "model-00001-of-00002.safetensors").read_bytes() == FILES["model-00001-of-00002.safetensors"]
    assert not (folder / "README.md").exists() and not (folder / "extra.safetensors").exists()
    assert seen[-1] == 1.0 and all(0 <= p <= 1 for p in seen)
    status = weights_download.status("Qwen/Qwen3.5-9B", root=tmp_path)
    assert status["complete"] and status["downloaded_bytes"] > 5000


def test_download_checks_free_disk_first(tmp_path, monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "disk_usage", lambda p: type("U", (), {"free": 1024, "total": 2048, "used": 1024})())
    with pytest.raises(OSError, match="free"):
        weights_download.download("Qwen/Qwen3.5-9B", root=tmp_path, http=FakeHub().client())


def test_tuned_model_names():
    assert tuned_name("qwen3.5:9b") == "gradeforge-qwen3-5-9b"
    assert tuned_name("Gemma4:E4B") == "gradeforge-gemma4-e4b"


def test_import_safetensors_uses_ollamas_converter_and_the_base_chat_format(tmp_path):
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text("{}")
    (merged / "model-00001-of-00001.safetensors").write_bytes(b"x")
    commands = []

    class Done:
        returncode, stdout, stderr = 0, "success", ""

    def run(cmd, **kwargs):
        commands.append(cmd)
        return Done()

    record = import_safetensors(merged, "gradeforge-qwen3-5-9b", "qwen3.5:9b", "v3", root=tmp_path / "ck",
                                probe=FakeProbe(), run=run, meta={"examples": 42})
    folder = tmp_path / "ck" / "gradeforge-qwen3-5-9b" / "v3"
    assert record["ollama_name"] == "gradeforge-qwen3-5-9b:v3" and record["examples"] == 42
    modelfile = (folder / "Modelfile").read_text()
    assert modelfile.startswith(f"FROM {merged.resolve()}\n")
    assert "RENDERER qwen3.5" in modelfile and "PARSER qwen3.5" in modelfile and "LICENSE" not in modelfile
    assert commands[0][1:] == ["create", "gradeforge-qwen3-5-9b:v3", "-f", str(folder / "Modelfile"),
                               "--quantize", "q4_K_M"]
    assert json.loads((folder / "meta.json").read_text())["base_model"] == "qwen3.5:9b"
    assert QWEN_MODELFILE  # the stand-in base model's Modelfile supplied those lines

    with pytest.raises(ValueError, match="safetensors model folder"):
        import_safetensors(tmp_path, "x", "qwen3.5:9b", "v1", root=tmp_path / "ck", probe=FakeProbe(), run=run)
