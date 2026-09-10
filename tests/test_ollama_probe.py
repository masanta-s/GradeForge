import pytest

from src import config
from src.models import ollama_probe
from src.models.ollama_probe import VRAMMeasurement


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1:11434", "http://127.0.0.1:11434"),
        ("0.0.0.0", "http://127.0.0.1:11434"),
        ("http://localhost", "http://localhost:11434"),
        ("https://ollama.lan:8443", "https://ollama.lan:8443"),
    ],
)
def test_normalise_ollama_url(host, expected):
    assert config._normalise_ollama_url(host) == expected


def test_vram_measurement_placement():
    on_gpu = VRAMMeasurement("m", 8192, size_bytes=100, size_vram_bytes=100, measured_at=0)
    split = VRAMMeasurement("m", 8192, size_bytes=100, size_vram_bytes=60, measured_at=0)
    empty = VRAMMeasurement("m", 8192, size_bytes=0, size_vram_bytes=0, measured_at=0)

    assert on_gpu.fully_on_gpu and on_gpu.gpu_fraction == 1.0
    assert not split.fully_on_gpu and split.gpu_fraction == pytest.approx(0.6)
    assert not empty.fully_on_gpu and empty.gpu_fraction == 0.0


def test_measurement_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "RESOLUTION_CACHE_DB", tmp_path / "cache.db")

    first = VRAMMeasurement("qwen3.5:9b", 8192, 7_000, 7_000, 1.0)
    ollama_probe.save_measurement(first)
    assert ollama_probe.load_measurement("qwen3.5:9b", 8192) == first
    assert ollama_probe.load_measurement("qwen3.5:9b", 32768) is None

    updated = VRAMMeasurement("qwen3.5:9b", 8192, 7_500, 7_000, 2.0)
    ollama_probe.save_measurement(updated)
    assert ollama_probe.load_measurement("qwen3.5:9b", 8192) == updated
