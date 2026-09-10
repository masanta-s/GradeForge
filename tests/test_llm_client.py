from types import SimpleNamespace

import pytest

from src import config
from src.knowledge import llm_client
from src.knowledge.llm_client import CloudConsentRequired, LLMClient


@pytest.fixture
def captured(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content='{"ok": true}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(llm_client.litellm, "completion", fake_completion)
    return calls


def _local(can_think=True, **kwargs):
    client = LLMClient(**kwargs)
    client.__dict__["can_think"] = can_think  # skip the Ollama capability lookup
    return client


def test_local_call_forwards_ollama_options(captured):
    out = _local().complete([{"role": "user", "content": "hi"}], schema={"type": "object"})
    [kw] = captured
    assert out == '{"ok": true}'
    assert kw["model"] == f"ollama_chat/{config.DEFAULT_LLM}"
    assert kw["api_base"] == config.OLLAMA_URL
    assert kw["num_ctx"] == config.LLM_NUM_CTX == 8192
    assert kw["think"] is False  # thinking costs ~55 s per grading call
    assert kw["temperature"] == 0.0
    assert kw["response_format"]["json_schema"]["schema"] == {"type": "object"}


def test_think_not_sent_to_models_without_the_capability(captured):
    _local(can_think=False).complete([{"role": "user", "content": "hi"}])
    assert "think" not in captured[0]


def test_callable_protocol_used_by_grading(captured):
    _local()([{"role": "user", "content": "hi"}], schema={"type": "object"})
    assert "response_format" in captured[0]


def test_cloud_requires_explicit_consent():
    with pytest.raises(CloudConsentRequired):
        LLMClient("gpt-4o", provider="cloud", api_key="sk-test")


def test_cloud_call_shape(captured):
    client = LLMClient("gemini/gemini-2.5-flash", provider="cloud", api_key="key", cloud_consent=True)
    client.complete([{"role": "user", "content": "hi"}])
    [kw] = captured
    assert kw["model"] == "gemini/gemini-2.5-flash"
    assert kw["api_key"] == "key"
    assert "api_base" not in kw and "think" not in kw and "num_ctx" not in kw
