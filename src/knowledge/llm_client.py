"""LiteLLM interface to local (Ollama) and opt-in cloud models.

Measured with qwen3.5:9b through LiteLLM 1.100 (2026-09-10): thinking is ON by default and
turns a 3-second grading call into ~55 s / ~1800 tokens, so `think=False` is the default here.
`think`, `num_ctx` and JSON-schema `response_format` are all forwarded to Ollama by LiteLLM.

An LLMClient instance is a plain `messages -> str` callable, which is all the network-free
grading package ever sees (see src/grading/structured_output.CompletionFn).
"""
from __future__ import annotations

from functools import cached_property
from typing import Literal

from src import config  # noqa: F401  (must precede litellm: disables its GitHub price-map fetch)

import litellm  # noqa: E402

Provider = Literal["ollama", "cloud"]


class CloudConsentRequired(PermissionError):
    pass


class LLMClient:
    def __init__(
        self,
        model: str = config.DEFAULT_LLM,
        provider: Provider = "ollama",
        *,
        api_key: str | None = None,
        cloud_consent: bool = False,
        temperature: float = 0.0,
        num_ctx: int = config.LLM_NUM_CTX,
        timeout: float = 180.0,
    ):
        if provider == "cloud" and not cloud_consent:
            raise CloudConsentRequired(
                "cloud models send exam data off this machine — the teacher must explicitly consent first"
            )
        self.model = model
        self.provider = provider
        self.api_key = api_key
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout = timeout

    @property
    def litellm_model(self) -> str:
        # ollama_chat/ -> /api/chat (chat template, tools, JSON schema); never bare ollama/.
        return f"ollama_chat/{self.model}" if self.provider == "ollama" else self.model

    @cached_property
    def can_think(self) -> bool:
        if self.provider != "ollama":
            return False
        from src.models.ollama_probe import OllamaModelProbe

        return OllamaModelProbe().probe(self.model).can_think

    def complete(
        self,
        messages: list[dict],
        *,
        schema: dict | None = None,
        think: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        kwargs: dict = {
            "model": self.litellm_model,
            "messages": messages,
            "temperature": self.temperature,
            "timeout": self.timeout,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if schema:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": "response", "schema": schema}}
        if self.provider == "ollama":
            kwargs["api_base"] = config.OLLAMA_URL
            kwargs["num_ctx"] = self.num_ctx
            if self.can_think:  # Ollama rejects `think` for models without the capability
                kwargs["think"] = think
        else:
            kwargs["api_key"] = self.api_key

        response = litellm.completion(**kwargs)
        return response.choices[0].message.content or ""

    def __call__(self, messages: list[dict], schema: dict | None = None) -> str:
        return self.complete(messages, schema=schema)
