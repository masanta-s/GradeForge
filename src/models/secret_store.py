"""Every credential comes from one place: the .env file in the project folder.

`src.config` loads it on import, so any process (server, scripts, tests) sees the same values, and
real environment variables still win for CI or a one-off shell. `.env` is git-ignored; the
committed `.env.example` lists the same names with empty values.

Nothing is written to the OS credential store, and GradeForge never writes secrets itself: the
teacher edits .env. The app only ever reports whether a name is set, and a masked hint.
"""
from __future__ import annotations

import os

# GradeForge's name for a credential -> the variable in .env
ENV_NAMES = {
    "kaggle": "KAGGLE_API_TOKEN",
    "huggingface": "HF_TOKEN",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
ENV_FILE = ".env"


class SecretStore:
    """Read-only view of the credentials in .env."""

    def variable(self, name: str) -> str:
        return ENV_NAMES.get(name, name.upper())

    def get(self, name: str) -> str | None:
        return (os.environ.get(self.variable(name), "") or "").strip() or None

    def source(self, name: str) -> str:
        """Where a key came from, for the settings page. Never the key itself."""
        from src import config

        variable = self.variable(name)
        if not self.get(name):
            return ""
        return f"{variable} in {ENV_FILE}" if variable in config.ENV_FILE_VARS else f"{variable} (environment)"

    def masked(self, name: str) -> str | None:
        """'sk-…aBcD': enough to recognise which key is set, never enough to use it."""
        value = self.get(name)
        if not value:
            return None
        return f"{value[:3]}…{value[-4:]}" if len(value) > 10 else "•" * len(value)

    def status(self, name: str) -> dict:
        return {"name": name, "variable": self.variable(name), "set": self.get(name) is not None,
                "masked": self.masked(name), "source": self.source(name)}


class MemorySecretStore(SecretStore):
    """For tests and demos: values live in this object only, never in the environment."""

    def __init__(self):
        self._values: dict[str, str] = {}

    def get(self, name: str) -> str | None:
        return self._values.get(name)

    def source(self, name: str) -> str:
        return "memory" if self._values.get(name) else ""

    def set(self, name: str, value: str) -> None:
        self._values[name] = value.strip()

    def delete(self, name: str) -> None:
        self._values.pop(name, None)
