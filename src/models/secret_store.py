"""API keys and the optional HuggingFace token, kept in the OS credential store via `keyring`
(Windows Credential Manager, DPAPI-protected), never in data/ or settings.json.

One credential per provider (service "gradeforge", username = provider). Windows caps a
credential blob at 2560 bytes, about 1280 characters once keyring UTF-16-encodes it, so keys
are never bundled together into one JSON credential.
"""
from __future__ import annotations

SERVICE = "gradeforge"
MAX_SECRET_CHARS = 1280


class SecretStore:
    def __init__(self, service: str = SERVICE):
        self.service = service

    def get(self, name: str) -> str | None:
        import keyring

        return keyring.get_password(self.service, name)

    def set(self, name: str, value: str) -> None:
        value = value.strip()
        if not value:
            raise ValueError("empty key")
        if len(value) > MAX_SECRET_CHARS:
            raise ValueError(f"key is longer than {MAX_SECRET_CHARS} characters, which Windows can't store")
        self._write(name, value)

    def delete(self, name: str) -> None:
        import keyring
        from keyring.errors import PasswordDeleteError

        try:
            keyring.delete_password(self.service, name)
        except PasswordDeleteError:
            pass  # nothing stored

    def masked(self, name: str) -> str | None:
        """'sk-…aBcD': enough to recognise which key is stored, never enough to use it."""
        value = self.get(name)
        if not value:
            return None
        return f"{value[:3]}…{value[-4:]}" if len(value) > 10 else "•" * len(value)

    def _write(self, name: str, value: str) -> None:
        import keyring

        keyring.set_password(self.service, name, value)


class MemorySecretStore(SecretStore):
    """In-process store for tests (and anywhere the OS credential store must not be touched)."""

    def __init__(self):
        super().__init__("memory")
        self._values: dict[str, str] = {}

    def get(self, name: str) -> str | None:
        return self._values.get(name)

    def delete(self, name: str) -> None:
        self._values.pop(name, None)

    def _write(self, name: str, value: str) -> None:
        self._values[name] = value
