"""Kaggle credentials, read from the environment (i.e. the .env file at the project root).

Kaggle's own client reads KAGGLE_API_TOKEN, so nothing has to be copied anywhere: src.config loads
.env on import, and .env is git-ignored (.env.example is the shared template). The value may be the
token itself or the path of a file containing it. Deliberately not the OS credential store: the
token stays in the project folder the teacher controls.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import os

from src import config  # noqa: F401  (loads .env before this module reads the environment)

TOKEN_ENV = "KAGGLE_API_TOKEN"


class KaggleNotConfigured(RuntimeError):
    pass


@dataclass(frozen=True)
class Quota:
    username: str
    gpu_hours_left: float | None
    gpu_hours_total: float | None
    tpu_hours_left: float | None
    refreshes: str | None


def stored_token() -> tuple[str | None, str]:
    """(token, where it came from). The token itself never leaves this process."""
    value = (os.environ.get(TOKEN_ENV) or "").strip()
    if not value:
        return None, ""
    path = Path(value)
    if len(value) < 260 and path.exists():
        return (path.read_text(encoding="utf-8").strip() or None), str(path)
    return value, f"{TOKEN_ENV} (.env)"


def source() -> str:
    return stored_token()[1]


def configured() -> bool:
    return bool(stored_token()[0])


def api():
    """An authenticated Kaggle client, or KaggleNotConfigured with what to do about it."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    token, _ = stored_token()
    if not token:
        raise KaggleNotConfigured(
            f"no Kaggle token: put {TOKEN_ENV}=... in the .env file in the project folder "
            "(copy .env.example), then restart GradeForge")
    os.environ[TOKEN_ENV] = token   # the client reads the value, not a file path
    client = KaggleApi()
    client.authenticate()
    return client


def check() -> Quota:
    """Prove the token works and report what's left of this week's free GPU time."""
    client = api()
    response = client.quota_view()

    def hours(quota):
        if quota is None:
            return None, None
        total = quota.total_time_allowed.total_seconds() / 3600
        return round(total, 1), round(max(0.0, total - quota.time_used.total_seconds() / 3600), 1)

    gpu_total, gpu_left = hours(response.gpu_quota)
    _, tpu_left = hours(response.tpu_quota)
    return Quota(username=client.config_values.get("username", ""), gpu_hours_left=gpu_left,
                 gpu_hours_total=gpu_total, tpu_hours_left=tpu_left,
                 refreshes=response.quota_refresh_time.isoformat() if response.quota_refresh_time else None)
