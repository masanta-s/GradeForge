"""Credentials come from the .env file in the project folder: one place, never written by the app."""
import json
from pathlib import Path

import pytest

from src import config
from src.models import kaggle_auth
from src.models.secret_store import ENV_NAMES
from tests.test_models_api import api  # noqa: F401  (fixture)
from tests.test_server import client  # noqa: F401  (the api fixture builds on it)


@pytest.fixture(autouse=True)
def no_inherited_token(monkeypatch):
    monkeypatch.delenv(kaggle_auth.TOKEN_ENV, raising=False)


def test_the_committed_example_lists_every_variable_with_empty_values():
    example = (config.PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    values = dict(line.split("=", 1) for line in example.splitlines() if "=" in line and not line.startswith("#"))
    assert set(values) == set(ENV_NAMES.values())
    assert not any(v.strip() for v in values.values())            # shared as a template, never with secrets
    ignored = (config.PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in ignored and ".env.example" not in ignored    # the real file stays out of git


def test_kaggle_token_read_from_the_environment(monkeypatch):
    assert kaggle_auth.stored_token() == (None, "") and not kaggle_auth.configured()
    monkeypatch.setenv(kaggle_auth.TOKEN_ENV, "  kg_secret_value  ")
    assert kaggle_auth.stored_token() == ("kg_secret_value", "KAGGLE_API_TOKEN (.env)")
    assert kaggle_auth.configured()


def test_kaggle_token_may_be_a_file_path(monkeypatch, tmp_path):
    path = tmp_path / "kaggle_token.txt"
    path.write_text("kg_from_file\n", encoding="utf-8")
    monkeypatch.setenv(kaggle_auth.TOKEN_ENV, str(path))
    assert kaggle_auth.stored_token() == ("kg_from_file", str(path))


def test_without_a_token_the_error_says_what_to_do():
    with pytest.raises(kaggle_auth.KaggleNotConfigured, match=r"\.env"):
        kaggle_auth.api()


def test_env_file_is_loaded_on_import(tmp_path, monkeypatch):
    from dotenv import load_dotenv

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("# a comment\nANTHROPIC_API_KEY=sk-ant-from-file\n", encoding="utf-8")
    load_dotenv(env_file, override=False)                          # what src.config does at import
    from src.models.secret_store import SecretStore

    assert SecretStore().get("anthropic") == "sk-ant-from-file"
    assert "_load_env_file" in Path(config.__file__).read_text(encoding="utf-8")


def test_kaggle_status_and_check_endpoints(api, monkeypatch):
    status = api.get("/api/models/kaggle").json()
    assert status == {"has_token": False, "source": "", "variable": "KAGGLE_API_TOKEN"}
    assert ".env" in api.post("/api/models/kaggle/check").json()["detail"]

    monkeypatch.setenv(kaggle_auth.TOKEN_ENV, "kg_test")
    assert api.get("/api/models/kaggle").json()["has_token"]
    monkeypatch.setattr(kaggle_auth, "check", lambda: kaggle_auth.Quota("masanta-s", 21.5, 30.0, 20.0, None))
    quota = api.post("/api/models/kaggle/check").json()
    assert quota["username"] == "masanta-s" and quota["gpu_hours_left"] == 21.5
    assert "kg_test" not in json.dumps(quota)

    def rejected():
        raise RuntimeError("401 Unauthorized")

    monkeypatch.setattr(kaggle_auth, "check", rejected)
    assert api.post("/api/models/kaggle/check").status_code == 502
