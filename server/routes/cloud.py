"""Opt-in cloud models: key test, cost estimate, and switching grading to a cloud model, which
requires the teacher's explicit consent. API keys are read from .env, never saved by the app."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from server.deps import get_services, not_found
from server.serialize import to_jsonable
from server.services import Services

router = APIRouter(prefix="/api/cloud", tags=["cloud"])


class TestIn(BaseModel):
    provider: str
    model: str = Field(min_length=1)


class EstimateIn(BaseModel):
    model: str = Field(min_length=1)
    papers: int = Field(default=30, ge=1, le=10_000)
    exam_id: str | None = None


class CloudSettingsIn(BaseModel):
    provider: Literal["ollama", "cloud"]
    cloud_provider: str = "openai"
    cloud_model: str = ""
    consent: bool = False


def _settings_view(services: Services) -> dict:
    s = services.settings
    return {"provider": s["provider"], "cloud_provider": s["cloud_provider"], "cloud_model": s["cloud_model"],
            "cloud_consent": s["cloud_consent"], "active": services.cloud_active}


@router.get("")
def overview(services: Services = Depends(get_services)) -> dict:
    from src.models.cloud_provider import PROVIDERS, chat_models

    return {"providers": services.cloud.status(), "models": {p: chat_models(p) for p in PROVIDERS},
            "settings": _settings_view(services)}


@router.get("/model-info")
def model_info(model: str) -> dict:
    from src.models.cloud_provider import model_info

    return model_info(model)


@router.post("/test")
def test_key(body: TestIn, services: Services = Depends(get_services)) -> dict:
    """One 5-token request with no student data, to check the key and model name."""
    return to_jsonable(services.cloud.validate_api_key(body.provider, body.model))


@router.post("/estimate")
def estimate(body: EstimateIn, services: Services = Depends(get_services)) -> dict:
    from src.models.cloud_provider import estimate_cost

    if body.exam_id:
        with not_found("exam"):
            key = services.storage.get_key(body.exam_id)
        if key is None:
            raise HTTPException(status_code=409, detail="this exam has no answer key yet")
        basis = key.exam_name
    else:
        from demo.samples import SAMPLE_KEY as key

        basis = "a typical 6-question paper (the demo exam)"
    return to_jsonable(estimate_cost(body.model, list(key.questions), body.papers)) | {"basis": basis}


@router.put("/settings")
def save_settings(body: CloudSettingsIn, services: Services = Depends(get_services)) -> dict:
    if body.provider == "cloud":
        if not body.consent:
            raise HTTPException(status_code=403, detail="cloud grading sends exam data off this computer: "
                                                        "tick the consent box first")
        if not body.cloud_model.strip():
            raise HTTPException(status_code=422, detail="choose a cloud model")
        if services.cloud.api_key(body.cloud_provider) is None:
            variable = services.secrets.variable(body.cloud_provider)
            raise HTTPException(status_code=409, detail=f"set {variable} in the .env file first")
    services.update_settings({"provider": body.provider, "cloud_provider": body.cloud_provider,
                              "cloud_model": body.cloud_model.strip(), "cloud_consent": body.consent})
    return _settings_view(services)
