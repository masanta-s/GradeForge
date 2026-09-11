"""Settings, health and job status."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from server.deps import get_services
from server.serialize import to_jsonable
from server.services import Services

router = APIRouter(prefix="/api", tags=["system"])


class SettingsUpdate(BaseModel):
    strictness: float | None = Field(default=None, ge=0, le=100)
    model: str | None = None
    teacher_name: str | None = None


@router.get("/health")
def health(services: Services = Depends(get_services)) -> dict:
    from src.models.ollama_probe import OllamaModelProbe

    try:
        ollama = OllamaModelProbe(timeout=3).version()
    except Exception:
        ollama = None
    return {"ok": True, "ollama": ollama, "model": services.model_name,
            "provider": "cloud" if services.cloud_active else "ollama"}


@router.get("/settings")
def get_settings(services: Services = Depends(get_services)) -> dict:
    return services.settings


@router.put("/settings")
def put_settings(update: SettingsUpdate, services: Services = Depends(get_services)) -> dict:
    return services.update_settings(update.model_dump(exclude_none=True))


@router.get("/jobs/{job_id}")
def job_status(job_id: str, services: Services = Depends(get_services)) -> dict:
    job = services.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return to_jsonable(job)
