"""FastAPI dependencies: the shared Services instance and 404 handling for storage lookups."""
from __future__ import annotations

from contextlib import contextmanager

from fastapi import HTTPException, Request

from server.services import Services


def get_services(request: Request) -> Services:
    return request.app.state.services


@contextmanager
def not_found(what: str):
    try:
        yield
    except KeyError:
        raise HTTPException(status_code=404, detail=f"{what} not found") from None


def job_response(job) -> dict:
    return {"job_id": job.id, "status": job.status}
