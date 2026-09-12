"""Models: tiers, capability checks, HF training source, switching, downloads, fine-tune import,
and disk clean-up of old checkpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from server.deps import get_services, job_response
from server.serialize import to_jsonable
from server.services import Services
from src import config

router = APIRouter(prefix="/api", tags=["models"])


class NameIn(BaseModel):
    name: str = Field(min_length=1)


class RepoIn(BaseModel):
    name: str = Field(min_length=1)
    repo: str | None = None   # None: forget the manual choice and resolve automatically


class ImportIn(BaseModel):
    path: str = Field(min_length=1)
    name: str = "gradeforge-grader"
    base: str = config.SECONDARY_LLM


class CleanIn(BaseModel):
    keep: int = Field(default=2, ge=1, le=10)


def _ollama_errors(fn):
    from src.models.ollama_probe import OllamaUnavailable

    try:
        return fn()
    except OllamaUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from None


@router.get("/models")
def list_models(services: Services = Depends(get_services)) -> list[dict]:
    """Installed Ollama models with tier badges. Cache-only: no model runs, no network."""
    corrections = services.grade_correction_count()
    models = _ollama_errors(lambda: services.models.list_models(corrections))
    for m in models:
        m["active"] = not services.cloud_active and m["name"] == services.settings["model"]
    return models


@router.get("/models/card")
def model_card(name: str, services: Services = Depends(get_services)) -> dict:
    from src.models.model_manager import HF_TOKEN_NAME

    card = _ollama_errors(lambda: services.models.card(name, services.grade_correction_count()))
    return card | {"active": not services.cloud_active and name == services.settings["model"],
                   "has_hf_token": services.secrets.get(HF_TOKEN_NAME) is not None}


@router.get("/models/gpu")
def gpu() -> dict:
    from src.models.model_manager import gpu_usage

    return gpu_usage() or {}


@router.post("/models/probe")
def probe(body: NameIn, services: Services = Depends(get_services)) -> dict:
    def work(job):
        report = services.models.run_probe(body.name, services.make_llm(body.name), progress=job.report)
        if body.name == services.settings["model"]:
            services.reset_llm()  # diagram vision follows the new result
        return report.to_dict()

    return job_response(services.jobs.submit("probe_model", work))


@router.post("/models/resolve")
def resolve(body: NameIn, services: Services = Depends(get_services)) -> dict:
    """Find the HF repo with this model's trainable weights. Sends only the model's name."""
    return job_response(services.jobs.submit("resolve_model", lambda job: services.models.resolve(body.name).to_dict()))


@router.put("/models/resolution")
def set_resolution(body: RepoIn, services: Services = Depends(get_services)) -> dict:
    """The teacher's "Not right?" choice (verified on HF), or back to automatic."""
    return job_response(services.jobs.submit(
        "resolve_model", lambda job: services.models.set_repo(body.name, body.repo).to_dict()))


@router.post("/models/switch")
def switch(body: NameIn, services: Services = Depends(get_services)) -> dict:
    """Use this model for grading. A model that was never checked is checked first, and one
    that fails a critical check is refused: grading stays on the current model."""
    def work(job):
        digests = services.models.probe.digests()
        if body.name not in digests:
            raise ValueError(f"{body.name} is not installed in Ollama")
        report = services.models.report(body.name, digests[body.name])
        if report is None:
            report = services.models.run_probe(body.name, services.make_llm(body.name),
                                               progress=lambda p, m: job.report(p * 0.95, m))
        if not report.critical_passed:
            failed = ", ".join(r.label.lower() for r in report.failed_critical)
            raise RuntimeError(f"{body.name} failed the {failed} check; still grading with "
                               f"{services.model_name}")
        services.update_settings({"model": body.name, "provider": "ollama"})
        return {"model": body.name, "probe": report.to_dict()}

    return job_response(services.jobs.submit("switch_model", work))


@router.post("/models/pull")
def pull(body: NameIn, services: Services = Depends(get_services)) -> dict:
    """Download a model through Ollama (into Ollama's own model folder), then measure its VRAM."""
    def work(job):
        from src.models.ollama_probe import save_measurement

        def on_progress(status: str, completed: int, total: int) -> None:
            if total:
                job.report(0.9 * completed / total, f"{status} ({completed / 2**30:.1f} of {total / 2**30:.1f} GiB)")
            elif status:
                job.report(job.progress, status)

        probe = services.models.probe
        probe.pull(body.name, on_progress)
        job.report(0.92, "Measuring GPU memory")
        measurement = probe.measure_vram(body.name, config.LLM_NUM_CTX)
        save_measurement(measurement, services.model_cache.path)
        probe.unload(body.name)
        return to_jsonable(measurement)

    return job_response(services.jobs.submit("pull_model", work))


@router.post("/models/import")
def import_model(body: ImportIn, services: Services = Depends(get_services)) -> dict:
    """Register a GGUF fine-tuned on Colab/Kaggle with Ollama, reusing the base model's template."""
    def work(job):
        from pathlib import Path

        from src.models.model_manager import import_gguf

        resolution = services.models.resolution(body.base)
        return import_gguf(Path(body.path.strip().strip('"')), body.name, body.base, probe=services.models.probe,
                           hf_repo=resolution.repo if resolution else None, progress=job.report)

    return job_response(services.jobs.submit("import_model", work))


@router.get("/models/secrets")
def secrets_status(services: Services = Depends(get_services)) -> list[dict]:
    """Which credentials are set in .env. Never returns a key, only a masked hint."""
    from src.models.secret_store import ENV_NAMES

    return [services.secrets.status(name) for name in ENV_NAMES]


@router.get("/models/kaggle")
def kaggle_status(services: Services = Depends(get_services)) -> dict:
    """Whether a Kaggle token is set in .env. Never returns the token itself."""
    from src.models.kaggle_auth import TOKEN_ENV, source

    where = source()
    return {"has_token": bool(where), "source": where, "variable": TOKEN_ENV}


@router.post("/models/kaggle/check")
def check_kaggle(services: Services = Depends(get_services)) -> dict:
    """Ask Kaggle who the token belongs to and how much free GPU time is left this week."""
    from src.models.kaggle_auth import KaggleNotConfigured, check

    try:
        return to_jsonable(check())
    except KaggleNotConfigured as e:
        raise HTTPException(status_code=409, detail=str(e)) from None
    except Exception as e:
        raise HTTPException(status_code=502,
                            detail=f"Kaggle rejected the token or is unreachable: {str(e)[:200]}") from None


# --- disk clean-up ------------------------------------------------------------------------

def _protected(services: Services) -> set[str]:
    return {services.settings["model"]}


@router.get("/storage")
def storage(keep: int = 2, services: Services = Depends(get_services)) -> dict:
    gc = services.checkpoint_gc()
    plan = gc.plan(keep, _protected(services))
    return {"disk": to_jsonable(gc.get_disk_report()), "keep": keep, "plan": to_jsonable(plan),
            "reclaimable_bytes": sum(i.size_bytes for i in plan)}


@router.post("/storage/clean")
def clean(body: CleanIn, services: Services = Depends(get_services)) -> dict:
    if services.jobs.busy("train_trocr", "import_model"):
        raise HTTPException(status_code=409, detail="training or an import is running; clean up afterwards")
    result = services.checkpoint_gc().collect(body.keep, _protected(services))
    return to_jsonable(result)
