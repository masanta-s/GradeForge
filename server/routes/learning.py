"""The four learning mechanisms: status, TrOCR fine-tuning, and the LLM fine-tuning notebook."""
from __future__ import annotations

import json
from collections import Counter

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from server.deps import get_services, job_response
from server.serialize import to_jsonable
from server.services import Services
from src import config

router = APIRouter(prefix="/api/learning", tags=["learning"])


class TrainIn(BaseModel):
    writer: str | None = None  # one student's handwriting, or everyone's corrections


class ExportIn(BaseModel):
    consent: bool = False
    model: str | None = None   # the Ollama model to fine-tune; default: the one grading now


@router.get("/status")
def status(services: Services = Depends(get_services)) -> dict:
    from src.learning.llm_finetune_export import RECOMMENDED_CORRECTIONS
    from src.learning.ocr_fine_tuner import ACTIVE_FILE, MIN_SAMPLES

    grades = services.corrections.grade_corrections()
    pairs = sorted({(c.model, c.subject) for c in grades if c.ai_quality is not None})
    ocr = services.corrections.ocr_corrections()
    active_file = config.TROCR_FINETUNED_DIR / ACTIVE_FILE
    active = json.loads(active_file.read_text(encoding="utf-8")) if active_file.exists() else None
    trainable = sum(c.teacher_quality is not None for c in grades)
    reports = [services.calibrator.report(model, subject) for model, subject in pairs]
    return {
        "few_shot": {"active": bool(grades), "corrections": len(grades)},
        "calibration": [to_jsonable(r) | {"needed": r.needed} for r in reports],
        "ocr": {"corrections": len(ocr), "by_writer": dict(Counter(c.writer_id for c in ocr)),
                "min_samples": MIN_SAMPLES, "can_train": len(ocr) >= MIN_SAMPLES,
                "active_version": active["version"] if active else None,
                "active_cer": active["cer"] if active else None},
        "llm": {"examples": trainable, "recommended": RECOMMENDED_CORRECTIONS, "can_export": trainable > 0},
        "history": services.corrections.training_history(),
    }


@router.post("/trocr/train")
def train_trocr(body: TrainIn, services: Services = Depends(get_services)) -> dict:
    from src.learning.ocr_fine_tuner import MIN_SAMPLES

    corrections = services.corrections.ocr_corrections(body.writer)
    if len(corrections) < MIN_SAMPLES:
        raise HTTPException(status_code=409, detail=f"need {MIN_SAMPLES} corrected lines, have {len(corrections)}")

    def work(job):
        from src.learning.ocr_fine_tuner import train_trocr_lora
        from src.models.ollama_probe import OllamaModelProbe

        job.report(0.01, "Freeing the GPU")
        services.reload_ocr()  # drop the in-memory TrOCR
        try:
            if not services.cloud_active:
                OllamaModelProbe(timeout=10).unload(services.settings["model"])
        except Exception:
            pass  # Ollama not running: nothing to unload
        samples = []
        for c in corrections:
            image = cv2.imdecode(np.fromfile(c.image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                samples.append((image, c.corrected_text))
        result = train_trocr_lora(samples, progress=job.report)
        services.corrections.log_training(
            mechanism="trocr_lora", model="trocr-base-handwritten", version=result.version, metric_name="cer",
            metric_before=result.baseline_cer, metric_after=result.tuned_cer, num_samples=len(samples),
            promoted=result.promoted, notes=f"writer: {body.writer or 'all'}; baseline {result.baseline}")
        if result.promoted:
            services.reload_ocr()  # next OCR run loads the new adapter
        return to_jsonable(result)

    return job_response(services.jobs.submit("train_trocr", work))


@router.get("/llm/plan")
def llm_plan(model: str | None = None, services: Services = Depends(get_services)) -> dict:
    """Where the chosen model could be fine-tuned (cache-only). `needs_source`: its HF source
    hasn't been looked up yet; the page offers to look it up."""
    from src.models.capability_prober import assign_tier
    from src.models.ollama_probe import OllamaUnavailable

    name = model or services.settings["model"]
    try:
        identity = services.models.probe.probe(name)
    except OllamaUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from None
    plan = services.models.training_plan(identity, services.grade_correction_count())
    report = services.models.report(name)
    return {"model": name, "needs_source": plan is None, "plan": plan.to_dict() if plan else None,
            "tier": assign_tier(report, plan).__dict__, "hardware": services.models.hardware.__dict__}


@router.post("/llm/export")
def export_notebook(body: ExportIn, services: Services = Depends(get_services)) -> Response:
    if not body.consent:
        raise HTTPException(status_code=403, detail="exporting sends student answers to Colab/Kaggle: consent required")
    from src.learning.llm_finetune_export import build_notebook
    from src.models.hf_resolver import ResolverUnavailable
    from src.models.ollama_probe import OllamaUnavailable

    name = body.model or services.settings["model"]
    corrections = services.corrections.grade_corrections()
    try:
        identity = services.models.probe.probe(name)
        if services.models.resolution(name) is None:
            services.models.resolve(name)  # sends only the model's name to huggingface.co
        plan = services.models.training_plan(identity, services.grade_correction_count())
    except (OllamaUnavailable, ResolverUnavailable) as e:
        raise HTTPException(status_code=503, detail=str(e)) from None
    if plan is None or plan.recommended is None:
        raise HTTPException(status_code=409, detail=plan.blocked_reason if plan else "no training plan")
    try:
        notebook = build_notebook(corrections, base_model=plan.repo, load_in_4bit=plan.estimate.method == "qlora",
                                  ollama_base=name)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from None
    return Response(json.dumps(notebook, indent=1, ensure_ascii=False), media_type="application/x-ipynb+json",
                    headers={"Content-Disposition": 'attachment; filename="gradeforge_finetune.ipynb"'})
