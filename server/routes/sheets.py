"""Student answer sheets: upload + OCR, teacher corrections, grading, what-if strictness."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from server.deps import get_services, job_response, not_found
from server.serialize import to_jsonable
from server.services import Services
from server.whatif import rescore

router = APIRouter(prefix="/api/exams/{exam_id}/sheets", tags=["sheets"])

SHEET_TYPES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


class LineFix(BaseModel):
    text: str


class GradeIn(BaseModel):
    strictness: float | None = Field(default=None, ge=0, le=100)


class WhatIf(BaseModel):
    strictness: float = Field(ge=0, le=100)
    save: bool = False  # apply it: the rescored result becomes the sheet's result (still no model calls)


class MarksFix(BaseModel):
    marks: float = Field(ge=0)
    note: str = ""
    teacher_name: str


@router.get("")
def list_sheets(exam_id: str, services: Services = Depends(get_services)) -> list[dict]:
    with not_found("exam"):
        return services.storage.list_sheets(exam_id)


@router.post("")
async def upload_sheet(exam_id: str, student: str = Form(...), file: UploadFile = File(...),
                       services: Services = Depends(get_services)) -> dict:
    suffix = "." + (file.filename or "").rsplit(".", 1)[-1].lower()
    if suffix not in SHEET_TYPES:
        raise HTTPException(status_code=415, detail=f"unsupported file type {suffix}")
    if not student.strip():
        raise HTTPException(status_code=422, detail="student name or roll number is required")
    with not_found("exam"):
        sheet_id, path = services.storage.create_sheet(exam_id, student.strip(), file.filename, await file.read())

    def work(job):
        job.report(0.1, "Reading handwriting")
        pages = services.ocr.run_file(path)
        services.storage.save_ocr(exam_id, sheet_id, pages)
        lines = sum(len(p.lines) for p in pages)
        return {"sheet_id": sheet_id, "lines": lines, "diagrams": sum(len(p.diagrams) for p in pages)}

    return {**job_response(services.jobs.submit("read_sheet", work)), "sheet_id": sheet_id}


@router.get("/{sheet_id}")
def get_sheet(exam_id: str, sheet_id: str, services: Services = Depends(get_services)) -> dict:
    with not_found("sheet"):
        return {"id": sheet_id, **services.storage.get_ocr(exam_id, sheet_id),
                "result": services.storage.get_result(exam_id, sheet_id)}


@router.get("/{sheet_id}/source")
def get_source(exam_id: str, sheet_id: str, services: Services = Depends(get_services)):
    with not_found("sheet"):
        return FileResponse(services.storage.source_file(exam_id, sheet_id))


@router.get("/{sheet_id}/{kind}/{name}")
def get_asset(exam_id: str, sheet_id: str, kind: str, name: str, services: Services = Depends(get_services)):
    with not_found("image"):
        return FileResponse(services.storage.asset(exam_id, sheet_id, kind, name), media_type="image/png")


@router.put("/{sheet_id}/lines/{page}/{index}")
def fix_line(exam_id: str, sheet_id: str, page: int, index: int, body: LineFix,
             services: Services = Depends(get_services)) -> dict:
    """Teacher corrects an OCR line. The TrOCR reading stays in `ocr_text` (training pair)."""
    with not_found("line"):
        meta = services.storage.get_ocr(exam_id, sheet_id)
        stored_page = next((p for p in meta["pages"] if p["page_index"] == page), None)
        if stored_page is None or not 0 <= index < len(stored_page["lines"]):
            raise KeyError("line")
    line = stored_page["lines"][index]
    line["text"] = body.text
    line["corrected"] = body.text != line["ocr_text"]
    services.storage.update_ocr(exam_id, sheet_id, meta)
    return line


@router.post("/{sheet_id}/grade")
def grade_sheet(exam_id: str, sheet_id: str, body: GradeIn | None = None,
                services: Services = Depends(get_services)) -> dict:
    with not_found("sheet"):
        key = services.storage.get_key(exam_id)
        meta = services.storage.get_ocr(exam_id, sheet_id)
    if key is None or not key.finalized:
        raise HTTPException(status_code=409, detail="finalize the answer key before grading")
    if not meta["pages"]:
        raise HTTPException(status_code=409, detail="the sheet has not been read yet")
    strictness = body.strictness if body and body.strictness is not None else services.settings["strictness"]

    def work(job):
        from src.grading.grading_engine import GradingEngine
        from src.ocr.answer_segmenter import segment_answers

        job.report(0.05, "Matching answers to questions")
        segmentation = segment_answers(services.storage.load_pages(exam_id, sheet_id), key.question_ids)
        engine = GradingEngine(key, subjective_grader=services.subjective,
                               diagram_evaluator=services.diagram_evaluator, llm=services.llm,
                               strictness=strictness)
        job.report(0.15, "Grading")
        paper = engine.grade_segmentation(segmentation)
        result = {**to_jsonable(paper), "student": meta["student"], "model": services.settings["model"],
                  "graded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        services.storage.save_result(exam_id, sheet_id, result)
        return {"total": result["total"], "max_total": result["max_total"]}

    return job_response(services.jobs.submit("grade_sheet", work))


@router.post("/{sheet_id}/what-if")
def what_if(exam_id: str, sheet_id: str, body: WhatIf, services: Services = Depends(get_services)) -> dict:
    """Marks at another strictness, from stored qualities: instant, no model calls.
    Preview by default; `save` applies it to the sheet."""
    with not_found("result"):
        result = services.storage.get_result(exam_id, sheet_id)
        if result is None:
            raise KeyError("result")
    rescored = rescore(result, body.strictness)
    if body.save:
        services.storage.save_result(exam_id, sheet_id, rescored)
    return rescored


@router.put("/{sheet_id}/questions/{question_id}")
def fix_marks(exam_id: str, sheet_id: str, question_id: str, body: MarksFix,
              services: Services = Depends(get_services)) -> dict:
    """Teacher overrides a question's marks. Kept as teacher intent for the learning phase."""
    if not body.teacher_name.strip():
        raise HTTPException(status_code=422, detail="teacher name is required")
    with not_found("question"):
        result = services.storage.get_result(exam_id, sheet_id)
        if result is None:
            raise KeyError("result")
        question = next((q for q in result["questions"] if q["question_id"] == question_id), None)
        if question is None:
            raise KeyError(question_id)
    if body.marks > question["max_marks"]:
        raise HTTPException(status_code=422, detail=f"marks cannot exceed {question['max_marks']:g}")
    question.update(ai_marks=question.get("ai_marks", question["marks"]), marks=body.marks,
                    teacher_marks=body.marks, teacher_note=body.note, corrected_by=body.teacher_name,
                    corrected_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    result["total"] = sum(q["marks"] for q in result["questions"])
    result["percentage"] = 100 * result["total"] / result["max_total"] if result["max_total"] else 0.0
    services.storage.save_result(exam_id, sheet_id, result)
    return question
