"""Exams, question papers, answer keys and answer-key disputes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from server.deps import get_services, job_response, not_found
from server.serialize import to_jsonable
from server.services import Services
from src.grading.answer_key import AnswerKey
from src.knowledge.question_parser import ParsedQuestion

router = APIRouter(prefix="/api", tags=["exams"])

PAPER_TYPES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


class NewExam(BaseModel):
    name: str
    subject: str


class QuestionIn(BaseModel):
    id: str
    text: str
    marks: float | None = None
    options: dict[str, str] = {}


class DiscussIn(BaseModel):
    message: str


class ResolveIn(BaseModel):
    teacher_name: str
    action: str  # "accept" | "insist"
    save_correction: bool = False


def _question_json(q: ParsedQuestion) -> dict:
    return {**to_jsonable(q), "qtype": q.qtype, "wants_diagram": q.wants_diagram}


@router.get("/exams")
def list_exams(services: Services = Depends(get_services)) -> list[dict]:
    return services.storage.list_exams()


@router.post("/exams", status_code=201)
def create_exam(body: NewExam, services: Services = Depends(get_services)) -> dict:
    if not body.name.strip():
        raise HTTPException(status_code=422, detail="exam name is required")
    return services.storage.create_exam(body.name.strip(), body.subject.strip())


@router.get("/exams/{exam_id}")
def get_exam(exam_id: str, services: Services = Depends(get_services)) -> dict:
    with not_found("exam"):
        exam = services.storage.get_exam(exam_id)
        key = services.storage.get_key(exam_id)
        return {**exam, "answer_key": to_jsonable(key) if key else None,
                "key_problems": key.validate() if key else [],
                "validations": services.storage.get_validations(exam_id),
                "sheets": services.storage.list_sheets(exam_id)}


@router.post("/exams/{exam_id}/paper")
async def upload_paper(exam_id: str, file: UploadFile = File(...), services: Services = Depends(get_services)) -> dict:
    with not_found("exam"):
        exam = services.storage.get_exam(exam_id)
    suffix = "." + (file.filename or "").rsplit(".", 1)[-1].lower()
    if suffix not in PAPER_TYPES:
        raise HTTPException(status_code=415, detail=f"unsupported file type {suffix}")
    path = services.storage.save_paper(exam_id, file.filename, await file.read())

    def work(job):
        from src.knowledge.question_parser import parse_question_paper, parse_with_llm

        job.report(0.1, "Reading the question paper")
        pages = services.ocr.run_file(path)
        job.report(0.7, "Finding questions")
        questions = parse_question_paper(pages)
        if not questions:
            job.report(0.8, "Layout not recognised, asking the AI to split the paper")
            questions = parse_with_llm(pages, services.llm)
        exam.update(paper_file=path.name, questions=[_question_json(q) for q in questions],
                    paper_text="\n".join(line.text for p in pages for line in p.lines))
        services.storage.update_exam(exam)
        return {"questions": exam["questions"]}

    return job_response(services.jobs.submit("read_paper", work))


@router.get("/exams/{exam_id}/paper")
def get_paper(exam_id: str, services: Services = Depends(get_services)):
    with not_found("paper"):
        exam = services.storage.get_exam(exam_id)
        if not exam.get("paper_file"):
            raise KeyError("paper")
        return FileResponse(services.storage.exam_dir(exam_id) / "paper" / exam["paper_file"])


@router.put("/exams/{exam_id}/questions")
def put_questions(exam_id: str, questions: list[QuestionIn], services: Services = Depends(get_services)) -> list:
    with not_found("exam"):
        exam = services.storage.get_exam(exam_id)
    exam["questions"] = [_question_json(ParsedQuestion(q.id, q.text, q.marks, q.options)) for q in questions]
    services.storage.update_exam(exam)
    return exam["questions"]


# --- answer key --------------------------------------------------------------------------

@router.post("/exams/{exam_id}/answer-key/generate")
def generate_key(exam_id: str, services: Services = Depends(get_services)) -> dict:
    with not_found("exam"):
        exam = services.storage.get_exam(exam_id)
    if not exam["questions"]:
        raise HTTPException(status_code=409, detail="upload the question paper first")
    parsed = [ParsedQuestion(q["id"], q["text"], q["marks"], q["options"]) for q in exam["questions"]]

    def work(job):
        from src.knowledge.answer_generator import generate_answer_key, review_needed

        key = generate_answer_key(
            parsed, services.llm, subject=exam["subject"], exam=exam["name"],
            on_progress=lambda i, n, q: job.report(i / n, f"Answered question {q.id} ({i}/{n})"),
        )
        services.storage.save_key(exam_id, key)
        return {"review": [{"question_id": q.id, "reason": reason} for q, reason in review_needed(key)]}

    return job_response(services.jobs.submit("generate_key", work))


@router.get("/exams/{exam_id}/answer-key")
def get_key(exam_id: str, services: Services = Depends(get_services)) -> dict:
    with not_found("answer key"):
        key = services.storage.get_key(exam_id)
        if key is None:
            raise KeyError("key")
    return {"answer_key": to_jsonable(key), "problems": key.validate()}


@router.put("/exams/{exam_id}/answer-key")
def put_key(exam_id: str, body: dict, services: Services = Depends(get_services)) -> dict:
    with not_found("exam"):
        services.storage.get_exam(exam_id)
    try:
        key = AnswerKey.from_dict(body)
    except TypeError as e:
        raise HTTPException(status_code=422, detail=f"malformed answer key: {e}") from None
    key.finalized = False  # any edit re-opens the key
    services.storage.save_key(exam_id, key)
    return {"answer_key": to_jsonable(key), "problems": key.validate()}


@router.post("/exams/{exam_id}/answer-key/validate")
def validate_key(exam_id: str, services: Services = Depends(get_services)) -> dict:
    with not_found("answer key"):
        exam = services.storage.get_exam(exam_id)
        key = services.storage.get_key(exam_id)
        if key is None:
            raise KeyError("key")

    def work(job):
        from src.knowledge.answer_validator import validate_question

        validations = {}
        for i, q in enumerate(key.questions, 1):
            job.report(i / len(key.questions), f"Checking question {q.id}")
            result = validate_question(q, services.llm, exam["subject"])
            validations[q.id] = {**to_jsonable(result), "flagged": result.flagged,
                                 "status": "open" if result.flagged else "resolved", "conversation": []}
        services.storage.save_validations(exam_id, validations)
        return {"flagged": [qid for qid, v in validations.items() if v["flagged"]]}

    return job_response(services.jobs.submit("validate_key", work))


@router.post("/exams/{exam_id}/answer-key/finalize")
def finalize_key(exam_id: str, services: Services = Depends(get_services)) -> dict:
    with not_found("answer key"):
        key = services.storage.get_key(exam_id)
        if key is None:
            raise KeyError("key")
    if problems := key.validate():
        raise HTTPException(status_code=409, detail={"problems": problems})
    open_disputes = [qid for qid, v in services.storage.get_validations(exam_id).items() if v["status"] == "open"]
    if open_disputes:
        raise HTTPException(status_code=409, detail={"open_disputes": open_disputes})
    key.finalized = True
    services.storage.save_key(exam_id, key)
    return {"finalized": True}


# --- disputes ----------------------------------------------------------------------------

def _dispute(exam_id: str, question_id: str, services: Services):
    from src.knowledge.answer_validator import ValidationResult
    from src.knowledge.dispute_manager import Dispute

    exam = services.storage.get_exam(exam_id)
    key = services.storage.get_key(exam_id)
    validations = services.storage.get_validations(exam_id)
    if key is None or question_id not in validations:
        raise KeyError(question_id)
    stored = validations[question_id]
    fields = {k: stored[k] for k in ("question_id", "verdict", "teacher_answer", "ai_answer",
                                     "justification", "confidence")}
    dispute = Dispute(key.get(question_id), ValidationResult(**fields), subject=exam["subject"],
                      exam=exam["name"], llm=services.llm, log=services.dispute_log)
    if stored["conversation"]:
        dispute.conversation = stored["conversation"]
    return dispute, key, validations


@router.post("/exams/{exam_id}/disputes/{question_id}/discuss")
def discuss(exam_id: str, question_id: str, body: DiscussIn, services: Services = Depends(get_services)) -> dict:
    with not_found("dispute"):
        dispute, _, validations = _dispute(exam_id, question_id, services)
    reply = dispute.discuss(body.message)
    validations[question_id]["conversation"] = dispute.conversation
    validations[question_id]["ai_conceded"] = dispute.ai_conceded
    services.storage.save_validations(exam_id, validations)
    return {"reply": reply, "ai_conceded": dispute.ai_conceded, "conversation": dispute.conversation}


@router.post("/exams/{exam_id}/disputes/{question_id}/resolve")
def resolve(exam_id: str, question_id: str, body: ResolveIn, services: Services = Depends(get_services)) -> dict:
    if not body.teacher_name.strip():
        raise HTTPException(status_code=422, detail="teacher name is required for the audit log")
    if body.action not in ("accept", "insist"):
        raise HTTPException(status_code=422, detail="action must be 'accept' or 'insist'")
    with not_found("dispute"):
        dispute, key, validations = _dispute(exam_id, question_id, services)
    updated = (dispute.accept_ai(body.teacher_name) if body.action == "accept"
               else dispute.insist(body.teacher_name, save_correction=body.save_correction))
    key.questions = [updated if q.id == question_id else q for q in key.questions]
    key.finalized = False
    services.storage.save_key(exam_id, key)
    validations[question_id]["status"] = "resolved"
    validations[question_id]["conversation"] = dispute.conversation
    services.storage.save_validations(exam_id, validations)
    return {"question": to_jsonable(updated)}


@router.get("/disputes")
def search_disputes(teacher: str | None = None, subject: str | None = None, text: str | None = None,
                    since: str | None = None, until: str | None = None,
                    services: Services = Depends(get_services)) -> list[dict]:
    records = services.dispute_log.search(teacher=teacher, subject=subject, text=text, since=since, until=until)
    return [to_jsonable(r) for r in records]


@router.get("/disputes/export.csv")
def export_disputes(services: Services = Depends(get_services)):
    from src import config

    path = services.dispute_log.export_csv(config.DATA_DIR / "dispute_logs" / "disputes.csv")
    return FileResponse(path, media_type="text/csv", filename="disputes.csv")
