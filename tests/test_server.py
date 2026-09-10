"""API workflow tests: FastAPI TestClient with stand-in models (no GPU, no Ollama)."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from server.main import create_app
from server.services import Services
from server.storage import Storage
from src.grading.subjective_grader import SubjectiveGrader
from src.knowledge.dispute_logger import DisputeLog
from src.ocr.pipeline import OCRLine, PageOCR
from tests.test_subjective_grader import BagOfWordsEmbedder

PAPER_LINES = [
    "Class 10 Biology - Unit Test",
    "Q1. Which organelle is the powerhouse of the cell? [1]",
    "(A) Nucleus (B) Mitochondria (C) Ribosome",
    "Q2. What is photosynthesis? [4]",
]
SHEET_LINES = ["Name: Riya", "Q1. B", "Q2. Plants use sunlight to make glucose", "and release oxgen."]


class FakeOCR:
    extractor = None

    def run_file(self, path):
        lines = PAPER_LINES if path.parent.name == "paper" else SHEET_LINES
        return [PageOCR(0, "handwriting", 0.0,
                        [OCRLine(t, 0.95, (0, i * 50, 300, 40), 0) for i, t in enumerate(lines)])]


class FakeLLM:
    """Answers by schema shape, like the real prompts."""

    def __call__(self, messages, schema=None):
        required = set(schema["required"]) if schema else set()
        if required == {"correct_option", "explanation", "confidence"}:          # generate MCQ
            return json.dumps({"correct_option": "A", "explanation": "wrong on purpose", "confidence": 0.9})
        if required == {"model_answer", "key_points", "confidence"}:            # generate written
            return json.dumps({"model_answer": "Plants use sunlight, water and carbon dioxide to make glucose "
                                               "and oxygen.", "key_points": ["Uses sunlight", "Makes glucose"],
                               "keywords": ["sunlight", "glucose", "oxygen"], "confidence": 0.9})
        if required == {"correct_option", "reasoning", "confidence"}:           # validate MCQ (blind solve)
            return json.dumps({"correct_option": "B", "reasoning": "ATP is made in mitochondria.",
                               "confidence": 0.95})
        if required == {"is_correct", "problems", "suggested_answer", "confidence"}:
            return json.dumps({"is_correct": True, "problems": [], "suggested_answer": "", "confidence": 0.9})
        if required == {"concede", "response"}:
            return json.dumps({"concede": False, "response": "You make a fair point, but it's mitochondria."})
        if required == {"quality", "feedback", "missing_points"}:               # grade written
            return json.dumps({"quality": 0.6, "feedback": "Mentions sunlight and glucose.",
                               "missing_points": ["water", "carbon dioxide"]})
        raise AssertionError(f"unexpected schema {required}")


@pytest.fixture
def client(tmp_path):
    services = Services(storage=Storage(tmp_path / "exams"), settings_path=tmp_path / "settings.json")
    services.ocr = FakeOCR()
    services.llm = FakeLLM()
    services.subjective = SubjectiveGrader(embedder=BagOfWordsEmbedder())
    services.dispute_log = DisputeLog(tmp_path / "disputes.db")
    with TestClient(create_app(services)) as c:
        yield c


def wait(client, job_id, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            assert job["status"] == "done", job["error"]
            return job["result"]
        time.sleep(0.05)
    raise TimeoutError(job_id)


def test_full_teacher_workflow(client):
    exam = client.post("/api/exams", json={"name": "Unit Test 2", "subject": "Biology"}).json()
    base = f"/api/exams/{exam['id']}"

    # 1. question paper -> questions
    job = client.post(f"{base}/paper", files={"file": ("paper.pdf", b"%PDF-fake", "application/pdf")}).json()
    questions = wait(client, job["job_id"])["questions"]
    assert [(q["id"], q["qtype"]) for q in questions] == [("1", "mcq"), ("2", "descriptive")]

    # 2-3. AI answer key (deliberately wrong MCQ) -> validation flags it
    wait(client, client.post(f"{base}/answer-key/generate").json()["job_id"])
    key = client.get(f"{base}/answer-key").json()
    assert key["answer_key"]["questions"][0]["correct_option"] == "A" and key["problems"] == []
    flagged = wait(client, client.post(f"{base}/answer-key/validate").json()["job_id"])["flagged"]
    assert flagged == ["1"]
    assert client.post(f"{base}/answer-key/finalize").status_code == 409  # open dispute

    # 4. teacher accepts the AI's correction -> key updated, dispute logged
    reply = client.post(f"{base}/disputes/1/discuss", json={"message": "Are you sure?"}).json()
    assert not reply["ai_conceded"] and len(reply["conversation"]) == 3
    assert client.post(f"{base}/disputes/1/resolve", json={"teacher_name": "", "action": "accept"}).status_code == 422
    resolved = client.post(f"{base}/disputes/1/resolve",
                           json={"teacher_name": "Mrs. Iyer", "action": "accept"}).json()
    assert resolved["question"]["correct_option"] == "B"
    assert client.post(f"{base}/answer-key/finalize").json() == {"finalized": True}
    [logged] = client.get("/api/disputes", params={"teacher": "iyer"}).json()
    assert logged["teacher_decision"] == "accepted_ai" and len(logged["conversation"]) == 3

    # 5. student sheet -> OCR -> teacher fixes a line
    upload = client.post(f"{base}/sheets", data={"student": "Riya Sharma"},
                         files={"file": ("sheet.png", b"fake", "image/png")}).json()
    sheet_id = upload["sheet_id"]
    assert wait(client, upload["job_id"])["lines"] == 4
    fixed = client.put(f"{base}/sheets/{sheet_id}/lines/0/3", json={"text": "and release oxygen."}).json()
    assert fixed["corrected"] and fixed["ocr_text"] == "and release oxgen."

    # 6. grade
    wait(client, client.post(f"{base}/sheets/{sheet_id}/grade", json={"strictness": 50}).json()["job_id"])
    sheet = client.get(f"{base}/sheets/{sheet_id}").json()
    result = sheet["result"]
    q1, q2 = result["questions"]
    assert (q1["marks"], q1["max_marks"]) == (1.0, 1)
    assert q2["answer_text"] == "Plants use sunlight to make glucose\nand release oxygen."  # corrected text used
    assert result["total"] == q1["marks"] + q2["marks"]
    assert client.get(f"/api/exams/{exam['id']}/sheets").json()[0]["total"] == result["total"]

    # 7. what-if: instant, no model calls, MCQ unaffected
    lenient = client.post(f"{base}/sheets/{sheet_id}/what-if", json={"strictness": 0}).json()
    strict = client.post(f"{base}/sheets/{sheet_id}/what-if", json={"strictness": 100}).json()
    assert lenient["questions"][0]["marks"] == strict["questions"][0]["marks"] == 1.0
    assert lenient["total"] > result["total"] > strict["total"]

    # 8. teacher overrides a mark; what-if never changes it
    override = client.put(f"{base}/sheets/{sheet_id}/questions/2",
                          json={"marks": 3.5, "note": "good enough", "teacher_name": "Mrs. Iyer"}).json()
    assert (override["teacher_marks"], override["ai_marks"]) == (3.5, q2["marks"])
    again = client.post(f"{base}/sheets/{sheet_id}/what-if", json={"strictness": 100}).json()
    assert again["questions"][1]["marks"] == 3.5
    assert client.put(f"{base}/sheets/{sheet_id}/questions/2",
                      json={"marks": 9, "teacher_name": "x"}).status_code == 422

    # 9. audit export
    csv = client.get("/api/disputes/export.csv")
    assert csv.status_code == 200 and "Mrs. Iyer" in csv.text

    # 10. applying a strictness saves the rescored result (teacher's mark still kept)
    applied = client.post(f"{base}/sheets/{sheet_id}/what-if", json={"strictness": 0, "save": True}).json()
    saved = client.get(f"{base}/sheets/{sheet_id}").json()["result"]
    assert saved["strictness"] == 0 and saved["total"] == applied["total"]
    assert saved["questions"][1]["marks"] == 3.5

    # 11. class analytics
    stats = client.get(f"{base}/analytics").json()
    assert stats["students"] == 1 and stats["percentages"] == [round(saved["percentage"], 1)]
    q2 = next(q for q in stats["questions"] if q["question_id"] == "2")
    assert (q2["average"], q2["teacher_changed"]) == (3.5, 1)


def test_errors(client):
    assert client.get("/api/exams/nope").status_code == 404
    assert client.post("/api/exams", json={"name": " ", "subject": "x"}).status_code == 422
    exam = client.post("/api/exams", json={"name": "E", "subject": "S"}).json()
    base = f"/api/exams/{exam['id']}"
    assert client.post(f"{base}/answer-key/generate").status_code == 409     # no paper yet
    assert client.post(f"{base}/paper", files={"file": ("x.docx", b"x")}).status_code == 415
    assert client.get(f"{base}/sheets/nope").status_code == 404
    assert client.get(f"{base}/sheets/x/lines/..%2F..%2Fexam.json").status_code == 404  # no path traversal
    upload = client.post(f"{base}/sheets", data={"student": "A"}, files={"file": ("s.png", b"x")}).json()
    assert client.post(f"{base}/sheets/{upload['sheet_id']}/grade").status_code == 409  # no key


def test_settings_roundtrip(client):
    assert client.get("/api/settings").json()["strictness"] == 50
    updated = client.put("/api/settings", json={"strictness": 70, "teacher_name": "Mrs. Iyer"}).json()
    assert updated["strictness"] == 70 and client.get("/api/settings").json()["teacher_name"] == "Mrs. Iyer"
    assert client.put("/api/settings", json={"strictness": 150}).status_code == 422
