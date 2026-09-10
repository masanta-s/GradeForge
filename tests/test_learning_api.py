"""Learning through the API: corrections are captured from normal teacher actions."""
import json

import cv2
import numpy as np
import pytest

from server.routes import learning as learning_routes
from src.learning.correction_store import CorrectionStore
from src.learning.ocr_fine_tuner import OCRTrainingResult
from tests.test_learning import HashingEmbedder
from tests.test_server import client, wait  # noqa: F401  (fixtures + helper)


@pytest.fixture
def api(client, tmp_path):
    services = client.app.state.services
    services.corrections = CorrectionStore(tmp_path / "corrections.db")
    services.subjective._embedder = HashingEmbedder()
    return client


def _graded_sheet(client):
    exam = client.post("/api/exams", json={"name": "Unit Test", "subject": "Biology"}).json()
    base = f"/api/exams/{exam['id']}"
    wait(client, client.post(f"{base}/paper", files={"file": ("paper.pdf", b"x")}).json()["job_id"])
    wait(client, client.post(f"{base}/answer-key/generate").json()["job_id"])
    wait(client, client.post(f"{base}/answer-key/validate").json()["job_id"])
    client.post(f"{base}/disputes/1/resolve", json={"teacher_name": "T", "action": "accept"})
    client.post(f"{base}/answer-key/finalize")
    upload = client.post(f"{base}/sheets", data={"student": "Riya"}, files={"file": ("s.png", b"x")}).json()
    wait(client, upload["job_id"])
    return base, upload["sheet_id"]


def test_teacher_actions_feed_learning(api):
    base, sheet = _graded_sheet(api)
    services = api.app.state.services
    # crops are needed for OCR training pairs: give the stored line an image
    folder = services.storage.sheet_dir(base.rsplit("/", 1)[1], sheet)
    (folder / "lines").mkdir(exist_ok=True)
    cv2.imencode(".png", np.full((40, 200), 255, np.uint8))[1].tofile(folder / "lines" / "p0_l3.png")
    meta = services.storage.get_ocr(base.rsplit("/", 1)[1], sheet)
    meta["pages"][0]["lines"][3]["crop"] = "p0_l3.png"
    services.storage.update_ocr(base.rsplit("/", 1)[1], sheet, meta)

    api.put(f"{base}/sheets/{sheet}/lines/0/3", json={"text": "and release oxygen."})
    wait(api, api.post(f"{base}/sheets/{sheet}/grade", json={"strictness": 50}).json()["job_id"])
    api.put(f"{base}/sheets/{sheet}/questions/2", json={"marks": 2.0, "note": "too short", "teacher_name": "Mrs. Iyer"})

    [ocr] = services.corrections.ocr_corrections("Riya")
    assert (ocr.ocr_text, ocr.corrected_text) == ("and release oxgen.", "and release oxygen.")
    [grade] = services.corrections.grade_corrections()
    assert (grade.teacher_marks, grade.note, grade.subject) == (2.0, "too short", "Biology")
    assert grade.ai_quality == 0.6 and grade.teacher_quality is not None  # plain written question

    status = api.get("/api/learning/status").json()
    assert status["few_shot"] == {"active": True, "corrections": 1}
    assert status["calibration"][0]["corrections"] == 1 and status["calibration"][0]["needed"] == 14
    assert status["ocr"]["by_writer"] == {"Riya": 1} and not status["ocr"]["can_train"]
    assert status["llm"]["examples"] == 1

    # the correction is now a few-shot example when grading a similar answer again
    [example] = services.retriever.retrieve("What is photosynthesis?", "Plants use sunlight to make glucose",
                                            subject="Biology")
    assert (example.marks, example.note) == (2.0, "too short")


def test_trocr_training_needs_enough_lines(api):
    assert api.post("/api/learning/trocr/train", json={}).status_code == 409


def test_trocr_training_job_logs_and_reports(api, monkeypatch, tmp_path):
    services = api.app.state.services
    image = tmp_path / "line.png"
    cv2.imencode(".png", np.full((40, 200), 255, np.uint8))[1].tofile(image)
    for i in range(25):
        services.corrections.add_ocr_correction(exam_id="e", sheet_id="s", writer_id="Riya", page=0, line_index=i,
                                                image_path=str(image), ocr_text="x", corrected_text=f"line {i}")
    seen = {}

    def fake_train(samples, progress):
        seen["n"] = len(samples)
        progress(0.5, "training")
        return OCRTrainingResult("v1", "base", 0.23, 0.02, True, 19, 6, 20.0, "dir")

    monkeypatch.setattr("src.learning.ocr_fine_tuner.train_trocr_lora", fake_train)
    result = wait(api, api.post("/api/learning/trocr/train", json={"writer": "Riya"}).json()["job_id"])
    assert seen["n"] == 25 and result["promoted"]
    [history] = api.get("/api/learning/status").json()["history"]
    assert (history["metric_before"], history["metric_after"], history["promoted"]) == (0.23, 0.02, True)


def test_notebook_export_requires_consent_and_data(api):
    assert api.post("/api/learning/llm/export", json={"consent": False}).status_code == 403
    assert api.post("/api/learning/llm/export", json={"consent": True}).status_code == 409  # nothing yet
    api.app.state.services.corrections.add_grade_correction(
        exam_id="e", sheet_id="riya-sharma-1a2b3c", question_id="2", subject="Biology", model="qwen3.5:9b",
        question_text="What is photosynthesis?", model_answer="Plants make glucose using sunlight.",
        student_answer="plants make food", max_marks=4, ai_marks=3, teacher_marks=1, ai_quality=0.7,
        strictness=50, note="too vague", teacher_name="Mrs. Iyer")
    response = api.post("/api/learning/llm/export", json={"consent": True})
    assert response.status_code == 200 and "attachment" in response.headers["content-disposition"]
    notebook = json.loads(response.text)
    text = json.dumps(notebook)
    assert notebook["nbformat"] == 4 and "plants make food" in text and "too vague" in text
    assert "riya" not in text.lower() and "Mrs. Iyer" not in text  # no student or teacher identity
    assert "Only 1 examples" in text  # below the recommended 200
