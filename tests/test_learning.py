import json
import sqlite3

import numpy as np
import pytest

from src.grading.answer_key import AnswerKey, Question
from src.grading.grading_engine import GradingEngine
from src.grading.strictness_curve import apply_strictness
from src.grading.subjective_grader import SubjectiveGrader
from src.learning.correction_retriever import CorrectionRetriever
from src.learning.correction_store import CorrectionStore, implied_quality
from src.learning.score_calibrator import MIN_CORRECTIONS, ScoreCalibrator
from tests.test_subjective_grader import BagOfWordsEmbedder


class HashingEmbedder:
    """Fixed-size bag-of-words vectors (like a real embedder, the size never changes)."""

    def __init__(self, dim=256):
        self.dim = dim

    def encode(self, sentences, normalize_embeddings=True):
        import re
        import zlib

        out = np.zeros((len(sentences), self.dim), np.float32)
        for row, s in enumerate(sentences):
            for w in re.findall(r"[a-z]+", s.lower()):
                out[row, zlib.crc32(w.encode()) % self.dim] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


@pytest.fixture
def store(tmp_path):
    return CorrectionStore(tmp_path / "corrections.db")


def _add(store, i=0, *, subject="Biology", model="qwen3.5:9b", answer="plants use sunlight to make glucose",
         ai_quality=0.8, teacher_marks=2.0, max_marks=4.0, strictness=50, sheet=None, question="What is photosynthesis?"):
    store.add_grade_correction(
        exam_id="e1", sheet_id=sheet or f"s{i}", question_id="2", subject=subject, model=model,
        question_text=question, model_answer="Plants use sunlight, water and CO2 to make glucose and oxygen.",
        student_answer=answer, max_marks=max_marks, ai_marks=3.0, teacher_marks=teacher_marks,
        ai_quality=ai_quality, strictness=strictness, note="too brief", teacher_name="Mrs. Iyer")


# --- store ---------------------------------------------------------------------------------

@pytest.mark.parametrize("strictness", [0, 30, 50, 80, 100])
@pytest.mark.parametrize("quality", [0.0, 0.25, 0.6, 1.0])
def test_implied_quality_inverts_the_strictness_curve(strictness, quality):
    marks = 4 * apply_strictness(quality, strictness)
    assert implied_quality(marks, 4, strictness) == pytest.approx(quality, abs=1e-9)


def test_recorrecting_replaces(store):
    _add(store, sheet="s1", teacher_marks=2.0)
    _add(store, sheet="s1", teacher_marks=3.0)
    [c] = store.grade_corrections()
    assert c.teacher_marks == 3.0
    assert c.teacher_quality == pytest.approx(implied_quality(3.0, 4.0, 50))


def test_ocr_corrections_upsert_and_revert(store):
    kwargs = dict(exam_id="e", sheet_id="s", writer_id="riya", page=0, line_index=3, image_path="x.png", ocr_text="oxgen")
    store.add_ocr_correction(**kwargs, corrected_text="oxygen")
    store.add_ocr_correction(**kwargs, corrected_text="oxygen gas")
    assert [c.corrected_text for c in store.ocr_corrections("riya")] == ["oxygen gas"]
    store.add_ocr_correction(**kwargs, corrected_text="oxgen")  # teacher undid their fix
    assert store.ocr_corrections() == []


def test_training_history(store):
    store.log_training(mechanism="trocr_lora", model="trocr-base", version="v1", metric_name="cer",
                       metric_before=0.2, metric_after=0.1, num_samples=30, promoted=True)
    [h] = store.training_history("trocr_lora")
    assert (h["version"], h["promoted"], h["metric_after"]) == ("v1", True, 0.1)


# --- retrieval (mechanism 1) ---------------------------------------------------------------

def test_retrieves_similar_corrections_only(store):
    _add(store, 1, answer="plants use sunlight to make glucose", teacher_marks=2)
    _add(store, 2, answer="the french revolution began in 1789", question="When did the revolution start?")
    _add(store, 3, answer="plants use sunlight", subject="Physics")
    retriever = CorrectionRetriever(store, embedder=HashingEmbedder())
    [example] = retriever.retrieve("What is photosynthesis?", "plants use sunlight to make food", subject="Biology")
    assert (example.answer, example.marks, example.max_marks, example.note) == (
        "plants use sunlight to make glucose", 2.0, 4.0, "too brief")
    assert retriever.retrieve("What is photosynthesis?", "plants make food", subject="Chemistry") == []


def test_exclude_sheet(store):
    _add(store, 1, sheet="riya")
    retriever = CorrectionRetriever(store, embedder=HashingEmbedder())
    assert retriever.retrieve("What is photosynthesis?", "plants use sunlight to make glucose", exclude_sheet="riya") == []


def test_embedder_change_re_embeds(store):
    _add(store, 1)
    CorrectionRetriever(store, embedder=HashingEmbedder(), embedder_name="old-model").sync()
    retriever = CorrectionRetriever(store, embedder=HashingEmbedder(), embedder_name="new-model")
    assert retriever.sync() == 1  # stale vectors are re-embedded, never compared across spaces
    assert retriever.sync() == 0
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute("SELECT embedder FROM correction_embeddings").fetchall() == [("new-model",)]


def test_same_name_different_size_is_re_embedded_not_a_crash(store):
    _add(store, 1)
    CorrectionRetriever(store, embedder=HashingEmbedder(256)).sync()
    swapped = CorrectionRetriever(store, embedder=HashingEmbedder(128))  # model files replaced, same name
    [example] = swapped.retrieve("What is photosynthesis?", "plants use sunlight to make glucose")
    assert example.marks == 2.0


# --- calibration (mechanism 2) -------------------------------------------------------------

def _biased_history(store, n, bias=0.2, model="qwen3.5:9b", subject="Biology", strictness=50):
    rng = np.random.default_rng(0)
    for i, teacher_q in enumerate(rng.uniform(0.1, 0.7, n)):
        teacher_marks = 4 * apply_strictness(teacher_q, strictness)
        _add(store, i, model=model, subject=subject, ai_quality=min(1.0, teacher_q + bias),
             teacher_marks=teacher_marks, strictness=strictness)


def test_calibrator_inactive_below_threshold(store):
    _biased_history(store, MIN_CORRECTIONS - 1)
    calibrator = ScoreCalibrator(store)
    assert calibrator.calibrate("qwen3.5:9b", "Biology", 0.8) == 0.8
    report = calibrator.report("qwen3.5:9b", "Biology")
    assert not report.active and report.needed == 1


def test_calibrator_removes_systematic_bias(store):
    _biased_history(store, 40, bias=0.2)
    calibrator = ScoreCalibrator(store)
    report = calibrator.report("qwen3.5:9b", "Biology")
    assert report.active and report.mean_bias == pytest.approx(0.2, abs=0.01)
    for teacher_q in (0.2, 0.4, 0.6):
        assert calibrator.calibrate("qwen3.5:9b", "Biology", teacher_q + 0.2) == pytest.approx(teacher_q, abs=0.06)
    # per model and per subject
    assert calibrator.calibrate("gemma4:e4b", "Biology", 0.6) == 0.6
    assert calibrator.calibrate("qwen3.5:9b", "Physics", 0.6) == 0.6


def test_calibration_is_strictness_independent(store):
    # Same teacher judgement recorded at different strictness settings -> same calibration.
    for strictness, sheet_prefix in ((10, "a"), (90, "b")):
        rng = np.random.default_rng(1)
        for i, teacher_q in enumerate(rng.uniform(0.1, 0.7, 20)):
            _add(store, sheet=f"{sheet_prefix}{i}", ai_quality=teacher_q + 0.2,
                 teacher_marks=4 * apply_strictness(teacher_q, strictness), strictness=strictness)
    qualities = [c.teacher_quality - c.ai_quality for c in store.grade_corrections()]
    assert np.allclose(qualities, -0.2)


# --- OCR fine-tune gate (mechanism 3) --------------------------------------------------------

@pytest.mark.parametrize(("baseline", "tuned", "promote"),
                         [(0.20, 0.05, True), (0.20, 0.197, False), (0.10, 0.10, False), (0.10, 0.30, False)])
def test_should_promote(baseline, tuned, promote):
    from src.learning.ocr_fine_tuner import should_promote

    assert should_promote(baseline, tuned) is promote


# --- grading integration -------------------------------------------------------------------

def test_engine_uses_examples_and_calibration():
    question = Question(id="2", text="What is photosynthesis?", qtype="short", max_marks=4,
                        model_answer="Plants use sunlight to make glucose.", keywords=["sunlight", "glucose"])
    seen = {}

    def llm(messages, schema=None):
        seen["prompt"] = messages[1]["content"]
        return json.dumps({"quality": 0.8, "feedback": "ok", "missing_points": []})

    from src.grading.subjective_grader import GradedExample

    engine = GradingEngine(
        AnswerKey("E", "Biology", [question]), subjective_grader=SubjectiveGrader(embedder=BagOfWordsEmbedder()),
        llm=llm, strictness=50,
        examples_for=lambda q, a: [GradedExample(q.text, "sunlight makes food", 1, 4, "too vague")],
        calibrate=lambda q: q - 0.3,
    )
    grade = engine.grade_answers({"2": "Plants use sunlight to make glucose."}).questions[0]
    assert "Teacher's marks: 1/4 (too vague)" in seen["prompt"]
    assert grade.detail.llm_quality == 0.8 and grade.detail.quality == pytest.approx(0.5)
    assert grade.marks == pytest.approx(round(4 * apply_strictness(0.5, 50) * 2) / 2)
