import json

import pytest

from src.grading.answer_key import AnswerKey, Question
from src.grading.grading_engine import GradingEngine
from src.grading.subjective_grader import SubjectiveGrader
from src.ocr.answer_segmenter import segment_answers
from src.ocr.pipeline import OCRLine, PageOCR
from tests.test_subjective_grader import BagOfWordsEmbedder

KEY = AnswerKey(
    exam_name="Biology Unit Test",
    subject="Biology",
    questions=[
        Question(id="1", text="Powerhouse of the cell?", qtype="mcq", max_marks=1,
                 options={"A": "Nucleus", "B": "Mitochondria"}, correct_option="B"),
        Question(id="2", text="What is photosynthesis?", qtype="short", max_marks=4,
                 model_answer="Plants use sunlight, carbon dioxide and water to make glucose and oxygen.",
                 keywords=["sunlight", "carbon dioxide", "glucose", "oxygen"]),
        Question(id="3", text="Define osmosis.", qtype="short", max_marks=2,
                 model_answer="Movement of water across a semi-permeable membrane from low to high solute "
                              "concentration.", keywords=["water", "semi-permeable membrane"]),
    ],
)


class FixedLLM:
    def __init__(self, quality):
        self.quality = quality

    def __call__(self, messages, schema=None):
        return json.dumps({"quality": self.quality, "feedback": "ok", "missing_points": []})


@pytest.fixture
def engine():
    return GradingEngine(KEY, subjective_grader=SubjectiveGrader(embedder=BagOfWordsEmbedder()),
                         llm=FixedLLM(0.5), strictness=50)


def _line(text, conf=0.95, page=0):
    return OCRLine(text=text, confidence=conf, bbox=(0, 0, 10, 10), page=page)


def test_grade_answers_routes_and_totals(engine):
    paper = engine.grade_answers({"1": "(b)", "2": "Plants use sunlight to make glucose and oxygen."})
    q1, q2, q3 = paper.questions
    assert (q1.marks, q1.feedback) == (1.0, "Correct option.")
    assert q2.status == "graded" and 0 < q2.marks < 4
    assert (q3.status, q3.marks) == ("not_attempted", 0.0)
    assert paper.max_total == 7
    assert paper.total == q1.marks + q2.marks
    assert paper.percentage == pytest.approx(100 * paper.total / 7)


def test_wrong_mcq_feedback_names_answer(engine):
    q1 = engine.grade_answers({"1": "A"}).questions[0]
    assert (q1.marks, q1.feedback, q1.needs_review) == (0.0, "Correct option: B.", False)


def test_ambiguous_mcq_goes_to_review(engine):
    q1 = engine.grade_answers({"1": "A, B"}).questions[0]
    assert q1.review_reasons == ["more than one option marked"]


def test_segmented_sheet_end_to_end(engine):
    pages = [PageOCR(0, "handwriting", lines=[
        _line("Name: Riya"),
        _line("Q1. B"),
        _line("Q2 Plants use sunlight"),
        _line("to make glucose", conf=0.4),  # shaky OCR
    ])]
    paper = engine.grade_segmentation(segment_answers(pages, KEY.question_ids))
    q1, q2, q3 = paper.questions
    assert q1.marks == 1.0
    assert q2.answer_text == "Plants use sunlight\nto make glucose"
    assert q2.review_reasons[0] == "1 line(s) read with low OCR confidence"
    assert paper.unplaced_text == "Name: Riya"
    assert q3.status == "not_attempted"
    assert "wasn't matched" in q3.review_reasons[0]
    assert {q.question_id for q in paper.review_queue} == {"2", "3"}


def test_strictness_affects_subjective_only():
    lenient = GradingEngine(KEY, subjective_grader=SubjectiveGrader(embedder=BagOfWordsEmbedder()),
                            llm=FixedLLM(0.5), strictness=0)
    strict = GradingEngine(KEY, subjective_grader=SubjectiveGrader(embedder=BagOfWordsEmbedder()),
                           llm=FixedLLM(0.5), strictness=100)
    answers = {"1": "B", "2": "Plants use sunlight to make glucose."}
    a, b = lenient.grade_answers(answers), strict.grade_answers(answers)
    assert a.questions[0].marks == b.questions[0].marks == 1.0
    assert a.questions[1].marks > b.questions[1].marks
