"""'Choose the correct option AND justify' questions — MCQ and subjective in one."""
import json

import pytest

from src.grading.answer_key import AnswerKey, Question
from src.grading.grading_engine import GradingEngine, MixedResult
from src.grading.mcq_grader import split_mixed_answer
from src.grading.subjective_grader import SubjectiveGrader
from tests.test_subjective_grader import BagOfWordsEmbedder

OPTIONS = {"A": "Ribosome", "B": "Mitochondria", "C": "Vacuole", "D": "Nucleus"}
MIXED = Question(
    id="5", text="Which organelle releases energy from food? Choose and justify.", qtype="mixed",
    max_marks=3, option_marks=1, options=OPTIONS, correct_option="B",
    model_answer="Mitochondria carry out aerobic respiration, releasing energy from glucose as ATP.",
    keywords=["respiration", "ATP", "glucose"],
)


@pytest.mark.parametrize(
    ("answer", "option", "justification"),
    [
        ("B) Mitochondria, because they do respiration", "B", "Mitochondria, because they do respiration"),
        ("B because respiration makes ATP", "B", "because respiration makes ATP"),
        ("(b)\nThey carry out respiration\nand make ATP", "(b)", "They carry out respiration\nand make ATP"),
        ("Ans: B\nRespiration happens there", "B", "Respiration happens there"),
        ("Mitochondria\nIt makes ATP from glucose", "Mitochondria", "It makes ATP from glucose"),
        ("The answer is B as it does respiration", "B", "The answer is B as it does respiration"),
        ("It makes energy by respiration", "", "It makes energy by respiration"),
        ("", "", ""),
    ],
)
def test_split_mixed_answer(answer, option, justification):
    assert split_mixed_answer(answer, OPTIONS) == (option, justification)


def test_article_is_not_mistaken_for_option_a():
    # "a" is option A's label AND an English article; without a separator it's just text.
    option, justification = split_mixed_answer("a mitochondria makes energy", OPTIONS)
    assert option == ""


class FixedLLM:
    def __init__(self, quality):
        self.quality = quality

    def __call__(self, messages, schema=None):
        return json.dumps({"quality": self.quality, "feedback": "Explains respiration.", "missing_points": []})


def _engine(quality):
    key = AnswerKey("Mixed test", "Biology", [MIXED])
    return GradingEngine(key, subjective_grader=SubjectiveGrader(embedder=BagOfWordsEmbedder()),
                         llm=FixedLLM(quality), strictness=50)


def test_correct_option_and_full_justification():
    grade = _engine(1.0).grade_answers({"5": "B) Mitochondria, because respiration releases ATP from glucose"}).questions[0]
    assert isinstance(grade.detail, MixedResult)
    assert grade.detail.choice.marks == 1.0 and grade.detail.choice.correct
    assert grade.detail.justification.max_marks == 2
    assert grade.marks == 3.0
    assert grade.feedback.startswith("Correct option.")
    assert not grade.needs_review


def test_justification_prompt_knows_the_options_and_correct_answer():
    llm = FixedLLM(0.5)
    calls = []
    engine = _engine(0.5)
    engine.llm = lambda messages, schema=None: calls.append(messages) or llm(messages, schema)
    engine.grade_answers({"5": "B because respiration"})
    user = calls[0][1]["content"]
    assert "Correct option: B) Mitochondria" in user and "Grade ONLY the student's justification" in user
    assert "<student_answer>\nbecause respiration\n</student_answer>" in user


def test_wrong_option_with_credited_justification_is_flagged():
    grade = _engine(0.8).grade_answers({"5": "A because it makes ATP by respiration"}).questions[0]
    assert grade.detail.choice.marks == 0.0 and grade.detail.choice.detected_option == "A"
    assert grade.detail.justification.marks > 0
    assert grade.marks == grade.detail.justification.marks
    assert grade.feedback.startswith("Correct option: B.")
    assert any("wrong option" in r for r in grade.review_reasons)


def test_missing_option_is_flagged_but_justification_still_graded():
    grade = _engine(0.8).grade_answers({"5": "It makes energy by respiration"}).questions[0]
    assert grade.detail.choice.marks == 0.0
    assert "could not find which option was chosen" in grade.review_reasons


def test_paper_mixing_all_question_types():
    key = AnswerKey("All types", "Biology", [
        Question(id="1", text="Powerhouse?", qtype="mcq", max_marks=1, options=OPTIONS, correct_option="B"),
        Question(id="2", text="Define ATP.", qtype="short", max_marks=2,
                 model_answer="ATP is the energy currency of the cell.", keywords=["energy currency"]),
        MIXED,
    ])
    engine = GradingEngine(key, subjective_grader=SubjectiveGrader(embedder=BagOfWordsEmbedder()),
                           llm=FixedLLM(1.0), strictness=50)
    paper = engine.grade_answers({"1": "B", "2": "ATP is the energy currency", "5": "B because respiration makes ATP"})
    assert [q.qtype for q in paper.questions] == ["mcq", "short", "mixed"]
    assert paper.total == paper.max_total == 6


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"option_marks": 0}, "option_marks"),
        ({"option_marks": 3}, "option_marks"),
        ({"model_answer": ""}, "model answer"),
        ({"correct_option": "E"}, "correct_option"),
    ],
)
def test_mixed_question_validation(changes, problem):
    question = Question(**{**MIXED.__dict__, **changes})
    assert any(problem in p for p in question.validate())
    assert MIXED.validate() == [] and MIXED.justification_marks == 2
