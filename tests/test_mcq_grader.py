import pytest

from src.grading.answer_key import Question
from src.grading.mcq_grader import detect_option, grade_mcq

OPTIONS = {"A": "Nucleus", "B": "Mitochondria", "C": "Ribosome", "D": "Golgi body"}
QUESTION = Question(id="4", text="Powerhouse of the cell?", qtype="mcq", max_marks=1,
                    options=OPTIONS, correct_option="B")


@pytest.mark.parametrize(
    "answer",
    ["B", "b", "(b)", "b)", "B.", "Option B", "Ans: B", "ans - b", "B) Mitochondria", "(b) mitochondria",
     "8"],  # "8" is how TrOCR often reads a handwritten B
)
def test_letter_forms(answer):
    assert detect_option(answer, OPTIONS) == ("B", "letter")


@pytest.mark.parametrize("answer", ["Mitochondria", "mitochondria", "mitochondira", "a mitochondria"])
def test_option_text_instead_of_letter(answer):
    assert detect_option(answer, OPTIONS) == ("B", "text_match")


@pytest.mark.parametrize("answer", ["B, C", "b or d", "(a) & (c)", "A/B"])
def test_several_letters_are_ambiguous(answer):
    assert detect_option(answer, OPTIONS) == (None, "ambiguous")


@pytest.mark.parametrize("answer", ["", "   ", "xyz qrs", "E", "cell",
                                    "nucleolus"])  # a different organelle, must not match "Nucleus"
def test_unreadable(answer):
    assert detect_option(answer, OPTIONS) == (None, "unreadable")


def test_numeric_option_labels_are_not_letter_corrected():
    options = {"1": "Two", "2": "Four", "3": "Eight", "4": "Sixteen"}
    assert detect_option("4", options) == ("4", "letter")
    assert detect_option("(2)", options) == ("2", "letter")


def test_grade_correct_and_wrong():
    right = grade_mcq(QUESTION, "(b)")
    wrong = grade_mcq(QUESTION, "C")
    assert (right.correct, right.marks, right.needs_review) == (True, 1, False)
    assert (wrong.correct, wrong.marks, wrong.detected_option) == (False, 0.0, "C")


def test_ambiguous_and_text_matches_go_to_review():
    assert grade_mcq(QUESTION, "B, C").needs_review
    text = grade_mcq(QUESTION, "Mitochondria")
    assert text.correct and text.needs_review  # correct, but a human should confirm
