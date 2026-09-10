import json

import pytest

from src.knowledge.question_parser import ParsedQuestion, parse_question_paper, parse_with_llm, split_marks
from src.ocr.pipeline import OCRLine, PageOCR


def _paper(*lines: str) -> list[PageOCR]:
    return [PageOCR(0, "pdf_text", lines=[OCRLine(t, 1.0, (0, i * 20, 100, 18)) for i, t in enumerate(lines)])]


@pytest.mark.parametrize(
    ("text", "body", "marks"),
    [
        ("Define osmosis. [2 marks]", "Define osmosis.", 2.0),
        ("Explain photosynthesis. (5)", "Explain photosynthesis.", 5.0),
        ("State Newton's law - 3 marks", "State Newton's law", 3.0),
        ("What is ATP? 1M", "What is ATP?", 1.0),
        ("What is ATP? [1.5]", "What is ATP?", 1.5),
        ("Name 3 organelles.", "Name 3 organelles.", None),  # a number, but not marks
    ],
)
def test_split_marks(text, body, marks):
    assert split_marks(text) == (body, marks)


def test_full_paper():
    questions = parse_question_paper(_paper(
        "Class 10 Biology - Unit Test",
        "Time: 1 hour   Max marks: 20",
        "General instructions: answer all questions.",
        "Q1. Which organelle is the powerhouse of the cell? [1 mark]",
        "(A) Nucleus",
        "(B) Mitochondria",
        "(C) Ribosome",
        "(D) Golgi body",
        "Q2. Which gas is released in photosynthesis? (A) Oxygen (B) Nitrogen (C) Carbon dioxide (D) Helium [1]",
        "Q3. Explain photosynthesis with its equation.",
        "Mention the role of chlorophyll. [5 marks]",
        "Q4. Which blood cells fight infection? Justify your answer. [3]",
        "a) Red blood cells",
        "b) White blood cells",
        "c) Platelets",
        "Q5. Draw a neat labelled diagram of an animal cell. [4]",
        "Q6. Answer the following:",
        "(a) Define diffusion. [2 marks]",
        "(b) Give one example of osmosis in plants. [2 marks]",
    ))
    by_id = {q.id: q for q in questions}
    assert list(by_id) == ["1", "2", "3", "4", "5", "6a", "6b"]

    assert by_id["1"].options == {"A": "Nucleus", "B": "Mitochondria", "C": "Ribosome", "D": "Golgi body"}
    assert (by_id["1"].marks, by_id["1"].qtype) == (1.0, "mcq")
    assert by_id["2"].text == "Which gas is released in photosynthesis?"
    assert by_id["2"].options["C"] == "Carbon dioxide" and by_id["2"].marks == 1.0
    assert by_id["3"].text == "Explain photosynthesis with its equation. Mention the role of chlorophyll."
    assert (by_id["3"].marks, by_id["3"].qtype) == (5.0, "descriptive")
    assert by_id["4"].qtype == "mixed" and by_id["4"].options["B"] == "White blood cells"
    assert by_id["5"].wants_diagram and by_id["5"].marks == 4.0
    assert (by_id["6a"].text, by_id["6a"].marks) == ("Define diffusion.", 2.0)
    assert by_id["6b"].qtype == "short"


def test_numbered_instructions_are_not_questions_when_paper_uses_q_markers():
    questions = parse_question_paper(_paper(
        "General instructions:",
        "1. All questions are compulsory.",
        "2. Draw diagrams wherever necessary.",
        "Q1. What is a gene? [2]",
        "Q2. Define mutation. [2]",
    ))
    assert [(q.id, q.text) for q in questions] == [("1", "What is a gene?"), ("2", "Define mutation.")]


def test_numbered_instructions_are_dropped_when_numbering_restarts():
    questions = parse_question_paper(_paper(
        "1. All questions are compulsory.",
        "2. Marks are given in brackets.",
        "1. What is a gene? [2]",
        "2. Define mutation. [2]",
        "3. What is DNA? [1]",
    ))
    assert [(q.id, q.text) for q in questions] == [
        ("1", "What is a gene?"), ("2", "Define mutation."), ("3", "What is DNA?")]


def test_roman_subparts():
    questions = parse_question_paper(_paper(
        "Q7. Answer briefly:",
        "(i) What is a tissue? [1]",
        "(ii) What is an organ? [1]",
    ))
    assert [q.id for q in questions] == ["7i", "7.ii"]


def test_nothing_recognised():
    assert parse_question_paper(_paper("Some notes", "without any numbering")) == []


def test_llm_fallback_structures_the_paper():
    reply = json.dumps({"questions": [
        {"id": "1", "text": "Define osmosis.", "marks": 2, "options": {}},
        {"id": "2", "text": "Powerhouse of the cell?", "marks": None, "options": {"a": "Nucleus", "b": "Mitochondria"}},
    ]})
    calls = []

    def llm(messages, schema=None):
        calls.append(messages)
        return reply

    questions = parse_with_llm(_paper("Define osmosis (2 marks) ... Powerhouse? a Nucleus b Mitochondria"), llm)
    assert questions == [
        ParsedQuestion("1", "Define osmosis.", 2.0, {}),
        ParsedQuestion("2", "Powerhouse of the cell?", None, {"A": "Nucleus", "B": "Mitochondria"}),
    ]
    assert "<question_paper>" in calls[0][0]["content"]
