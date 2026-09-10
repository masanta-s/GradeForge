"""Integration: question paper PDF -> parsed questions -> qwen3.5:9b answer key -> validation."""
import pymupdf
import pytest

from src import config
from src.grading.answer_key import Question
from tests.test_grading_integration import _ollama_has_model

pytestmark = pytest.mark.skipif(not _ollama_has_model(), reason=f"Ollama with {config.DEFAULT_LLM} not available")

PAPER = [
    "Class 10 Biology - Unit Test 2        Max marks: 12",
    "General instructions: all questions are compulsory.",
    "Q1. Which organelle is known as the powerhouse of the cell? [1]",
    "(A) Nucleus   (B) Mitochondria   (C) Ribosome   (D) Golgi body",
    "Q2. Which gas is released during photosynthesis? [1]",
    "Q3. Explain the process of osmosis with one example. [3]",
    "Q4. Which blood cells help fight infection? Justify your answer. [3]",
    "(A) Red blood cells   (B) White blood cells   (C) Platelets",
    "Q5. Draw a neat labelled diagram of an animal cell. [4]",
]


@pytest.fixture(scope="module")
def llm():
    from src.knowledge.llm_client import LLMClient

    return LLMClient()


@pytest.fixture(scope="module")
def key(tmp_path_factory, llm):
    from src.knowledge.answer_generator import generate_answer_key
    from src.knowledge.question_parser import parse_question_paper
    from src.ocr.pipeline import OCRPipeline

    path = tmp_path_factory.mktemp("paper") / "paper.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        for i, line in enumerate(PAPER):
            page.insert_text((50, 60 + i * 22), line, fontsize=10)
        doc.save(path)
    questions = parse_question_paper(OCRPipeline().run_file(path))
    key = generate_answer_key(questions, llm, subject="Biology", exam="Class 10 Unit Test 2")
    for q in key.questions:
        print(f"\nQ{q.id} [{q.qtype}] {q.max_marks:g}m conf={q.ai_confidence}: option={q.correct_option} "
              f"answer={q.model_answer[:90]!r} keywords={q.keywords[:5]} "
              f"diagram={q.diagram.required_labels if q.diagram else None}")
    return key


def test_paper_parsed_and_key_generated(key):
    assert [q.id for q in key.questions] == ["1", "2", "3", "4", "5"]
    assert key.validate() == []
    assert key.total_marks == 12


def test_generated_answers_are_right(key):
    q = {q.id: q for q in key.questions}
    assert q["1"].correct_option == "B"
    assert "oxygen" in q["2"].model_answer.lower()
    assert q["4"].qtype == "mixed" and q["4"].correct_option == "B"
    assert q["5"].diagram is not None
    labels = " ".join(q["5"].diagram.required_labels).lower()
    assert "nucleus" in labels and "membrane" in labels


@pytest.fixture(scope="module")
def validations(llm):
    from src.knowledge.answer_validator import validate_question

    entries = {
        "mcq_wrong": Question(id="1", text="Which organelle is known as the powerhouse of the cell?", qtype="mcq",
                              max_marks=1, options={"A": "Nucleus", "B": "Mitochondria", "C": "Ribosome"},
                              correct_option="A"),
        "mcq_right": Question(id="1", text="Which organelle is known as the powerhouse of the cell?", qtype="mcq",
                              max_marks=1, options={"A": "Nucleus", "B": "Mitochondria", "C": "Ribosome"},
                              correct_option="B"),
        "written_wrong": Question(id="2", text="Which gas is released during photosynthesis?", qtype="short",
                                  max_marks=1, model_answer="Carbon dioxide is released during photosynthesis."),
        "written_right": Question(id="2", text="Which gas is released during photosynthesis?", qtype="short",
                                  max_marks=1, model_answer="Oxygen is released during photosynthesis."),
    }
    out = {name: validate_question(q, llm, "Biology") for name, q in entries.items()}
    for name, v in out.items():
        print(f"\n{name:13} {v.verdict:9} ai={v.ai_answer!r} conf={v.confidence}\n              {v.justification}")
    return out, entries


def test_wrong_teacher_answers_are_flagged(validations):
    results, _ = validations
    assert results["mcq_wrong"].flagged and results["mcq_wrong"].ai_answer.startswith("B")
    assert results["written_wrong"].flagged and "oxygen" in results["written_wrong"].ai_answer.lower()


def test_correct_teacher_answers_pass(validations):
    results, _ = validations
    assert results["mcq_right"].verdict == "agrees"
    assert results["written_right"].verdict == "agrees"


def test_ai_holds_position_against_a_wrong_argument(validations, llm, tmp_path):
    from src.knowledge.dispute_logger import DisputeLog
    from src.knowledge.dispute_manager import Dispute

    results, entries = validations
    dispute = Dispute(entries["mcq_wrong"], results["mcq_wrong"], subject="Biology", exam="Unit Test 2",
                      llm=llm, log=DisputeLog(tmp_path / "d.db"))
    reply = dispute.discuss("The nucleus controls the cell, so it must be the powerhouse.")
    print(f"\nAI reply: {reply}")
    assert not dispute.ai_conceded
