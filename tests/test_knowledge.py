import csv
import json

import pytest

from src.grading.answer_key import Question
from src.knowledge.answer_generator import generate_answer_key, generate_question, review_needed
from src.knowledge.answer_validator import validate_question
from src.knowledge.dispute_logger import DisputeLog, DisputeRecord
from src.knowledge.dispute_manager import Dispute
from src.knowledge.question_parser import ParsedQuestion


class SchemaLLM:
    """Replies according to which JSON schema was requested (identified by its required keys)."""

    def __init__(self, **replies):
        self.replies = replies
        self.calls = []

    def __call__(self, messages, schema=None):
        self.calls.append(messages)
        required = tuple(sorted(schema["required"])) if schema else ()
        for key, reply in self.replies.items():
            if key in required:
                return json.dumps(reply) if not isinstance(reply, str) else reply
        raise AssertionError(f"no scripted reply for schema {required}")


MCQ = ParsedQuestion("1", "Powerhouse of the cell?", 1.0, {"A": "Nucleus", "B": "Mitochondria"})


# --- generation ------------------------------------------------------------------------

def test_generate_mcq():
    llm = SchemaLLM(correct_option={"correct_option": "(b)", "explanation": "Makes ATP.", "confidence": 0.95})
    q = generate_question(MCQ, llm, "Biology", "Unit test")
    assert (q.qtype, q.correct_option, q.explanation, q.source) == ("mcq", "B", "Makes ATP.", "ai_generated")
    assert q.validate() == []
    assert "Unit test" in llm.calls[0][0]["content"] and "B) Mitochondria" in llm.calls[0][1]["content"]


def test_generate_written_uses_suggested_marks_only_when_paper_has_none():
    reply = {"model_answer": "Osmosis is ...", "key_points": ["water", "membrane"], "suggested_marks": 3,
             "confidence": 0.9}
    stated = generate_question(ParsedQuestion("2", "Define osmosis.", 2.0), SchemaLLM(model_answer=reply), "Bio", "T")
    unstated = generate_question(ParsedQuestion("2", "Define osmosis.", None), SchemaLLM(model_answer=reply), "Bio", "T")
    assert (stated.max_marks, unstated.max_marks) == (2.0, 3.0)
    assert stated.keywords == ["water", "membrane"]


def test_generate_mixed_splits_marks():
    pq = ParsedQuestion("3", "Which cells fight infection? Justify.", 3.0, {"A": "RBC", "B": "WBC"})
    llm = SchemaLLM(correct_option={"correct_option": "B", "model_answer": "WBCs make antibodies.",
                                    "key_points": ["antibodies"], "confidence": 0.9})
    q = generate_question(pq, llm, "Bio", "T")
    assert (q.qtype, q.correct_option, q.option_marks, q.justification_marks) == ("mixed", "B", 1.0, 2.0)


def test_generate_diagram_question():
    pq = ParsedQuestion("4", "Draw a labelled diagram of a neuron.", 3.0)
    llm = SchemaLLM(required_labels={"description": "A neuron.", "required_labels": ["Dendrite", "Axon"],
                                     "diagram_marks": 3, "confidence": 0.8})
    q = generate_question(pq, llm, "Bio", "T")
    assert q.diagram.required_labels == ["Dendrite", "Axon"] and q.diagram.marks == 3.0
    assert q.written_marks == 0 and q.validate() == []


def test_failed_generation_keeps_paper_options_and_is_flagged():
    key = generate_answer_key([MCQ], SchemaLLM(correct_option="garbage"), subject="Bio", exam="T")
    [q] = key.questions
    assert q.options == MCQ.options  # from the paper, so the teacher needn't retype them
    assert q.correct_option is None and "please write" in q.explanation
    [(flagged, reason)] = review_needed(key)
    assert flagged.id == "1" and "correct_option" in reason


def test_low_confidence_is_flagged():
    llm = SchemaLLM(correct_option={"correct_option": "B", "explanation": "?", "confidence": 0.4})
    key = generate_answer_key([MCQ], llm, subject="Bio", exam="T")
    [(q, reason)] = review_needed(key)
    assert reason == "model confidence 40%"


# --- validation ------------------------------------------------------------------------

TEACHER_MCQ = Question(id="1", text="Powerhouse of the cell?", qtype="mcq", max_marks=1,
                       options={"A": "Nucleus", "B": "Mitochondria"}, correct_option="A")


def test_mcq_validation_solves_blind_and_disagrees():
    llm = SchemaLLM(correct_option={"correct_option": "B", "reasoning": "ATP is made in mitochondria.",
                                    "confidence": 0.97})
    result = validate_question(TEACHER_MCQ, llm, "Biology")
    assert result.flagged and result.ai_answer == "B) Mitochondria" and result.teacher_answer == "A) Nucleus"
    prompt = llm.calls[0][1]["content"]
    assert "Nucleus" in prompt and "answer key" not in prompt.lower()  # teacher's choice never shown


def test_mcq_validation_agrees():
    llm = SchemaLLM(correct_option={"correct_option": "A", "reasoning": "ok", "confidence": 0.9})
    assert validate_question(TEACHER_MCQ, llm, "Biology").verdict == "agrees"


def test_written_validation_reports_problems():
    q = Question(id="2", text="What gas do plants release in photosynthesis?", qtype="short", max_marks=1,
                 model_answer="Carbon dioxide")
    llm = SchemaLLM(is_correct={"is_correct": False, "problems": ["Plants release oxygen, not CO2."],
                                "suggested_answer": "Oxygen", "confidence": 0.95})
    result = validate_question(q, llm, "Biology")
    assert (result.verdict, result.ai_answer) == ("disagrees", "Oxygen")
    assert "oxygen, not CO2" in result.justification
    assert "<answer_key_entry>\nCarbon dioxide\n</answer_key_entry>" in llm.calls[0][1]["content"]


def test_unusable_validation_is_unsure():
    assert validate_question(TEACHER_MCQ, SchemaLLM(correct_option="???"), "Bio").verdict == "unsure"


# --- dispute log -----------------------------------------------------------------------

def _record(**changes):
    base = dict(teacher_name="Mrs. Iyer", exam_name="Unit test", subject="Biology", question_id="1",
                question_text="Powerhouse?", teacher_answer="A) Nucleus", ai_answer="B) Mitochondria",
                ai_justification="ATP", teacher_decision="one_time_override",
                conversation=[{"role": "ai", "content": "Maybe wrong", "at": "t"}])
    return DisputeRecord(**{**base, **changes})


def test_log_roundtrip_search_and_export(tmp_path):
    log = DisputeLog(tmp_path / "disputes.db")
    first = log.add(_record())
    log.add(_record(teacher_name="Mr. Das", subject="Physics", question_text="Unit of force?",
                    teacher_decision="save_correction", use_for_training=True))
    assert log.get(first).conversation[0]["content"] == "Maybe wrong"
    assert [r.teacher_name for r in log.search(subject="Physics")] == ["Mr. Das"]
    assert [r.teacher_name for r in log.search(teacher="iyer")] == ["Mrs. Iyer"]
    assert [r.question_text for r in log.search(text="force")] == ["Unit of force?"]
    assert [r.teacher_name for r in log.search(training_only=True)] == ["Mr. Das"]

    path = log.export_csv(tmp_path / "out" / "disputes.csv")
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    assert len(rows) == 2 and rows[0]["conversation"] == "[ai] Maybe wrong"


def test_log_requires_teacher_name(tmp_path):
    with pytest.raises(ValueError):
        DisputeLog(tmp_path / "d.db").add(_record(teacher_name="  "))


# --- dispute flow ----------------------------------------------------------------------

@pytest.fixture
def dispute(tmp_path):
    validation = validate_question(TEACHER_MCQ, SchemaLLM(
        correct_option={"correct_option": "B", "reasoning": "ATP is made in mitochondria.", "confidence": 0.97}),
        "Biology")
    llm = SchemaLLM(concede={"concede": False, "response": "Respiration happens in mitochondria."})
    return Dispute(TEACHER_MCQ, validation, subject="Biology", exam="Unit test", llm=llm,
                   log=DisputeLog(tmp_path / "d.db"))


def test_accept_ai_updates_key_and_logs(dispute):
    updated = dispute.accept_ai("Mrs. Iyer")
    assert (updated.correct_option, updated.source) == ("B", "ai_suggestion_accepted")
    assert TEACHER_MCQ.correct_option == "A"  # original untouched
    [record] = dispute.log.search()
    assert (record.teacher_decision, record.use_for_training) == ("accepted_ai", False)


def test_discussion_is_recorded_and_insist_saves_correction(dispute):
    reply = dispute.discuss("In our textbook the nucleus is called the control centre.")
    assert reply == "Respiration happens in mitochondria." and not dispute.ai_conceded
    kept = dispute.insist("Mrs. Iyer", save_correction=True)
    assert (kept.correct_option, kept.source) == ("A", "teacher_override")
    [record] = dispute.log.search()
    assert record.teacher_decision == "save_correction" and record.use_for_training
    assert [t["role"] for t in record.conversation] == ["ai", "teacher", "ai"]


def test_ai_can_concede(dispute):
    dispute.llm = SchemaLLM(concede={"concede": True, "response": "You're right, both are accepted."})
    dispute.discuss("Our board accepts both answers.")
    assert dispute.ai_conceded
