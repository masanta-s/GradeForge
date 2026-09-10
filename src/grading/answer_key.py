"""Answer key model: questions, marks, model answers, and where each answer came from.

Question types:
  mcq          choose an option                          (options + correct_option)
  short        written answer                            (model_answer)
  descriptive  long written answer                       (model_answer)
  mixed        choose an option AND justify it           (all of the above + option_marks;
               the option earns option_marks, the justification the rest)

A paper can freely mix all types. Stored as JSON under data/answer_keys/. Question ids follow
the answer segmenter's format: "3", "3b", "3.ii".
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

QuestionType = Literal["mcq", "short", "descriptive", "mixed"]
AnswerSource = Literal["teacher", "ai_generated", "ai_suggestion_accepted", "teacher_override"]


@dataclass
class Question:
    id: str
    text: str
    qtype: QuestionType
    max_marks: float
    model_answer: str = ""
    keywords: list[str] = field(default_factory=list)
    options: dict[str, str] = field(default_factory=dict)  # MCQ: {"A": "Mitochondria", ...}
    correct_option: str | None = None                      # MCQ: "B"
    option_marks: float = 0.0                              # mixed: marks for choosing correctly
    source: AnswerSource = "teacher"

    @property
    def has_options(self) -> bool:
        return self.qtype in ("mcq", "mixed")

    @property
    def has_written_part(self) -> bool:
        return self.qtype != "mcq"

    @property
    def justification_marks(self) -> float:
        return self.max_marks - self.option_marks if self.qtype == "mixed" else 0.0

    def validate(self) -> list[str]:
        problems = []
        if self.max_marks <= 0:
            problems.append(f"Q{self.id}: max_marks must be positive")
        if self.has_options:
            if not self.options:
                problems.append(f"Q{self.id}: {self.qtype} question needs options")
            elif self.correct_option not in self.options:
                problems.append(f"Q{self.id}: correct_option {self.correct_option!r} is not one of {sorted(self.options)}")
        if self.has_written_part and not self.model_answer.strip():
            problems.append(f"Q{self.id}: {self.qtype} question needs a model answer")
        if self.qtype == "mixed" and not 0 < self.option_marks < self.max_marks:
            problems.append(f"Q{self.id}: mixed question needs 0 < option_marks < max_marks")
        return problems


@dataclass
class AnswerKey:
    exam_name: str
    subject: str
    questions: list[Question]
    finalized: bool = False

    @property
    def total_marks(self) -> float:
        return sum(q.max_marks for q in self.questions)

    @property
    def question_ids(self) -> list[str]:
        return [q.id for q in self.questions]

    def get(self, question_id: str) -> Question:
        for q in self.questions:
            if q.id == question_id:
                return q
        raise KeyError(question_id)

    def validate(self) -> list[str]:
        problems = [p for q in self.questions for p in q.validate()]
        seen: set[str] = set()
        for q in self.questions:
            if q.id in seen:
                problems.append(f"duplicate question id {q.id!r}")
            seen.add(q.id)
        return problems

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> AnswerKey:
        data = json.loads(path.read_text(encoding="utf-8"))
        questions = [Question(**q) for q in data.pop("questions")]
        key = cls(questions=questions, **data)
        if problems := key.validate():
            raise ValueError(f"invalid answer key {path.name}: " + "; ".join(problems))
        return key
