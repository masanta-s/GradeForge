"""Answer key model: questions, marks, model answers, and where each answer came from.

Question types:
  mcq          choose an option                          (options + correct_option)
  short        written answer                            (model_answer)
  descriptive  long written answer                       (model_answer)
  mixed        choose an option AND justify it           (all of the above + option_marks;
               the option earns option_marks, the justification the rest)

Any question may also carry a `diagram` part (DiagramSpec) worth some of its marks; the
written part earns what's left. A diagram-only question needs no written model answer.

A paper can freely mix all types. Stored as JSON under data/answer_keys/. Question ids follow
the answer segmenter's format: "3", "3b", "3.ii".
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from src.diagram.weightage import DiagramSpec

QuestionType = Literal["mcq", "short", "descriptive", "mixed"]
AnswerSource = Literal["teacher", "ai_generated", "ai_suggestion_accepted", "teacher_override"]


@dataclass
class Question:
    id: str
    text: str
    qtype: QuestionType
    max_marks: float
    model_answer: str = ""
    keywords: list[str] = field(default_factory=list)      # 1-3 word terms for the local cross-check
    key_points: list[str] = field(default_factory=list)    # phrases telling the LLM what earns marks
    options: dict[str, str] = field(default_factory=dict)  # MCQ: {"A": "Mitochondria", ...}
    correct_option: str | None = None                      # MCQ: "B"
    option_marks: float = 0.0                              # mixed: marks for choosing correctly
    diagram: DiagramSpec | None = None                     # optional drawn part (Phase 4)
    source: AnswerSource = "teacher"
    ai_confidence: float | None = None                     # model's self-reported confidence, if AI-made
    explanation: str = ""                                  # why this is the answer (shown to the teacher)

    @property
    def has_options(self) -> bool:
        return self.qtype in ("mcq", "mixed")

    @property
    def diagram_marks(self) -> float:
        return self.diagram.marks if self.diagram else 0.0

    @property
    def written_marks(self) -> float:
        """Marks for written text: the answer (short/descriptive) or the justification (mixed)."""
        if self.qtype == "mcq":
            return 0.0
        return self.max_marks - self.diagram_marks - (self.option_marks if self.qtype == "mixed" else 0.0)

    @property
    def has_written_part(self) -> bool:
        return self.written_marks > 0

    @property
    def justification_marks(self) -> float:
        return self.written_marks if self.qtype == "mixed" else 0.0

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
        if self.qtype == "mixed" and not (self.option_marks > 0 and self.written_marks > 0):
            problems.append(f"Q{self.id}: mixed question needs 0 < option_marks < max_marks")
        if self.diagram:
            problems += self.diagram.validate(self.id, self.max_marks)
            if self.written_marks < 0:
                problems.append(f"Q{self.id}: option and diagram marks exceed max_marks")
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
    def from_dict(cls, data: dict) -> AnswerKey:
        """Build without validating — drafts under review may still be incomplete."""
        data = dict(data)
        questions = []
        for q in data.pop("questions"):
            q = dict(q)
            if q.get("diagram"):
                q["diagram"] = DiagramSpec(**q["diagram"])
            questions.append(Question(**q))
        return cls(questions=questions, **data)

    @classmethod
    def load(cls, path: Path, validate: bool = True) -> AnswerKey:
        key = cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if validate and (problems := key.validate()):
            raise ValueError(f"invalid answer key {path.name}: " + "; ".join(problems))
        return key
