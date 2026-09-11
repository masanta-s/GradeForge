"""Training data for fine-tuning the grading LLM: the grader's own prompt -> the teacher's score.

Each example is the exact prompt the grader sends (system prompt, question, model answer, key
points, few-shot examples, student answer) followed by the start of the JSON reply. The grader's
schema puts "quality" first, so the loss is on `{"quality": 0.42` (the teacher's score), plus the
feedback only when the teacher wrote a note. Untrained parts are simply not in the target, so the
model is never taught to write filler feedback or empty "missing_points", and its own feedback
style is kept.

Only the training split is used (see evaluation.split_of), so held-out answers stay unseen and
the benchmark stays honest. Incremental rounds train on answers added since the last version
plus a random replay of older ones, so earlier lessons aren't forgotten.
"""
from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from src.grading.answer_key import Question
from src.grading.subjective_grader import GradedExample, build_messages
from src.learning.correction_store import GradeCorrection
from src.learning.evaluation import measurable, question_from_correction, split_of

REPLAY_RATIO = 2.0   # older examples replayed per new one in an incremental round


@dataclass(frozen=True)
class TrainingExample:
    correction_id: int
    created_at: str
    messages: list[dict]     # system + user: exactly what the grader sends
    target: str              # the reply prefix the model learns


def target_for(c: GradeCorrection) -> str:
    target = f'{{"quality": {c.teacher_quality:.2f}'
    if c.note.strip():
        target += f', "feedback": {json.dumps(c.note.strip(), ensure_ascii=False)}'
    return target


def build_examples(corrections: Sequence[GradeCorrection], *, splits: tuple[str, ...] = ("train",),
                   question_for: Callable[[GradeCorrection], Question] = question_from_correction,
                   examples_for: Callable[[GradeCorrection], Sequence[GradedExample]] | None = None,
                   ) -> list[TrainingExample]:
    """`examples_for(c)`: few-shot examples as grading would show them (the caller excludes `c`
    itself and the held-out answers)."""
    out = []
    for c in corrections:
        if not measurable(c) or split_of(c) not in splits:
            continue
        shots = examples_for(c) if examples_for else ()
        out.append(TrainingExample(c.id, c.created_at, build_messages(question_for(c), c.student_answer, shots),
                                   target_for(c)))
    return out


def select_round(examples: Sequence[TrainingExample], since: str | None, replay_ratio: float = REPLAY_RATIO,
                 seed: int = 0) -> list[TrainingExample]:
    """Everything for a first version; afterwards, new examples plus a replay of older ones."""
    if since is None:
        return list(examples)
    new = [e for e in examples if e.created_at > since]
    old = [e for e in examples if e.created_at <= since]
    if not new:
        return []
    replay = random.Random(seed).sample(old, min(len(old), round(replay_ratio * len(new))))
    return new + replay
