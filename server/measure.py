"""The held-out benchmark, wired to the live services.

A model "setup" is measured exactly as it grades: the answer key's question (with key points),
the grading prompt, and for the "learned" setup the teacher's few-shot examples and this model's
calibration. Held-out answers are excluded from both, so the setup never sees its own test.
"""
from __future__ import annotations

from collections.abc import Callable

from src.grading.answer_key import Question
from src.learning.correction_store import GradeCorrection
from src.learning.evaluation import GradeFn, holdout_ids, holdout_items, measure, question_from_correction
from src.learning.score_calibrator import ScoreCalibrator

SETUPS = {"alone": "the model alone", "learned": "with your examples and calibration"}


def question_lookup(services) -> Callable[[GradeCorrection], Question]:
    """The answer key's version of the question (key points included), when it still exists."""
    keys: dict = {}

    def question_for(c: GradeCorrection) -> Question:
        if c.exam_id not in keys:
            try:
                keys[c.exam_id] = services.storage.get_key(c.exam_id)
            except KeyError:
                keys[c.exam_id] = None
        key = keys[c.exam_id]
        try:
            return key.get(c.question_id) if key is not None else question_from_correction(c)
        except KeyError:
            return question_from_correction(c)

    return question_for


def grade_fn(services, llm, model_name: str, *, learned: bool, holdout: frozenset[int]) -> GradeFn:
    calibrator = ScoreCalibrator(services.corrections, exclude_ids=holdout)
    question_for = question_lookup(services)

    def grade(c: GradeCorrection) -> float:
        question = question_for(c)
        examples = (services.retriever.retrieve(question.text, c.student_answer, subject=c.subject, k=4,
                                                exclude_ids=holdout) if learned else ())
        calibrate = (lambda q: calibrator.calibrate(model_name, c.subject, q)) if learned else None
        return services.subjective.grade(question, c.student_answer, c.strictness, llm=llm, examples=examples,
                                         calibrate=calibrate).marks

    return grade


def run_benchmark(services, *, model_name: str, llm, setups=("alone", "learned"), trigger: str = "manual",
                  progress: Callable[[float, str], None] | None = None) -> list[dict]:
    report = progress or (lambda p, m: None)
    corrections = services.corrections.grade_corrections()
    items = holdout_items(corrections)
    if not items:
        raise ValueError("no held-out answers yet: correct or approve a few graded sheets first")
    holdout = holdout_ids(corrections)
    total = len(holdout_items(corrections, limit=None))
    runs = []
    for i, setup in enumerate(setups):
        label = f"{model_name}, {SETUPS[setup]}"
        result, _ = measure(items, grade_fn(services, llm, model_name, learned=setup == "learned", holdout=holdout),
                            progress=lambda p, m, i=i, label=label: report((i + p) / len(setups), f"{label}: {m}"))
        run_id = services.corrections.add_benchmark_run(label=label, model=model_name, setup=setup,
                                                        holdout_total=total, trigger=trigger, **result.to_dict())
        runs.append({"id": run_id, "label": label, "setup": setup, **result.to_dict()})
    return runs
