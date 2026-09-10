"""Re-score a graded paper at a different strictness without calling any model.

Strictness only changes the curve from quality (stored per question) to marks, so moving the
slider after grading is instant. MCQ marks don't depend on strictness; marks a teacher set by
hand are never changed.
"""
from __future__ import annotations

import copy

from src.grading.strictness_curve import apply_strictness
from src.grading.subjective_grader import round_marks


def _curve(part: dict | None, strictness: float) -> float:
    if not part or part.get("quality") is None:
        return 0.0
    return round_marks(part["max_marks"] * apply_strictness(part["quality"], strictness), part["max_marks"])


def rescore(result: dict, strictness: float) -> dict:
    out = copy.deepcopy(result)
    for q in out["questions"]:
        if q["status"] != "graded" or q.get("teacher_marks") is not None:
            continue
        detail = q.get("detail") or {}
        if q["qtype"] == "mcq":
            marks = detail.get("marks", q["marks"])
        elif q["qtype"] == "mixed" and "choice" in detail:
            marks = detail["choice"]["marks"] + _curve(detail["justification"], strictness)
        else:
            marks = _curve(detail, strictness)
        if q.get("diagram"):
            marks += _curve(q["diagram"], strictness)
        q["marks"] = marks
    out["strictness"] = strictness
    out["total"] = sum(q["marks"] for q in out["questions"])
    out["percentage"] = 100 * out["total"] / out["max_total"] if out["max_total"] else 0.0
    return out
