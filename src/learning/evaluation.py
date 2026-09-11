"""How closely the AI's marks match the teacher's, and whether that is improving.

Two measures, answering two questions:

1. Day by day ("is it getting better as I use it?"): every approved sheet records how many of
   the AI's marks the teacher changed. Grouped by week, the share of marks accepted as-is and the
   average change show the trend with no extra model runs.

2. Held-out benchmark ("how much did this version improve?"): a fixed 20% of teacher-checked
   answers is never used for training, few-shot examples or calibration while being measured.
   Each model setup re-marks the same answers and is compared with the teacher: exact agreement,
   agreement within half a mark (or 10% of the question), average difference, and bias.

The split is a hash of (sheet, question), so an answer stays in the same split forever and new
answers join the held-out set at the same 20% rate as the data grows. The training split also
sets aside 10% as a validation set that early stopping watches, so the held-out set is only
ever used to judge finished versions.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import datetime

from src.grading.answer_key import Question
from src.learning.correction_store import GradeCorrection

HOLDOUT_EVERY = 5        # 1 in 5 answers is held out
VALIDATION_EVERY = 10    # 1 in 10 of the rest watches for overfitting during training
DEFAULT_SAMPLE = 150     # held-out answers re-marked per measurement (~3 s each with qwen3.5:9b)
MIN_GAIN_PCT = 1.0       # a new version must cut the average difference by 1 point of the max marks


def _bucket(c: GradeCorrection) -> int:
    return int(hashlib.sha1(f"{c.sheet_id}/{c.question_id}".encode()).hexdigest()[:12], 16)


def split_of(c: GradeCorrection) -> str:
    bucket = _bucket(c)
    if bucket % HOLDOUT_EVERY == 0:
        return "holdout"
    return "validation" if (bucket // HOLDOUT_EVERY) % VALIDATION_EVERY == 0 else "train"


def measurable(c: GradeCorrection) -> bool:
    """Plain written answers with a score the LLM produces (MCQs are marked by rule)."""
    return c.teacher_quality is not None and bool(c.student_answer.strip())


def holdout_items(corrections: Iterable[GradeCorrection], limit: int | None = DEFAULT_SAMPLE) -> list[GradeCorrection]:
    """A stable sample: the same answers are chosen run after run while the pool is unchanged."""
    pool = sorted((c for c in corrections if measurable(c) and split_of(c) == "holdout"), key=_bucket)
    return pool if limit is None else pool[:limit]


def holdout_ids(corrections: Iterable[GradeCorrection]) -> frozenset[int]:
    return frozenset(c.id for c in corrections if split_of(c) == "holdout")


def question_from_correction(c: GradeCorrection) -> Question:
    return Question(id=c.question_id, text=c.question_text, qtype="short", max_marks=c.max_marks,
                    model_answer=c.model_answer)


# --- metrics --------------------------------------------------------------------------------

@dataclass(frozen=True)
class Agreement:
    n: int
    exact: float      # share of answers where the AI gave exactly the teacher's mark
    close: float      # ... within half a mark (or 10% of the question's marks, if larger)
    mae: float        # average difference, in marks
    mae_pct: float    # average difference, as % of each question's max marks
    bias: float       # average (AI - teacher) in marks: + means the AI is more generous

    def to_dict(self) -> dict:
        return asdict(self)


def agreement(pairs: Iterable[tuple[float, float, float]]) -> Agreement:
    """`pairs`: (ai_marks, teacher_marks, max_marks) per answer."""
    rows = [(a, t, m) for a, t, m in pairs if m > 0]
    if not rows:
        return Agreement(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    diffs = [a - t for a, t, _ in rows]
    n = len(rows)
    return Agreement(
        n=n,
        exact=sum(abs(d) < 0.25 for d in diffs) / n,
        close=sum(abs(d) <= max(0.5, 0.1 * m) + 1e-9 for d, (_, _, m) in zip(diffs, rows)) / n,
        mae=sum(abs(d) for d in diffs) / n,
        mae_pct=100 * sum(abs(d) / m for d, (_, _, m) in zip(diffs, rows)) / n,
        bias=sum(diffs) / n,
    )


def is_better(new: Agreement, current: Agreement, min_gain_pct: float = MIN_GAIN_PCT) -> bool:
    """Promote only a clear gain: a smaller average difference, without fewer close marks."""
    return new.n > 0 and new.mae_pct <= current.mae_pct - min_gain_pct and new.close >= current.close - 0.02


@dataclass(frozen=True)
class ItemResult:
    correction_id: int
    ai_marks: float
    teacher_marks: float
    max_marks: float


GradeFn = Callable[[GradeCorrection], float]   # a held-out answer -> the AI's marks for it


def measure(items: list[GradeCorrection], grade: GradeFn,
            progress: Callable[[float, str], None] | None = None) -> tuple[Agreement, list[ItemResult]]:
    report = progress or (lambda p, m: None)
    results = []
    for i, c in enumerate(items):
        report(i / max(1, len(items)), f"Re-marking held-out answer {i + 1} of {len(items)}")
        results.append(ItemResult(c.id, float(grade(c)), c.teacher_marks, c.max_marks))
    return agreement((r.ai_marks, r.teacher_marks, r.max_marks) for r in results), results


# --- day by day -----------------------------------------------------------------------------

def weekly_agreement(reviews: Iterable[dict]) -> list[dict]:
    """Approved sheets grouped by ISO week: how many AI marks the teacher kept, on average."""
    weeks: dict[str, dict] = defaultdict(lambda: {"sheets": 0, "judged": 0, "changed": 0, "abs_diff": 0.0,
                                                   "models": set()})
    for r in reviews:
        year, week, _ = datetime.fromisoformat(r["created_at"]).isocalendar()
        w = weeks[f"{year}-W{week:02d}"]
        w["sheets"] += 1
        w["judged"] += r["judged"]
        w["changed"] += r["changed"]
        w["abs_diff"] += r["abs_diff"]
        w["models"].add(r["model"])
    out = []
    for key in sorted(weeks):
        w = weeks[key]
        judged = w["judged"] or 1
        out.append({"week": key, "sheets": w["sheets"], "judged": w["judged"], "changed": w["changed"],
                    "accepted_pct": round(100 * (1 - w["changed"] / judged), 1),
                    "avg_change": round(w["abs_diff"] / judged, 2), "models": sorted(w["models"])})
    return out
