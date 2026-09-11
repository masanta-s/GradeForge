"""Measuring agreement with the teacher: splits, metrics, trend, training data."""
import sqlite3
from contextlib import closing

import pytest

from src.learning.correction_store import CorrectionStore, GradeCorrection
from src.learning.evaluation import (Agreement, agreement, holdout_ids, holdout_items, is_better, split_of,
                                     weekly_agreement)
from src.learning.llm_dataset import build_examples, select_round, target_for


def correction(i, *, sheet=None, teacher=2.0, ai=3.0, note="", created="2026-09-01T10:00:00+00:00", kind="changed",
               quality=0.5):
    return GradeCorrection(id=i, created_at=created, exam_id="e", sheet_id=sheet or f"s{i}", question_id="2",
                           subject="Biology", model="qwen3.5:9b", question_text="What is photosynthesis?",
                           model_answer="Plants make glucose.", student_answer=f"answer {i}", max_marks=4,
                           ai_marks=ai, teacher_marks=teacher, ai_quality=0.6, teacher_quality=quality,
                           strictness=50, note=note, teacher_name="T", kind=kind)


POOL = [correction(i) for i in range(2000)]


def test_split_is_stable_and_about_one_in_five():
    splits = [split_of(c) for c in POOL]
    assert 0.17 < splits.count("holdout") / len(POOL) < 0.23
    assert 0.06 < splits.count("validation") / len(POOL) < 0.10
    # the same answer always lands in the same split, whatever else is in the pool
    assert [split_of(c) for c in POOL[:50]] == splits[:50]
    assert split_of(correction(7, note="changed later")) == split_of(POOL[7])


def test_holdout_sample_is_deterministic_and_growing_pool_keeps_most_items():
    first = holdout_items(POOL[:1000], limit=100)
    assert [c.id for c in first] == [c.id for c in holdout_items(POOL[:1000], limit=100)]
    grown = {c.id for c in holdout_items(POOL, limit=100)}
    assert len(grown & {c.id for c in first}) > 30   # a stable-ish sample as data grows
    assert holdout_ids(POOL) == {c.id for c in POOL if split_of(c) == "holdout"}
    assert all(split_of(c) == "holdout" for c in first)


def test_agreement_metrics():
    # (ai, teacher, max): exact, off by 0.5, off by 2 on a 10-mark question (within 10%? no: 1.0 allowed)
    result = agreement([(3, 3, 4), (2.5, 3, 4), (6, 8, 10), (1, 1, 2)])
    assert result.n == 4 and result.exact == 0.5 and result.close == 0.75
    assert result.mae == pytest.approx((0 + 0.5 + 2 + 0) / 4)
    assert result.mae_pct == pytest.approx(100 * (0 + 0.125 + 0.2 + 0) / 4)
    assert result.bias == pytest.approx((0 - 0.5 - 2 + 0) / 4)        # the AI marks lower than the teacher
    assert agreement([]) == Agreement(0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_only_clear_gains_are_promoted():
    current = Agreement(100, 0.40, 0.70, 0.6, 15.0, 0.2)
    assert is_better(Agreement(100, 0.45, 0.75, 0.5, 12.0, 0.1), current)
    assert not is_better(Agreement(100, 0.41, 0.71, 0.58, 14.5, 0.1), current)   # within noise
    assert not is_better(Agreement(100, 0.45, 0.60, 0.5, 12.0, 0.1), current)    # fewer close marks
    assert not is_better(Agreement(0, 0, 0, 0, 0, 0), current)


def test_weekly_trend():
    reviews = [
        {"created_at": "2026-09-01T09:00:00+00:00", "judged": 10, "changed": 4, "abs_diff": 3.0, "model": "qwen3.5:9b"},
        {"created_at": "2026-09-02T09:00:00+00:00", "judged": 10, "changed": 2, "abs_diff": 1.0, "model": "qwen3.5:9b"},
        {"created_at": "2026-09-09T09:00:00+00:00", "judged": 8, "changed": 1, "abs_diff": 0.5, "model": "gf:v1"},
    ]
    first, second = weekly_agreement(reviews)
    assert (first["week"], first["sheets"], first["accepted_pct"], first["avg_change"]) == ("2026-W36", 2, 70.0, 0.2)
    assert (second["accepted_pct"], second["models"]) == (87.5, ["gf:v1"])


def test_training_targets_teach_the_score_not_filler():
    assert target_for(correction(1, quality=0.4167)) == '{"quality": 0.42'
    assert target_for(correction(1, quality=0.5, note='Mentions "glucose" only')) == \
        '{"quality": 0.50, "feedback": "Mentions \\"glucose\\" only"'


def test_training_data_never_includes_held_out_answers():
    examples = build_examples(POOL[:300])
    ids = {e.correction_id for e in examples}
    assert ids and not ids & holdout_ids(POOL[:300])
    assert not ids & {c.id for c in POOL[:300] if split_of(c) == "validation"}
    validation = build_examples(POOL[:300], splits=("validation",))
    assert validation and all(split_of(POOL[e.correction_id]) == "validation" for e in validation)
    assert examples[0].messages[-1]["content"].endswith(f"<student_answer>\n{POOL[examples[0].correction_id].student_answer}\n</student_answer>")


def test_incremental_rounds_replay_older_answers():
    old = [correction(i, created="2026-09-01T10:00:00+00:00") for i in range(100)]
    new = [correction(i, created="2026-09-10T10:00:00+00:00") for i in range(100, 110)]
    examples = build_examples(old + new)
    first = select_round(examples, since=None)
    assert len(first) == len(examples)
    later = select_round(examples, since="2026-09-05T00:00:00+00:00")
    fresh = [e for e in later if e.created_at > "2026-09-05"]
    assert len(fresh) == len([e for e in examples if e.created_at > "2026-09-05"])
    assert len(later) - len(fresh) == min(len(examples) - len(fresh), 2 * len(fresh))
    assert select_round(examples, since="2026-12-01T00:00:00+00:00") == []   # nothing new: no round


def test_old_databases_gain_the_kind_column(tmp_path):
    path = tmp_path / "old.db"
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("""CREATE TABLE grade_corrections (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
            exam_id TEXT NOT NULL, sheet_id TEXT NOT NULL, question_id TEXT NOT NULL, subject TEXT NOT NULL,
            model TEXT NOT NULL, question_text TEXT NOT NULL, model_answer TEXT NOT NULL, student_answer TEXT NOT NULL,
            max_marks REAL NOT NULL, ai_marks REAL NOT NULL, teacher_marks REAL NOT NULL, ai_quality REAL,
            teacher_quality REAL, strictness REAL NOT NULL, note TEXT NOT NULL, teacher_name TEXT NOT NULL,
            UNIQUE (sheet_id, question_id))""")
        conn.execute("INSERT INTO grade_corrections VALUES (1,'t','e','s','2','B','m','q','a','x',4,3,1,0.7,0.2,50,'','T')")
    store = CorrectionStore(path)
    [old] = store.grade_corrections()
    assert old.kind == "changed" and old.teacher_marks == 1
    store.add_grade_correction(exam_id="e", sheet_id="s2", question_id="2", subject="B", model="m", question_text="q",
                               model_answer="a", student_answer="y", max_marks=4, ai_marks=3, teacher_marks=3,
                               ai_quality=0.7, strictness=50, kind="confirmed")
    assert [c.kind for c in store.grade_corrections()] == ["changed", "confirmed"]


def test_few_shot_and_calibration_can_leave_answers_out(tmp_path):
    from src.learning.correction_retriever import CorrectionRetriever
    from src.learning.score_calibrator import ScoreCalibrator
    from tests.test_learning import HashingEmbedder

    store = CorrectionStore(tmp_path / "c.db")
    for i in range(20):
        store.add_grade_correction(exam_id="e", sheet_id=f"s{i}", question_id="2", subject="Biology", model="m",
                                   question_text="What is photosynthesis?", model_answer="Plants make glucose.",
                                   student_answer="plants use sunlight to make glucose", max_marks=4, ai_marks=3,
                                   teacher_marks=2, ai_quality=0.8, strictness=50)
    ids = {c.id for c in store.grade_corrections()}
    retriever = CorrectionRetriever(store, embedder=HashingEmbedder())
    assert retriever.retrieve("What is photosynthesis?", "plants use sunlight to make glucose", k=4)
    assert retriever.retrieve("What is photosynthesis?", "plants use sunlight to make glucose", k=4,
                              exclude_ids=frozenset(ids)) == []
    assert ScoreCalibrator(store).calibrate("m", "Biology", 0.8) != 0.8
    assert ScoreCalibrator(store, exclude_ids=frozenset(ids)).calibrate("m", "Biology", 0.8) == 0.8
