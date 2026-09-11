"""The permanent asset: teacher corrections, stored as intent, never as model weights.

    grade_corrections   teacher-checked marks: kind "changed" (the teacher overrode the AI) or
                        "confirmed" (the teacher approved the sheet and kept the AI's mark).
                        Without confirmations every measurement would only see the AI's
                        mistakes, and calibration and training would learn only from them.
    ocr_corrections     the teacher fixed a line TrOCR misread (line image + corrected text)
    sheet_reviews       one row per approved sheet: how many AI marks the teacher changed
    benchmark_runs      agreement with the teacher measured on held-out answers, per model setup
    training_history    every fine-tune attempt, with before/after metrics, promoted or not

Every learning mechanism reads from here, so switching models never loses what was learned.

Scores are stored as quality (0..1) as well as marks. The strictness curve maps quality to
marks (marks = max * quality ** exponent), so the teacher's mark is converted back to the
quality it implies at the strictness in use. That keeps calibration valid at any strictness.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.grading.strictness_curve import strictness_exponent

_SCHEMA = """
CREATE TABLE IF NOT EXISTS grade_corrections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    exam_id         TEXT NOT NULL,
    sheet_id        TEXT NOT NULL,
    question_id     TEXT NOT NULL,
    subject         TEXT NOT NULL,
    model           TEXT NOT NULL,
    question_text   TEXT NOT NULL,
    model_answer    TEXT NOT NULL,
    student_answer  TEXT NOT NULL,
    max_marks       REAL NOT NULL,
    ai_marks        REAL NOT NULL,
    teacher_marks   REAL NOT NULL,
    ai_quality      REAL,
    teacher_quality REAL,
    strictness      REAL NOT NULL,
    note            TEXT NOT NULL,
    teacher_name    TEXT NOT NULL,
    kind            TEXT NOT NULL DEFAULT 'changed',
    UNIQUE (sheet_id, question_id)
);
CREATE TABLE IF NOT EXISTS sheet_reviews (
    sheet_id        TEXT PRIMARY KEY,
    exam_id         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    teacher_name    TEXT NOT NULL,
    model           TEXT NOT NULL,
    judged          INTEGER NOT NULL,
    changed         INTEGER NOT NULL,
    abs_diff        REAL NOT NULL,
    max_total       REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS benchmark_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    label           TEXT NOT NULL,
    model           TEXT NOT NULL,
    setup           TEXT NOT NULL,
    n               INTEGER NOT NULL,
    exact           REAL NOT NULL,
    close           REAL NOT NULL,
    mae             REAL NOT NULL,
    mae_pct         REAL NOT NULL,
    bias            REAL NOT NULL,
    holdout_total   INTEGER NOT NULL,
    trigger         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ocr_corrections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    exam_id         TEXT NOT NULL,
    sheet_id        TEXT NOT NULL,
    writer_id       TEXT NOT NULL,
    page            INTEGER NOT NULL,
    line_index      INTEGER NOT NULL,
    image_path      TEXT NOT NULL,
    ocr_text        TEXT NOT NULL,
    corrected_text  TEXT NOT NULL,
    UNIQUE (sheet_id, page, line_index)
);
CREATE TABLE IF NOT EXISTS training_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    mechanism       TEXT NOT NULL,
    model           TEXT NOT NULL,
    version         TEXT NOT NULL,
    metric_name     TEXT NOT NULL,
    metric_before   REAL,
    metric_after    REAL,
    num_samples     INTEGER NOT NULL,
    promoted        INTEGER NOT NULL,
    notes           TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def implied_quality(marks: float, max_marks: float, strictness: float) -> float:
    """Invert the strictness curve: the quality that earns `marks` at this strictness."""
    if max_marks <= 0:
        return 0.0
    ratio = min(max(marks / max_marks, 0.0), 1.0)
    return ratio ** (1 / strictness_exponent(strictness))


@dataclass(frozen=True)
class GradeCorrection:
    id: int
    created_at: str
    exam_id: str
    sheet_id: str
    question_id: str
    subject: str
    model: str
    question_text: str
    model_answer: str
    student_answer: str
    max_marks: float
    ai_marks: float
    teacher_marks: float
    ai_quality: float | None
    teacher_quality: float | None
    strictness: float
    note: str
    teacher_name: str
    kind: str = "changed"   # changed | confirmed


@dataclass(frozen=True)
class OCRCorrection:
    id: int
    created_at: str
    exam_id: str
    sheet_id: str
    writer_id: str
    page: int
    line_index: int
    image_path: str
    ocr_text: str
    corrected_text: str


class CorrectionStore:
    def __init__(self, db_path: Path = config.CORRECTIONS_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(grade_corrections)")}
            if columns and "kind" not in columns:  # databases from before confirmations existed
                conn.execute("ALTER TABLE grade_corrections ADD COLUMN kind TEXT NOT NULL DEFAULT 'changed'")
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    # --- grades ------------------------------------------------------------------------
    def add_grade_correction(self, *, exam_id: str, sheet_id: str, question_id: str, subject: str, model: str,
                             question_text: str, model_answer: str, student_answer: str, max_marks: float,
                             ai_marks: float, teacher_marks: float, ai_quality: float | None, strictness: float,
                             note: str = "", teacher_name: str = "", kind: str = "changed") -> None:
        """Re-checking the same question on the same sheet replaces the earlier entry."""
        teacher_quality = implied_quality(teacher_marks, max_marks, strictness) if ai_quality is not None else None
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO grade_corrections (created_at, exam_id, sheet_id, question_id, subject, model,"
                " question_text, model_answer, student_answer, max_marks, ai_marks, teacher_marks, ai_quality,"
                " teacher_quality, strictness, note, teacher_name, kind)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (_now(), exam_id, sheet_id, question_id, subject, model, question_text, model_answer,
                 student_answer, max_marks, ai_marks, teacher_marks, ai_quality, teacher_quality, strictness,
                 note, teacher_name, kind),
            )

    def grade_corrections(self, *, subject: str | None = None, model: str | None = None) -> list[GradeCorrection]:
        clauses, params = [], []
        if subject:
            clauses.append("subject = ?")
            params.append(subject)
        if model:
            clauses.append("model = ?")
            params.append(model)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        columns = ", ".join(GradeCorrection.__dataclass_fields__)
        with closing(self._connect()) as conn:
            rows = conn.execute(f"SELECT {columns} FROM grade_corrections {where} ORDER BY id", params).fetchall()
        return [GradeCorrection(*row) for row in rows]

    # --- sheet reviews (the day-by-day agreement measure) --------------------------------
    def add_sheet_review(self, *, sheet_id: str, exam_id: str, teacher_name: str, model: str, judged: int,
                         changed: int, abs_diff: float, max_total: float) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("INSERT OR REPLACE INTO sheet_reviews VALUES (?,?,?,?,?,?,?,?,?)",
                         (sheet_id, exam_id, _now(), teacher_name, model, judged, changed, abs_diff, max_total))

    def sheet_reviews(self) -> list[dict]:
        with closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute("SELECT * FROM sheet_reviews ORDER BY created_at")]

    # --- benchmark runs ------------------------------------------------------------------
    def add_benchmark_run(self, *, label: str, model: str, setup: str, n: int, exact: float, close: float,
                          mae: float, mae_pct: float, bias: float, holdout_total: int, trigger: str) -> int:
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                "INSERT INTO benchmark_runs (created_at, label, model, setup, n, exact, close, mae, mae_pct, bias,"
                " holdout_total, trigger) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (_now(), label, model, setup, n, exact, close, mae, mae_pct, bias, holdout_total, trigger))
            return cursor.lastrowid

    def benchmark_runs(self) -> list[dict]:
        with closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute("SELECT * FROM benchmark_runs ORDER BY id")]

    # --- OCR ---------------------------------------------------------------------------
    def add_ocr_correction(self, *, exam_id: str, sheet_id: str, writer_id: str, page: int, line_index: int,
                           image_path: str, ocr_text: str, corrected_text: str) -> None:
        with closing(self._connect()) as conn, conn:
            if corrected_text.strip() == ocr_text.strip():  # teacher reverted their fix
                conn.execute("DELETE FROM ocr_corrections WHERE sheet_id=? AND page=? AND line_index=?",
                             (sheet_id, page, line_index))
                return
            conn.execute(
                "INSERT OR REPLACE INTO ocr_corrections (created_at, exam_id, sheet_id, writer_id, page, line_index,"
                " image_path, ocr_text, corrected_text) VALUES (?,?,?,?,?,?,?,?,?)",
                (_now(), exam_id, sheet_id, writer_id, page, line_index, image_path, ocr_text, corrected_text),
            )

    def ocr_corrections(self, writer_id: str | None = None) -> list[OCRCorrection]:
        with closing(self._connect()) as conn:
            if writer_id:
                rows = conn.execute("SELECT * FROM ocr_corrections WHERE writer_id=? ORDER BY id", (writer_id,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM ocr_corrections ORDER BY id").fetchall()
        return [OCRCorrection(*row) for row in rows]

    # --- training history --------------------------------------------------------------
    def log_training(self, *, mechanism: str, model: str, version: str, metric_name: str,
                     metric_before: float | None, metric_after: float | None, num_samples: int,
                     promoted: bool, notes: str = "") -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO training_history (created_at, mechanism, model, version, metric_name, metric_before,"
                " metric_after, num_samples, promoted, notes) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (_now(), mechanism, model, version, metric_name, metric_before, metric_after, num_samples,
                 int(promoted), notes),
            )

    def training_history(self, mechanism: str | None = None) -> list[dict]:
        with closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row
            query = "SELECT * FROM training_history" + (" WHERE mechanism = ?" if mechanism else "") + " ORDER BY id"
            rows = conn.execute(query, (mechanism,) if mechanism else ()).fetchall()
        return [dict(r) | {"promoted": bool(r["promoted"])} for r in rows]
