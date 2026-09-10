"""SQLite audit trail of teacher-AI disagreements about the answer key (data/disputes.db).

Every decision is stored with the teacher's name, a UTC timestamp and the full conversation,
and can be searched by teacher, subject, date range or text, and exported to CSV.
"""
from __future__ import annotations

import csv
import json
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from src import config

Decision = Literal["accepted_ai", "save_correction", "one_time_override"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS disputes (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    teacher_name      TEXT NOT NULL,
    timestamp         TEXT NOT NULL,
    exam_name         TEXT NOT NULL,
    subject           TEXT NOT NULL,
    question_id       TEXT NOT NULL,
    question_text     TEXT NOT NULL,
    teacher_answer    TEXT NOT NULL,
    ai_answer         TEXT NOT NULL,
    ai_justification  TEXT NOT NULL,
    teacher_decision  TEXT NOT NULL,
    conversation      TEXT NOT NULL,
    use_for_training  INTEGER NOT NULL,
    resolved          INTEGER NOT NULL
)
"""
_COLUMNS = ("id", "teacher_name", "timestamp", "exam_name", "subject", "question_id", "question_text",
            "teacher_answer", "ai_answer", "ai_justification", "teacher_decision", "conversation",
            "use_for_training", "resolved")


@dataclass
class DisputeRecord:
    teacher_name: str
    exam_name: str
    subject: str
    question_id: str
    question_text: str
    teacher_answer: str
    ai_answer: str
    ai_justification: str
    teacher_decision: Decision
    conversation: list[dict] = field(default_factory=list)
    use_for_training: bool = False
    resolved: bool = True
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    id: int | None = None


class DisputeLog:
    def __init__(self, db_path: Path = config.DISPUTES_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def add(self, record: DisputeRecord) -> int:
        if not record.teacher_name.strip():
            raise ValueError("a teacher name is required for the audit trail")
        row = asdict(record)
        row["conversation"] = json.dumps(record.conversation, ensure_ascii=False)
        row.pop("id")
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                f"INSERT INTO disputes ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                [int(v) if isinstance(v, bool) else v for v in row.values()],
            )
            record.id = cursor.lastrowid
        return record.id

    @staticmethod
    def _to_record(row: tuple) -> DisputeRecord:
        data = dict(zip(_COLUMNS, row))
        data["conversation"] = json.loads(data["conversation"])
        data["use_for_training"] = bool(data["use_for_training"])
        data["resolved"] = bool(data["resolved"])
        return DisputeRecord(**data)

    def get(self, dispute_id: int) -> DisputeRecord:
        with closing(self._connect()) as conn:
            row = conn.execute(f"SELECT {', '.join(_COLUMNS)} FROM disputes WHERE id = ?", (dispute_id,)).fetchone()
        if row is None:
            raise KeyError(dispute_id)
        return self._to_record(row)

    def search(self, *, teacher: str | None = None, subject: str | None = None, since: str | None = None,
               until: str | None = None, text: str | None = None,
               training_only: bool = False) -> list[DisputeRecord]:
        clauses, params = [], []
        if teacher:
            clauses.append("teacher_name LIKE ?")
            params.append(f"%{teacher}%")
        if subject:
            clauses.append("subject = ?")
            params.append(subject)
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until:
            clauses.append("timestamp <= ?")
            params.append(until)
        if text:
            clauses.append("(question_text LIKE ? OR teacher_answer LIKE ? OR ai_answer LIKE ?)")
            params += [f"%{text}%"] * 3
        if training_only:
            clauses.append("use_for_training = 1")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with closing(self._connect()) as conn:
            rows = conn.execute(f"SELECT {', '.join(_COLUMNS)} FROM disputes {where} ORDER BY timestamp, id",
                                params).fetchall()
        return [self._to_record(r) for r in rows]

    def export_csv(self, path: Path, records: list[DisputeRecord] | None = None) -> Path:
        records = self.search() if records is None else records
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as f:  # BOM: Excel opens UTF-8 correctly
            writer = csv.DictWriter(f, fieldnames=list(_COLUMNS))
            writer.writeheader()
            for record in records:
                row = asdict(record)
                row["conversation"] = "\n".join(f"[{t['role']}] {t['content']}" for t in record.conversation)
                writer.writerow(row)
        return path
