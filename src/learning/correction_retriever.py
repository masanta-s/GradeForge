"""Mechanism 1: dynamic few-shot retrieval. Works with every model, from the first correction.

Each grade correction is embedded (question + student answer) with the sentence embedder.
At grading time the most similar past corrections go into the grading prompt as "how this
teacher graded similar answers", which is how any model, including cloud models that can
never be fine-tuned here, picks up the teacher's marking style.

Vectors are stored with the embedder's name. If the embedder changes (different vector space,
maybe different size), stale vectors are re-embedded, never silently compared.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing

import numpy as np

from src import config
from src.grading.subjective_grader import GradedExample
from src.learning.correction_store import CorrectionStore, GradeCorrection

_SCHEMA = """
CREATE TABLE IF NOT EXISTS correction_embeddings (
    correction_id INTEGER PRIMARY KEY,
    embedder      TEXT NOT NULL,
    vector        BLOB NOT NULL
)
"""


def _text(question_text: str, answer: str) -> str:
    return f"Question: {question_text}\nAnswer: {answer}"


class CorrectionRetriever:
    def __init__(self, store: CorrectionStore, embedder=None, embedder_name: str = config.DEFAULT_EMBEDDER):
        self.store = store
        self._embedder = embedder
        self.embedder_name = embedder_name
        with closing(sqlite3.connect(store.db_path)) as conn, conn:
            conn.execute(_SCHEMA)

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(str(config.EMBEDDERS_DIR / self.embedder_name))
        return self._embedder

    def sync(self) -> int:
        """Embed corrections that have no vector from the current embedder. Returns how many."""
        corrections = {c.id: c for c in self.store.grade_corrections()}
        with closing(sqlite3.connect(self.store.db_path)) as conn, conn:
            current = {cid for (cid,) in conn.execute(
                "SELECT correction_id FROM correction_embeddings WHERE embedder = ?", (self.embedder_name,))}
            missing = [c for cid, c in corrections.items() if cid not in current]
            if missing:
                vectors = self.embedder.encode([_text(c.question_text, c.student_answer) for c in missing],
                                               normalize_embeddings=True)
                conn.executemany(
                    "INSERT OR REPLACE INTO correction_embeddings VALUES (?, ?, ?)",
                    [(c.id, self.embedder_name, np.asarray(v, np.float32).tobytes()) for c, v in zip(missing, vectors)],
                )
        return len(missing)

    def retrieve(self, question_text: str, answer: str, *, subject: str | None = None, k: int = 4,
                 min_similarity: float = 0.5, exclude_sheet: str | None = None,
                 _retried: bool = False) -> list[GradedExample]:
        corrections = [c for c in self.store.grade_corrections(subject=subject) if c.sheet_id != exclude_sheet]
        if not corrections or not answer.strip():
            return []
        self.sync()
        with closing(sqlite3.connect(self.store.db_path)) as conn:
            stored = dict(conn.execute("SELECT correction_id, vector FROM correction_embeddings WHERE embedder = ?",
                                       (self.embedder_name,)).fetchall())
        pool = [c for c in corrections if c.id in stored]
        if not pool:
            return []
        query = np.asarray(self.embedder.encode([_text(question_text, answer)], normalize_embeddings=True)[0],
                           np.float32)
        vectors = [np.frombuffer(stored[c.id], np.float32) for c in pool]
        if any(v.shape != query.shape for v in vectors):
            # Same name, different vector size: the model files were swapped. Re-embed once; if
            # sizes still disagree the embedder itself is inconsistent, so retrieve nothing.
            if _retried:
                return []
            self._forget_vectors()
            return self.retrieve(question_text, answer, subject=subject, k=k, min_similarity=min_similarity,
                                 exclude_sheet=exclude_sheet, _retried=True)
        scores = np.stack(vectors) @ query
        best = [i for i in np.argsort(-scores)[:k] if scores[i] >= min_similarity]
        return [self._example(pool[i]) for i in best]

    def _forget_vectors(self) -> None:
        with closing(sqlite3.connect(self.store.db_path)) as conn, conn:
            conn.execute("DELETE FROM correction_embeddings WHERE embedder = ?", (self.embedder_name,))

    @staticmethod
    def _example(c: GradeCorrection) -> GradedExample:
        return GradedExample(question=c.question_text, answer=c.student_answer, marks=c.teacher_marks,
                             max_marks=c.max_marks, note=c.note)
