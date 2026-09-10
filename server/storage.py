"""On-disk layout for exams, answer keys, student sheets and results (all under data/, on F:).

    data/exams/<exam_id>/
        exam.json                 name, subject, parsed questions
        paper/<file>              the uploaded question paper
        answer_key.json           AnswerKey
        validations.json          AI checks of the teacher's key, by question id
        sheets/<sheet_id>/
            source/<file>         the uploaded answer sheet
            ocr.json              lines (TrOCR text + teacher-corrected text) and drawings
            lines/*.png           line crops (for review and OCR fine-tuning)
            diagrams/*.png        drawing image / ink / strokes (to re-evaluate without re-OCR)
            result.json           the graded paper
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src import config
from src.diagram.detector import DiagramRegion
from src.grading.answer_key import AnswerKey
from src.ocr.pipeline import OCRLine, PageOCR


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "item"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Background jobs write JSON while request threads read it. On Windows, replacing a file that
# another thread has open fails with PermissionError (Linux allows it), so file access is
# serialised within the process. The retry covers antivirus / search indexers briefly holding
# a freshly written file open.
_IO_LOCK = threading.RLock()


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, ensure_ascii=False)
    with _IO_LOCK:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        for attempt in range(20):
            try:
                tmp.replace(path)  # atomic: a crash never leaves half a file
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05)


def _read_json(path: Path):
    with _IO_LOCK:
        return json.loads(path.read_text(encoding="utf-8"))


def _save_png(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", image)[1].tofile(path)
    return path.name


def _load_png(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


class Storage:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else config.DATA_DIR / "exams"
        self.root.mkdir(parents=True, exist_ok=True)

    # --- exams -------------------------------------------------------------------------
    def exam_dir(self, exam_id: str) -> Path:
        path = (self.root / exam_id).resolve()
        if self.root.resolve() not in path.parents or not path.is_dir():
            raise KeyError(exam_id)
        return path

    def create_exam(self, name: str, subject: str) -> dict:
        exam_id = f"{_slug(name)}-{uuid.uuid4().hex[:6]}"
        exam = {"id": exam_id, "name": name, "subject": subject, "created_at": _now(),
                "paper_file": None, "questions": []}
        _write_json(self.root / exam_id / "exam.json", exam)
        return exam

    def list_exams(self) -> list[dict]:
        exams = [_read_json(p) for p in self.root.glob("*/exam.json")]
        return sorted(exams, key=lambda e: e["created_at"], reverse=True)

    def get_exam(self, exam_id: str) -> dict:
        return _read_json(self.exam_dir(exam_id) / "exam.json")

    def update_exam(self, exam: dict) -> None:
        _write_json(self.exam_dir(exam["id"]) / "exam.json", exam)

    def save_paper(self, exam_id: str, filename: str, content: bytes) -> Path:
        folder = self.exam_dir(exam_id) / "paper"
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        path = folder / f"paper{Path(filename).suffix.lower()}"
        path.write_bytes(content)
        return path

    # --- answer key --------------------------------------------------------------------
    def save_key(self, exam_id: str, key: AnswerKey) -> None:
        key.save(self.exam_dir(exam_id) / "answer_key.json")

    def get_key(self, exam_id: str) -> AnswerKey | None:
        path = self.exam_dir(exam_id) / "answer_key.json"
        return AnswerKey.load(path, validate=False) if path.exists() else None  # drafts may be incomplete

    def save_validations(self, exam_id: str, validations: dict) -> None:
        _write_json(self.exam_dir(exam_id) / "validations.json", validations)

    def get_validations(self, exam_id: str) -> dict:
        path = self.exam_dir(exam_id) / "validations.json"
        return _read_json(path) if path.exists() else {}

    # --- sheets ------------------------------------------------------------------------
    def sheet_dir(self, exam_id: str, sheet_id: str) -> Path:
        path = (self.exam_dir(exam_id) / "sheets" / sheet_id).resolve()
        if not path.is_dir():
            raise KeyError(sheet_id)
        return path

    def create_sheet(self, exam_id: str, student: str, filename: str, content: bytes) -> tuple[str, Path]:
        sheet_id = f"{_slug(student)}-{uuid.uuid4().hex[:6]}"
        folder = self.exam_dir(exam_id) / "sheets" / sheet_id / "source"
        folder.mkdir(parents=True)
        path = folder / f"sheet{Path(filename).suffix.lower()}"
        path.write_bytes(content)
        _write_json(folder.parent / "ocr.json", {"student": student, "status": "uploaded", "pages": []})
        return sheet_id, path

    def list_sheets(self, exam_id: str) -> list[dict]:
        sheets = []
        for ocr in sorted((self.exam_dir(exam_id) / "sheets").glob("*/ocr.json")):
            meta = _read_json(ocr)
            result = ocr.parent / "result.json"
            summary = _read_json(result) if result.exists() else None
            sheets.append({"id": ocr.parent.name, "student": meta["student"], "status": meta["status"],
                           "total": summary["total"] if summary else None,
                           "max_total": summary["max_total"] if summary else None,
                           "review_count": len([q for q in summary["questions"] if q["review_reasons"]])
                           if summary else None})
        return sheets

    def source_file(self, exam_id: str, sheet_id: str) -> Path:
        return next((self.sheet_dir(exam_id, sheet_id) / "source").iterdir())

    def save_ocr(self, exam_id: str, sheet_id: str, pages: list[PageOCR]) -> None:
        folder = self.sheet_dir(exam_id, sheet_id)
        meta = _read_json(folder / "ocr.json")
        stored_pages = []
        for page in pages:
            lines = []
            for i, line in enumerate(page.lines):
                crop = _save_png(folder / "lines" / f"p{page.page_index}_l{i}.png", line.crop) \
                    if line.crop is not None else None
                lines.append({"text": line.text, "ocr_text": line.text, "confidence": line.confidence,
                              "bbox": list(map(int, line.bbox)), "page": line.page, "crop": crop})
            diagrams = []
            for i, d in enumerate(page.diagrams):
                stem = f"p{page.page_index}_d{i}"
                for part in ("image", "ink", "strokes"):
                    _save_png(folder / "diagrams" / f"{stem}_{part}.png", getattr(d, part))
                diagrams.append({"stem": stem, "bbox": list(map(int, d.bbox)), "confidence": d.confidence,
                                 "page": d.page, "line_height": d.line_height})
            stored_pages.append({"page_index": page.page_index, "source": page.source,
                                 "skew_angle": page.skew_angle, "lines": lines, "diagrams": diagrams})
        meta.update(pages=stored_pages, status="read")
        _write_json(folder / "ocr.json", meta)

    def get_ocr(self, exam_id: str, sheet_id: str) -> dict:
        return _read_json(self.sheet_dir(exam_id, sheet_id) / "ocr.json")

    def update_ocr(self, exam_id: str, sheet_id: str, meta: dict) -> None:
        _write_json(self.sheet_dir(exam_id, sheet_id) / "ocr.json", meta)

    def load_pages(self, exam_id: str, sheet_id: str) -> list[PageOCR]:
        """Rebuild PageOCR objects (with teacher-corrected text) for grading."""
        folder = self.sheet_dir(exam_id, sheet_id)
        pages = []
        for p in self.get_ocr(exam_id, sheet_id)["pages"]:
            lines = [OCRLine(text=ln["text"], confidence=ln["confidence"], bbox=tuple(ln["bbox"]),
                             page=ln["page"]) for ln in p["lines"]]
            diagrams = [DiagramRegion(
                bbox=tuple(d["bbox"]), confidence=d["confidence"], page=d["page"], line_height=d["line_height"],
                **{part: _load_png(folder / "diagrams" / f"{d['stem']}_{part}.png")
                   for part in ("image", "ink", "strokes")},
            ) for d in p["diagrams"]]
            pages.append(PageOCR(p["page_index"], p["source"], p["skew_angle"], lines, diagrams))
        return pages

    def save_result(self, exam_id: str, sheet_id: str, result: dict) -> None:
        _write_json(self.sheet_dir(exam_id, sheet_id) / "result.json", result)
        meta = self.get_ocr(exam_id, sheet_id)
        meta["status"] = "graded"
        self.update_ocr(exam_id, sheet_id, meta)

    def get_result(self, exam_id: str, sheet_id: str) -> dict | None:
        path = self.sheet_dir(exam_id, sheet_id) / "result.json"
        return _read_json(path) if path.exists() else None

    def asset(self, exam_id: str, sheet_id: str, kind: str, name: str) -> Path:
        if kind not in ("lines", "diagrams") or not re.fullmatch(r"[\w.-]+\.png", name):
            raise KeyError(name)
        path = self.sheet_dir(exam_id, sheet_id) / kind / name
        if not path.exists():
            raise KeyError(name)
        return path
