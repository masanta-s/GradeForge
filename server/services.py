"""Long-lived services for the API: storage, jobs, settings and lazily-loaded models.

Models load on first use and are shared: the OCR pipeline and the diagram evaluator use the
same TrOCR instance, so it sits in VRAM once (see the measured VRAM budget in the plan).
Tests replace any of these attributes with fakes.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from src import config
from server.jobs import JobRunner
from server.storage import Storage

DEFAULT_SETTINGS = {"strictness": 50, "model": config.DEFAULT_LLM, "teacher_name": ""}


class Services:
    def __init__(self, storage: Storage | None = None, settings_path: Path | None = None):
        self.storage = storage or Storage()
        self.jobs = JobRunner()
        self.settings_path = settings_path or config.DATA_DIR / "settings.json"
        self._lock = threading.Lock()
        self._ocr = self._llm = self._subjective = self._diagram_evaluator = self._dispute_log = None
        self._corrections = self._retriever = self._calibrator = None

    # --- settings ----------------------------------------------------------------------
    @property
    def settings(self) -> dict:
        if self.settings_path.exists():
            return {**DEFAULT_SETTINGS, **json.loads(self.settings_path.read_text(encoding="utf-8"))}
        return dict(DEFAULT_SETTINGS)

    def update_settings(self, changes: dict) -> dict:
        settings = {**self.settings, **{k: v for k, v in changes.items() if k in DEFAULT_SETTINGS}}
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        if "model" in changes:
            self._llm = self._diagram_evaluator = None  # rebuilt with the new model on next use
        return settings

    # --- models (lazy) -----------------------------------------------------------------
    @property
    def ocr(self):
        with self._lock:
            if self._ocr is None:
                from src.ocr.pipeline import OCRPipeline

                self._ocr = OCRPipeline()
            return self._ocr

    @ocr.setter
    def ocr(self, value):
        self._ocr = value

    @property
    def llm(self):
        if self._llm is None:
            from src.knowledge.llm_client import LLMClient

            self._llm = LLMClient(self.settings["model"])
        return self._llm

    @llm.setter
    def llm(self, value):
        self._llm = value

    @property
    def subjective(self):
        if self._subjective is None:
            from src.grading.subjective_grader import SubjectiveGrader

            self._subjective = SubjectiveGrader()
        return self._subjective

    @subjective.setter
    def subjective(self, value):
        self._subjective = value

    @property
    def diagram_evaluator(self):
        if self._diagram_evaluator is None:
            from src.diagram.evaluator import DiagramEvaluator

            self._diagram_evaluator = DiagramEvaluator(extractor=self.ocr.extractor, vlm=self.llm)
        return self._diagram_evaluator

    @diagram_evaluator.setter
    def diagram_evaluator(self, value):
        self._diagram_evaluator = value

    @property
    def dispute_log(self):
        if self._dispute_log is None:
            from src.knowledge.dispute_logger import DisputeLog

            self._dispute_log = DisputeLog()
        return self._dispute_log

    @dispute_log.setter
    def dispute_log(self, value):
        self._dispute_log = value

    # --- learning ----------------------------------------------------------------------
    @property
    def corrections(self):
        if self._corrections is None:
            from src.learning.correction_store import CorrectionStore

            self._corrections = CorrectionStore()
        return self._corrections

    @corrections.setter
    def corrections(self, value):
        self._corrections = value
        self._retriever = self._calibrator = None

    @property
    def retriever(self):
        if self._retriever is None:
            from src.learning.correction_retriever import CorrectionRetriever

            # Reuse the grader's embedder: one copy of MiniLM in memory.
            self._retriever = CorrectionRetriever(self.corrections, embedder=self.subjective.embedder)
        return self._retriever

    @property
    def calibrator(self):
        if self._calibrator is None:
            from src.learning.score_calibrator import ScoreCalibrator

            self._calibrator = ScoreCalibrator(self.corrections)
        return self._calibrator

    def reload_ocr(self) -> None:
        """After a TrOCR fine-tune is promoted: next use loads the new adapter."""
        extractor = getattr(self._ocr, "_extractor", None)
        if extractor is not None:
            extractor.unload()  # free its VRAM now rather than when garbage-collected
        self._ocr = self._diagram_evaluator = None
