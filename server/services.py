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
from src.models.cache import ModelCache
from src.models.secret_store import SecretStore
from server.jobs import JobRunner
from server.storage import Storage

DEFAULT_SETTINGS = {
    "strictness": 50, "model": config.DEFAULT_LLM, "teacher_name": "",
    # Cloud is opt-in: used only when provider == "cloud" AND the teacher consented.
    "provider": "ollama", "cloud_provider": "openai", "cloud_model": "", "cloud_consent": False,
}
_LLM_KEYS = {"model", "provider", "cloud_provider", "cloud_model", "cloud_consent"}


class Services:
    def __init__(self, storage: Storage | None = None, settings_path: Path | None = None,
                 model_cache: ModelCache | None = None, secrets: SecretStore | None = None):
        self.storage = storage or Storage()
        self.jobs = JobRunner()
        self.settings_path = settings_path or config.DATA_DIR / "settings.json"
        self.model_cache = model_cache or ModelCache()
        self.secrets = secrets or SecretStore()
        self._lock = threading.Lock()
        self._ocr = self._llm = self._subjective = self._diagram_evaluator = self._dispute_log = None
        self._corrections = self._retriever = self._calibrator = None
        self._models = self._cloud = None

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
        if _LLM_KEYS & changes.keys():
            self._llm = self._diagram_evaluator = None  # rebuilt with the new model on next use
        return settings

    @property
    def cloud_active(self) -> bool:
        s = self.settings
        return s["provider"] == "cloud" and bool(s["cloud_consent"]) and bool(s["cloud_model"])

    @property
    def model_name(self) -> str:
        """The grading model in use: an Ollama tag, or a cloud model name."""
        return self.settings["cloud_model"] if self.cloud_active else self.settings["model"]

    def vision_allowed(self) -> bool:
        """Send diagrams to the model only if it can really see them. A failed vision check
        (e.g. gemma4:e4b on Ollama, which reports vision but gets no image) switches diagram
        marking to labels and shape. Unchecked models are trusted: a meaningless reply is
        caught by the diagram evaluator and flagged for review."""
        if self.cloud_active:
            from src.models.cloud_provider import model_info

            info = model_info(self.settings["cloud_model"])
            return info["vision"] or not info["known"]
        from src.models.capability_prober import cached_report

        report = cached_report(self.settings["model"], cache=self.model_cache)
        return report is None or report.vision_ok

    # --- models (lazy) -----------------------------------------------------------------
    @property
    def models(self):
        if self._models is None:
            from src.models.model_manager import ModelManager

            self._models = ModelManager(cache=self.model_cache, secrets=self.secrets)
        return self._models

    @models.setter
    def models(self, value):
        self._models = value

    @property
    def cloud(self):
        if self._cloud is None:
            from src.models.cloud_provider import CloudProvider

            self._cloud = CloudProvider(self.secrets)
        return self._cloud

    @cloud.setter
    def cloud(self, value):
        self._cloud = value

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

            if self.cloud_active:
                from src.models.cloud_provider import litellm_model

                s = self.settings
                self._llm = LLMClient(litellm_model(s["cloud_provider"], s["cloud_model"]), provider="cloud",
                                      api_key=self.cloud.api_key(s["cloud_provider"]), cloud_consent=True)
            else:
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

            vlm = self.llm if self.vision_allowed() else None
            self._diagram_evaluator = DiagramEvaluator(extractor=self.ocr.extractor, vlm=vlm)
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

    def make_llm(self, model: str):
        """A client for any installed Ollama model (probing a model before switching to it)."""
        from src.knowledge.llm_client import LLMClient

        return LLMClient(model)

    def reset_llm(self) -> None:
        """After a capability check of the current model: its vision result may have changed."""
        self._llm = self._diagram_evaluator = None

    def checkpoint_gc(self):
        from src.models.checkpoint_gc import CheckpointGC

        return CheckpointGC(config.LLM_CHECKPOINTS_DIR, config.TROCR_FINETUNED_DIR, config.CACHE_DIR / "train",
                            remove_ollama=self.models.probe.remove)

    def grade_correction_count(self) -> int:
        return sum(c.teacher_quality is not None for c in self.corrections.grade_corrections())

    def reload_ocr(self) -> None:
        """After a TrOCR fine-tune is promoted: next use loads the new adapter."""
        extractor = getattr(self._ocr, "_extractor", None)
        if extractor is not None:
            extractor.unload()  # free its VRAM now rather than when garbage-collected
        self._ocr = self._diagram_evaluator = None
