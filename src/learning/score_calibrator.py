"""Mechanism 2: score calibration. Works with every model once there are 15 corrections.

Learns the systematic gap between a model's quality judgements and the teacher's (implied)
quality, separately per model and subject, with isotonic regression: monotonic (a better
answer never gets a lower calibrated score), non-parametric, and it fits in milliseconds on the CPU.

Example: if a model consistently rates Biology answers about 15% above what this teacher
awards, calibration pulls its scores down to match, before the strictness curve is applied.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression

from src.learning.correction_store import CorrectionStore

MIN_CORRECTIONS = 15


@dataclass(frozen=True)
class BiasReport:
    model: str
    subject: str
    corrections: int
    active: bool
    mean_bias: float | None  # model quality minus teacher quality, on average (+ = model too generous)

    @property
    def needed(self) -> int:
        return max(0, MIN_CORRECTIONS - self.corrections)


class ScoreCalibrator:
    def __init__(self, store: CorrectionStore, min_corrections: int = MIN_CORRECTIONS):
        self.store = store
        self.min_corrections = min_corrections
        self._fitted: dict[tuple[str, str], tuple[int, IsotonicRegression]] = {}

    def _pairs(self, model: str, subject: str) -> tuple[np.ndarray, np.ndarray]:
        rows = [c for c in self.store.grade_corrections(model=model, subject=subject)
                if c.ai_quality is not None and c.teacher_quality is not None]
        return (np.array([c.ai_quality for c in rows], float), np.array([c.teacher_quality for c in rows], float))

    def _model(self, model: str, subject: str) -> IsotonicRegression | None:
        ai, teacher = self._pairs(model, subject)
        if len(ai) < self.min_corrections:
            return None
        cached = self._fitted.get((model, subject))
        if cached and cached[0] == len(ai):  # refit only when new corrections arrive
            return cached[1]
        fitted = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(ai, teacher)
        self._fitted[(model, subject)] = (len(ai), fitted)
        return fitted

    def calibrate(self, model: str, subject: str, quality: float) -> float:
        fitted = self._model(model, subject)
        return float(fitted.predict([quality])[0]) if fitted is not None else quality

    def report(self, model: str, subject: str) -> BiasReport:
        ai, teacher = self._pairs(model, subject)
        return BiasReport(model, subject, len(ai), len(ai) >= self.min_corrections,
                          float(np.mean(ai - teacher)) if len(ai) else None)
