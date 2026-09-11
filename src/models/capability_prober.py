"""Capability probe: a ~30 s self-test before a model is trusted with grading.

1. JSON adherence (critical): three schema-constrained tasks, parsed exactly as grading
   parses them (fences stripped, repair turn allowed). Also records whether the model's
   output was valid JSON natively, without any repair.
2. Instruction following (critical): exact-format replies ("only the number", one per line).
3. Rubric reasoning (critical): the real grading prompt on a known 5-mark question with a
   full, a partial and a wrong (on-topic: it describes respiration) answer. Each score must
   be within 1.5 marks of the expected one.
4. Vision (optional): read a word from an image. Failing it only switches diagram marking to
   the label/shape path; OCR is unaffected (TrOCR is fixed). Ollama's "vision" capability flag
   is not proof: measured 2026-09-11, gemma4:e4b reports vision but answers "I cannot see the
   image" (also via /api/chat directly), while qwen3.5:9b reads the word.
5. Long context (optional): find one fact in ~5k tokens, within the 8K window used here.

Reports are cached per model and Ollama digest, so a re-pulled model is checked again.
"""
from __future__ import annotations

import base64
import json
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from src.grading.answer_key import Question
from src.grading.structured_output import parse_json_response
from src.grading.subjective_grader import LLM_SCHEMA, build_messages
from src.models.cache import ModelCache

PROBE_VERSION = 1
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)

Progress = Callable[[float, str], None]


@dataclass(frozen=True)
class ProbeResult:
    name: str
    label: str
    status: str        # pass | fail | skipped
    critical: bool
    detail: str
    seconds: float = 0.0


@dataclass(frozen=True)
class ProbeReport:
    model: str
    digest: str | None
    results: tuple[ProbeResult, ...]
    native_json: bool
    created_at: float = field(default_factory=time.time)
    version: int = PROBE_VERSION

    @property
    def critical_passed(self) -> bool:
        return all(r.status == "pass" for r in self.results if r.critical)

    @property
    def vision_ok(self) -> bool:
        return any(r.name == "vision" and r.status == "pass" for r in self.results)

    @property
    def failed_critical(self) -> list[ProbeResult]:
        return [r for r in self.results if r.critical and r.status != "pass"]

    def to_dict(self) -> dict:
        return asdict(self) | {"critical_passed": self.critical_passed, "vision_ok": self.vision_ok}

    @classmethod
    def from_dict(cls, data: dict) -> ProbeReport:
        return cls(data["model"], data.get("digest"), tuple(ProbeResult(**r) for r in data["results"]),
                   data["native_json"], data["created_at"], data.get("version", 0))


# --- test material ------------------------------------------------------------------------

JSON_TASKS = [
    ("Extract the details from this sentence: 'Riya scored 14 out of 20 in Biology.'",
     {"type": "object", "properties": {"name": {"type": "string"}, "score": {"type": "number"},
                                       "out_of": {"type": "number"}, "subject": {"type": "string"}},
      "required": ["name", "score", "out_of", "subject"]},
     lambda d: float(d["score"]) == 14 and float(d["out_of"]) == 20),
    ("Classify the tone of this teacher comment: 'The explanation was clear and well organised.'",
     {"type": "object", "properties": {"tone": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                                       "reason": {"type": "string"}},
      "required": ["tone", "reason"]},
     lambda d: str(d["tone"]).lower() == "positive"),
    ("List the three states of matter.",
     {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "string"}}},
      "required": ["items"]},
     lambda d: len(d["items"]) == 3),
]

INSTRUCTION_TASKS = [
    ("Reply with only the word BANANA in capital letters, and nothing else.",
     lambda r: r.strip().rstrip(".") == "BANANA"),
    ("What is 17 + 25? Reply with only the number.",
     lambda r: re.fullmatch(r"\s*42\.?\s*", r) is not None),
    ("Name exactly three primary colours of light, one per line, with no numbering, bullets or other text.",
     lambda r: (lines := [x.strip() for x in r.strip().splitlines() if x.strip()]) and len(lines) == 3
     and all(not re.match(r"^[\d\-*•.)]", x) and len(x.split()) <= 3 for x in lines)),
]

RUBRIC_QUESTION = Question(
    id="probe", text="Explain how photosynthesis produces food in plants.", qtype="short", max_marks=5,
    model_answer="Plants use chlorophyll in their chloroplasts to absorb light energy. This energy converts "
                 "carbon dioxide and water into glucose, and oxygen is released as a by-product.",
    key_points=["chlorophyll absorbs light", "happens in chloroplasts", "carbon dioxide and water are used",
                "glucose is produced", "oxygen is released"],
)
RUBRIC_CASES = [  # (name, answer, expected marks out of 5)
    ("full", "Photosynthesis takes place in the chloroplasts. Chlorophyll absorbs light energy, which is used "
             "to turn carbon dioxide and water into glucose. Oxygen is given off as a by-product.", 5.0),
    ("partial", "Plants make their food using sunlight. They take in carbon dioxide from the air.", 2.0),
    ("wrong", "Photosynthesis is when plants break down glucose using oxygen to release energy, "
              "producing carbon dioxide and water.", 0.0),
]
RUBRIC_TOLERANCE = 1.5

VISION_WORD = "MITOSIS"
NEEDLE = "The access code for the biology lab is 4172."


def _clean(reply: str) -> str:
    return _THINK.sub("", reply).strip()


def _vision_image() -> np.ndarray:
    image = np.full((220, 620, 3), 255, np.uint8)
    cv2.putText(image, VISION_WORD, (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 3.2, (20, 20, 20), 8, cv2.LINE_AA)
    cv2.rectangle(image, (15, 25), (605, 195), (60, 60, 60), 3)
    return image


def _haystack(sentences: int = 300) -> str:
    """~5.7k tokens at 300 sentences (measured with qwen3.5:9b), inside the 8K window."""
    rooms = ["library", "canteen", "gym", "art room", "office", "workshop", "music room", "garden"]
    items = ["projector", "fan", "window", "cupboard", "whiteboard", "heater", "clock", "shelf"]
    lines = [f"Note {i}: the {items[i % 8]} in the {rooms[(i * 3) % 8]} was checked on day {i % 28 + 1}."
             for i in range(sentences)]
    lines.insert(int(sentences * 0.6), NEEDLE)
    return " ".join(lines)


# --- probes ---------------------------------------------------------------------------------

PROBES = [  # (name, label, critical)
    ("json", "JSON adherence", True),
    ("instructions", "Instruction following", True),
    ("rubric", "Rubric reasoning", True),
    ("vision", "Vision", False),
    ("long_context", "Long context", False),
]


class CapabilityProber:
    def __init__(self, llm, cache: ModelCache | None = None):
        """`llm`: a `messages, schema=None -> str` callable (an LLMClient)."""
        self.llm = llm
        self.cache = cache or ModelCache()
        self._native = False

    def probe_json(self) -> ProbeResult:
        parsed, correct, native = 0, 0, 0
        for prompt, schema, check in JSON_TASKS:
            messages = [{"role": "system", "content": "Answer with a JSON object only."},
                        {"role": "user", "content": prompt}]
            raw = self.llm(messages, schema=schema)
            try:
                native += isinstance(json.loads(raw), dict)
            except json.JSONDecodeError:
                pass
            outcome = parse_json_response(raw, schema["required"], complete=self.llm, messages=messages,
                                          schema=schema)
            if outcome.data is not None and outcome.layer <= 3:
                parsed += 1
                try:
                    correct += bool(check(outcome.data))
                except (KeyError, TypeError, ValueError):
                    pass
        self._native = native == len(JSON_TASKS)
        ok =parsed == len(JSON_TASKS) and correct >= 2
        how = "natively" if native == len(JSON_TASKS) else f"{native} natively, the rest after clean-up/repair"
        return ProbeResult("json", "JSON adherence", "pass" if ok else "fail", True,
                           f"{parsed}/{len(JSON_TASKS)} valid JSON ({how}); {correct}/{len(JSON_TASKS)} correct")

    def probe_instructions(self) -> ProbeResult:
        passed = []
        for prompt, check in INSTRUCTION_TASKS:
            reply = _clean(self.llm([{"role": "user", "content": prompt}]))
            passed.append(bool(check(reply)))
        ok = sum(passed) >= 2
        return ProbeResult("instructions", "Instruction following", "pass" if ok else "fail", True,
                           f"{sum(passed)}/{len(passed)} exact-format replies")

    def probe_rubric(self) -> ProbeResult:
        scores, ok = [], True
        for name, answer, expected in RUBRIC_CASES:
            messages = build_messages(RUBRIC_QUESTION, answer)
            outcome = parse_json_response(self.llm(messages, schema=LLM_SCHEMA), {"quality", "feedback"},
                                          complete=self.llm, messages=messages, schema=LLM_SCHEMA)
            if outcome.data is None:
                scores.append(f"{name}: no score")
                ok = False
                continue
            marks = float(np.clip(float(outcome.data.get("quality", outcome.data.get("score", 0))), 0, 1)) * 5
            scores.append(f"{name} {marks:.1f}/5")
            ok &= abs(marks - expected) <= RUBRIC_TOLERANCE
        return ProbeResult("rubric", "Rubric reasoning", "pass" if ok else "fail", True,
                           ", ".join(scores) + f" (expected 5, 2, 0 ±{RUBRIC_TOLERANCE:g})")

    def probe_vision(self) -> ProbeResult:
        _, png = cv2.imencode(".png", _vision_image())
        url = "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()
        reply = _clean(self.llm([{"role": "user", "content": [
            {"type": "text", "text": "What single word is written in this image? Reply with only the word."},
            {"type": "image_url", "image_url": {"url": url}},
        ]}]))
        ok = VISION_WORD.lower() in reply.lower()
        detail = ("read the test word, so diagrams get a visual judgement" if ok
                  else f"read {reply[:40]!r}; diagrams will be marked on labels and shape only")
        return ProbeResult("vision", "Vision", "pass" if ok else "fail", False, detail)

    def probe_long_context(self) -> ProbeResult:
        text = _haystack()
        reply = _clean(self.llm([{"role": "user", "content": text + "\n\nWhat is the access code for the "
                                                             "biology lab? Reply with only the code."}]))
        ok = "4172" in reply
        tokens = len(text) / 3.6  # chars per token measured on this text
        return ProbeResult("long_context", "Long context", "pass" if ok else "fail", False,
                           f"{'found' if ok else 'missed'} one fact in ~{tokens / 1000:.0f}k tokens")

    def _run(self, name: str, label: str, critical: bool) -> ProbeResult:
        start = time.time()
        try:
            result = getattr(self, f"probe_{name}")()
        except Exception as e:  # a crash is a failed probe, reported rather than raised
            result = ProbeResult(name, label, "fail", critical, f"error: {type(e).__name__}: {e}"[:300])
        return ProbeResult(**{**asdict(result), "seconds": round(time.time() - start, 1)})

    def probe(self, model: str, digest: str | None = None, has_vision: bool | None = None,
              progress: Progress | None = None) -> ProbeReport:
        report = progress or (lambda p, m: None)
        self._native = False
        results = []
        for i, (name, label, critical) in enumerate(PROBES):
            report(i / len(PROBES), f"Check {i + 1} of {len(PROBES)}: {label}")
            if name == "vision" and has_vision is False:
                results.append(ProbeResult("vision", "Vision", "skipped", False,
                                           "the model has no vision; diagrams are marked on labels and shape only"))
                continue
            results.append(self._run(name, label, critical))
        result = ProbeReport(model, digest, tuple(results), self._native)
        self.cache.put("probe", model, result.to_dict())
        return result


def cached_report(model: str, digest: str | None = None, cache: ModelCache | None = None) -> ProbeReport | None:
    data = (cache or ModelCache()).get("probe", model)
    if not data or data.get("version") != PROBE_VERSION:
        return None
    if digest and data.get("digest") and data["digest"] != digest:
        return None  # the model was re-pulled or replaced: check it again
    return ProbeReport.from_dict(data)


# --- tiers ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelTier:
    level: str     # green | yellow | red | unknown
    label: str
    where: str | None
    reason: str


YELLOW_NOTE = "3 of 4 learning mechanisms active. Few-shot retrieval + score calibration provide continuous learning."


def assign_tier(report: ProbeReport | None, plan=None) -> ModelTier:
    """`plan`: the TrainingPlan, or None when the HF source hasn't been checked yet."""
    if report is None:
        return ModelTier("unknown", "Not checked", None, "Run the capability check before grading with this model.")
    if not report.critical_passed:
        failed = ", ".join(r.label.lower() for r in report.failed_critical)
        return ModelTier("red", "Incompatible", None, f"Failed {failed}: not safe to grade with.")
    if plan is None:
        return ModelTier("yellow", "Inference-only", None,
                         f"{YELLOW_NOTE} Weight fine-tuning not checked yet: find its HuggingFace source.")
    if plan.recommended:
        return ModelTier("green", "Trainable", plan.where, f"Checks passed; fine-tunable ({plan.where}).")
    return ModelTier("yellow", "Inference-only", None, f"{YELLOW_NOTE} {plan.blocked_reason}")
