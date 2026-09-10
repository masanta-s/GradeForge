"""Map OCR'd lines of a student's answer sheet to question numbers (plan: Known Gap #1).

Rule-based first pass:
- Prefixed markers ("Q3", "Ans 3", "Question 3(b)") are strong: always accepted, and OCR
  confusions in the number are repaired ("Ql" -> Q1, "QO5" -> Q05).
- Bare markers ("3.", "3)", "3(b)") are weak, because students number points inside answers.
  They are accepted only if the question exists, isn't already answered, and its number is
  higher than the current question's (points restart at 1, so they're rejected).
- Anything unresolved (missing questions, orphan text) sets `needs_llm`, so the LLM pass can
  re-segment the sheet.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace

from src.ocr.pipeline import OCRLine, PageOCR

_PREFIX = re.compile(r"^\s*(?:q(?:ue(?:s(?:tion)?)?)?|ans(?:wer)?)\s*(?:n[o0]\.?)?\s*[.:\-]?\s*", re.I)
_PREFIXED_NUM = re.compile(r"[0-9lIO|]{1,2}")
_BARE_NUM = re.compile(r"\s*[0-9]{1,2}")
_SUBPART = re.compile(r"\s*(?:\((?P<p>[a-h]|[ivx]{1,4})\)|(?P<l>[a-h])(?=[.)\s:]|$))", re.I)
_SEPARATOR = re.compile(r"\s*[.):\-]")
_OCR_DIGITS = str.maketrans({"l": "1", "I": "1", "|": "1", "O": "0"})
_ROMAN = re.compile(r"[ivx]{1,4}", re.I)


@dataclass(frozen=True)
class Marker:
    question_id: str   # "3", "3b", "3.ii"
    number: int
    strong: bool       # had a Q/Ans prefix
    rest: str          # text after the marker (the start of the answer)


@dataclass
class AnswerSegment:
    question_id: str
    lines: list[OCRLine] = field(default_factory=list)
    marker_confidence: float = 1.0
    diagrams: list = field(default_factory=list)  # list[DiagramRegion], in reading order

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines if line.text)


@dataclass
class SegmentationResult:
    answers: dict[str, AnswerSegment]
    unassigned: list[OCRLine]
    missing: list[str]
    unassigned_diagrams: list = field(default_factory=list)

    @property
    def needs_llm(self) -> bool:
        return bool(self.missing) or any(line.text.strip() for line in self.unassigned)


def parse_marker(text: str) -> Marker | None:
    prefix = _PREFIX.match(text)
    if prefix:
        num = _PREFIXED_NUM.match(text, prefix.end())
        strong = True
    else:
        num = _BARE_NUM.match(text)
        strong = False
    if not num:
        return None
    digits = num.group().strip().translate(_OCR_DIGITS)
    if not digits.isdigit() or int(digits) == 0:
        return None
    pos = num.end()

    sub = _SUBPART.match(text, pos)
    subpart = ""
    if sub:
        part = (sub["p"] or sub["l"]).lower()
        # Letters a-h never overlap roman numerals (i, v, x), so "(ii)" -> "3.ii", "(b)" -> "3b".
        subpart = f".{part}" if _ROMAN.fullmatch(part) else part
        pos = sub.end()

    sep = _SEPARATOR.match(text, pos)
    if sep:
        pos = sep.end()
    elif not strong and not sub:
        return None  # a bare number needs "3." / "3)" / "3(a)" — not "3 moles of ..."
    elif pos < len(text) and not text[pos].isspace():
        return None  # "Q12abc" — marker must end at a boundary

    number = int(digits)
    return Marker(question_id=f"{number}{subpart}", number=number, strong=strong, rest=text[pos:].strip())


def _resolve_id(marker: Marker, expected: set[str] | None) -> str | None:
    """The answer key id this marker refers to, or None if the paper has no such question."""
    if expected is None or marker.question_id in expected:
        return marker.question_id
    if str(marker.number) in expected:
        return str(marker.number)  # "3b" written, but the paper only has "3"
    if any(re.fullmatch(rf"{marker.number}\D.*", e) for e in expected):
        return marker.question_id  # "Q3" written, paper has "3a"/"3b" — sub-parts flagged missing
    return None


def _accepts(marker: Marker, qid: str, seen: set[str], current: int) -> bool:
    if marker.strong:
        return True
    return qid not in seen and marker.number > current


def segment_answers(pages: Iterable[PageOCR], expected_ids: Iterable[str] | None = None) -> SegmentationResult:
    expected = {q.lower() for q in expected_ids} if expected_ids is not None else None
    answers: dict[str, AnswerSegment] = {}
    unassigned: list[OCRLine] = []
    unassigned_diagrams: list = []
    current: AnswerSegment | None = None
    current_number = 0

    for page in pages:
        # Reading order by vertical position: a drawing belongs to the question it sits under.
        items = sorted([(line.bbox[1], 0, line) for line in page.lines] +
                       [(d.bbox[1], 1, d) for d in page.diagrams], key=lambda item: (item[0], item[1]))
        for _, is_diagram, item in items:
            if is_diagram:
                (current.diagrams if current else unassigned_diagrams).append(item)
                continue
            line = item
            marker = parse_marker(line.text)
            qid = _resolve_id(marker, expected) if marker else None
            if qid and _accepts(marker, qid, set(answers), current_number):
                current = answers.setdefault(qid, AnswerSegment(qid, marker_confidence=line.confidence))
                current_number = marker.number
                if marker.rest:
                    current.lines.append(replace(line, text=marker.rest))
                continue
            (current.lines if current else unassigned).append(line)

    missing = sorted(expected - set(answers), key=_sort_key) if expected is not None else []
    return SegmentationResult(answers=answers, unassigned=unassigned, missing=missing,
                              unassigned_diagrams=unassigned_diagrams)


def _sort_key(question_id: str) -> tuple[int, str]:
    digits = re.match(r"\d+", question_id)
    return (int(digits.group()) if digits else 0, question_id)
