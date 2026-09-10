"""MCQ grading from OCR'd student answers.

Accepts "B", "b)", "(B)", "Option B", "Ans: B", the option text instead of the letter
("Mitochondria"), and repairs common handwriting-OCR confusions (8 -> B). Several different
letters ("B, C") or unreadable input are marked wrong *and* flagged for teacher review.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from rapidfuzz import fuzz, process, utils

from src.grading.answer_key import Question

# Handwritten capitals TrOCR commonly misreads as digits.
_LETTER_CONFUSIONS = {"8": "B", "6": "G", "0": "D", "4": "A", "1": "I", "5": "S"}
_NOISE = re.compile(r"^\s*(?:ans(?:wer)?|option|opt|choice)\s*[.:\-]?\s*", re.I)
_LONE_LABEL = re.compile(r"\(?([A-Za-z0-9])\)?[.:]?")
_LEADING_LABEL = re.compile(r"\(?([A-Za-z0-9])[).:\-]\s*(.+)", re.S)
_CONJUNCTIONS = re.compile(r"\b(?:and|or)\b", re.I)
_LABEL_THEN_CONNECTOR = re.compile(
    r"\(?([A-Za-z0-9])\)?[.,:]?\s+((?:because|as|since|so|due|reason|justification)\b.*)", re.I | re.S)
_STATED_OPTION = re.compile(
    r"\b(?:answer|option|ans|choice)\s*(?:is|=|:|-)?\s*\(?([A-Ha-h])\)?(?![A-Za-z0-9])", re.I)
_LIST_ITEM = r"\(?[A-Za-z0-9]\)?\.?"
_LIST_SEP = r"(?:\s*(?:[,/&]|\band\b|\bor\b)\s*|\s+)"
_LABEL_LIST = re.compile(rf"{_LIST_ITEM}(?:{_LIST_SEP}{_LIST_ITEM})+", re.I)
# 90, not 85: at 85 "nucleolus" (a different organelle) matches "Nucleus" (score 88).
_TEXT_MATCH_THRESHOLD = 90
_MIN_TEXT_CHARS = 3


@dataclass(frozen=True)
class MCQResult:
    question_id: str
    marks: float
    correct: bool
    detected_option: str | None
    method: Literal["letter", "text_match", "unreadable", "ambiguous"]
    needs_review: bool


def _as_label(token: str, labels: set[str]) -> str | None:
    letter = token.upper()
    if letter not in labels:
        letter = _LETTER_CONFUSIONS.get(token, letter)
    return letter if letter in labels else None


def detect_option(answer: str, options: dict[str, str]) -> tuple[str | None, str]:
    """Return (option letter or None, method)."""
    labels = {k.upper() for k in options}
    cleaned = _NOISE.sub("", answer).strip()

    # The whole answer is one label: "B", "(b)", "b."
    lone = _LONE_LABEL.fullmatch(cleaned)
    if lone and (label := _as_label(lone[1], labels)):
        return label, "letter"

    # Only labels and separators: "B, C", "b or d", "(a) & (c)"
    if _LABEL_LIST.fullmatch(cleaned):
        found = {label for tok in re.findall(r"[A-Za-z0-9]", _CONJUNCTIONS.sub(" ", cleaned))
                 if (label := _as_label(tok, labels))}
        if len(found) > 1:
            return None, "ambiguous"
        if found:
            return found.pop(), "letter"

    # A label plus separator, then text: "B) Mitochondria". A bare leading word like
    # "a mitochondria" is NOT a label.
    leading = _LEADING_LABEL.fullmatch(cleaned)
    if leading and (label := _as_label(leading[1], labels)):
        return label, "letter"

    # No letter: maybe the student wrote the option text. Needs a few letters to compare —
    # WRatio's partial matching scores a lone "E" at 90 against "Nucleus".
    if len(re.sub(r"[^A-Za-z0-9]", "", cleaned)) >= _MIN_TEXT_CHARS:
        choices = {k.upper(): v for k, v in options.items() if v}
        match = process.extractOne(cleaned, choices, scorer=fuzz.WRatio, processor=utils.default_process)
        if match and match[1] >= _TEXT_MATCH_THRESHOLD:
            return match[2], "text_match"
    return None, "unreadable"


def split_mixed_answer(answer: str, options: dict[str, str]) -> tuple[str, str]:
    """Split a "choose and justify" answer into (option part, justification).

    Handles "B) Mitochondria, because ...", "B because ...", "B" then the explanation on the
    next lines, "Mitochondria" then the explanation, and "The answer is B as ...".
    """
    lines = [line for line in answer.strip().splitlines() if line.strip()]
    if not lines:
        return "", ""
    first, rest = _NOISE.sub("", lines[0]).strip(), lines[1:]
    labels = {k.upper() for k in options}

    def joined(*parts: str) -> str:
        return "\n".join(p.strip() for p in parts if p.strip())

    if _LONE_LABEL.fullmatch(first) and _as_label(_LONE_LABEL.fullmatch(first)[1], labels):
        return first, joined(*rest)
    if (m := _LABEL_THEN_CONNECTOR.match(first)) and _as_label(m[1], labels):
        return m[1], joined(m[2], *rest)
    if (m := _LEADING_LABEL.fullmatch(first)) and _as_label(m[1], labels):
        return m[1], joined(m[2], *rest)
    if rest and detect_option(first, options)[1] == "text_match":
        return first, joined(*rest)
    if m := _STATED_OPTION.search(answer):  # "the answer is B because ..."
        return m[1], answer.strip()
    return "", answer.strip()


def grade_mcq(question: Question, student_answer: str, marks: float | None = None) -> MCQResult:
    """Grade the option choice. `marks` overrides max_marks (mixed questions pass option_marks)."""
    marks = question.max_marks if marks is None else marks
    option, method = detect_option(student_answer, question.options)
    correct = option is not None and option == (question.correct_option or "").upper()
    return MCQResult(
        question_id=question.id,
        marks=marks if correct else 0.0,
        correct=correct,
        detected_option=option,
        method=method,
        needs_review=method in ("unreadable", "ambiguous") or method == "text_match",
    )
