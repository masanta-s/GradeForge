"""Turn a question paper's text into structured questions.

Rule-based first (deterministic, testable), with an LLM fallback for layouts the rules can't
read. Handles "Q1." / "1)" / "Question 1" numbering, "(a)" / "(ii)" sub-parts, marks written
as "[5 marks]", "(2)", "5M", "- 3 marks", MCQ options one per line or several on one line,
and guesses each question's type from its wording.

"(a) ..." is ambiguous: an MCQ option or a sub-question. Options are short and come as a set
of three or more (or share one line); sub-questions are sentences and often carry marks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.grading.structured_output import CompletionFn, parse_json_response
from src.ocr.answer_segmenter import parse_marker
from src.ocr.pipeline import PageOCR

_MARKS = re.compile(
    r"\s*(?:[\[(]\s*(\d+(?:\.\d+)?)\s*(?:marks?|m)?\s*[\])]|[-–—]?\s*(\d+(?:\.\d+)?)\s*(?:marks?|m)\b\.?)\s*$", re.I)
_SUBPART = re.compile(r"^\s*\(?([a-h]|[ivx]{1,4})\)\s+(.+)$", re.I)
_OPTION_LINE = re.compile(r"^\s*\(?([A-Da-d])[).]\s+(.+)$")
_INLINE_OPTIONS = re.compile(r"\(?([A-Da-d])\)\s*(.+?)(?=\s+\(?[A-Da-d]\)\s|$)")
_JUSTIFY = re.compile(r"\b(justify|give (?:a )?reasons?|explain your (?:choice|answer)|with reason)\b", re.I)
_DIAGRAM = re.compile(r"\b(draw|diagram|sketch|label(?:led|ed)?)\b", re.I)
_OPTION_MAX_CHARS = 60


@dataclass
class ParsedQuestion:
    id: str
    text: str
    marks: float | None = None
    options: dict[str, str] = field(default_factory=dict)

    @property
    def qtype(self) -> str:
        if self.options:
            return "mixed" if _JUSTIFY.search(self.text) else "mcq"
        return "descriptive" if (self.marks or 0) >= 4 else "short"

    @property
    def wants_diagram(self) -> bool:
        return bool(_DIAGRAM.search(self.text))


def split_marks(text: str) -> tuple[str, float | None]:
    match = _MARKS.search(text)
    if not match:
        return text.strip(), None
    return text[:match.start()].strip(), float(match.group(1) or match.group(2))


def _inline_options(text: str) -> dict[str, str] | None:
    """'(A) Nucleus (B) Mitochondria (C) Ribosome' -> options, if at least two are present."""
    found = _INLINE_OPTIONS.findall(text)
    if len(found) >= 2:
        return {label.upper(): value.strip() for label, value in found}
    return None


def parse_question_paper(pages: list[PageOCR]) -> list[ParsedQuestion]:
    lines = [line.text.strip() for page in pages for line in page.lines if line.text.strip()]
    questions: list[ParsedQuestion] = []
    current: ParsedQuestion | None = None
    pending: list[tuple[str, str]] = []  # lettered items not yet known to be options or sub-parts

    def flush_pending() -> None:
        nonlocal current
        if not pending or current is None:
            pending.clear()
            return
        short = all(len(text) <= _OPTION_MAX_CHARS and split_marks(text)[1] is None for _, text in pending)
        if len(pending) >= 3 and short:
            current.options.update({label.upper(): text for label, text in pending})
        else:
            parent = current
            for label, text in pending:
                body, marks = split_marks(text)
                sub = label.lower() if not re.fullmatch(r"[ivx]{2,4}", label, re.I) else f".{label.lower()}"
                questions.append(ParsedQuestion(f"{parent.id}{sub}", body, marks))
            current = questions[-1]
        pending.clear()

    # Numbered instructions ("1. All questions are compulsory.") look like questions. If the
    # paper uses "Q1"/"Question 1" anywhere, only those start questions; with bare numbering,
    # a restart at "1" means everything before it was instructions.
    uses_prefix = any((m := parse_marker(t)) and m.strong for t in lines)

    for text in lines:
        marker = parse_marker(text)
        starts = marker and marker.question_id.isdigit() and (
            marker.strong or (not uses_prefix and marker.rest))
        if starts:
            flush_pending()
            if not uses_prefix and marker.number == 1 and questions:
                questions.clear()
            body, marks = split_marks(marker.rest)
            current = ParsedQuestion(marker.question_id, body, marks)
            if (opts := _inline_options(body)) is not None:
                current.text = body[:_INLINE_OPTIONS.search(body).start()].strip()
                current.options = opts
            questions.append(current)
            continue
        if current is None:
            continue  # title, instructions, "Max marks: 80" ...
        if (opts := _inline_options(text)) is not None:
            flush_pending()
            current.options.update(opts)
            continue
        item = _OPTION_LINE.match(text) or _SUBPART.match(text)
        if item:
            pending.append((item.group(1), item.group(2).strip()))
            continue
        flush_pending()
        body, marks = split_marks(text)
        current.text = f"{current.text} {body}".strip()
        if marks is not None and current.marks is None:
            current.marks = marks
    flush_pending()

    # A question split into sub-parts is a heading, not something students answer on its own.
    parents = {q.id for q in questions if any(o.id != q.id and o.id.startswith(q.id) and not o.id[len(q.id)].isdigit()
                                               for o in questions)}
    return [q for q in questions if q.id not in parents or q.options]


LLM_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"}, "text": {"type": "string"},
            "marks": {"type": ["number", "null"]},
            "options": {"type": "object", "additionalProperties": {"type": "string"}},
        },
        "required": ["id", "text"],
    }}},
    "required": ["questions"],
}

LLM_PROMPT = """Split this exam question paper into its individual questions.
Use ids like "1", "2", "3a", "3b" (sub-parts get a letter). Put MCQ options in "options" as
{"A": "...", "B": "..."}. Include marks if the paper states them, else null. Skip titles and
instructions. Reply with JSON only: {"questions": [...]}.

<question_paper>
%s
</question_paper>"""


def parse_with_llm(pages: list[PageOCR], llm: CompletionFn) -> list[ParsedQuestion]:
    """Fallback for layouts the rules miss. The paper text is data, not instructions."""
    text = "\n".join(line.text for page in pages for line in page.lines)
    messages = [{"role": "user", "content": LLM_PROMPT % text}]
    outcome = parse_json_response(llm(messages, schema=LLM_SCHEMA), {"questions"},
                                  complete=llm, messages=messages, schema=LLM_SCHEMA)
    if outcome.data is None:
        return []
    return [ParsedQuestion(str(q["id"]).strip().lower(), str(q["text"]).strip(),
                           float(q["marks"]) if q.get("marks") is not None else None,
                           {str(k).upper(): str(v) for k, v in (q.get("options") or {}).items()})
            for q in outcome.data["questions"] if str(q.get("text", "")).strip()]
