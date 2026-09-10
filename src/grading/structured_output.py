"""Four-layer defence against malformed model output (plan: Structured-Output Robustness).

    1. Native constrained decoding — the caller passes a JSON schema (Ollama `format`).
    2. Tolerant parser — strip <think> blocks and ``` fences, brace-match the outermost object.
    3. Repair turn — send the broken output back, ask for JSON only (max 2 retries).
    4. Degraded fallback — regex-extract `SCORE:` / `FEEDBACK:`, mark reliable_json=False.

This package is network-free: the LLM arrives as an injected `CompletionFn`.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

CompletionFn = Callable[..., str]  # (messages, schema=None) -> str

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)
_SCORE = re.compile(r"(?:score|quality|marks?)\s*[:=]\s*(-?\d+(?:\.\d+)?)(?:\s*/\s*(\d+(?:\.\d+)?))?", re.I)
_FEEDBACK = re.compile(r"feedback\s*[:=]\s*(.+)", re.I | re.S)

REPAIR_PROMPT = (
    "Your previous response was not valid JSON. Return ONLY the JSON object, with no explanation, "
    "no markdown fences, and these keys: {keys}."
)


@dataclass(frozen=True)
class ParseOutcome:
    data: dict | None
    layer: int            # which layer produced `data` (1-4); 0 = total failure
    reliable_json: bool
    raw: str


def strip_noise(text: str) -> str:
    text = _THINK.sub("", text)
    fenced = _FENCE.search(text)
    return fenced.group(1) if fenced else text


def extract_json_object(text: str) -> str | None:
    """Outermost balanced {...}, respecting braces inside strings."""
    start = text.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def _try_parse(text: str, required: set[str]) -> tuple[dict | None, int]:
    try:
        data = json.loads(text)
        if isinstance(data, dict) and required <= data.keys():
            return data, 1
    except json.JSONDecodeError:
        pass
    candidate = extract_json_object(strip_noise(text))
    if candidate:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict) and required <= data.keys():
                return data, 2
        except json.JSONDecodeError:
            pass
    return None, 0


def regex_fallback(text: str) -> dict | None:
    text = _THINK.sub("", text)
    score = _SCORE.search(text)
    if not score:
        return None
    value = float(score.group(1))
    if score.group(2):  # "3.5/5" -> ratio
        value = value / float(score.group(2))
    feedback = _FEEDBACK.search(text)
    return {"score": value, "feedback": feedback.group(1).strip() if feedback else ""}


def parse_json_response(
    raw: str,
    required_keys: Iterable[str],
    *,
    complete: CompletionFn | None = None,
    messages: list[dict] | None = None,
    schema: dict | None = None,
    max_repairs: int = 2,
) -> ParseOutcome:
    required = set(required_keys)
    data, layer = _try_parse(raw, required)
    if data is not None:
        return ParseOutcome(data, layer, True, raw)

    attempts = [raw]
    if complete is not None and messages is not None:
        conversation = list(messages)
        for _ in range(max_repairs):
            # New list each turn: callers (e.g. audit logs) may keep the one they were given.
            conversation = conversation + [
                {"role": "assistant", "content": attempts[-1]},
                {"role": "user", "content": REPAIR_PROMPT.format(keys=", ".join(sorted(required)))},
            ]
            attempts.append(complete(conversation, schema=schema))
            data, _ = _try_parse(attempts[-1], required)
            if data is not None:
                return ParseOutcome(data, 3, True, attempts[-1])

    for attempt in attempts:  # the original reply may have a readable SCORE: line even if repairs didn't
        fallback = regex_fallback(attempt)
        if fallback is not None:
            return ParseOutcome(fallback, 4, False, attempt)
    return ParseOutcome(None, 0, False, attempts[-1])
