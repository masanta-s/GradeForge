"""Strictness slider: maps an answer-quality score (0..1) to a marks ratio (0..1).

    marks = quality ** exponent,  exponent = 0.3 * (2.5 / 0.3) ** (strictness / 100)

strictness 0 -> exponent 0.30 (lenient), 50 -> ~0.87, 100 -> 2.50 (strict). The curve is
continuous and monotonic in both arguments: better answers never score lower, and a stricter
setting never awards more.

Raw embedding cosine similarity is a poor quality score on its own — an on-topic but wrong
answer (or one that just restates the question) still scores ~0.5 — so `rescale_similarity`
maps the range [floor, 1] onto [0, 1] first (plan: Known Gap #5).
"""
from __future__ import annotations

MIN_EXPONENT = 0.3
MAX_EXPONENT = 2.5


def strictness_exponent(strictness: float) -> float:
    if not 0 <= strictness <= 100:
        raise ValueError(f"strictness must be in [0, 100], got {strictness}")
    return MIN_EXPONENT * (MAX_EXPONENT / MIN_EXPONENT) ** (strictness / 100)


def apply_strictness(quality: float, strictness: float) -> float:
    """Marks ratio for an answer of the given quality (0..1) at the given strictness (0..100)."""
    quality = min(max(quality, 0.0), 1.0)
    return quality ** strictness_exponent(strictness)


def rescale_similarity(similarity: float, floor: float, ceiling: float = 1.0) -> float:
    """Map cosine similarity from [floor, ceiling] onto [0, 1], clamping outside the range.

    `floor` is the similarity a worthless answer reaches for this question — e.g. the
    similarity between the question text itself and the model answer.
    """
    if not floor < ceiling:
        raise ValueError(f"floor ({floor}) must be below ceiling ({ceiling})")
    return min(max((similarity - floor) / (ceiling - floor), 0.0), 1.0)


def strictness_label(strictness: float) -> str:
    for upper, label in ((12.5, "Very lenient"), (37.5, "Lenient"), (62.5, "Medium"), (87.5, "Strict")):
        if strictness < upper:
            return label
    return "Very strict"
