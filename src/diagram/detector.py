"""Find hand-drawn diagrams on an answer sheet and separate them from text lines.

Key observation: a handwritten letter or cursive word is at most ~1.5x the text-line height,
while diagram strokes (a cell outline, a flowchart box, an arrow shaft) are far taller. So
ink components taller than 2x the typical line height seed a drawing; everything within about
a line height of those strokes (labels, leader lines, small parts) joins it.

Runs on `PreprocessedPage.drawing_ink`, which keeps a drawing's long straight edges (the text
mask removes them as if they were ruled lines). No network access (air-gapped package).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.ocr.preprocessing import PreprocessedPage, segment_lines

TALL_FACTOR = 2.0
VERY_TALL_FACTOR = 4.0
MAX_STROKE_FILL = 0.2


@dataclass(frozen=True)
class DiagramRegion:
    bbox: tuple[int, int, int, int]  # x, y, w, h in deskewed page coordinates
    image: np.ndarray                # grayscale crop from the unwhitened page
    ink: np.ndarray                  # drawing-ink mask crop
    strokes: np.ndarray              # mask crop of the drawing strokes only (no labels)
    confidence: float                # share of the region's ink that is drawing strokes
    page: int = 0
    line_height: float = 0.0         # the page's typical text-line height (label sizing)


def _is_drawing_stroke(stat: np.ndarray, typical: float) -> bool:
    """Tall AND sparse. Measured: outline drawings fill 4-7% of their bounding box, while text
    — even letters chained across touching lines — fills 27-36%. Very tall components count
    regardless (shaded drawings)."""
    w, h, area = stat[cv2.CC_STAT_WIDTH], stat[cv2.CC_STAT_HEIGHT], stat[cv2.CC_STAT_AREA]
    if h < TALL_FACTOR * typical or w < 0.5 * typical:
        return False
    return area / (w * h) < MAX_STROKE_FILL or h >= VERY_TALL_FACTOR * typical


def typical_line_height(mask: np.ndarray) -> float:
    lines = segment_lines(mask)
    if not lines:
        return max(12.0, mask.shape[0] / 40)
    return float(np.median([e - s for s, e in lines]))


def _merge_boxes(boxes: list[list[int]]) -> list[list[int]]:
    """Merge overlapping [x0, y0, x1, y1] boxes until none overlap."""
    boxes = [b[:] for b in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return boxes


def detect_diagrams(page: PreprocessedPage, page_index: int = 0) -> list[DiagramRegion]:
    ink = page.drawing_ink if page.drawing_ink is not None else page.ink
    original = page.original if page.original is not None else page.gray
    typical = typical_line_height(page.ink)

    # Removing a ruled line leaves a small gap wherever a stroke crossed it; re-close those so
    # an outline isn't cut into arcs shorter than the tall-stroke threshold.
    gap = max(5, int(0.2 * typical)) | 1
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, gap)))
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (gap, 3)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    tall = [i for i in range(1, count) if _is_drawing_stroke(stats[i], typical)]
    if not tall:
        return []

    strokes = np.isin(labels, tall)
    reach = max(3, int(1.2 * typical)) | 1
    grown = cv2.dilate(strokes.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach)))
    region_count, region_labels = cv2.connectedComponents(grown, connectivity=8)
    # Letters joined into words: a label whose first letters are near the drawing is included
    # whole, not cut off after the letters within reach.
    words = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(0.6 * typical)), 1)))
    _, word_labels = cv2.connectedComponents(words, connectivity=8)

    boxes = []
    for region in range(1, region_count):
        touching_words = np.unique(word_labels[(region_labels == region) & (words > 0)])
        in_words = np.isin(word_labels, touching_words[touching_words > 0]) & (closed > 0)
        members = np.unique(labels[in_words])
        members = members[members > 0]
        x0 = int(stats[members, cv2.CC_STAT_LEFT].min())
        y0 = int(stats[members, cv2.CC_STAT_TOP].min())
        x1 = int((stats[members, cv2.CC_STAT_LEFT] + stats[members, cv2.CC_STAT_WIDTH]).max())
        y1 = int((stats[members, cv2.CC_STAT_TOP] + stats[members, cv2.CC_STAT_HEIGHT]).max())
        boxes.append([x0, y0, x1, y1])

    regions = []
    h, w = ink.shape
    pad = int(0.3 * typical)
    for x0, y0, x1, y1 in _merge_boxes(boxes):
        x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
        region_ink = ink[y0:y1, x0:x1]
        region_strokes = (strokes[y0:y1, x0:x1] & (region_ink > 0)).astype(np.uint8) * 255
        total = max(1, int(np.count_nonzero(region_ink)))
        regions.append(DiagramRegion(
            bbox=(x0, y0, x1 - x0, y1 - y0),
            image=original[y0:y1, x0:x1],
            ink=region_ink,
            strokes=region_strokes,
            confidence=round(float(np.count_nonzero(region_strokes)) / total, 3),
            page=page_index,
            line_height=typical,
        ))
    return regions


def mask_out(mask: np.ndarray, regions: list[DiagramRegion]) -> np.ndarray:
    """Copy of `mask` with diagram regions blanked, so text-line segmentation ignores drawings."""
    out = mask.copy()
    for x, y, w, h in (r.bbox for r in regions):
        out[y:y + h, x:x + w] = 0
    return out
