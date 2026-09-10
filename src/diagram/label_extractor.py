"""Read the text labels written inside or beside a drawing.

Labels are the region's ink that is NOT drawing strokes. Letters are grouped into label
blocks by horizontal dilation (about half a line height, the usual gap between letters and
words), size-filtered, cropped from the unwhitened image and read with TrOCR. No network access.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.diagram.detector import DiagramRegion


@dataclass(frozen=True)
class DiagramLabel:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]  # within the diagram crop


def label_boxes(region: DiagramRegion) -> list[tuple[int, int, int, int]]:
    line_h = region.line_height or max(12.0, region.ink.shape[0] / 10)
    labels = cv2.bitwise_and(region.ink, cv2.bitwise_not(region.strokes))
    labels = cv2.morphologyEx(labels, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    joined = cv2.dilate(labels, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(0.6 * line_h)), 3)))
    count, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)

    boxes = []
    h, w = labels.shape
    for i in range(1, count):
        x, y, bw, bh, _ = stats[i]
        if not (0.3 * line_h <= bh <= 2.0 * line_h and bw >= 0.4 * line_h):
            continue  # specks, or leftover stroke fragments
        pad = int(0.2 * line_h)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        boxes.append((x0, y0, min(w, x + bw + pad) - x0, min(h, y + bh + pad) - y0))
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def extract_labels(region: DiagramRegion, extractor) -> list[DiagramLabel]:
    boxes = label_boxes(region)
    if not boxes:
        return []
    crops = [region.image[y:y + h, x:x + w] for x, y, w, h in boxes]
    readings = extractor.read_lines(crops)
    # Small drawn parts (organelle ovals, arrowheads) aren't tall enough to count as strokes and
    # get read as "0", "o," — a real label has at least two letters.
    return [DiagramLabel(r.text, r.confidence, box) for r, box in zip(readings, boxes)
            if sum(c.isalpha() for c in r.text) >= 2]
