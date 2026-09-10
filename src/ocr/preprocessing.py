"""Scan preprocessing and text-line segmentation for handwritten answer sheets.

TrOCR is a *line* recogniser (trained on single IAM lines), so a page must be split into
lines before recognition. Pipeline:

    grayscale -> illumination flattening (phone-photo shadows) -> deskew (projection-profile
    search) -> ink mask (adaptive threshold) -> ruled-line + page-edge removal -> line
    segmentation (horizontal projection, over-tall bands split at their weakest row) -> crops

Pure OpenCV/NumPy. No network access anywhere under src/ocr/.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LineRegion:
    bbox: tuple[int, int, int, int]  # x, y, w, h in deskewed page coordinates
    image: np.ndarray                # grayscale crop, dark ink on white


@dataclass(frozen=True)
class PreprocessedPage:
    gray: np.ndarray      # illumination-flattened, deskewed; ruled lines whitened (what TrOCR reads)
    ink: np.ndarray       # uint8 mask, 255 = ink; any long straight line removed (text segmentation)
    skew_angle: float     # degrees the page was rotated by to straighten it
    # Drawings keep their own long straight edges (box sides, graph axes), which the aggressive
    # rule removal above would erase: only page-spanning rules are removed here.
    original: np.ndarray | None = None      # deskewed grayscale, nothing whitened (diagram crops)
    drawing_ink: np.ndarray | None = None   # ink mask with only page-wide rules removed


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def flatten_illumination(gray: np.ndarray) -> np.ndarray:
    """Divide out the paper background so shadows and uneven phone lighting disappear."""
    kernel = max(15, (min(gray.shape) // 40) | 1)
    background = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), kernel)
    return cv2.divide(gray, background, scale=255)


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Adaptive threshold -> uint8 mask with ink = 255."""
    block = max(15, (min(gray.shape) // 50) | 1)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    mask = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 15
    )
    # Drop salt noise: specks smaller than a pen dot.
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def ruled_lines(mask: np.ndarray, min_fraction: float = 1 / 12, bridge: int = 0) -> np.ndarray:
    """Mask of horizontal/vertical straight lines at least `min_fraction` of the page long.

    1/12 catches broken or faint ruled lines (and also a drawing's long edges); 1/2 catches only
    page-spanning rules and margins. `bridge` first closes gaps of up to that many pixels along
    each direction, so a faint rule that thresholding broke into fragments still counts as one.
    """
    h, w = mask.shape
    along_x, along_y = mask, mask
    if bridge:
        along_x = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (bridge, 1)))
        along_y = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, bridge)))
    horizontal = cv2.morphologyEx(
        along_x, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, int(w * min_fraction)), 1))
    )
    vertical = cv2.morphologyEx(
        along_y, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, int(h * min_fraction))))
    )
    return cv2.dilate(cv2.bitwise_or(horizontal, vertical), np.ones((3, 3), np.uint8))


def extend_full_span(rules: np.ndarray, min_share: float = 0.25) -> np.ndarray:
    """Ruled lines span the whole page, so any row (column) where a detected rule covers at
    least `min_share` of the width (height) is ruled edge to edge. This also catches rule
    fragments that thresholding cut off next to thick drawing strokes."""
    h, w = rules.shape
    full = rules.copy()
    full[(rules > 0).sum(axis=1) >= min_share * w, :] = 255
    full[:, (rules > 0).sum(axis=0) >= min_share * h] = 255
    return full


def drop_page_edges(mask: np.ndarray) -> np.ndarray:
    """Remove long components touching the image border: paper edges, desk shadows, scan borders.

    Short border-touching components (a letter written at the very edge) are kept, but a thin
    margin is always cleared: OpenCV erosion treats out-of-image pixels as ink, so 1-px seams
    on the border survive the speck filter in `ink_mask`.
    """
    h, w = mask.shape
    cleaned = mask.copy()
    margin = max(3, min(h, w) // 200)
    cleaned[:margin] = 0
    cleaned[-margin:] = 0
    cleaned[:, :margin] = 0
    cleaned[:, -margin:] = 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    for label in range(1, count):
        x, y, cw, ch, _ = stats[label]
        touches_border = x <= margin or y <= margin or x + cw >= w - margin or y + ch >= h - margin
        if touches_border and (cw > w / 8 or ch > h / 8):
            cleaned[labels == label] = 0
    return cleaned


def _rotate(image: np.ndarray, angle: float, border: int) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border
    )


def estimate_skew(mask: np.ndarray, max_angle: float = 10.0) -> float:
    """Angle (degrees) that makes text rows horizontal: maximises horizontal-projection variance."""
    scale = min(1.0, 800 / mask.shape[1])
    small = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if not small.any():
        return 0.0

    def score(angle: float) -> float:
        return float(np.var(_rotate(small, angle, 0).sum(axis=1, dtype=np.float64)))

    coarse = max(np.arange(-max_angle, max_angle + 0.01, 0.5), key=score)
    fine = max(np.arange(coarse - 0.5, coarse + 0.51, 0.1), key=score)
    return round(float(fine), 1)


def preprocess_page(image: np.ndarray, deskew: bool = True) -> PreprocessedPage:
    gray = flatten_illumination(to_grayscale(image))
    # Deskew first: rule removal uses horizontal/vertical kernels, which miss tilted rules.
    # Ruled lines help the skew estimate, so it runs on the raw ink mask.
    angle = estimate_skew(ink_mask(gray)) if deskew else 0.0
    if angle:
        gray = _rotate(gray, angle, 255)  # flattened paper is ~255, so no artificial edge
    raw = ink_mask(gray)
    # 50% share: a broken ruled line is still mostly detected, but a diagram's box edge (well
    # under half the width) must not wipe its whole row.
    rules = extend_full_span(ruled_lines(raw), min_share=0.5)
    page_rules = extend_full_span(ruled_lines(raw, min_fraction=0.5, bridge=max(9, min(raw.shape) // 100)))
    # Remove rules from both the mask (segmentation) and the grayscale image (what TrOCR reads).
    mask = drop_page_edges(cv2.bitwise_and(raw, cv2.bitwise_not(rules)))
    drawing_ink = drop_page_edges(cv2.bitwise_and(raw, cv2.bitwise_not(page_rules)))
    whitened = np.where(rules > 0, 255, gray).astype(np.uint8)
    return PreprocessedPage(gray=whitened, ink=mask, skew_angle=angle, original=gray, drawing_ink=drawing_ink)


def _runs(active: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of consecutive True values."""
    padded = np.concatenate(([False], active, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _split_tall(band: tuple[int, int], profile: np.ndarray, typical: float) -> list[tuple[int, int]]:
    """Split a band holding several touching lines at its weakest interior row."""
    start, end = band
    if end - start < 1.7 * typical:
        return [band]
    margin = int(typical * 0.5)
    interior = profile[start + margin:end - margin]
    if interior.size == 0:
        return [band]
    cut = start + margin + int(np.argmin(interior))
    return _split_tall((start, cut), profile, typical) + _split_tall((cut, end), profile, typical)


def line_pitch(profile: np.ndarray, min_lag: int = 12) -> float | None:
    """Distance between consecutive text lines, from the autocorrelation of the row profile.

    Works even when every line touches its neighbours (one merged ink band), where band heights
    say nothing about line height.
    """
    x = profile - profile.mean()
    n = len(x)
    if n < 3 * min_lag or not x.any():
        return None
    spectrum = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
    ac = ac / ac[0]
    window = ac[min_lag:n // 2]
    if window.size < 3:
        return None
    peaks = [i for i in range(1, window.size - 1) if window[i] >= window[i - 1] and window[i] > window[i + 1]]
    if not peaks:
        return None
    best = max(window[i] for i in peaks)
    if best < 0.15:  # no periodic line structure
        return None
    first = next(i for i in peaks if window[i] >= 0.7 * best)  # the fundamental, not a harmonic
    return float(first + min_lag)


def segment_lines(mask: np.ndarray, min_height: int = 8) -> list[tuple[int, int]]:
    """Vertical [y0, y1) extents of text lines in an ink mask.

    Candidates are runs of rows with *any* ink (a low threshold), so a short line like
    "Q1. B" gets its full height. A page-relative threshold keeps only the densest rows of
    such a line, cropping it in half. Touching lines form one tall run, which is split at its
    weakest row.
    """
    profile = mask.sum(axis=1, dtype=np.float64) / 255
    if not profile.any():
        return []
    window = max(3, mask.shape[0] // 300)
    smooth = np.convolve(profile, np.ones(window) / window, mode="same")
    p90 = np.percentile(smooth[smooth > 0], 90)

    candidates = [b for b in _runs(smooth > max(1.0, 0.02 * p90)) if b[1] - b[0] >= 3]
    if not candidates:
        return []

    # Typical line height from the candidates, capped by the line pitch: when lines touch so
    # much that they merge into a few giant runs, the pitch still gives the real line height.
    typical = float(np.median([e - s for s, e in candidates]))
    pitch = line_pitch(smooth)
    if pitch is not None:
        typical = min(typical, pitch)

    # Merge candidates separated by tiny gaps (i-dots, accents, broken strokes).
    merged = [candidates[0]]
    for start, end in candidates[1:]:
        if start - merged[-1][1] < 0.25 * typical:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    lines = [piece for band in merged for piece in _split_tall(band, smooth, typical)]
    # Drop fragments far smaller than a real line (stray marks between lines).
    return [(s, e) for s, e in lines if e - s >= max(min_height, 0.3 * typical)]


def extract_lines(page: PreprocessedPage, pad_ratio: float = 0.25) -> list[LineRegion]:
    """Crop each text line (with ascender/descender padding) from the grayscale page."""
    h, w = page.ink.shape
    extents = segment_lines(page.ink)
    regions = []
    for i, (y0, y1) in enumerate(extents):
        pad = int((y1 - y0) * pad_ratio)
        top = max(0, y0 - pad, extents[i - 1][1] if i else 0)
        bottom = min(h, y1 + pad, extents[i + 1][0] if i + 1 < len(extents) else h)
        columns = np.flatnonzero(page.ink[y0:y1].any(axis=0))
        if columns.size == 0:
            continue
        x0 = max(0, columns[0] - pad)
        x1 = min(w, columns[-1] + 1 + pad)
        regions.append(LineRegion(bbox=(x0, top, x1 - x0, bottom - top), image=page.gray[top:bottom, x0:x1]))
    return regions
