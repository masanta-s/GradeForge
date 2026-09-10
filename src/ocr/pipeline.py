"""End-to-end OCR: file -> pages -> preprocessing -> line segmentation -> TrOCR.

PDFs with a real text layer (printed / digital question papers) are read directly with
PyMuPDF instead of TrOCR, which is a handwriting model. Scanned PDFs are rendered to images.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from src.ocr.preprocessing import extract_lines, preprocess_page

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_RENDER_DPI = 200
MIN_TEXT_LAYER_CHARS = 20
LOW_CONFIDENCE = 0.60


@dataclass(frozen=True)
class OCRLine:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]
    page: int = 0
    crop: np.ndarray | None = None  # kept so low-confidence lines can be re-read / corrected

    @property
    def needs_review(self) -> bool:
        return self.confidence < LOW_CONFIDENCE


@dataclass
class PageOCR:
    page_index: int
    source: Literal["handwriting", "pdf_text"]
    skew_angle: float = 0.0
    lines: list[OCRLine] = field(default_factory=list)
    diagrams: list = field(default_factory=list)  # list[DiagramRegion]

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def low_confidence_lines(self) -> list[OCRLine]:
        return [line for line in self.lines if line.needs_review]


def load_image(path: Path) -> np.ndarray:
    # np.fromfile + imdecode handles non-ASCII Windows paths, unlike cv2.imread.
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"not a readable image: {path}")
    return image


class OCRPipeline:
    def __init__(self, extractor=None, detect_diagrams: bool = True):
        self._extractor = extractor
        self.detect_diagrams = detect_diagrams

    @property
    def extractor(self):
        if self._extractor is None:  # lazy: loading TrOCR takes GPU memory and ~seconds
            from src.learning.ocr_fine_tuner import active_adapter
            from src.ocr.text_extractor import TrOCRExtractor

            # The promoted fine-tune (learning mechanism 3), if any; otherwise base TrOCR.
            self._extractor = TrOCRExtractor(adapter_dir=active_adapter())
        return self._extractor

    def run_image(self, image: np.ndarray, page_index: int = 0) -> PageOCR:
        from src.diagram.detector import detect_diagrams, mask_out

        page = preprocess_page(image)
        diagrams = detect_diagrams(page, page_index) if self.detect_diagrams else []
        # Drawings are blanked out so their strokes and labels aren't read as text lines.
        text_page = replace(page, ink=mask_out(page.ink, diagrams)) if diagrams else page
        regions = extract_lines(text_page)
        readings = self.extractor.read_lines([r.image for r in regions]) if regions else []
        return PageOCR(
            page_index=page_index,
            source="handwriting",
            skew_angle=page.skew_angle,
            lines=[
                OCRLine(text=rd.text, confidence=rd.confidence, bbox=rg.bbox, page=page_index, crop=rg.image)
                for rg, rd in zip(regions, readings)
            ],
            diagrams=diagrams,
        )

    def run_file(self, path: str | Path) -> list[PageOCR]:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._run_pdf(path)
        if suffix in IMAGE_SUFFIXES:
            return [self.run_image(load_image(path))]
        raise ValueError(f"unsupported file type: {path.suffix}")

    def _run_pdf(self, path: Path) -> list[PageOCR]:
        import pymupdf

        pages = []
        with pymupdf.open(path) as doc:
            for index, page in enumerate(doc):
                text_page = _pdf_text_page(page, index)
                if text_page is not None:
                    pages.append(text_page)
                    continue
                pix = page.get_pixmap(dpi=PDF_RENDER_DPI, colorspace=pymupdf.csRGB)
                rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                pages.append(self.run_image(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), page_index=index))
        return pages


def _pdf_text_page(page, index: int) -> PageOCR | None:
    """Use the PDF's own text layer when it has one (digital question papers)."""
    scale = PDF_RENDER_DPI / 72
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"]).strip()
            if not text:
                continue
            x0, y0, x1, y1 = (v * scale for v in line["bbox"])
            lines.append(OCRLine(text=text, confidence=1.0,
                                 bbox=(int(x0), int(y0), int(x1 - x0), int(y1 - y0)), page=index))
    if sum(len(line.text) for line in lines) < MIN_TEXT_LAYER_CHARS:
        return None
    return PageOCR(page_index=index, source="pdf_text", lines=lines)
