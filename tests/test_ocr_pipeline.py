import cv2
import numpy as np
import pymupdf

from src.ocr.pipeline import LOW_CONFIDENCE, OCRPipeline
from src.ocr.text_extractor import LineReading
from tests.conftest import SAMPLE_LINES, make_page


class FakeExtractor:
    """Stands in for TrOCR: returns canned readings, records what it was asked to read."""

    def __init__(self, confidences=None):
        self.calls: list[int] = []
        self.confidences = confidences

    def read_lines(self, images):
        self.calls.append(len(images))
        confs = self.confidences or [0.95] * len(images)
        return [LineReading(f"line {i}", confs[i % len(confs)], confs[i % len(confs)]) for i in range(len(images))]


def test_run_image_reads_each_segmented_line():
    extractor = FakeExtractor()
    page = OCRPipeline(extractor).run_image(make_page(skew=3.0, gradient=True), page_index=2)
    assert page.source == "handwriting"
    assert extractor.calls == [len(SAMPLE_LINES)]
    assert [line.text for line in page.lines] == [f"line {i}" for i in range(len(SAMPLE_LINES))]
    assert all(line.page == 2 and line.crop is not None for line in page.lines)
    assert abs(page.skew_angle + 3.0) <= 0.5


def test_low_confidence_lines_are_flagged():
    extractor = FakeExtractor(confidences=[0.9, 0.3])
    page = OCRPipeline(extractor).run_image(make_page())
    flagged = page.low_confidence_lines
    assert flagged and all(line.confidence < LOW_CONFIDENCE for line in flagged)
    assert len(flagged) == len(SAMPLE_LINES) // 2


def test_image_file_with_unicode_path(tmp_path):
    path = tmp_path / "उत्तर_answer.png"  # non-ASCII names break cv2.imread on Windows
    ok, encoded = cv2.imencode(".png", make_page())
    encoded.tofile(path)
    pages = OCRPipeline(FakeExtractor()).run_file(path)
    assert len(pages) == 1 and len(pages[0].lines) == len(SAMPLE_LINES)


def test_digital_pdf_uses_text_layer_not_trocr(tmp_path):
    path = tmp_path / "question_paper.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 100), "Q1. Explain photosynthesis. [5 marks]", fontsize=12)
        page.insert_text((72, 130), "Q2. State Newton's first law. [3 marks]", fontsize=12)
        doc.save(path)

    extractor = FakeExtractor()
    pages = OCRPipeline(extractor).run_file(path)
    assert extractor.calls == []  # TrOCR never ran
    assert pages[0].source == "pdf_text"
    assert [line.text for line in pages[0].lines] == [
        "Q1. Explain photosynthesis. [5 marks]",
        "Q2. State Newton's first law. [3 marks]",
    ]
    assert all(line.confidence == 1.0 for line in pages[0].lines)


def test_scanned_pdf_is_rendered_and_ocrd(tmp_path):
    path = tmp_path / "scan.pdf"
    ok, png = cv2.imencode(".png", make_page())
    with pymupdf.open() as doc:
        page = doc.new_page(width=850, height=550)
        page.insert_image(page.rect, stream=png.tobytes())
        doc.save(path)

    extractor = FakeExtractor()
    pages = OCRPipeline(extractor).run_file(path)
    assert pages[0].source == "handwriting"
    assert len(pages[0].lines) == len(SAMPLE_LINES)


def test_unsupported_file_type(tmp_path):
    path = tmp_path / "notes.docx"
    path.write_bytes(b"x")
    try:
        OCRPipeline(FakeExtractor()).run_file(path)
    except ValueError as e:
        assert ".docx" in str(e)
    else:
        raise AssertionError("expected ValueError")
