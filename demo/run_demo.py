"""End-to-end demo: answer sheet image -> OCR -> question mapping -> grading report.

    . .\\env.ps1
    python -m demo.run_demo                          # renders a sample handwritten-style sheet
    python -m demo.run_demo --sheet photo.jpg        # grade your own sheet against the sample key
    python -m demo.run_demo --strictness 80 --no-llm

Everything runs locally: TrOCR + MiniLM on the GPU, qwen3.5:9b through Ollama.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src import config
from src.diagram.weightage import DiagramSpec
from src.grading.answer_key import AnswerKey, Question

HANDWRITING_FONT = r"C:\Windows\Fonts\segoepr.ttf"

SAMPLE_KEY = AnswerKey(
    exam_name="Class 10 Biology - Unit Test (demo)",
    subject="Biology",
    questions=[
        Question(id="1", text="Which organelle is the powerhouse of the cell?", qtype="mcq", max_marks=1,
                 options={"A": "Nucleus", "B": "Mitochondria", "C": "Ribosome", "D": "Golgi body"},
                 correct_option="B"),
        Question(id="2", text="What is photosynthesis?", qtype="short", max_marks=4,
                 model_answer="Photosynthesis is the process by which green plants use sunlight, water and "
                              "carbon dioxide to make glucose and release oxygen, using chlorophyll.",
                 keywords=["sunlight", "carbon dioxide", "glucose", "oxygen", "chlorophyll"]),
        Question(id="3", text="Define osmosis.", qtype="short", max_marks=2,
                 model_answer="Osmosis is the movement of water molecules through a semi-permeable membrane "
                              "from a dilute solution to a concentrated solution.",
                 keywords=["water", "semi-permeable membrane", "concentrated"]),
        Question(id="4", text="Name the process by which plants lose water through leaves.", qtype="short",
                 max_marks=1, model_answer="Transpiration.", keywords=["transpiration"]),
        Question(id="5", text="Which organelle releases energy from food? Choose the correct option and "
                              "justify your answer.", qtype="mixed", max_marks=3, option_marks=1,
                 options={"A": "Ribosome", "B": "Mitochondria", "C": "Vacuole", "D": "Nucleus"},
                 correct_option="B",
                 model_answer="Mitochondria carry out aerobic respiration, which releases energy from "
                              "glucose in the form of ATP.",
                 keywords=["respiration", "glucose", "ATP"]),
        Question(id="6", text="Draw a neat labelled diagram of an animal cell.", qtype="short", max_marks=4,
                 diagram=DiagramSpec(marks=4, required_labels=["Nucleus", "Cell membrane", "Mitochondria",
                                                               "Cytoplasm"],
                                     description="An animal cell showing the cell membrane, nucleus, "
                                                 "mitochondria and cytoplasm.")),
    ],
)

SAMPLE_SHEET = [
    "Name: Riya Sharma   Roll 12",
    "Q1. B",
    "Q2. Plants use sunlight, water and carbon",
    "dioxide to make glucose. Oxygen is released.",
    "Chlorophyll absorbs the light.",
    "Q3. Water moves across a membrane.",
    "Q4. Respiration",
    "Q5. B because respiration in mitochondria",
    "breaks down glucose to release energy.",
    "Q6. Diagram of an animal cell:",
]
SAMPLE_CELL_LABELS = ["Nucleus", "Cell membrane", "Mitochondria"]  # student forgot "Cytoplasm"


def _font(size: int):
    try:
        return ImageFont.truetype(HANDWRITING_FONT, size)
    except OSError:
        return ImageFont.load_default(size=size)


def _draw_cell(draw: ImageDraw.ImageDraw, top: int) -> None:
    """A labelled animal cell like a student's sketch: membrane, nucleus, two mitochondria."""
    draw.ellipse((220, top, 900, top + 560), outline=25, width=6)
    draw.ellipse((480, top + 190, 640, top + 330), outline=25, width=5)
    draw.ellipse((300, top + 120, 400, top + 180), outline=25, width=4)
    draw.ellipse((700, top + 380, 810, top + 440), outline=25, width=4)
    anchors = [(640, top + 260), (890, top + 300), (400, top + 150)]
    for i, (label, anchor) in enumerate(zip(SAMPLE_CELL_LABELS, anchors)):
        y = top + 70 + i * 150
        draw.line([anchor, (1080, y)], fill=25, width=3)
        draw.text((1100, y + 14), label, font=_font(36), fill=30, anchor="ls")


def render_sample_sheet(path: Path) -> Path:
    font = _font(44)
    width, spacing, drawing_height = 1700, 88, 660
    text_height = 160 + spacing * (len(SAMPLE_SHEET) + 1)
    page = Image.new("L", (width, text_height + drawing_height), 246)
    draw = ImageDraw.Draw(page)
    for y in range(120, page.height - 40, spacing):
        draw.line([(40, y), (width - 40, y)], fill=175, width=2)
    for i, text in enumerate(SAMPLE_SHEET):
        draw.text((140, 120 + (i + 1) * spacing - 18), text, font=font, fill=30, anchor="ls")
    _draw_cell(draw, top=text_height - 40)
    image = np.array(page, dtype=np.float32) * np.linspace(0.7, 1.0, width, dtype=np.float32)[None, :]
    h, w = image.shape
    image = cv2.warpAffine(np.clip(image, 0, 255).astype(np.uint8),
                           cv2.getRotationMatrix2D((w / 2, h / 2), 2.5, 1.0), (w, h), borderValue=246)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", image)[1].tofile(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sheet", type=Path, help="answer sheet image or PDF (default: render a sample)")
    parser.add_argument("--strictness", type=float, default=50)
    parser.add_argument("--no-llm", action="store_true", help="grade with local embeddings only")
    args = parser.parse_args()

    from src.grading.grading_engine import GradingEngine
    from src.ocr.answer_segmenter import segment_answers
    from src.ocr.pipeline import OCRPipeline

    sheet = args.sheet or render_sample_sheet(config.DATA_DIR / "demo" / "sample_sheet.png")
    print(f"Sheet: {sheet}")

    start = time.time()
    pages = OCRPipeline().run_file(sheet)
    print(f"\n-- OCR ({time.time() - start:.1f}s) --")
    for page in pages:
        print(f"page {page.page_index} ({page.source}, deskewed {page.skew_angle:+.1f} deg)")
        for line in page.lines:
            flag = "  <- low confidence" if line.needs_review else ""
            print(f"  [{line.confidence:.2f}] {line.text}{flag}")
        for diagram in page.diagrams:
            print(f"  [diagram] at {diagram.bbox}, detection confidence {diagram.confidence:.2f}")

    segmentation = segment_answers(pages, SAMPLE_KEY.question_ids)
    llm = None
    if not args.no_llm:
        from src.knowledge.llm_client import LLMClient

        llm = LLMClient()
    engine = GradingEngine(SAMPLE_KEY, llm=llm, strictness=args.strictness)

    start = time.time()
    paper = engine.grade_segmentation(segmentation)
    print(f"\n-- Grading ({time.time() - start:.1f}s, strictness {args.strictness:g}, "
          f"{'LLM ' + config.DEFAULT_LLM if llm else 'embeddings only'}) --")
    for q in paper.questions:
        print(f"Q{q.question_id} [{q.qtype}] {q.marks:g}/{q.max_marks:g}  answer: {q.answer_text!r}")
        if q.diagram:
            print(f"     diagram {q.diagram.marks:g}/{q.diagram.max_marks:g}: labels {q.diagram.matched_labels}, "
                  f"missing {q.diagram.missing_labels}")
        if q.feedback:
            print(f"     feedback: {q.feedback}")
        for reason in q.review_reasons:
            print(f"     REVIEW: {reason}")
    if paper.unplaced_text:
        print(f"Not matched to any question: {paper.unplaced_text!r}")
    print(f"\nTOTAL {paper.total:g}/{paper.max_total:g} ({paper.percentage:.0f}%) - "
          f"{len(paper.review_queue)} question(s) flagged for teacher review")


if __name__ == "__main__":
    main()
