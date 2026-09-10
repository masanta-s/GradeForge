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

from demo.samples import SAMPLE_KEY, STUDENTS, render_sheet
from src import config


def render_sample_sheet(path: Path) -> Path:
    lines, labels = STUDENTS["Riya Sharma"]
    return render_sheet(path, lines, labels)


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
