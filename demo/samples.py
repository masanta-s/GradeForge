"""Sample exam for the demo: question paper, teacher's answer key, and three students' sheets.

Sheets are rendered in a handwriting-style font on ruled paper, tilted and unevenly lit like a
phone photo, with a hand-drawn cell diagram. The students are fictional.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.diagram.weightage import DiagramSpec
from src.grading.answer_key import AnswerKey, Question

HANDWRITING_FONT = r"C:\Windows\Fonts\segoepr.ttf"

SAMPLE_KEY = AnswerKey(
    exam_name="Class 10 Biology - Unit Test (demo)",
    subject="Biology",
    finalized=True,
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

PAPER_LINES = [
    "Class 10 Biology - Unit Test (demo)                        Max marks: 15",
    "Answer all questions.",
    "Q1. Which organelle is the powerhouse of the cell? [1]",
    "(A) Nucleus (B) Mitochondria (C) Ribosome (D) Golgi body",
    "Q2. What is photosynthesis? [4]",
    "Q3. Define osmosis. [2]",
    "Q4. Name the process by which plants lose water through leaves. [1]",
    "Q5. Which organelle releases energy from food? Choose the correct option and justify your answer. [3]",
    "(A) Ribosome (B) Mitochondria (C) Vacuole (D) Nucleus",
    "Q6. Draw a neat labelled diagram of an animal cell. [4]",
]

# name -> (lines written, labels on the cell drawing or None for no drawing)
STUDENTS: dict[str, tuple[list[str], list[str] | None]] = {
    "Riya Sharma": ([
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
    ], ["Nucleus", "Cell membrane", "Mitochondria"]),       # forgot "Cytoplasm"
    "Arjun Mehta": ([
        "Name: Arjun Mehta   Roll 7",
        "Q1. A",
        "Q2. Plants take food from the soil",
        "and water to grow.",
        "Q3. Water goes out of the cell.",
        "Q4. Evaporation",
        "Q5. C because the vacuole stores food.",
        "Q6. Animal cell:",
    ], ["Nucleus"]),
    "Meera Nair": ([
        "Name: Meera Nair   Roll 21",
        "Q1. B",
        "Q2. Green plants make glucose from carbon",
        "dioxide and water using sunlight and",
        "chlorophyll, and give out oxygen.",
        "Q3. Movement of water through a semi",
        "permeable membrane from a dilute to a",
        "concentrated solution.",
        "Q4. Transpiration",
        "Q5. B. Mitochondria do respiration and",
        "release energy as ATP.",
    ], None),                                                # did not attempt the diagram
}


def _font(size: int):
    try:
        return ImageFont.truetype(HANDWRITING_FONT, size)
    except OSError:
        return ImageFont.load_default(size=size)


def _draw_cell(draw: ImageDraw.ImageDraw, top: int, labels: list[str]) -> None:
    """A labelled animal cell like a student's sketch: membrane, nucleus, two mitochondria."""
    draw.ellipse((220, top, 900, top + 560), outline=25, width=6)
    draw.ellipse((480, top + 190, 640, top + 330), outline=25, width=5)
    draw.ellipse((300, top + 120, 400, top + 180), outline=25, width=4)
    draw.ellipse((700, top + 380, 810, top + 440), outline=25, width=4)
    anchors = {"Nucleus": (640, top + 260), "Cell membrane": (890, top + 300),
               "Mitochondria": (400, top + 150), "Cytoplasm": (560, top + 470)}
    for i, label in enumerate(labels):
        y = top + 70 + i * 150
        draw.line([anchors.get(label, (700, top + 280)), (1080, y)], fill=25, width=3)
        draw.text((1100, y + 14), label, font=_font(36), fill=30, anchor="ls")


def render_sheet(path: Path, lines: list[str], cell_labels: list[str] | None, skew: float = 2.5) -> Path:
    font = _font(44)
    width, spacing = 1700, 88
    text_height = 160 + spacing * (len(lines) + 1)
    page = Image.new("L", (width, text_height + (660 if cell_labels else 80)), 246)
    draw = ImageDraw.Draw(page)
    for y in range(120, page.height - 40, spacing):
        draw.line([(40, y), (width - 40, y)], fill=175, width=2)
    for i, text in enumerate(lines):
        draw.text((140, 120 + (i + 1) * spacing - 18), text, font=font, fill=30, anchor="ls")
    if cell_labels:
        _draw_cell(draw, text_height - 40, cell_labels)
    image = np.array(page, dtype=np.float32) * np.linspace(0.7, 1.0, width, dtype=np.float32)[None, :]
    h, w = image.shape
    image = cv2.warpAffine(np.clip(image, 0, 255).astype(np.uint8),
                           cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0), (w, h), borderValue=246)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", image)[1].tofile(path)
    return path


def render_paper_pdf(path: Path) -> Path:
    import pymupdf

    path.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open() as doc:
        page = doc.new_page()
        for i, line in enumerate(PAPER_LINES):
            page.insert_text((40, 60 + i * 24), line, fontsize=9)
        doc.save(path)
    return path
