"""Synthetic answer-sheet generator: handwriting-style text on ruled paper, with known
line count, skew, lighting gradient and noise — ground truth for preprocessing tests."""
import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

HANDWRITING_FONT = r"C:\Windows\Fonts\segoepr.ttf"

SAMPLE_LINES = [
    "Photosynthesis happens in the chloroplast",
    "Light energy is converted to chemical energy",
    "Carbon dioxide and water form glucose",
    "Oxygen is released as a by product",
    "Chlorophyll absorbs red and blue light",
    "The Calvin cycle fixes carbon",
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(HANDWRITING_FONT, size)
    except OSError:
        return ImageFont.load_default(size=size)


def make_page(
    lines: list[str] = SAMPLE_LINES,
    ruled: bool = True,
    skew: float = 0.0,
    gradient: bool = False,
    noise: float = 0.0,
    line_spacing: int = 90,
    size: tuple[int, int] = (1700, 1100),
    seed: int = 0,
) -> np.ndarray:
    """Return a BGR page image. Positive `skew` rotates counter-clockwise (OpenCV convention)."""
    width, height = size
    page = Image.new("L", (width, height), 245)
    draw = ImageDraw.Draw(page)
    if ruled:
        for y in range(120, height - 40, line_spacing):
            draw.line([(40, y), (width - 40, y)], fill=170, width=2)
        draw.line([(110, 40), (110, height - 40)], fill=150, width=2)  # margin
    font = _font(46)
    for i, text in enumerate(lines):
        baseline = 120 + (i + 1) * line_spacing - 18
        draw.text((140, baseline), text, font=font, fill=25, anchor="ls")

    image = np.array(page, dtype=np.float32)
    if gradient:
        image *= np.linspace(0.55, 1.0, width, dtype=np.float32)[None, :]
    if noise:
        image += np.random.default_rng(seed).normal(0, noise, image.shape).astype(np.float32)
    image = np.clip(image, 0, 255).astype(np.uint8)
    if skew:
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), skew, 1.0)
        image = cv2.warpAffine(image, matrix, (width, height), borderValue=245)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


@pytest.fixture
def page_factory():
    return make_page
