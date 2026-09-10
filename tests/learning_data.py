"""Synthetic 'student with unusual handwriting' for OCR fine-tuning tests.

A clean handwriting font is read almost perfectly by base TrOCR (measured: 0.4-2.1 % CER on nine
Windows script fonts), so it can't show whether fine-tuning helps. This writer has consistent
quirks, as a real student does: a heavy right slant, squashed letters, a wobbly baseline, thin
faint strokes and noise. Fine-tuning must learn to read this one person.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = r"C:\Windows\Fonts\Inkfree.ttf"
WORDS = ("cell membrane nucleus energy glucose oxygen water light plant leaf root stem enzyme protein "
         "carbon dioxide release store chloroplast mitochondria respiration diffusion osmosis tissue organ "
         "blood heart lungs food chain acid base salt heat force motion speed").split()


def sentences(n: int, seed: int = 0, words: tuple[int, int] = (3, 5)) -> list[str]:
    rng = np.random.default_rng(seed)
    return [" ".join(rng.choice(WORDS, rng.integers(*words, endpoint=True))).capitalize() for _ in range(n)]


def messy_line(text: str, seed: int = 0, strength: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    try:
        font = ImageFont.truetype(FONT, 48)
    except OSError:
        font = ImageFont.load_default(size=48)
    left, top, right, bottom = font.getbbox(text)
    img = Image.new("L", (right - left + 80, bottom - top + 40), 255)
    ImageDraw.Draw(img).text((40 - left, 20 - top), text, font=font, fill=0)
    ink = 255 - np.array(img, np.float32)
    h, w = ink.shape

    # squashed letters + heavy slant (this writer's consistent style)
    squash = 1 - 0.35 * strength
    shear = 0.55 * strength
    matrix = np.float32([[1, -shear, shear * h * squash], [0, squash, (1 - squash) * h / 2]])
    ink = cv2.warpAffine(ink, matrix, (w + int(shear * h), h))
    # wobbly baseline: a slow vertical sine wave along the line
    h, w = ink.shape
    xs = np.arange(w, dtype=np.float32)
    offset = (6 * strength * np.sin(xs / (40 + 10 * rng.random()) + rng.random() * 6)).astype(np.float32)
    map_x = np.tile(xs, (h, 1))
    map_y = (np.arange(h, dtype=np.float32)[:, None] + offset[None, :]).astype(np.float32)
    ink = cv2.remap(ink, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
    # thin, faint pen and some paper noise
    ink = cv2.erode(ink, np.ones((2, 2), np.uint8)) * (0.55 + 0.1 * rng.random())
    page = 250 - ink + rng.normal(0, 10 * strength, ink.shape)
    return np.clip(page, 0, 255).astype(np.uint8)
