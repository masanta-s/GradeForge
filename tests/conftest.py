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


CELL_LABELS = ["Nucleus", "Cell membrane", "Mitochondria", "Cytoplasm"]


def make_diagram_page(kind: str = "cell", ruled: bool = True, skew: float = 0.0,
                      labels: list[str] = CELL_LABELS) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """A page with text above and below a drawing. Returns (BGR image, drawing bbox x, y, w, h).

    kind="cell": ellipse membrane, nucleus, mitochondria, labels with leader lines.
    kind="flowchart": boxes with long straight edges joined by arrows.
    """
    width, height, spacing = 1700, 1500, 90
    page = Image.new("L", (width, height), 245)
    draw = ImageDraw.Draw(page)
    if ruled:
        for y in range(120, height - 40, spacing):
            draw.line([(40, y), (width - 40, y)], fill=170, width=2)
    font = _font(46)
    small = _font(36)
    draw.text((140, 192), "Q3. Draw a labelled diagram of an animal cell", font=font, fill=25, anchor="ls")
    draw.text((140, 282), "The cell is shown below.", font=font, fill=25, anchor="ls")

    top = 330
    if kind == "cell":
        box = (220, top, 900, top + 560)
        draw.ellipse(box, outline=20, width=6)                                 # membrane
        draw.ellipse((480, top + 190, 640, top + 330), outline=20, width=5)     # nucleus
        draw.ellipse((300, top + 120, 400, top + 180), outline=20, width=4)     # mitochondria
        draw.ellipse((700, top + 380, 810, top + 440), outline=20, width=4)
        anchors = [(640, top + 260), (890, top + 300), (400, top + 150), (560, top + 470)]
        for i, (label, anchor) in enumerate(zip(labels, anchors)):
            y = top + 70 + i * 130
            draw.line([anchor, (1080, y)], fill=20, width=3)
            draw.text((1100, y + 14), label, font=small, fill=25, anchor="ls")
        bbox = (220, top, 1100 + int(draw.textlength(max(labels, key=len), font=small)) - 220, 560)
    elif kind == "flowchart":
        boxes = [(300, top, 800, top + 110), (300, top + 220, 800, top + 330), (300, top + 440, 800, top + 550)]
        for b in boxes:
            draw.rectangle(b, outline=20, width=5)
        for (_, _, _, y1), (_, y0, _, _) in zip(boxes, boxes[1:]):
            draw.line([(550, y1), (550, y0)], fill=20, width=5)
            draw.polygon([(535, y0 - 20), (565, y0 - 20), (550, y0)], fill=20)
        for b, text in zip(boxes, ["Start", "Process", "End"]):
            draw.text((340, b[1] + 72), text, font=small, fill=25, anchor="ls")
        bbox = (300, top, 500, 550)
    else:
        raise ValueError(kind)

    below = top + 560 + 130
    draw.text((140, below), "Q4. The nucleus controls the cell.", font=font, fill=25, anchor="ls")
    image = np.array(page, dtype=np.uint8)
    if skew:
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), skew, 1.0)
        image = cv2.warpAffine(image, matrix, (width, height), borderValue=245)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), bbox


@pytest.fixture
def page_factory():
    return make_page


@pytest.fixture(scope="session", autouse=True)
def real_keyring_is_never_touched():
    """API keys live in Windows Credential Manager; tests get an in-memory keyring instead."""
    import keyring
    from keyring.backend import KeyringBackend
    from keyring.errors import PasswordDeleteError

    class MemoryKeyring(KeyringBackend):
        priority = 1

        def __init__(self):
            super().__init__()
            self.values = {}

        def get_password(self, service, username):
            return self.values.get((service, username))

        def set_password(self, service, username, password):
            self.values[(service, username)] = password

        def delete_password(self, service, username):
            if self.values.pop((service, username), None) is None:
                raise PasswordDeleteError(username)

    previous = keyring.get_keyring()
    keyring.set_keyring(MemoryKeyring())
    yield
    keyring.set_keyring(previous)


@pytest.fixture(scope="session", autouse=True)
def real_data_is_never_touched():
    """Fail the run if any test writes to the teacher's real databases or settings in data/."""
    from src import config

    watched = [*config.DATA_DIR.glob("*.db"), config.DATA_DIR / "settings.json"]
    before = {p: p.stat().st_mtime_ns for p in watched if p.exists()}
    yield
    changed = [p.name for p, mtime in before.items() if p.exists() and p.stat().st_mtime_ns != mtime]
    created = [p.name for p in config.DATA_DIR.glob("*.db") if p not in before]
    assert not changed and not created, f"tests modified real data: {changed + created}"
