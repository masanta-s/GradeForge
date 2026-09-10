import pytest

from src.diagram.detector import detect_diagrams, mask_out
from src.ocr.preprocessing import preprocess_page, segment_lines
from tests.conftest import make_diagram_page, make_page


def iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter)


@pytest.mark.parametrize(("kind", "ruled", "skew"), [("cell", True, 0.0), ("cell", False, 0.0),
                                                     ("flowchart", True, 3.0), ("cell", True, -2.0)])
def test_finds_the_drawing_and_nothing_else(kind, ruled, skew):
    image, truth = make_diagram_page(kind=kind, ruled=ruled)
    if skew:  # re-render skewed; truth only checked on unskewed pages
        image, _ = make_diagram_page(kind=kind, ruled=ruled, skew=skew)
    regions = detect_diagrams(preprocess_page(image))
    assert len(regions) == 1
    region = regions[0]
    if not skew:
        assert iou(region.bbox, truth) > 0.8, (region.bbox, truth)
    assert region.confidence > 0.5
    assert region.image.shape == region.ink.shape == region.strokes.shape == (region.bbox[3], region.bbox[2])


def test_flowchart_keeps_long_straight_edges():
    # Box edges are ~500 px long: the text mask removes them as "ruled lines", the drawing mask must not.
    image, _ = make_diagram_page(kind="flowchart")
    page = preprocess_page(image)
    [region] = detect_diagrams(page)
    x, y, w, h = region.bbox
    assert (page.drawing_ink[y:y + h, x:x + w] > 0).sum(axis=1).max() > 400
    assert (page.ink[y:y + h, x:x + w] > 0).sum(axis=1).max() < 400


@pytest.mark.parametrize("spacing", [90, 58])
def test_plain_text_pages_have_no_diagrams(spacing):
    # spacing=58 makes descenders touch the next line's ascenders (~2-line-tall components).
    assert detect_diagrams(preprocess_page(make_page(ruled=True, line_spacing=spacing))) == []


def test_masking_diagrams_leaves_only_text_lines():
    image, _ = make_diagram_page(kind="cell")
    page = preprocess_page(image)
    regions = detect_diagrams(page)
    lines = segment_lines(mask_out(page.ink, regions))
    assert len(lines) == 3  # the two lines above the drawing and the Q4 line below
