import numpy as np
import pytest

from src.ocr.preprocessing import (
    estimate_skew,
    extract_lines,
    ink_mask,
    preprocess_page,
    segment_lines,
    to_grayscale,
)
from tests.conftest import SAMPLE_LINES


def test_clean_page_finds_every_line(page_factory):
    page = preprocess_page(page_factory(ruled=False), deskew=False)
    assert len(segment_lines(page.ink)) == len(SAMPLE_LINES)


def test_ruled_lines_and_margin_are_removed(page_factory):
    page = preprocess_page(page_factory(ruled=True), deskew=False)
    assert len(segment_lines(page.ink)) == len(SAMPLE_LINES)
    # No row may be (nearly) all ink — that would be a surviving ruled line.
    assert (page.ink > 0).mean(axis=1).max() < 0.5


@pytest.mark.parametrize("skew", [-6.0, -2.5, 3.0, 7.0])
def test_skew_is_recovered(page_factory, skew):
    mask = ink_mask(to_grayscale(page_factory(ruled=False, skew=skew)))
    assert estimate_skew(mask) == pytest.approx(-skew, abs=0.5)


def test_skewed_photo_like_page_segments(page_factory):
    image = page_factory(ruled=True, skew=4.0, gradient=True, noise=8.0)
    page = preprocess_page(image)
    assert page.skew_angle == pytest.approx(-4.0, abs=0.5)
    assert len(segment_lines(page.ink)) == len(SAMPLE_LINES)


def test_page_edges_do_not_widen_line_crops(page_factory):
    # A skewed, unevenly lit photo has a hard diagonal edge where the page meets the border.
    page = preprocess_page(page_factory(ruled=True, skew=4.0, gradient=True, noise=8.0))
    regions = extract_lines(page)
    assert len(regions) == len(SAMPLE_LINES)
    assert all(r.bbox[0] > 60 for r in regions), [r.bbox for r in regions]


def test_short_lines_get_full_height_crops(page_factory):
    # A short line ("Q1. B") has little ink per row; a page-relative threshold used to keep only
    # its densest rows, cropping it to half height so neither TrOCR nor the VLM could read it.
    lines = ["Name: Riya Sharma Roll 12", "Q1. B", "Q2. Plants use sunlight, water and carbon", "Q4. Yes"]
    regions = extract_lines(preprocess_page(page_factory(lines=lines, ruled=True, skew=2.5)))
    assert len(regions) == len(lines)
    heights = [r.bbox[3] for r in regions]
    assert min(heights) >= 0.7 * max(heights), heights


def test_touching_lines_are_split(page_factory):
    # Tight spacing makes descenders/ascenders of neighbouring lines touch.
    page = preprocess_page(page_factory(ruled=False, line_spacing=58), deskew=False)
    assert len(segment_lines(page.ink)) == len(SAMPLE_LINES)


def test_extract_lines_crops_are_ordered_and_non_overlapping(page_factory):
    page = preprocess_page(page_factory(ruled=True, skew=2.0))
    regions = extract_lines(page)
    assert len(regions) == len(SAMPLE_LINES)
    for above, below in zip(regions, regions[1:]):
        assert above.bbox[1] + above.bbox[3] <= below.bbox[1]
    for region in regions:
        assert region.image.shape == (region.bbox[3], region.bbox[2])
        assert region.image.dtype == np.uint8


def test_blank_page_has_no_lines():
    blank = np.full((800, 600, 3), 240, np.uint8)
    page = preprocess_page(blank)
    assert segment_lines(page.ink) == []
    assert extract_lines(page) == []
