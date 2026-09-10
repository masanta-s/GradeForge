"""Integration: real TrOCR on the GPU. Skipped until setup_env.py has downloaded the model."""
import jiwer
import numpy as np
import pytest
import torch

from src import config
from src.ocr.pipeline import OCRPipeline
from tests.conftest import SAMPLE_LINES, make_page

pytestmark = [
    pytest.mark.skipif(not (config.TROCR_BASE_DIR / "model.safetensors").exists(), reason="run setup_env.py first"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]


@pytest.fixture(scope="module")
def extractor():
    from src.ocr.text_extractor import TrOCRExtractor

    ext = TrOCRExtractor()
    yield ext
    ext.unload()


def test_reads_synthetic_handwriting_page(extractor):
    page = OCRPipeline(extractor).run_image(make_page(ruled=True, skew=3.0, gradient=True, noise=6.0))
    predicted = [line.text for line in page.lines]
    cer = jiwer.cer(SAMPLE_LINES, predicted)
    print(f"\npage CER {cer:.1%}", *zip(predicted, (f"{l.confidence:.2f}" for l in page.lines)), sep="\n  ")
    assert len(predicted) == len(SAMPLE_LINES)
    assert cer < 0.10


def test_confidence_separates_text_from_noise(extractor):
    page = OCRPipeline(extractor).run_image(make_page(ruled=False))
    noise = np.random.default_rng(0).integers(0, 255, (60, 900), dtype=np.uint8)
    [garbage] = extractor.read_lines([noise])
    text_conf = min(line.confidence for line in page.lines)
    print(f"\nlowest text-line confidence {text_conf:.2f}, noise confidence {garbage.confidence:.2f}")
    assert text_conf > garbage.confidence


def test_batching_matches_single_reads(extractor):
    page = OCRPipeline(extractor).run_image(make_page(ruled=False))
    crops = [line.crop for line in page.lines]
    single = [extractor.read_lines([c])[0].text for c in crops]
    assert single == [line.text for line in page.lines]
