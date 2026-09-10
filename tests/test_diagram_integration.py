"""Integration: real TrOCR label reading + real qwen3.5:9b vision judging drawn diagrams."""
import pytest
import torch

from src import config
from src.diagram.detector import detect_diagrams
from src.diagram.evaluator import DiagramEvaluator, structural_similarity
from src.diagram.weightage import DiagramSpec
from src.ocr.preprocessing import preprocess_page
from tests.conftest import CELL_LABELS, make_diagram_page
from tests.test_grading_integration import _ollama_has_model

pytestmark = [
    pytest.mark.skipif(not (config.TROCR_BASE_DIR / "model.safetensors").exists(), reason="run setup_env.py"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not _ollama_has_model(), reason=f"Ollama with {config.DEFAULT_LLM} not available"),
]

SPEC = DiagramSpec(marks=4, required_labels=CELL_LABELS,
                   description="An animal cell with cell membrane, nucleus, mitochondria and cytoplasm.")
QUESTION = "Draw a neat labelled diagram of an animal cell."


def _regions(kind, labels=CELL_LABELS):
    image, _ = make_diagram_page(kind=kind, labels=labels)
    return detect_diagrams(preprocess_page(image))


@pytest.fixture(scope="module")
def scores():
    from src.knowledge.llm_client import LLMClient

    evaluator = DiagramEvaluator(vlm=LLMClient())
    drawings = {
        "complete": _regions("cell"),
        "one_label": _regions("cell", labels=["Nucleus", "", "", ""]),
        "wrong_diagram": _regions("flowchart"),
    }
    out = {name: evaluator.evaluate(QUESTION, SPEC, regions, strictness=50) for name, regions in drawings.items()}
    for name, s in out.items():
        print(f"\n{name:13} marks {s.marks}/{s.max_marks}  quality {s.quality:.2f}  components {s.components}\n"
              f"              labels read {s.labels_read}  matched {s.matched_labels}\n"
              f"              feedback: {s.feedback}  review: {s.review_reasons}")
    reference = drawings["complete"][0].ink
    print("\nSSIM vs complete cell:",
          {n: round(structural_similarity(d[0].ink, reference), 3) for n, d in drawings.items()})
    return out


def test_ranking(scores):
    assert scores["complete"].marks > scores["one_label"].marks > scores["wrong_diagram"].marks


def test_complete_drawing_scores_well(scores):
    assert scores["complete"].marks >= 3.0
    assert len(scores["complete"].matched_labels) >= 3


def test_wrong_diagram_scores_low(scores):
    assert scores["wrong_diagram"].marks <= 1.0
