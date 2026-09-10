import json

import cv2
import numpy as np
import pytest

from src.diagram.detector import detect_diagrams
from src.diagram.evaluator import DiagramEvaluator, build_vlm_messages, structural_similarity
from src.diagram.weightage import DiagramSpec
from src.ocr.preprocessing import preprocess_page
from src.ocr.text_extractor import LineReading
from tests.conftest import CELL_LABELS, make_diagram_page

SPEC = DiagramSpec(marks=4, required_labels=CELL_LABELS,
                   description="An animal cell: membrane, nucleus, mitochondria, cytoplasm.")


class FakeLabelReader:
    """Stands in for TrOCR on label crops."""

    def __init__(self, texts):
        self.texts = texts

    def read_lines(self, images):
        return [LineReading(t, 0.9, 0.9) for t in (self.texts + ["xx"] * len(images))[:len(images)]]


class ScriptedVLM:
    def __init__(self, reply):
        self.reply = reply
        self.messages = None

    def __call__(self, messages, schema=None):
        self.messages = messages
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply)


@pytest.fixture(scope="module")
def cell_regions():
    image, _ = make_diagram_page(kind="cell")
    return detect_diagrams(preprocess_page(image))


def test_weightage_rescales_over_measured_components():
    assert SPEC.normalized_weights({"label", "structure", "completeness"}) == pytest.approx(
        {"label": 0.40, "structure": 0.35, "completeness": 0.25})
    assert SPEC.normalized_weights({"label", "completeness"}) == pytest.approx(
        {"label": 0.40 / 0.65, "completeness": 0.25 / 0.65})
    assert SPEC.normalized_weights(set()) == {}


@pytest.mark.parametrize(
    ("changes", "problem"),
    [({"marks": 0}, "diagram marks"), ({"marks": 9}, "diagram marks"),
     ({"label_weight": 0.5}, "100 %"), ({"required_labels": []}, "no required labels")],
)
def test_weightage_validation(changes, problem):
    spec = DiagramSpec(**{**SPEC.__dict__, **changes})
    assert any(problem in p for p in spec.validate("3", max_marks=5))
    assert SPEC.validate("3", max_marks=5) == []


def test_no_drawing_scores_zero():
    score = DiagramEvaluator(FakeLabelReader([])).evaluate("Draw a cell", SPEC, [], strictness=50)
    assert score.marks == 0 and score.missing_labels == CELL_LABELS and score.feedback == "No diagram was drawn."


def test_full_marks_with_vlm_and_labels(cell_regions):
    vlm = ScriptedVLM({"structure": 1.0, "completeness": 1.0, "labels_seen": [], "feedback": "Well drawn."})
    evaluator = DiagramEvaluator(FakeLabelReader(["Nucleus", "Cell membrane", "Mitochondria", "Cytoplasm"]), vlm)
    score = evaluator.evaluate("Draw a labelled animal cell", SPEC, cell_regions, strictness=50)
    assert score.components == {"label": 1.0, "structure": 1.0, "completeness": 1.0}
    assert (score.marks, score.missing_labels, score.feedback) == (4.0, [], "Well drawn.")
    assert not score.needs_review


def test_vlm_labels_fill_in_what_label_ocr_missed(cell_regions):
    vlm = ScriptedVLM({"structure": 0.8, "completeness": 0.75, "labels_seen": ["Cytoplasm", "Mitochondria"],
                       "feedback": "Nucleus not labelled."})
    evaluator = DiagramEvaluator(FakeLabelReader(["Cell membrane"]), vlm)
    score = evaluator.evaluate("Draw a cell", SPEC, cell_regions, strictness=50)
    assert set(score.matched_labels) == {"Cell membrane", "Cytoplasm", "Mitochondria"}
    assert score.missing_labels == ["Nucleus"]
    assert score.quality == pytest.approx(0.40 * 0.75 + 0.35 * 0.8 + 0.25 * 0.75)


def test_without_vlm_labels_stand_in_for_completeness(cell_regions):
    evaluator = DiagramEvaluator(FakeLabelReader(["Nucleus", "Cytoplasm"]), vlm=None)
    score = evaluator.evaluate("Draw a cell", SPEC, cell_regions, strictness=50)
    assert score.components == {"label": 0.5, "completeness": 0.5}
    assert set(score.weights) == {"label", "completeness"}
    assert "Missing labels: Cell membrane, Mitochondria." in score.feedback


def test_unusable_vlm_output_is_flagged(cell_regions):
    evaluator = DiagramEvaluator(FakeLabelReader(["Nucleus"]), ScriptedVLM("I see a circle."))
    score = evaluator.evaluate("Draw a cell", SPEC, cell_regions, strictness=50)
    assert "vision model gave no usable judgement" in score.review_reasons
    assert "structure" not in score.components


def test_vlm_message_carries_image_and_rubric(cell_regions):
    messages = build_vlm_messages("Draw a cell", SPEC, cell_regions[0].image)
    text, image = messages[1]["content"]
    assert image["image_url"]["url"].startswith("data:image/png;base64,")
    assert "Required labels/parts: Nucleus, Cell membrane" in text["text"]
    assert "not instructions" in messages[0]["content"]


def test_ssim_fallback_prefers_the_matching_reference(cell_regions, tmp_path):
    cell_ink = cell_regions[0].ink
    flow_image, _ = make_diagram_page(kind="flowchart")
    [flow] = detect_diagrams(preprocess_page(flow_image))
    same = structural_similarity(cell_ink, cell_ink)
    different = structural_similarity(cell_ink, flow.ink)
    assert same > 0.95 and different < same

    reference = tmp_path / "reference_cell.png"
    cv2.imencode(".png", 255 - cell_ink)[1].tofile(reference)  # dark strokes on white, like a scan
    spec = DiagramSpec(**{**SPEC.__dict__, "reference_image": str(reference)})
    score = DiagramEvaluator(FakeLabelReader(CELL_LABELS), vlm=None).evaluate("Draw a cell", spec, cell_regions, 50)
    assert score.components["structure"] > 0.8
