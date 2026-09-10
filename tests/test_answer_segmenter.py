import pytest

from src.ocr.answer_segmenter import parse_marker, segment_answers
from src.ocr.pipeline import OCRLine, PageOCR


def _page(*texts: str, index: int = 0) -> PageOCR:
    return PageOCR(
        page_index=index,
        source="handwriting",
        lines=[OCRLine(text=t, confidence=0.9, bbox=(0, i * 50, 100, 40), page=index) for i, t in enumerate(texts)],
    )


@pytest.mark.parametrize(
    ("text", "qid", "strong", "rest"),
    [
        ("Q1. Photosynthesis is", "1", True, "Photosynthesis is"),
        ("Q 12) The answer", "12", True, "The answer"),
        ("Ans 3 Mitochondria", "3", True, "Mitochondria"),
        ("Ans. 4: yes", "4", True, "yes"),
        ("Question 5(b) Osmosis", "5b", True, "Osmosis"),
        ("Ql. Light reaction", "1", True, "Light reaction"),   # OCR read 1 as l
        ("QO5 Enzymes", "5", True, "Enzymes"),                 # OCR read 0 as O
        ("2. Newton's first law", "2", False, "Newton's first law"),
        ("3) Force", "3", False, "Force"),
        ("4(a) Velocity", "4a", False, "Velocity"),
        ("4(ii) Acceleration", "4.ii", False, "Acceleration"),
        ("6b. Momentum", "6b", False, "Momentum"),
    ],
)
def test_parse_marker(text, qid, strong, rest):
    marker = parse_marker(text)
    assert marker is not None
    assert (marker.question_id, marker.strong, marker.rest) == (qid, strong, rest)


@pytest.mark.parametrize(
    "text",
    [
        "3 moles of oxygen are produced",   # number without separator is content
        "The answer is 42.",
        "Quickly the reaction proceeds",
        "Answer lies in the cell wall",
        "0. nothing",
        "",
    ],
)
def test_parse_marker_rejects_content(text):
    assert parse_marker(text) is None


def test_basic_segmentation_across_pages():
    result = segment_answers(
        [
            _page("Q1. Photosynthesis happens", "in chloroplasts"),
            _page("continued text for Q1", "Q2 Newton's law", "states inertia", index=1),
        ],
        expected_ids=["1", "2"],
    )
    assert list(result.answers) == ["1", "2"]
    assert result.answers["1"].text == "Photosynthesis happens\nin chloroplasts\ncontinued text for Q1"
    assert result.answers["2"].text == "Newton's law\nstates inertia"
    assert [line.page for line in result.answers["1"].lines] == [0, 0, 1]
    assert not result.needs_llm


def test_numbered_points_inside_an_answer_are_not_questions():
    result = segment_answers(
        [_page("Q3. Advantages of solar power:", "1. Renewable", "2. No emissions", "Q4 Wind energy")],
        expected_ids=["1", "2", "3", "4"],
    )
    assert result.answers["3"].text == "Advantages of solar power:\n1. Renewable\n2. No emissions"
    assert set(result.answers) == {"3", "4"}
    assert result.missing == ["1", "2"]
    assert result.needs_llm


def test_bare_markers_advance_in_order():
    result = segment_answers([_page("1. Cell", "2. Tissue", "3. Organ")], expected_ids=["1", "2", "3"])
    assert {k: v.text for k, v in result.answers.items()} == {"1": "Cell", "2": "Tissue", "3": "Organ"}


def test_strong_marker_allows_out_of_order_answers():
    result = segment_answers([_page("Q2 second first", "Q1 then first")], expected_ids=["1", "2"])
    assert result.answers["2"].text == "second first"
    assert result.answers["1"].text == "then first"


def test_unknown_question_numbers_stay_as_answer_text():
    result = segment_answers([_page("Q1 energy", "7. is conserved")], expected_ids=["1", "2"])
    assert result.answers["1"].text == "energy\n7. is conserved"


def test_subparts_resolve_against_answer_key():
    result = segment_answers(
        [_page("Q3 (see below)", "Q5(a) Osmosis", "Q5(b) Diffusion")],
        expected_ids=["3", "5a", "5b"],
    )
    assert set(result.answers) == {"3", "5a", "5b"}
    assert result.answers["5b"].text == "Diffusion"


def test_text_before_first_marker_is_unassigned():
    result = segment_answers([_page("Name: Riya  Roll 12", "Q1 answer")], expected_ids=["1"])
    assert [line.text for line in result.unassigned] == ["Name: Riya  Roll 12"]
    assert result.needs_llm  # orphan text -> let the LLM pass decide


def test_no_expected_ids_accepts_any_marker():
    result = segment_answers([_page("Q9 anything", "Q10 more")])
    assert list(result.answers) == ["9", "10"]
    assert result.missing == []
