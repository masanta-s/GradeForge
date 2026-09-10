import pytest

from src.grading.strictness_curve import (
    apply_strictness,
    rescale_similarity,
    strictness_exponent,
    strictness_label,
)


@pytest.mark.parametrize(
    ("strictness", "exponent", "marks_at_70"),
    [(0, 0.30, 0.899), (25, 0.51, 0.834), (50, 0.866, 0.734), (75, 1.47, 0.592), (100, 2.50, 0.410)],
)
def test_plan_table_values(strictness, exponent, marks_at_70):
    assert strictness_exponent(strictness) == pytest.approx(exponent, abs=0.01)
    assert apply_strictness(0.70, strictness) == pytest.approx(marks_at_70, abs=0.005)


def test_endpoints_are_fixed():
    for s in (0, 33, 50, 100):
        assert apply_strictness(0.0, s) == 0.0
        assert apply_strictness(1.0, s) == 1.0


def test_monotonic_in_quality_and_strictness():
    grid = [i / 20 for i in range(21)]
    strictnesses = range(0, 101, 5)
    for s in strictnesses:
        marks = [apply_strictness(q, s) for q in grid]
        assert marks == sorted(marks)
    for q in grid[1:-1]:
        marks = [apply_strictness(q, s) for s in strictnesses]
        assert marks == sorted(marks, reverse=True)


def test_quality_is_clamped():
    assert apply_strictness(1.3, 50) == 1.0
    assert apply_strictness(-0.2, 50) == 0.0


def test_invalid_strictness():
    with pytest.raises(ValueError):
        strictness_exponent(101)


def test_rescale_similarity_removes_the_on_topic_floor():
    floor = 0.55  # e.g. similarity of the question text itself to the model answer
    assert rescale_similarity(0.50, floor) == 0.0      # restating the question earns nothing
    assert rescale_similarity(0.55, floor) == 0.0
    assert rescale_similarity(1.00, floor) == 1.0
    assert rescale_similarity(0.775, floor) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        rescale_similarity(0.8, floor=1.0)


def test_labels_cover_the_slider():
    labels = [strictness_label(s) for s in (0, 25, 50, 75, 100)]
    assert labels == ["Very lenient", "Lenient", "Medium", "Strict", "Very strict"]
    assert {strictness_label(s) for s in range(101)} == set(labels)
