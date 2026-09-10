import pytest

from src.grading.answer_key import AnswerKey, Question


def _key() -> AnswerKey:
    return AnswerKey(
        exam_name="Class 10 Biology — Unit Test 2",
        subject="Biology",
        questions=[
            Question(id="1", text="Define photosynthesis.", qtype="short", max_marks=2,
                     model_answer="Plants convert light energy into chemical energy.",
                     keywords=["light energy", "chemical energy", "chlorophyll"]),
            Question(id="2", text="Powerhouse of the cell?", qtype="mcq", max_marks=1,
                     options={"A": "Nucleus", "B": "Mitochondria"}, correct_option="B"),
        ],
    )


def test_roundtrip_preserves_everything(tmp_path):
    key = _key()
    path = tmp_path / "keys" / "bio.json"
    key.save(path)
    loaded = AnswerKey.load(path)
    assert loaded == key
    assert loaded.total_marks == 3
    assert loaded.question_ids == ["1", "2"]
    assert "—" in path.read_text(encoding="utf-8")  # non-ASCII kept readable


@pytest.mark.parametrize(
    ("mutate", "problem"),
    [
        (lambda k: setattr(k.questions[1], "correct_option", "E"), "correct_option"),
        (lambda k: setattr(k.questions[0], "model_answer", " "), "needs a model answer"),
        (lambda k: setattr(k.questions[0], "max_marks", 0), "max_marks"),
        (lambda k: setattr(k.questions[1], "id", "1"), "duplicate"),
    ],
)
def test_invalid_keys_are_rejected(tmp_path, mutate, problem):
    key = _key()
    mutate(key)
    assert any(problem in p for p in key.validate())
    path = tmp_path / "bad.json"
    key.save(path)
    with pytest.raises(ValueError, match=problem):
        AnswerKey.load(path)


def test_get_by_id():
    key = _key()
    assert key.get("2").correct_option == "B"
    with pytest.raises(KeyError):
        key.get("9")
