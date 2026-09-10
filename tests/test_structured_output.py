import pytest

from src.grading.structured_output import extract_json_object, parse_json_response, regex_fallback

KEYS = {"quality", "feedback"}
GOOD = '{"quality": 0.7, "feedback": "Mentions chlorophyll but not glucose."}'


class ScriptedLLM:
    """Returns canned replies in order; records every conversation it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.conversations = []

    def __call__(self, messages, schema=None):
        self.conversations.append(messages)
        return self.replies.pop(0)


def test_layer1_clean_json():
    out = parse_json_response(GOOD, KEYS)
    assert (out.layer, out.reliable_json, out.data["quality"]) == (1, True, 0.7)


@pytest.mark.parametrize(
    "raw",
    [
        f"```json\n{GOOD}\n```",
        f"Sure! Here is the grade:\n{GOOD}\nHope this helps.",
        f"<think>The student forgot glucose {{maybe}}...</think>\n{GOOD}",
        '{"quality": 0.7, "feedback": "uses {braces} and \\"quotes\\" inside"}',
    ],
)
def test_layer2_tolerant_parser(raw):
    out = parse_json_response(raw, KEYS)
    assert out.layer in (1, 2) and out.reliable_json and out.data["quality"] == 0.7


def test_missing_required_key_is_not_accepted():
    out = parse_json_response('{"quality": 0.5}', KEYS)
    assert out.data is None and out.layer == 0


def test_layer3_repair_turn():
    llm = ScriptedLLM("still not json", GOOD)
    out = parse_json_response("I think it deserves 70%", KEYS, complete=llm,
                              messages=[{"role": "user", "content": "grade this"}])
    assert (out.layer, out.data["quality"]) == (3, 0.7)
    assert len(llm.conversations) == 2
    first_repair = llm.conversations[0]
    assert first_repair[-2] == {"role": "assistant", "content": "I think it deserves 70%"}
    assert "feedback, quality" in first_repair[-1]["content"]


def test_layer4_regex_fallback_uses_original_reply():
    llm = ScriptedLLM("nope", "still nope")
    out = parse_json_response("SCORE: 3.5/5\nFEEDBACK: Good but incomplete.", KEYS, complete=llm,
                              messages=[{"role": "user", "content": "grade"}])
    assert (out.layer, out.reliable_json) == (4, False)
    assert out.data == {"score": 0.7, "feedback": "Good but incomplete."}


def test_total_failure():
    out = parse_json_response("I cannot grade this.", KEYS)
    assert (out.data, out.layer, out.reliable_json) == (None, 0, False)


def test_extract_json_object_skips_unbalanced_prefix():
    assert extract_json_object('text { broken and then {"a": {"b": 1}} tail') == '{"a": {"b": 1}}'


def test_regex_fallback_plain_score():
    assert regex_fallback("Quality = 0.4") == {"score": 0.4, "feedback": ""}
    assert regex_fallback("no numbers here") is None
