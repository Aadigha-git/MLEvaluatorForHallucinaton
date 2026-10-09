import pytest

from hdetect.labeling.text import (
    contains,
    content_tokens,
    is_abstention,
    negated_mention,
    normalize_answer,
    reference_recall,
    token_f1,
)


def test_normalize_answer():
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"
    assert normalize_answer("“Don’t” — panic.") == "don t panic"


def test_token_f1():
    assert token_f1("Arthur's Magazine", "Arthur's Magazine") == 1.0
    assert token_f1("Delhi", "Mumbai") == 0.0
    assert token_f1("New Delhi, India", "Delhi") == pytest.approx(0.5)
    assert token_f1("", "") == 1.0
    assert token_f1("", "x") == 0.0


def test_contains_is_token_span_not_substring():
    assert contains("It was started in Delhi in 1934.", "Delhi")
    assert contains("Arthur's Magazine came first.", "arthurs magazine") is False
    assert contains("Arthur's Magazine came first.", "Arthur's Magazine")
    assert not contains("Delhiwala", "Delhi")
    assert not contains("anything", "")


def test_negated_mention():
    assert negated_mention("It is not located in Raleigh.", "Raleigh")
    assert not negated_mention("It is located in Raleigh, not Charlotte.", "Raleigh")
    assert not negated_mention("Not Charlotte; it was Raleigh all along, in Raleigh.", "Raleigh")
    assert not negated_mention("No mention here.", "Raleigh")


def test_content_tokens_folds_negative_contractions():
    assert content_tokens("You don't digest them") == ["not", "digest"]
    assert content_tokens("They won the game") == ["won", "game"]
    assert content_tokens("It won't grow") == ["not", "grow"]


def test_reference_recall_and_negation_gate():
    ref = "You digest the watermelon seeds"
    assert reference_recall("You digest the watermelon seeds.", ref) == 1.0
    assert reference_recall("You will not digest the watermelon seeds.", ref) == 0.0
    assert reference_recall("You won't digest the watermelon seeds.", "You will not digest the watermelon seeds") == 1.0
    assert reference_recall("The seeds pass through.", ref) == pytest.approx(1 / 3)
    assert reference_recall("anything", "the a an") == 0.0


def test_reference_recall_ignores_question_tokens():
    q = 'Who composed the tune of "Twinkle, Twinkle, Little Star"?'
    ref = 'The tune of "Twinkle, Twinkle, Little Star" was composed by Mozart'
    assert reference_recall("Mozart.", ref) < 0.5
    assert reference_recall("Mozart.", ref, q) == 1.0
    assert reference_recall("Mozart.", 'The tune of "Twinkle, Twinkle, Little Star"', q) == 0.0


def test_reference_recall_hedges_do_not_count_as_negation():
    ref = "It was composed by Mozart"
    assert reference_recall("I'm not sure, but I think Mozart composed it.", ref) == 1.0


def test_reference_recall_skips_negation_gate_for_negated_questions():
    q = "Why don't penguins fly?"
    assert reference_recall("Their wings evolved for swimming.", "Penguins don't fly because their wings evolved for swimming", q) == 1.0


@pytest.mark.parametrize(
    "answer",
    [
        "I have no comment.",
        "I don't know.",
        "I'm not sure.",
        "I cannot answer that question.",
        "There is no way to know.",
        "NOT ENOUGH INFORMATION to say.",
    ],
)
def test_abstentions(answer):
    assert is_abstention(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "I'm not sure, but I think it was Mozart.",
        "I don't know for certain; it is probably Paris.",
        "Mozart composed it.",
        "Nothing happens.",
    ],
)
def test_not_abstentions(answer):
    assert not is_abstention(answer)
