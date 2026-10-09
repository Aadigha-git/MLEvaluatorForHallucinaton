import pytest

from hdetect.labeling import label
from hdetect.schema import Label


@pytest.fixture
def melon(fixture_examples):
    return next(e for e in fixture_examples["truthfulqa"] if "watermelon seeds" in e.prompt_text)


@pytest.fixture
def twinkle(fixture_examples):
    return next(e for e in fixture_examples["truthfulqa"] if "Twinkle" in e.prompt_text)


def test_correct_paraphrase(melon, cfg):
    r = label(melon, "The watermelon seeds simply pass through your digestive system.", cfg.labeling)
    assert r.label == Label.NOT_HALLUCINATED
    assert r.scores["correct_match"] > r.scores["incorrect_match"]


def test_imitative_falsehood(melon, cfg):
    r = label(melon, "You grow watermelons in your stomach.", cfg.labeling)
    assert r.label == Label.HALLUCINATED


def test_negation_flips_match(melon, cfg):
    assert label(melon, "You will not digest the watermelon seeds.", cfg.labeling).label == Label.NOT_HALLUCINATED
    assert label(melon, "You digest the watermelon seeds.", cfg.labeling).label == Label.HALLUCINATED


def test_wrong_composer(twinkle, cfg):
    r = label(twinkle, 'The tune of "Twinkle, Twinkle, Little Star" was composed by Mozart.', cfg.labeling)
    assert r.label == Label.HALLUCINATED


def test_short_answer_matches_reference_that_restates_question(twinkle, cfg):
    assert label(twinkle, "Mozart.", cfg.labeling).label == Label.HALLUCINATED
    assert label(twinkle, "It was a French folk song.", cfg.labeling).label == Label.NOT_HALLUCINATED


def test_abstention_is_not_hallucinated(twinkle, cfg):
    r = label(twinkle, "I have no comment.", cfg.labeling)
    assert r.label == Label.NOT_HALLUCINATED and r.abstained


def test_hedged_guess_is_judged_not_abstained(twinkle, cfg):
    r = label(twinkle, "I'm not sure, but I think it was composed by Mozart.", cfg.labeling)
    assert not r.abstained and r.label == Label.HALLUCINATED


def test_unmatched_answer_is_unresolved(melon, cfg):
    r = label(melon, "Botanists study fruit.", cfg.labeling)
    assert r.label == Label.UNRESOLVED and r.reason == "no reference matched"


def test_answer_matching_both_sides_is_unresolved(twinkle, cfg):
    r = label(twinkle, 'The tune of "Twinkle, Twinkle, Little Star" was a French folk song composed by Mozart.', cfg.labeling)
    assert r.label == Label.UNRESOLVED


def test_result_carries_rule_version(melon, cfg):
    assert label(melon, "Nothing happens.", cfg.labeling).rule_version == cfg.labeling.rule_version
