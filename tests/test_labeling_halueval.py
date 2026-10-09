import pytest

from hdetect.labeling import label
from hdetect.schema import Label


def _find(fixture_examples, needle):
    return next(e for e in fixture_examples["halueval"] if needle in e.prompt_text)


@pytest.fixture
def oberoi(fixture_examples):
    return _find(fixture_examples, "Oberoi")


@pytest.fixture
def magazine(fixture_examples):
    return _find(fixture_examples, "Arthur's Magazine")


@pytest.fixture
def yes_no(fixture_examples):
    return _find(fixture_examples, "Kings of Leon")


@pytest.mark.parametrize(
    "answer, expected, reason",
    [
        ("Delhi", Label.NOT_HALLUCINATED, "contains the right answer"),
        ("The head office is in Delhi, India.", Label.NOT_HALLUCINATED, "contains the right answer"),
        ("The Oberoi Group is headquartered in Mumbai.", Label.HALLUCINATED, "asserts a different answer"),
        ("I don't know.", Label.NOT_HALLUCINATED, "abstention"),
    ],
)
def test_short_answer_rules(oberoi, cfg, answer, expected, reason):
    r = label(oberoi, answer, cfg.labeling)
    assert (r.label, r.reason) == (expected, reason)


def test_comparison_question(magazine, cfg):
    assert label(magazine, "Arthur's Magazine was started first.", cfg.labeling).label == Label.NOT_HALLUCINATED
    assert label(magazine, "First for Women came first.", cfg.labeling).label == Label.HALLUCINATED


def test_comparison_option_mentioned_in_longer_claim_is_unresolved(magazine, cfg):
    r = label(magazine, "First for Women was founded long after Arthur's Magazine closed.", cfg.labeling)
    assert r.label == Label.UNRESOLVED
    assert r.reason == "right answer is one of the options named in the question"


def test_negated_mention_is_unresolved(oberoi, cfg):
    r = label(oberoi, "The head office is not in Delhi but in Mumbai.", cfg.labeling)
    assert (r.label, r.reason) == (Label.UNRESOLVED, "right answer is mentioned under a negation")


def test_partial_overlap_is_unresolved(magazine, cfg):
    r = label(magazine, "The magazine.", cfg.labeling)
    assert r.label == Label.UNRESOLVED
    assert cfg.labeling.halueval.unresolved_floor <= r.scores["f1_right"] < cfg.labeling.halueval.f1_threshold


def test_yes_no_questions(yes_no, cfg):
    expected = yes_no.correct_refs[0].lower()
    opposite = "no" if expected == "yes" else "yes"
    assert label(yes_no, f"{expected.capitalize()}, they are.", cfg.labeling).label == Label.NOT_HALLUCINATED
    assert label(yes_no, f"{opposite.capitalize()}.", cfg.labeling).label == Label.HALLUCINATED
    assert label(yes_no, "Both bands play rock music.", cfg.labeling).label == Label.UNRESOLVED


def test_scores_include_hallucinated_reference(oberoi, cfg):
    r = label(oberoi, "The Oberoi family's hotel company is based in Mumbai.", cfg.labeling)
    assert r.label == Label.HALLUCINATED
    assert r.scores["f1_hallucinated"] == 1.0
