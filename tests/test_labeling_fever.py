import pytest

from hdetect.labeling import label
from hdetect.labeling.fever import parse_verdict
from hdetect.schema import Label


@pytest.mark.parametrize(
    "answer, verdict",
    [
        ("SUPPORTED. The film was directed by Pablo Larraín.", "SUPPORTED"),
        ("**REFUTED** - it was Pablo Larraín.", "REFUTED"),
        ("Verdict: refuted. Larraín directed it.", "REFUTED"),
        ("NOT ENOUGH INFO: the evidence does not say.", "NOT ENOUGH INFO"),
        ("The claim is refuted by the film's credits.", "REFUTED"),
        ("REFUTED. Some sources claim it is supported, but the credits disagree.", "REFUTED"),
        ("It could be supported or refuted depending on the source.", None),
        ("Peter Jackson did not direct it.", None),
    ],
)
def test_parse_verdict(answer, verdict):
    assert parse_verdict(answer) == verdict


@pytest.fixture
def jackie(fixture_examples):
    return next(e for e in fixture_examples["fever"] if e.source_id == "70882")


def test_verdict_against_gold(jackie, cfg):
    assert jackie.gold_label == "REFUTES"
    assert label(jackie, "REFUTED. Pablo Larraín directed it.", cfg.labeling).label == Label.NOT_HALLUCINATED
    assert label(jackie, "SUPPORTED. Peter Jackson directed it.", cfg.labeling).label == Label.HALLUCINATED


def test_not_enough_info_is_abstention(jackie, cfg):
    r = label(jackie, "NOT ENOUGH INFO. I can't verify the director.", cfg.labeling)
    assert r.label == Label.NOT_HALLUCINATED and r.abstained


def test_unparseable_is_unresolved(jackie, cfg):
    r = label(jackie, "Peter Jackson is a director from New Zealand.", cfg.labeling)
    assert r.label == Label.UNRESOLVED and r.reason == "no parseable verdict"
