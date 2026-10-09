"""Labelers applied to answers whose truth the benchmark itself provides."""

import dataclasses

from hdetect.labeling import label
from hdetect.schema import Label


def test_halueval_right_answers_never_flagged(fixture_examples, cfg):
    for ex in fixture_examples["halueval"]:
        assert label(ex, ex.correct_refs[0], cfg.labeling).label == Label.NOT_HALLUCINATED, ex.prompt_text


def test_halueval_hallucinated_answers_mostly_rejected(fixture_examples, cfg):
    """Known limitation: an answer that contains the right entity plus false content is accepted."""
    examples = fixture_examples["halueval"]
    accepted = [
        ex.prompt_text for ex in examples
        if label(ex, ex.incorrect_refs[0], cfg.labeling).label == Label.NOT_HALLUCINATED
    ]
    assert len(accepted) <= len(examples) // 4, accepted


def test_fever_gold_and_opposite_verdicts(fixture_examples, cfg):
    word = {"SUPPORTS": "SUPPORTED", "REFUTES": "REFUTED"}
    flip = {"SUPPORTS": "REFUTES", "REFUTES": "SUPPORTS"}
    for ex in fixture_examples["fever"]:
        assert label(ex, f"{word[ex.gold_label]}. Per the evidence.", cfg.labeling).label == Label.NOT_HALLUCINATED
        assert label(ex, f"{word[flip[ex.gold_label]]}. Per the evidence.", cfg.labeling).label == Label.HALLUCINATED


def test_truthfulqa_leave_one_out_accuracy_when_resolved(fixture_examples, cfg):
    right = wrong = 0
    for ex in fixture_examples["truthfulqa"]:
        for field, expected in (("correct_refs", Label.NOT_HALLUCINATED), ("incorrect_refs", Label.HALLUCINATED)):
            refs = getattr(ex, field)
            for i, ref in enumerate(refs):
                rest = dataclasses.replace(ex, **{field: refs[:i] + refs[i + 1 :]})
                got = label(rest, ref, cfg.labeling).label
                if got != Label.UNRESOLVED:
                    right += got == expected
                    wrong += got != expected
    assert right + wrong >= 20
    assert right / (right + wrong) >= 0.8
