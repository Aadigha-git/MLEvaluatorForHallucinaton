import pytest

from hdetect.cleaning import clean_answer
from hdetect.labeling import label


@pytest.mark.parametrize(
    "raw, cleaned",
    [
        ("No, food has become *more* affordable.\n\n\n\n", "No, food has become more affordable."),
        ("**SUPPORTED**. Paris is the capital.", "SUPPORTED. Paris is the capital."),
        ("__Bold__ and _italic_ and `code`", "Bold and italic and code"),
        ("# Heading\n> quoted\n- one\n* two\n+ three", "Heading quoted one two three"),
        ("1. First\n2. Second", "1. First 2. Second"),
        ("snake_case_name and 5*3*2 stay", "snake_case_name and 5*3*2 stay"),
        ("Ｆｕｌｌ\u00a0width", "Full width"),
        ("   \n\t ", ""),
    ],
)
def test_clean_answer(raw, cleaned):
    assert clean_answer(raw) == cleaned


def _variants(answer: str) -> list[str]:
    head, _, tail = answer.partition(" ")
    return [
        answer,
        f"**{answer}**",
        f"*{head}* {tail}\n\n\n",
        f"- {answer}\n- {answer}",
        f"# {answer}",
        f"> `{answer}`",
        f"_{answer}_",
    ]


def _split_fever_verdict(ex, raw: str) -> bool:
    return ex.dataset == "fever" and raw.startswith("*NOT* ENOUGH INFO")


def test_cleaning_changes_labels_only_for_split_fever_verdicts(fixture_examples, cfg):
    checked = 0
    for dataset, examples in fixture_examples.items():
        for ex in examples:
            if dataset == "fever":
                answers = [
                    f"{v}. The claim matches Wikipedia." for v in ("SUPPORTED", "REFUTED", "NOT ENOUGH INFO")
                ] + ["Wikipedia says this is refuted."]
            else:
                answers = list(ex.correct_refs) + list(ex.incorrect_refs) + ["I have no comment."]
            for answer in answers:
                for raw in _variants(answer):
                    before, after = label(ex, raw, cfg.labeling), label(ex, clean_answer(raw), cfg.labeling)
                    if _split_fever_verdict(ex, raw):
                        assert (before.label, after.label) == ("unresolved", "not_hallucinated")
                    else:
                        assert before == after, raw
                    checked += 1
    assert checked == 1449
