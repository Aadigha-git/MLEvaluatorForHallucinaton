from __future__ import annotations

from hdetect.config import TruthfulQALabeling
from hdetect.labeling.text import is_abstention, reference_recall
from hdetect.schema import Example, Label, LabelResult


def label(example: Example, answer: str, cfg: TruthfulQALabeling, rule_version: str) -> LabelResult:
    """See docs/LABELING.md, "TruthfulQA"."""
    if is_abstention(answer):
        return LabelResult(Label.NOT_HALLUCINATED, "abstention", rule_version, abstained=True)

    correct = [r for r in example.correct_refs if not is_abstention(r)]
    incorrect = [r for r in example.incorrect_refs if not is_abstention(r)]
    q = example.prompt_text
    c = max((reference_recall(answer, r, q) for r in correct), default=0.0)
    i = max((reference_recall(answer, r, q) for r in incorrect), default=0.0)
    scores = {"correct_match": round(c, 4), "incorrect_match": round(i, 4)}

    if max(c, i) < cfg.min_match:
        return LabelResult(Label.UNRESOLVED, "no reference matched", rule_version, scores=scores)
    if c - i >= cfg.margin:
        return LabelResult(Label.NOT_HALLUCINATED, "closer to a correct reference", rule_version, scores=scores)
    if i - c >= cfg.margin:
        return LabelResult(Label.HALLUCINATED, "closer to an incorrect reference", rule_version, scores=scores)
    return LabelResult(Label.UNRESOLVED, "correct and incorrect references match about equally", rule_version, scores=scores)
