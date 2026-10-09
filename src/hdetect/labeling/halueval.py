from __future__ import annotations

from hdetect.config import HaluEvalLabeling
from hdetect.labeling.text import contains, is_abstention, negated_mention, normalize_answer, token_f1, tokens
from hdetect.schema import Example, Label, LabelResult

_YES_NO = {"yes", "no"}


def _yes_no_verdict(answer: str) -> str | None:
    toks = tokens(answer)
    if toks and toks[0] in _YES_NO:
        return toks[0]
    present = _YES_NO.intersection(toks)
    return present.pop() if len(present) == 1 else None


def label(example: Example, answer: str, cfg: HaluEvalLabeling, rule_version: str) -> LabelResult:
    """See docs/LABELING.md, "HaluEval"."""
    if is_abstention(answer):
        return LabelResult(Label.NOT_HALLUCINATED, "abstention", rule_version, abstained=True)

    (right,) = example.correct_refs
    f1 = token_f1(answer, right)
    scores = {"f1_right": round(f1, 4)}
    if example.incorrect_refs:
        scores["f1_hallucinated"] = round(token_f1(answer, example.incorrect_refs[0]), 4)

    expected = normalize_answer(right)
    if expected not in _YES_NO and contains(answer, right):
        if negated_mention(answer, right):
            return LabelResult(Label.UNRESOLVED, "right answer is mentioned under a negation", rule_version, scores=scores)
        if contains(example.prompt_text, right) and f1 < cfg.f1_threshold:
            # Comparison questions name every option, so a mention proves nothing on its own.
            return LabelResult(Label.UNRESOLVED, "right answer is one of the options named in the question", rule_version, scores=scores)
        return LabelResult(Label.NOT_HALLUCINATED, "contains the right answer", rule_version, scores=scores)

    if expected in _YES_NO:
        verdict = _yes_no_verdict(answer)
        if verdict is None:
            return LabelResult(Label.UNRESOLVED, "no yes/no verdict", rule_version, scores=scores)
        if verdict == expected:
            return LabelResult(Label.NOT_HALLUCINATED, "matching yes/no verdict", rule_version, scores=scores)
        return LabelResult(Label.HALLUCINATED, "opposite yes/no verdict", rule_version, scores=scores)

    if f1 >= cfg.f1_threshold:
        return LabelResult(Label.NOT_HALLUCINATED, "high token overlap with the right answer", rule_version, scores=scores)
    if f1 >= cfg.unresolved_floor:
        return LabelResult(Label.UNRESOLVED, "partial overlap with the right answer", rule_version, scores=scores)
    return LabelResult(Label.HALLUCINATED, "asserts a different answer", rule_version, scores=scores)
