from __future__ import annotations

import re

from hdetect.data.fever import (
    VERDICT_NOT_ENOUGH_INFO,
    VERDICT_REFUTED,
    VERDICT_SUPPORTED,
    VERDICT_TO_GOLD,
)
from hdetect.schema import Example, Label, LabelResult

_VERDICT = re.compile(
    r"\b(?:(support(?:s|ed))|(refute(?:s|d))|(not enough info(?:rmation)?))\b", re.IGNORECASE
)
_LEADING_NOISE = re.compile(r"^[\s*#>\"'`_\-:]*(?:(?:final\s+)?(?:answer|verdict|label)\s*[:\-]\s*)?[\s*\"'`_]*", re.IGNORECASE)


def _canonical(match: re.Match[str]) -> str:
    if match.group(1):
        return VERDICT_SUPPORTED
    if match.group(2):
        return VERDICT_REFUTED
    return VERDICT_NOT_ENOUGH_INFO


def parse_verdict(answer: str) -> str | None:
    """The verdict the answer opens with; otherwise the only verdict it mentions; otherwise None."""
    body = _LEADING_NOISE.sub("", answer, count=1)
    leading = _VERDICT.match(body)
    if leading:
        return _canonical(leading)
    found = {_canonical(m) for m in _VERDICT.finditer(answer)}
    return found.pop() if len(found) == 1 else None


def label(example: Example, answer: str, rule_version: str) -> LabelResult:
    """See docs/LABELING.md, "FEVER"."""
    verdict = parse_verdict(answer)
    scores = {}
    if verdict is None:
        return LabelResult(Label.UNRESOLVED, "no parseable verdict", rule_version, scores=scores)
    if verdict == VERDICT_NOT_ENOUGH_INFO:
        return LabelResult(Label.NOT_HALLUCINATED, "abstention", rule_version, abstained=True, scores=scores)
    if VERDICT_TO_GOLD[verdict] == example.gold_label:
        return LabelResult(Label.NOT_HALLUCINATED, "verdict matches gold label", rule_version, scores=scores)
    return LabelResult(Label.HALLUCINATED, "verdict contradicts gold label", rule_version, scores=scores)
