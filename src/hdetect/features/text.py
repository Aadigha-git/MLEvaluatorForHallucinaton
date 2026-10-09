"""Model-free features on cleaned text (F5–F8). Inputs are strings only; no labels exist here."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from hdetect.features.lexicon import booster_count, hedge_count
from hdetect.labeling.text import content_tokens, tokens

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_LEADING_PUNCT = re.compile(r"^[^\w]+")


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]


def lex_overlap_max(answer: str, question: str, passages: Sequence[str]) -> float:
    """F5: the largest share of the answer's own content tokens (not in the question) found in a passage."""
    own = set(content_tokens(answer)) - set(content_tokens(question))
    if not own:
        return 0.0
    return max(len(own & set(content_tokens(p))) / len(own) for p in passages)


def hedge_rates(answer: str) -> tuple[float, float]:
    """F6: (hedge matches / n, booster matches / n), both 0 when n = 0."""
    toks = tokens(answer)
    if not toks:
        return 0.0, 0.0
    return hedge_count(toks) / len(toks), booster_count(toks) / len(toks)


def log_length(answer: str) -> float:
    """F7: ln(1 + n) with n the number of normalized tokens."""
    return math.log1p(len(tokens(answer)))


def specificity(answer: str) -> float:
    """F7: (tokens with a digit + capitalized tokens not starting a sentence) / whitespace tokens.

    A token may count in both terms. Leading punctuation (quotes, brackets) is ignored when
    testing capitalization; a sentence starts at the first token or after a token ending in
    ".", "!", or "?"; the token "I" is not counted.
    """
    words = answer.split()
    if not words:
        return 0.0
    digits = capitalized = 0
    for j, word in enumerate(words):
        if any(ch.isdigit() for ch in word):
            digits += 1
        starts_sentence = j == 0 or words[j - 1][-1] in ".!?"
        core = _LEADING_PUNCT.sub("", word)
        if core and core[0].isupper() and not starts_sentence and core.rstrip(".,;:!?'\")") != "I":
            capitalized += 1
    return (digits + capitalized) / len(words)
