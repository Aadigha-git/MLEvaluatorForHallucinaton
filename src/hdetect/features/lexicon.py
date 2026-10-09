"""Lexicon `hedges-v1` (docs/PREREGISTRATION.md, "Features", F6). Never edit; add a new version."""

from __future__ import annotations

from hdetect.labeling.text import tokens

LEXICON_VERSION = "hedges-v1"

HEDGES_V1 = (
    "might", "may", "could", "possibly", "possible", "perhaps", "probably", "likely", "unlikely",
    "apparently", "seemingly", "seem", "seems", "appear", "appears", "suggest", "suggests",
    "reportedly", "allegedly", "supposedly", "roughly", "approximately", "estimated", "generally",
    "typically", "usually", "often", "sometimes", "somewhat", "unclear", "uncertain", "unknown",
    "debated", "disputed", "i think", "i believe", "not sure", "it is possible", "it depends",
    "some sources", "is thought", "is believed", "is said",
)

BOOSTERS_V1 = (
    "definitely", "certainly", "clearly", "undoubtedly", "always", "never", "absolutely", "indeed",
    "in fact", "obviously", "surely", "proven", "without doubt", "of course",
)


def _phrases(lexicon: tuple[str, ...]) -> list[tuple[str, ...]]:
    return sorted({tuple(tokens(p)) for p in lexicon}, key=lambda p: (-len(p), p))


_HEDGE_PHRASES = _phrases(HEDGES_V1)
_BOOSTER_PHRASES = _phrases(BOOSTERS_V1)


def count_matches(toks: list[str], phrases: list[tuple[str, ...]]) -> int:
    """Non-overlapping matches, scanning left to right and trying the longest phrase first."""
    count, i = 0, 0
    while i < len(toks):
        for phrase in phrases:
            if tuple(toks[i : i + len(phrase)]) == phrase:
                count += 1
                i += len(phrase)
                break
        else:
            i += 1
    return count


def hedge_count(toks: list[str]) -> int:
    return count_matches(toks, _HEDGE_PHRASES)


def booster_count(toks: list[str]) -> int:
    return count_matches(toks, _BOOSTER_PHRASES)
