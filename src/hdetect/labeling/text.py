from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans({c: " " for c in string.punctuation + "“”‘’—–"})

NEGATIONS = frozenset({"not", "no", "never", "none", "nothing", "nobody", "neither", "nor", "cannot", "without"})

# Function words dropped before reference matching; negations are deliberately kept.
STOPWORDS = frozenset(
    """
    a an the and or but if of to in on at by for with from as into onto about over than then
    is are was were be been being am do does did done has have had having will would shall should
    can could may might must it its this that these those there here which who whom whose what
    when where why how i you he she we they me him her us them my your his our their
    so very just also too some any such only own same other because s t
    """.split()
)

_NEGATIVE_CONTRACTION_STEMS = frozenset(
    {"don", "doesn", "didn", "isn", "aren", "wasn", "weren", "won", "can", "couldn",
     "shouldn", "wouldn", "hasn", "haven", "hadn", "mustn", "needn", "ain"}
)

_ABSTENTION = re.compile(
    r"\b("
    r"i have no comment|no comment"
    r"|i (do not|don ?t) know|i m not sure|i am not sure|i m unsure|i am unsure"
    r"|i (cannot|can ?t|am unable to|m unable to) (answer|say|determine|verify|know|tell)"
    r"|there is no way to (know|tell|determine)"
    r"|not enough information|insufficient information"
    r"|it is (impossible|not possible) to (know|say|tell|determine)"
    r")\b"
)

# Words that turn "I'm not sure" into a hedged assertion rather than an abstention.
_ASSERTION_AFTER_ABSTENTION = re.compile(
    r"\b(but|however|though|although|i think|i believe|probably|likely|my (best )?guess|it (might|could|may) be)\b"
)


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation and articles, collapse whitespace.

    Apostrophes become spaces, so "don't" normalizes to "don t".
    """
    text = unicodedata.normalize("NFKC", text).lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def tokens(text: str) -> list[str]:
    return normalize_answer(text).split()


def content_tokens(text: str) -> list[str]:
    """Normalized tokens minus stopwords, with negative contractions ("don t") folded into "not"."""
    toks = tokens(text)
    out, i = [], 0
    while i < len(toks):
        if toks[i] in _NEGATIVE_CONTRACTION_STEMS and i + 1 < len(toks) and toks[i + 1] == "t":
            out.append("not")
            i += 2
            continue
        if toks[i] not in STOPWORDS:
            out.append(toks[i])
        i += 1
    return out


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = tokens(prediction), tokens(reference)
    if not pred or not ref:
        return float(pred == ref)
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def find_spans(answer: str, reference: str) -> list[int]:
    """Start indices where the normalized reference appears as a contiguous token span of the answer."""
    ans, ref = tokens(answer), tokens(reference)
    if not ref:
        return []
    return [i for i in range(len(ans) - len(ref) + 1) if ans[i : i + len(ref)] == ref]


def contains(answer: str, reference: str) -> bool:
    return bool(find_spans(answer, reference))


def negated_mention(answer: str, reference: str, window: int = 3) -> bool:
    """Whether every mention of the reference has a negation within `window` tokens before it."""
    spans = find_spans(answer, reference)
    ans = tokens(answer)
    return bool(spans) and all(has_negation(ans[max(0, i - window) : i]) for i in spans)


def has_negation(toks: list[str]) -> bool:
    return any(t in NEGATIONS for t in toks)


def strip_hedges(answer: str) -> str:
    """Normalized answer with abstention/hedge phrases removed, so "not sure" does not read as a negation."""
    return " ".join(_ABSTENTION.sub(" ", normalize_answer(answer)).split())


def reference_recall(answer: str, reference: str, question: str = "") -> float:
    """Share of the reference's answer-specific content tokens present in the answer.

    Tokens the reference shares with the question are ignored, since references often restate
    the question; a reference with nothing beyond the question scores 0. The score is zeroed
    when answer and reference disagree on negation, unless the question itself is negated.
    """
    ans = content_tokens(strip_hedges(answer))
    q = content_tokens(question)
    q_set = set(q)
    ref = [t for t in content_tokens(reference) if t not in q_set]
    if not ref:
        return 0.0
    if not has_negation(q) and has_negation(ans) != has_negation(ref):
        return 0.0
    ans_counts = Counter(ans)
    hits = sum(min(n, ans_counts[t]) for t, n in Counter(ref).items())
    return hits / len(ref)


def is_abstention(answer: str) -> bool:
    """True when the answer declines to answer; a hedged guess ("not sure, but X") is not an abstention."""
    norm = normalize_answer(answer)
    match = _ABSTENTION.search(norm)
    if match is None:
        return False
    return _ASSERTION_AFTER_ABSTENTION.search(norm[match.end() :]) is None
