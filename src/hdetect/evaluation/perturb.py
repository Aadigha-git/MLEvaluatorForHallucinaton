"""Rule-based answer perturbations for the Step 10 adversarial tests (docs/SPEC.md, "Adversarial tests").

Each function takes a cleaned answer and returns the perturbed text, or None when the rule does not
apply (nothing to strip, one sentence only, no swappable entity). Only the answer changes: the
question, retrieved passages, and resamples stay as they were. No labels are used.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Sequence

from hdetect.features.lexicon import HEDGES_V1
from hdetect.features.text import split_sentences

HEDGE_PREFIX = "I think "
PAD_SENTENCE = "Let me know if you have any other questions."
PERTURBATIONS = ("hedge_strip", "hedge_insert", "length_pad", "truncate", "entity_swap")

_HEDGE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(h) for h in sorted(HEDGES_V1, key=len, reverse=True)) + r")\b", re.IGNORECASE)
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?])")
_DANGLING = re.compile(r"[,;:]+([.!?])")
_ENTITY_RUN = re.compile(r"\b[A-Z][\w'’\-]*(?:\s+[A-Z][\w'’\-]*)*")
_SENTENCE_START = re.compile(r"(?:^|[.!?]\s+)$")


def _tidy(text: str) -> str:
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", re.sub(r"\s{2,}", " ", text)).strip()
    return _DANGLING.sub(r"\1", text).lstrip(",;: ")


def hedge_strip(answer: str) -> str | None:
    """Remove every `hedges-v1` phrase (boosters are left alone)."""
    out = _tidy(_HEDGE_RE.sub("", answer))
    return out if out != answer and out else None


def hedge_insert(answer: str) -> str | None:
    return HEDGE_PREFIX + answer if answer else None


def length_pad(answer: str) -> str | None:
    return f"{answer} {PAD_SENTENCE}" if answer else None


def truncate(answer: str) -> str | None:
    """Keep the first sentence (the feature pipeline's sentence splitter)."""
    sentences = split_sentences(answer)
    return sentences[0].strip() if len(sentences) > 1 else None


def entities(answer: str, question: str) -> list[str]:
    """Runs of capitalized words that do not start a sentence and do not appear in the question."""
    q = question.lower()
    out = []
    for m in _ENTITY_RUN.finditer(answer):
        if _SENTENCE_START.search(answer[:m.start()]):
            continue
        run = m.group(0)
        if run.lower() not in q and run not in out:
            out.append(run)
    return out


def entity_pool(answers: Iterable[tuple[str, str]]) -> list[str]:
    """Every swappable entity across (answer, question) pairs, sorted for determinism."""
    return sorted({e for a, q in answers for e in entities(a, q)})


def entity_swap(answer: str, question: str, pool: Sequence[str], key: str, seed: int) -> str | None:
    """Replace every occurrence of the answer's first swappable entity with one drawn from `pool`.

    The draw is seeded by `seed` and the answer's `key`, so it does not depend on processing order.
    """
    own = entities(answer, question)
    if not own:
        return None
    target = own[0]
    choices = [e for e in pool if e.lower() != target.lower() and e.lower() not in question.lower()]
    if not choices:
        return None
    replacement = random.Random(f"{seed}:{key}").choice(choices)
    return re.sub(rf"\b{re.escape(target)}\b", replacement, answer)


def perturb(name: str, answer: str, question: str, pool: Sequence[str], key: str, seed: int) -> str | None:
    if name == "entity_swap":
        return entity_swap(answer, question, pool, key, seed)
    return {"hedge_strip": hedge_strip, "hedge_insert": hedge_insert, "length_pad": length_pad,
            "truncate": truncate}[name](answer)
