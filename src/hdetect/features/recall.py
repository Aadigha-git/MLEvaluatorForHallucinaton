"""Retrieval recall@5 (docs/PREREGISTRATION.md, "Retrieval recall"). Descriptive only.

Uses gold evidence, never labels or generated answers. TruthfulQA has no gold evidence, so its
recall is reported as not measurable rather than approximated.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Sequence

from hdetect.data.fever import page_title
from hdetect.features.access import QuestionInput
from hdetect.features.retrieval import Passage
from hdetect.features.text import split_sentences
from hdetect.labeling.text import tokens

NOT_MEASURABLE = "n/a: no gold evidence"


def _norm_title(title: str) -> str:
    return unicodedata.normalize("NFKC", title).casefold()


def _contains_span(haystack: list[str], needle: list[str]) -> bool:
    n = len(needle)
    return n > 0 and any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


def fever_gold_sentences(q: QuestionInput) -> list[str]:
    """Gold passages are "Title: sentence"; the sentence follows the page title and ": "."""
    out = []
    for passage, page in zip(q.gold_passages, q.gold_pages, strict=True):
        prefix = f"{page_title(page)}: "
        if not passage.startswith(prefix):
            raise ValueError(f"{q.question_id}: gold passage does not start with its page title")
        out.append(passage[len(prefix) :])
    return out


def fever_page_hit(q: QuestionInput, passages: Sequence[Passage]) -> bool:
    gold = {_norm_title(page_title(p)) for p in q.gold_pages}
    return any(_norm_title(p.title) in gold for p in passages)


def sentence_hit(sentences: Iterable[str], passages: Sequence[Passage]) -> bool:
    texts = [tokens(p.formatted()) for p in passages]
    return any(_contains_span(t, tokens(s)) for s in sentences for t in texts)


def recall(questions: Sequence[QuestionInput], retrieved: dict[str, list[Passage]]) -> dict:
    """Recall@k per dataset present in `questions`."""
    by_dataset: dict[str, list[QuestionInput]] = {}
    for q in questions:
        by_dataset.setdefault(q.dataset, []).append(q)
    out = {}
    for dataset, qs in sorted(by_dataset.items()):
        n = len(qs)
        if dataset == "fever":
            page = sum(fever_page_hit(q, retrieved[q.question_id]) for q in qs)
            sent = sum(sentence_hit(fever_gold_sentences(q), retrieved[q.question_id]) for q in qs)
            out[dataset] = {"n": n, "page_recall@5": page / n, "sentence_recall@5": sent / n}
        elif dataset == "halueval":
            hit = sum(
                sentence_hit(split_sentences(q.gold_passages[0]), retrieved[q.question_id]) for q in qs
            )
            out[dataset] = {"n": n, "knowledge_sentence_recall@5": hit / n}
        else:
            out[dataset] = {"n": n, "recall": NOT_MEASURABLE}
    return out
