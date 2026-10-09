from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer

from hdetect.config import DedupConfig
from hdetect.schema import Example

_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def normalize_question(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return _NON_ALNUM.sub(" ", text).strip()


def question_text(example: Example) -> str:
    """The text a reader would call "the question"; FEVER prompts wrap the claim in a template."""
    return example.meta.get("claim", example.prompt_text) if example.dataset == "fever" else example.prompt_text


@dataclass(frozen=True)
class DuplicatePair:
    a: str
    b: str
    similarity: float


@dataclass
class DedupResult:
    kept: list[Example]
    cluster_of: dict[str, str]
    pairs: list[DuplicatePair] = field(default_factory=list)
    dropped: list[dict[str, str]] = field(default_factory=list)


class _UnionFind:
    def __init__(self, items: Sequence[str]):
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def near_duplicate_pairs(
    texts: Sequence[str],
    ngram_range: tuple[int, int],
    threshold: float,
    chunk_size: int = 500,
) -> list[tuple[int, int, float]]:
    """Index pairs (i < j) whose character n-gram TF-IDF cosine similarity is >= threshold."""
    if len(texts) < 2:
        return []
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range).fit_transform(texts)
    pairs = []
    for start in range(0, matrix.shape[0], chunk_size):
        sims = (matrix[start : start + chunk_size] @ matrix.T).toarray()
        for local_i, row in enumerate(sims):
            i = start + local_i
            for j in (row[i + 1 :] >= threshold - 1e-9).nonzero()[0]:
                pairs.append((i, i + 1 + int(j), float(row[i + 1 + j])))
    return pairs


def deduplicate(examples: Sequence[Example], cfg: DedupConfig) -> DedupResult:
    """Collapse exact duplicates, cluster near duplicates, and resolve cross-dataset clusters.

    Every kept example gets a cluster id; split assignment must keep each cluster whole.
    """
    examples = sorted(examples, key=lambda e: e.question_id)
    dropped: list[dict[str, str]] = []

    first_by_text: dict[tuple[str, str], str] = {}
    unique: list[Example] = []
    for ex in examples:
        key = (ex.dataset, normalize_question(question_text(ex)))
        if key in first_by_text:
            dropped.append(
                {"question_id": ex.question_id, "reason": "exact_duplicate", "duplicate_of": first_by_text[key]}
            )
        else:
            first_by_text[key] = ex.question_id
            unique.append(ex)

    ids = [ex.question_id for ex in unique]
    texts = [normalize_question(question_text(ex)) for ex in unique]
    uf = _UnionFind(ids)
    pairs = []
    for i, j, sim in near_duplicate_pairs(texts, cfg.ngram_range, cfg.threshold):
        uf.union(ids[i], ids[j])
        pairs.append(DuplicatePair(ids[i], ids[j], round(sim, 4)))

    members: dict[str, list[Example]] = {}
    for ex in unique:
        members.setdefault(uf.find(ex.question_id), []).append(ex)

    priority = {name: rank for rank, name in enumerate(cfg.keep_priority)}
    kept: list[Example] = []
    cluster_of: dict[str, str] = {}
    for root in sorted(members):
        group = members[root]
        winner = min((ex.dataset for ex in group), key=priority.__getitem__)
        survivors = [ex for ex in group if ex.dataset == winner]
        for ex in group:
            if ex.dataset != winner:
                dropped.append(
                    {"question_id": ex.question_id, "reason": "cross_dataset_duplicate", "duplicate_of": root}
                )
        cluster_id = min(ex.question_id for ex in survivors)
        for ex in survivors:
            cluster_of[ex.question_id] = cluster_id
        kept.extend(survivors)

    kept.sort(key=lambda e: e.question_id)
    return DedupResult(kept=kept, cluster_of=cluster_of, pairs=pairs, dropped=dropped)
