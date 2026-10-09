from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hdetect.config import SPLIT_ROLES, Config, SplitSpec
from hdetect.data.io import read_jsonl, write_jsonl
from hdetect.schema import DATASETS, Example


class LeakageError(AssertionError):
    pass


@dataclass(frozen=True)
class Assignment:
    question_id: str
    dataset: str
    split: str
    generator: str
    cluster_id: str
    stratum: str | None = None


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def _stratum(example: Example, key: str | None) -> str | None:
    if key is None:
        return None
    value = getattr(example, key, None)
    if value is None:
        value = example.meta.get(key)
    if value is None:
        raise KeyError(f"{example.question_id} has no stratification field {key!r}")
    return str(value)


def _largest_remainder(weights: dict[str, int], total: int) -> dict[str, int]:
    pool = sum(weights.values())
    if pool == 0:
        return {k: 0 for k in weights}
    raw = {k: total * w / pool for k, w in weights.items()}
    quotas = {k: int(v) for k, v in raw.items()}
    leftover = total - sum(quotas.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - quotas[k]), k))[:leftover]:
        quotas[k] += 1
    return quotas


def _clusters_by_stratum(
    examples: Sequence[Example], cluster_of: dict[str, str], stratify: str | None
) -> dict[str | None, list[list[Example]]]:
    clusters: dict[str, list[Example]] = {}
    for ex in sorted(examples, key=lambda e: e.question_id):
        clusters.setdefault(cluster_of[ex.question_id], []).append(ex)
    by_stratum: dict[str | None, list[list[Example]]] = {}
    for cid in sorted(clusters):
        members = clusters[cid]
        by_stratum.setdefault(_stratum(members[0], stratify), []).append(members)
    return by_stratum


def _split_by_fraction(
    by_stratum: dict[str | None, list[list[Example]]], specs: Sequence[SplitSpec], rng: np.random.Generator
) -> dict[str, SplitSpec]:
    """Greedy per-stratum assignment of whole clusters to the split furthest below its target."""
    out: dict[str, SplitSpec] = {}
    for stratum in sorted(by_stratum, key=str):
        clusters = by_stratum[stratum]
        total = sum(len(c) for c in clusters)
        counts = {s.name: 0 for s in specs}
        for idx in rng.permutation(len(clusters)):
            cluster = clusters[idx]
            spec = max(specs, key=lambda s: s.fraction * total - counts[s.name])
            counts[spec.name] += len(cluster)
            for ex in cluster:
                out[ex.question_id] = spec
    return out


def _sample(
    by_stratum: dict[str | None, list[list[Example]]], spec: SplitSpec, rng: np.random.Generator
) -> dict[str, SplitSpec]:
    sizes = {k: sum(len(c) for c in v) for k, v in by_stratum.items()}
    pool = sum(sizes.values())
    target = pool if spec.n is None else min(spec.n, pool)
    quotas = _largest_remainder({str(k): v for k, v in sizes.items()}, target)
    out: dict[str, SplitSpec] = {}
    for stratum in sorted(by_stratum, key=str):
        clusters = by_stratum[stratum]
        quota, taken = quotas[str(stratum)], 0
        for idx in rng.permutation(len(clusters)):
            cluster = clusters[idx]
            if taken + len(cluster) > quota:
                continue
            taken += len(cluster)
            for ex in cluster:
                out[ex.question_id] = spec
    return out


def assign_splits(examples: Sequence[Example], cluster_of: dict[str, str], cfg: Config) -> list[Assignment]:
    assignments = []
    for dataset in DATASETS:
        pool = [ex for ex in examples if ex.dataset == dataset]
        specs = cfg.splits_for(dataset)
        if not pool or not specs:
            continue
        stratify = specs[0].stratify
        if any(s.stratify != stratify for s in specs):
            raise ValueError(f"splits drawn from {dataset} must share one stratify key")
        rng = np.random.default_rng([cfg.seed, _stable_int(dataset)])
        by_stratum = _clusters_by_stratum(pool, cluster_of, stratify)
        if specs[0].fraction is not None:
            chosen = _split_by_fraction(by_stratum, specs, rng)
        else:
            chosen = {}
            for spec in specs:
                remaining = {
                    k: [c for c in clusters if c[0].question_id not in chosen]
                    for k, clusters in by_stratum.items()
                }
                chosen.update(_sample(remaining, spec, rng))
        for ex in pool:
            spec = chosen.get(ex.question_id)
            if spec is not None:
                assignments.append(
                    Assignment(
                        question_id=ex.question_id,
                        dataset=dataset,
                        split=spec.name,
                        generator=spec.generator,
                        cluster_id=cluster_of[ex.question_id],
                        stratum=_stratum(ex, stratify),
                    )
                )
    assignments.sort(key=lambda a: (a.split, a.question_id))
    check_integrity(assignments)
    return assignments


def check_integrity(assignments: Iterable[Assignment]) -> None:
    """Raise LeakageError if a question or a near-duplicate cluster touches more than one split."""
    split_of_question: dict[str, str] = {}
    split_of_cluster: dict[str, str] = {}
    for a in assignments:
        if a.question_id in split_of_question:
            raise LeakageError(f"{a.question_id} assigned twice")
        split_of_question[a.question_id] = a.split
        prior = split_of_cluster.setdefault(a.cluster_id, a.split)
        if prior != a.split:
            raise LeakageError(f"cluster {a.cluster_id} spans {prior} and {a.split}")
        if SPLIT_ROLES.get(a.split) != (a.dataset, a.generator):
            raise LeakageError(f"{a.question_id}: {a.dataset}/{a.generator} not allowed in {a.split}")


class SplitManifest:
    """The frozen question -> split mapping. Every downstream row looks its split up here."""

    def __init__(self, assignments: Iterable[Assignment]):
        self.assignments = sorted(assignments, key=lambda a: (a.split, a.question_id))
        check_integrity(self.assignments)
        self._by_id = {a.question_id: a for a in self.assignments}

    def __len__(self) -> int:
        return len(self.assignments)

    def split_of(self, question_id: str) -> str:
        return self._by_id[question_id].split

    def generator_of(self, question_id: str) -> str:
        return self._by_id[question_id].generator

    def question_ids(self, split: str) -> list[str]:
        return [a.question_id for a in self.assignments if a.split == split]

    def attach(self, rows: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """Tag rows (answers, resamples, features) with their question's split."""
        for row in rows:
            yield {**row, "split": self.split_of(row["question_id"])}

    def counts(self) -> dict[str, int]:
        return dict(sorted(Counter(a.split for a in self.assignments).items()))

    def write(self, path: str | Path) -> None:
        write_jsonl(path, (asdict(a) for a in self.assignments))

    @classmethod
    def read(cls, path: str | Path) -> SplitManifest:
        return cls(Assignment(**row) for row in read_jsonl(path))
