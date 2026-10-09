"""The only reads of questions and stored generations for feature extraction.

Test splits are refused unless the caller passes `allow_test_splits=True`, which only the
evaluation entry point under ml/evaluation/ does (enforced by tests). Generation files are
verified against the committed manifest before any row is read.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from hdetect.data.dedup import question_text
from hdetect.data.io import read_jsonl
from hdetect.data.splits import SplitManifest
from hdetect.features import GOLD_DATASETS, TEST_SPLITS
from hdetect.generation.report import sha256_file
from hdetect.schema import Example


class SplitAccessError(PermissionError):
    pass


class ManifestMismatch(RuntimeError):
    pass


def guard(split: str, allow_test_splits: bool) -> None:
    if split in TEST_SPLITS and not allow_test_splits:
        raise SplitAccessError(f"{split} may only be read through ml/evaluation/")


@dataclass(frozen=True)
class QuestionInput:
    """Everything feature extraction may know about a question. No labels or reference answers."""

    question_id: str
    dataset: str
    split: str
    generator: str
    query: str
    gold_passages: tuple[str, ...]
    gold_pages: tuple[str, ...]


def gold_passages(example: Example) -> tuple[str, ...]:
    """F1: HaluEval's one knowledge passage; each FEVER gold sentence as "Title: sentence"."""
    return tuple(example.gold_evidence) if example.dataset in GOLD_DATASETS else ()


def to_input(example: Example, split: str, generator: str) -> QuestionInput:
    pages = tuple(p[0] for p in example.meta.get("evidence_pages", ())) if example.dataset == "fever" else ()
    return QuestionInput(
        question_id=example.question_id,
        dataset=example.dataset,
        split=split,
        generator=generator,
        query=question_text(example),
        gold_passages=gold_passages(example),
        gold_pages=pages,
    )


def load_inputs(
    split: str, examples_path: str | Path, manifest: SplitManifest, *, allow_test_splits: bool = False
) -> list[QuestionInput]:
    guard(split, allow_test_splits)
    wanted = {a.question_id: a for a in manifest.assignments if a.split == split}
    out = []
    for row in read_jsonl(examples_path):
        if row["question_id"] in wanted:
            a = wanted[row["question_id"]]
            out.append(to_input(Example.from_dict(row), split, a.generator))
    if len(out) != len(wanted):
        raise ManifestMismatch(f"{split}: {len(out)} of {len(wanted)} questions found")
    return sorted(out, key=lambda q: q.question_id)


@dataclass(frozen=True)
class Answers:
    """The stored primary answer and the 5 cached resamples for one question, raw text untouched."""

    question_id: str
    primary: dict
    resamples: tuple[dict, ...]


def verify_generation_file(split: str, generations_dir: str | Path, manifest_path: str | Path) -> Path:
    record = json.loads(Path(manifest_path).read_text())
    meta = record["files"][split]
    path = Path(generations_dir) / Path(meta["path"]).name
    if sha256_file(path) != meta["sha256"]:
        raise ManifestMismatch(f"{path} does not match the generation manifest")
    return path


def load_answers(
    split: str, generations_dir: str | Path, manifest_path: str | Path, *, allow_test_splits: bool = False
) -> dict[str, Answers]:
    guard(split, allow_test_splits)
    path = verify_generation_file(split, generations_dir, manifest_path)
    by_question: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in read_jsonl(path):
        if row["status"] == "ok":
            by_question[row["question_id"]][row["sample_index"]] = row
    out = {}
    for qid, samples in by_question.items():
        if 0 not in samples:
            continue
        out[qid] = Answers(qid, samples[0], tuple(samples[i] for i in range(1, 6) if i in samples))
    return out
