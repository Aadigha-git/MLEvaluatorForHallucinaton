"""Labels for generated primary answers: the frozen automatic rules, then hand labels for the rest.

Automatic labels are stored per rule version and split in `data/labels/v{version}/auto/{split}.jsonl`
(git-ignored). The
labeler's input is the cleaned primary answer; raw text stays in the generation files. Test splits
are labeled and read only with `allow_test_splits=True`, which only ml/evaluation/ passes.

Final labels (docs/PREREGISTRATION.md, "Labeling commitments"): an automatically resolved answer
keeps its automatic label; an unresolved answer takes its hand label; an answer the hand labeler
marks `undecidable` is excluded and counted. Training refuses to run while any unresolved answer
has no hand decision.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hdetect.cleaning import clean_answer
from hdetect.config import LabelingConfig
from hdetect.data.io import read_jsonl, write_jsonl
from hdetect.data.splits import SplitManifest
from hdetect.features.access import guard, load_answers
from hdetect.labeling import label
from hdetect.labeling.audit import UNDECIDABLE
from hdetect.schema import Example, Label

LABELS_ROOT = Path("data/labels")
EXCLUSION_FLAG_RATE = 0.05


def version_dir(rule_version: str, root: str | Path = LABELS_ROOT) -> Path:
    """Labels from different rule versions never share a directory: data/labels/v{version}/."""
    return Path(root) / f"v{rule_version}"


def audit_file(stem: str, rule_version: str, suffix: str) -> str:
    """File names under data/audit/ for one rule version; version 1 keeps its original names."""
    return f"{stem}{'' if rule_version == '1' else f'_v{rule_version}'}{suffix}"


class IncompleteLabels(RuntimeError):
    pass


class LabelFileMismatch(RuntimeError):
    pass


def auto_path(split: str, labels_dir: str | Path) -> Path:
    return Path(labels_dir) / "auto" / f"{split}.jsonl"


def label_split(
    split: str,
    examples: Mapping[str, Example],
    manifest: SplitManifest,
    generations_dir: str | Path,
    generation_manifest: str | Path,
    labeling: LabelingConfig,
    config_hash: str,
    *,
    allow_test_splits: bool = False,
) -> list[dict[str, Any]]:
    """Automatic label for every primary answer in `split`, in question_id order."""
    answers = load_answers(split, generations_dir, generation_manifest, allow_test_splits=allow_test_splits)
    expected = set(manifest.question_ids(split))
    if set(answers) != expected:
        raise LabelFileMismatch(f"{split}: {len(answers)} primary answers for {len(expected)} questions")
    rows = []
    for qid in sorted(answers):
        primary = answers[qid].primary
        answer = clean_answer(primary["text"])
        result = label(examples[qid], answer, labeling)
        rows.append(
            {
                "answer_id": primary["answer_id"],
                "question_id": qid,
                "dataset": examples[qid].dataset,
                "generator": primary["generator"],
                "split": split,
                "answer": answer,
                "raw_sha256": hashlib.sha256(primary["text"].encode("utf-8")).hexdigest(),
                "finish_reason": primary["finish_reason"],
                **result.to_dict(),
                "config_hash": config_hash,
            }
        )
    return rows


def write_auto_labels(rows: Sequence[dict[str, Any]], split: str, labels_dir: str | Path) -> Path:
    """Write once; rewriting is allowed only if the content is identical (labels are immutable)."""
    path = auto_path(split, labels_dir)
    if path.exists():
        if list(read_jsonl(path)) != [json.loads(json.dumps(r)) for r in rows]:
            raise LabelFileMismatch(f"{path} exists with different content; bump the rule version instead")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, rows)
    return path


def read_auto_labels(split: str, labels_dir: str | Path, *, allow_test_splits: bool = False) -> list[dict]:
    guard(split, allow_test_splits)
    return list(read_jsonl(auto_path(split, labels_dir)))


@dataclass(frozen=True)
class FinalLabels:
    split: str
    y: dict[str, int]
    excluded: tuple[str, ...]
    counts: dict[str, int]

    @property
    def exclusion_rate(self) -> float:
        total = len(self.y) + len(self.excluded)
        return len(self.excluded) / total if total else 0.0

    @property
    def flagged(self) -> bool:
        return self.exclusion_rate > EXCLUSION_FLAG_RATE

    def summary(self) -> dict[str, Any]:
        return {
            "split": self.split,
            **self.counts,
            "excluded": len(self.excluded),
            "exclusion_rate": round(self.exclusion_rate, 4),
            "flagged_over_5pct": self.flagged,
        }


def final_labels(split: str, auto_rows: Sequence[dict[str, Any]], hand: Mapping[str, str]) -> FinalLabels:
    """answer_id -> 1 (hallucinated) or 0, from automatic labels plus hand labels for unresolved answers."""
    unresolved = {r["answer_id"] for r in auto_rows if r["label"] == Label.UNRESOLVED.value}
    stray = set(hand) - unresolved
    split_ids = {r["answer_id"] for r in auto_rows}
    if stray & split_ids:
        raise LabelFileMismatch(f"{split}: hand labels given for {len(stray & split_ids)} automatically resolved answers")
    missing = sorted(unresolved - set(hand))
    if missing:
        raise IncompleteLabels(f"{split}: {len(missing)} of {len(unresolved)} unresolved answers have no hand label")
    y: dict[str, int] = {}
    excluded = []
    source = Counter()
    for r in auto_rows:
        aid = r["answer_id"]
        if aid in unresolved:
            value = hand[aid]
            if value == UNDECIDABLE:
                excluded.append(aid)
                continue
            source["hand_labeled"] += 1
        else:
            value = r["label"]
            source["auto_labeled"] += 1
        y[aid] = int(value == Label.HALLUCINATED.value)
    counts = {
        "answers": len(auto_rows),
        "labeled": len(y),
        "hallucinated": sum(y.values()),
        "not_hallucinated": len(y) - sum(y.values()),
        "auto_labeled": source["auto_labeled"],
        "hand_labeled": source["hand_labeled"],
        "abstained": sum(bool(r["abstained"]) for r in auto_rows if r["answer_id"] in y),
    }
    return FinalLabels(split, y, tuple(sorted(excluded)), counts)
