from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

DATASETS = ("truthfulqa", "halueval", "fever")


def make_question_id(dataset: str, source_id: str) -> str:
    digest = hashlib.sha256(f"{dataset}\x1f{source_id}".encode()).hexdigest()[:16]
    return f"{dataset}-{digest}"


@dataclass(frozen=True)
class Example:
    question_id: str
    dataset: str
    source_id: str
    prompt_text: str
    correct_refs: tuple[str, ...] = ()
    incorrect_refs: tuple[str, ...] = ()
    gold_evidence: tuple[str, ...] = ()
    gold_label: str | None = None
    meta: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("correct_refs", "incorrect_refs", "gold_evidence"):
            d[key] = list(d[key])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Example:
        return cls(
            question_id=d["question_id"],
            dataset=d["dataset"],
            source_id=d["source_id"],
            prompt_text=d["prompt_text"],
            correct_refs=tuple(d.get("correct_refs", ())),
            incorrect_refs=tuple(d.get("incorrect_refs", ())),
            gold_evidence=tuple(d.get("gold_evidence", ())),
            gold_label=d.get("gold_label"),
            meta=dict(d.get("meta", {})),
        )


class Label(StrEnum):
    HALLUCINATED = "hallucinated"
    NOT_HALLUCINATED = "not_hallucinated"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class LabelResult:
    label: Label
    reason: str
    rule_version: str
    abstained: bool = False
    scores: dict[str, float] = field(default_factory=dict, compare=False, hash=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": str(self.label),
            "reason": self.reason,
            "rule_version": self.rule_version,
            "abstained": self.abstained,
            "scores": dict(self.scores),
        }
