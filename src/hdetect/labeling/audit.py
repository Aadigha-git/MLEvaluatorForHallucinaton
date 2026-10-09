"""Hand-audit tooling: a blind stratified sample, hand-label import, and Cohen's kappa.

An answer record is a dict with at least answer_id, question_id, dataset, generator,
answer, and label (the automatic label).
"""

from __future__ import annotations

import csv
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score

from hdetect.schema import Example, Label

STRATA = ("dataset", "generator", "label")
HAND_LABELS = {Label.HALLUCINATED.value, Label.NOT_HALLUCINATED.value}
UNDECIDABLE = "undecidable"
BLIND_COLUMNS = (
    "answer_id", "question_id", "dataset", "generator", "question", "answer",
    "correct_refs", "incorrect_refs", "gold_evidence", "gold_label", "hand_label", "notes",
)
_JOIN = "\n---\n"


def _allocate(sizes: Mapping[tuple, int], total: int) -> dict[tuple, int]:
    pool = sum(sizes.values())
    raw = {k: total * v / pool for k, v in sizes.items()}
    quotas = {k: int(v) for k, v in raw.items()}
    leftover = total - sum(quotas.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - quotas[k]), k))[:leftover]:
        quotas[k] += 1
    return quotas


def audit_sample(records: Sequence[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Proportionally stratified sample of automatically resolved answers, for estimating kappa."""
    resolved = sorted(
        (r for r in records if r["label"] != Label.UNRESOLVED.value), key=lambda r: r["answer_id"]
    )
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in resolved:
        groups.setdefault(tuple(r[k] for k in STRATA), []).append(r)
    if not groups:
        return []
    quotas = _allocate({k: len(v) for k, v in groups.items()}, min(n, len(resolved)))
    rng = np.random.default_rng(seed)
    sample = []
    for key in sorted(groups):
        group = groups[key]
        idx = rng.choice(len(group), size=quotas[key], replace=False)
        sample.extend(group[i] for i in sorted(idx))
    return sorted(sample, key=lambda r: r["answer_id"])


def unresolved_queue(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted((r for r in records if r["label"] == Label.UNRESOLVED.value), key=lambda r: r["answer_id"])


def write_blind_csv(records: Sequence[dict[str, Any]], examples: Mapping[str, Example], path: str | Path) -> None:
    """Write answers for hand-labeling without their automatic labels, so auditors are not anchored."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BLIND_COLUMNS)
        writer.writeheader()
        for r in records:
            ex = examples[r["question_id"]]
            writer.writerow(
                {
                    "answer_id": r["answer_id"],
                    "question_id": r["question_id"],
                    "dataset": r["dataset"],
                    "generator": r["generator"],
                    "question": ex.meta.get("claim", ex.prompt_text),
                    "answer": r["answer"],
                    "correct_refs": _JOIN.join(ex.correct_refs),
                    "incorrect_refs": _JOIN.join(ex.incorrect_refs),
                    "gold_evidence": _JOIN.join(ex.gold_evidence),
                    "gold_label": ex.gold_label or "",
                    "hand_label": "",
                    "notes": "",
                }
            )


def read_hand_labels(path: str | Path, *, allow_undecidable: bool = False) -> dict[str, str]:
    """answer_id -> hand label. Blank rows are skipped; anything else unexpected is an error.

    `undecidable` is accepted only for the unresolved queue, where such answers are excluded and
    counted; the agreement sample needs a decision for every row.
    """
    allowed = HAND_LABELS | {UNDECIDABLE} if allow_undecidable else HAND_LABELS
    labels = {}
    with open(path, newline="", encoding="utf-8") as f:
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            value = (row.get("hand_label") or "").strip().lower()
            if not value:
                continue
            if value not in allowed:
                raise ValueError(f"{path}:{line_no}: hand_label must be one of {sorted(allowed)}, got {value!r}")
            labels[row["answer_id"]] = value
    return labels


@dataclass(frozen=True)
class Agreement:
    n: int
    n_questions: int
    raw_agreement: float
    kappa: float
    kappa_ci_low: float
    kappa_ci_high: float


def _kappa(a: Sequence[str], b: Sequence[str]) -> float:
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore")
        return float(cohen_kappa_score(a, b, labels=sorted(HAND_LABELS)))


def agreement(
    auto: Mapping[str, str],
    hand: Mapping[str, str],
    question_of: Mapping[str, str],
    n_boot: int,
    seed: int,
) -> Agreement:
    """Cohen's kappa between automatic and hand labels, with a bootstrap CI resampling questions."""
    ids = sorted(set(auto) & set(hand))
    if not ids:
        raise ValueError("no answers have both an automatic and a hand label")
    a = [auto[i] for i in ids]
    h = [hand[i] for i in ids]

    by_question: dict[str, list[int]] = {}
    for idx, answer_id in enumerate(ids):
        by_question.setdefault(question_of[answer_id], []).append(idx)
    questions = sorted(by_question)

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        rows = [i for q in rng.choice(len(questions), size=len(questions)) for i in by_question[questions[q]]]
        boots.append(_kappa([a[i] for i in rows], [h[i] for i in rows]))
    low, high = np.nanpercentile(boots, [2.5, 97.5]) if not np.all(np.isnan(boots)) else (np.nan, np.nan)

    return Agreement(
        n=len(ids),
        n_questions=len(questions),
        raw_agreement=float(np.mean([x == y for x, y in zip(a, h)])),
        kappa=_kappa(a, h),
        kappa_ci_low=float(low),
        kappa_ci_high=float(high),
    )
