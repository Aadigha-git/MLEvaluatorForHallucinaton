from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from hdetect.schema import Example, make_question_id

DATASET = "fever"

SUPPORTS = "SUPPORTS"
REFUTES = "REFUTES"
NOT_ENOUGH_INFO = "NOT ENOUGH INFO"

# The generator answers with one of these verdicts; NOT ENOUGH INFO is the abstention option.
VERDICT_SUPPORTED = "SUPPORTED"
VERDICT_REFUTED = "REFUTED"
VERDICT_NOT_ENOUGH_INFO = "NOT ENOUGH INFO"
VERDICT_TO_GOLD = {VERDICT_SUPPORTED: SUPPORTS, VERDICT_REFUTED: REFUTES}

PROMPT_TEMPLATE = (
    "Claim: {claim}\n"
    "Is this claim true according to Wikipedia? Start your answer with exactly one of "
    "SUPPORTED, REFUTED, or NOT ENOUGH INFO, then give a one-sentence justification."
)

_PTB_TOKENS = {
    "-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]",
    "-LCB-": "{", "-RCB-": "}", "-COLON-": ":",
}
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?)\]}'])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[{])\s+")


def detokenize(text: str) -> str:
    for token, char in _PTB_TOKENS.items():
        text = text.replace(token, char)
    text = text.replace("``", '"').replace("''", '"')
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    return " ".join(text.split())


def page_title(page_id: str) -> str:
    return detokenize(page_id.replace("_", " "))


def examples_from_rows(rows: Iterable[dict[str, Any]]) -> tuple[list[Example], dict[str, int]]:
    """Group the mirror's rows (one per evidence set) into one example per FEVER claim.

    Drops NOT ENOUGH INFO claims, whose evidence is machine-retrieved rather than gold,
    and claims whose rows disagree on the label. Returns the examples and drop counts.
    """
    grouped: dict[int, list[dict[str, Any]]] = {}
    n_rows = 0
    for row in rows:
        grouped.setdefault(int(row["original_id"]), []).append(row)
        n_rows += 1

    stats = {
        "rows_in": n_rows,
        "claims": len(grouped),
        "dropped_nei": 0,
        "dropped_conflicting_label": 0,
    }
    examples = []
    for original_id in sorted(grouped):
        group = grouped[original_id]
        labels = {r["label"] for r in group}
        if len(labels) > 1:
            stats["dropped_conflicting_label"] += 1
            continue
        (label,) = labels
        if label == NOT_ENOUGH_INFO:
            stats["dropped_nei"] += 1
            continue

        evidence: list[str] = []
        pages: list[list[str]] = []
        for row in group:
            for page_id, sent_idx, sentence in row["evidence"]:
                text = f"{page_title(page_id)}: {detokenize(sentence)}"
                if text not in evidence:
                    evidence.append(text)
                    pages.append([page_id, str(sent_idx)])

        claim = group[0]["claim"].strip()
        examples.append(
            Example(
                question_id=make_question_id(DATASET, str(original_id)),
                dataset=DATASET,
                source_id=str(original_id),
                prompt_text=PROMPT_TEMPLATE.format(claim=claim),
                gold_evidence=tuple(evidence),
                gold_label=label,
                meta={"claim": claim, "evidence_pages": pages},
            )
        )
    stats["examples_out"] = len(examples)
    return examples, stats
