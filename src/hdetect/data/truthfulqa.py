from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from hdetect.schema import Example, make_question_id

DATASET = "truthfulqa"


def examples_from_rows(rows: Iterable[dict[str, Any]]) -> list[Example]:
    """TruthfulQA ships no evidence passages, only a source URL, so gold_evidence is empty."""
    examples = []
    for row in rows:
        question = row["question"].strip()
        examples.append(
            Example(
                question_id=make_question_id(DATASET, question),
                dataset=DATASET,
                source_id=question,
                prompt_text=question,
                correct_refs=tuple(a.strip() for a in row["correct_answers"]),
                incorrect_refs=tuple(a.strip() for a in row["incorrect_answers"]),
                meta={
                    "category": row["category"],
                    "type": row["type"],
                    "best_answer": row["best_answer"].strip(),
                    "source_url": row["source"],
                },
            )
        )
    return examples
