from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from hdetect.schema import Example, make_question_id

DATASET = "halueval"

# HaluEval's knowledge field concatenates sentences without a space ("century.First").
_MISSING_SPACE = re.compile(r"(?<=[a-z0-9)\]\"'])\.(?=[A-Z])")


def clean_knowledge(text: str) -> str:
    return _MISSING_SPACE.sub(". ", text.strip())


def examples_from_rows(rows: Iterable[dict[str, Any]]) -> list[Example]:
    examples = []
    for row in rows:
        question = row["question"].strip()
        examples.append(
            Example(
                question_id=make_question_id(DATASET, question),
                dataset=DATASET,
                source_id=question,
                prompt_text=question,
                correct_refs=(row["right_answer"].strip(),),
                incorrect_refs=(row["hallucinated_answer"].strip(),),
                gold_evidence=(clean_knowledge(row["knowledge"]),),
            )
        )
    return examples
