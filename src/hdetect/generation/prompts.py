"""Versioned generation prompts. Never edit a template in place; add a new version."""

from __future__ import annotations

from hdetect.data.fever import PROMPT_TEMPLATE as FEVER_V1
from hdetect.schema import Example

QA_V1 = "Answer the following question in one or two sentences.\n\nQuestion: {question}"

PROMPTS = {
    "qa-v1": QA_V1,
    "fever-v1": FEVER_V1,
}


def render(prompt_version: str, example: Example) -> str:
    template = PROMPTS[prompt_version]
    if prompt_version.startswith("fever-"):
        return template.format(claim=example.meta["claim"])
    return template.format(question=example.prompt_text)


def messages(prompt: str) -> list[dict[str, str]]:
    """One user message and no system message: Gemma 3's chat template has no system role."""
    return [{"role": "user", "content": prompt}]
