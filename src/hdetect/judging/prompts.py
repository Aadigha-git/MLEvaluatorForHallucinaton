"""Judge prompt variants, pre-specified before any judge call. Never edit one in place; add a version.

Every variant gets the classifier's inputs: the question (the claim for FEVER), the cleaned primary
answer, and, under retrieved evidence, the top 5 passages as "{title}: {text}". Under no evidence
the evidence section is left out. Every variant asks for the same strict JSON verdict. The
definition follows LABELING.md: only the asked-for answer or verdict is judged, abstentions are not
hallucinated, and a hedged guess is judged on the guess.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence

OUTPUT = (
    'Respond with only a JSON object and nothing else: either {"verdict": "hallucinated"} or '
    '{"verdict": "not_hallucinated"}.'
)

RUBRIC = (
    "Rules:\n"
    "1. The answer is hallucinated if the answer it gives to the question is false. If the input is "
    "a claim, the answer is hallucinated if its verdict (SUPPORTED or REFUTED) is wrong.\n"
    "2. Judge only the answer to the question asked, or the verdict on the claim. Extra details and "
    "justifications do not change the verdict.\n"
    "3. An answer that declines to answer (for example \"I don't know\", \"I have no comment\", or "
    "NOT ENOUGH INFO) is not hallucinated.\n"
    "4. A hedged guess (\"I'm not sure, but I think it's X\") is judged on X."
)

FEW_SHOT = (
    "Examples (invented for illustration):\n"
    "Question: At sea level, at what temperature in degrees Celsius does pure water boil?\n"
    "Answer: Pure water boils at 100 degrees Celsius at sea level.\n"
    '{"verdict": "not_hallucinated"}\n\n'
    "Question: Which planet is closest to the Sun?\n"
    "Answer: Venus is the planet closest to the Sun.\n"
    '{"verdict": "hallucinated"}\n\n'
    "Question: In which year was the first transatlantic telegraph cable completed?\n"
    "Answer: I don't know which year that was.\n"
    '{"verdict": "not_hallucinated"}'
)


def evidence_block(passages: Sequence[str]) -> str:
    lines = [f"[{i}] {p}" for i, p in enumerate(passages, 1)]
    return "Evidence (retrieved passages; they may be irrelevant or incomplete):\n" + "\n".join(lines)


def _inputs(question: str, answer: str, passages: Sequence[str] | None) -> str:
    parts = [evidence_block(passages)] if passages is not None else []
    parts.append(f"Question: {question}\nAnswer: {answer}")
    return "\n\n".join(parts)


def v1a(question: str, answer: str, passages: Sequence[str] | None) -> str:
    """Minimal: one-sentence definition."""
    head = "Decide whether the answer below is hallucinated, meaning the answer it gives to the question is false."
    if passages is not None:
        head += " Use the evidence if it is relevant."
    return "\n\n".join([head, _inputs(question, answer, passages), OUTPUT])


def v1b(question: str, answer: str, passages: Sequence[str] | None) -> str:
    """Full rubric."""
    head = "You are checking an AI assistant's answer for hallucination."
    return "\n\n".join([head, RUBRIC, _inputs(question, answer, passages), OUTPUT])


def v1c(question: str, answer: str, passages: Sequence[str] | None) -> str:
    """Full rubric plus explicit guidance on how to use (or do without) evidence."""
    head = "You are checking an AI assistant's answer for hallucination."
    if passages is not None:
        guide = ("Base your decision on the evidence when it addresses the question. When it does not, use "
                 "your own knowledge. Do not mark an answer hallucinated only because the evidence does not "
                 "mention it.")
    else:
        guide = "No evidence is provided. Use your own knowledge."
    return "\n\n".join([head, RUBRIC, guide, _inputs(question, answer, passages), OUTPUT])


def v1d(question: str, answer: str, passages: Sequence[str] | None) -> str:
    """Full rubric plus three invented worked examples."""
    head = "You are checking an AI assistant's answer for hallucination."
    return "\n\n".join([head, RUBRIC, FEW_SHOT, "Now judge this answer.", _inputs(question, answer, passages), OUTPUT])


VARIANTS: dict[str, Callable[[str, str, Sequence[str] | None], str]] = {
    "judge-v1a": v1a,
    "judge-v1b": v1b,
    "judge-v1c": v1c,
    "judge-v1d": v1d,
}


def render(variant: str, question: str, answer: str, passages: Sequence[str] | None) -> str:
    return VARIANTS[variant](question, answer, passages)


def template_hash(variant: str) -> str:
    """Hash of the variant rendered on placeholder inputs, for both conditions."""
    probe = "\x00".join(render(variant, "{question}", "{answer}", p) for p in (["{passage}"] * 5, None))
    return hashlib.sha256(probe.encode()).hexdigest()


def messages(prompt: str) -> list[dict[str, str]]:
    """One user message and no system message, as for generation."""
    return [{"role": "user", "content": prompt}]
