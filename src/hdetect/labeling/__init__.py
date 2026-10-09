"""Automatic labels for generated answers, from reference answers or gold evidence only.

Nothing in this package may call an LLM judge, the NLI model, or the embedding model
used as features (enforced by tests/test_labeling_isolation.py).
"""

from __future__ import annotations

from hdetect.config import LabelingConfig
from hdetect.labeling import fever, halueval, truthfulqa
from hdetect.schema import Example, LabelResult


def label(example: Example, answer: str, cfg: LabelingConfig) -> LabelResult:
    if example.dataset == "truthfulqa":
        return truthfulqa.label(example, answer, cfg.truthfulqa, cfg.rule_version)
    if example.dataset == "halueval":
        return halueval.label(example, answer, cfg.halueval, cfg.rule_version)
    if example.dataset == "fever":
        return fever.label(example, answer, cfg.rule_version)
    raise ValueError(f"no labeler for dataset {example.dataset!r}")
