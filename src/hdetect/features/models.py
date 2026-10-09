"""Pinned embedding and NLI models (docs/PREREGISTRATION.md, "Retrieval and feature models").

Both run on CPU in float32, one input at a time, with deterministic algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PREREGISTERED_NLI_LABELS = {0: "contradiction", 1: "entailment", 2: "neutral"}


class LabelMappingError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str


def configure_torch(threads: int) -> None:
    import torch

    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


class Embedder:
    def __init__(self, spec: ModelSpec, max_seq_length: int, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.spec = spec
        self.model = SentenceTransformer(spec.model_id, revision=spec.revision, device=device)
        if self.model.max_seq_length != max_seq_length:
            raise RuntimeError(f"{spec.model_id} max_seq_length {self.model.max_seq_length} != {max_seq_length}")

    def embed(self, text: str) -> np.ndarray:
        """Unit-normalized embedding of one text (batch size 1)."""
        return self.model.encode([text], batch_size=1, normalize_embeddings=True, convert_to_numpy=True)[0]


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(u, v))


class NLI:
    def __init__(self, spec: ModelSpec, max_length: int, device: str = "cpu"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.spec = spec
        self.max_length = max_length
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            spec.model_id, revision=spec.revision, dtype=torch.float32
        ).to(device).eval()
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        if id2label != PREREGISTERED_NLI_LABELS:
            raise LabelMappingError(f"{spec.model_id}@{spec.revision} id2label {id2label}")
        self.index = {label: i for i, label in id2label.items()}

    def probs(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Softmax probabilities; only the premise is truncated to fit max_length."""
        enc = self.tokenizer(
            premise, hypothesis, truncation="only_first", max_length=self.max_length, return_tensors="pt"
        )
        with self._torch.no_grad():
            logits = self.model(**enc).logits[0]
        p = self._torch.softmax(logits, dim=-1).tolist()
        return {label: p[i] for label, i in self.index.items()}
