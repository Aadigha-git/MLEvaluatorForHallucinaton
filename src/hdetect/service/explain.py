"""Template explanations (amendment 2026-10-23, "Explanation generator").

For each feature f, Δf = s(x) − s(x with f set to its val_id median), with s the classifier's
uncalibrated score. A flag or escalate verdict names the two largest positive Δ; an accept verdict
names the two most negative. The text says what moved the score, not why an answer is wrong.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

NAMES = {
    "sim_max": "similarity to the retrieved evidence",
    "nli_entail_max": "support from the evidence (NLI entailment)",
    "nli_contra_max": "contradiction by the evidence (NLI)",
    "lex_overlap_max": "word overlap with the evidence",
    "hedge_rate": "hedging",
    "booster_rate": "certainty wording",
    "log_length": "answer length (log tokens)",
    "specificity": "specificity (numbers and names)",
    "self_contra": "internal contradiction",
    "sc_embed": "agreement with resamples (embedding)",
    "sc_entail": "agreement with resamples (NLI)",
    "sc_contra": "contradiction by resamples (NLI)",
}
VERDICT_TEXT = {"accept": "Accepted", "escalate": "Escalated", "flag": "Flagged"}


def contributions(x: Mapping[str, float], features: Sequence[str], medians: Mapping[str, float],
                  score: Callable[[np.ndarray], np.ndarray]) -> dict[str, float]:
    """Δf for every feature, in one batched call of `score` (one row per replaced feature)."""
    base = np.array([[x[f] for f in features]], dtype=float)
    rows = np.repeat(base, len(features), axis=0)
    for i, f in enumerate(features):
        rows[i, i] = medians[f]
    s = score(np.vstack([base, rows]))
    return {f: float(s[0] - s[i + 1]) for i, f in enumerate(features)}


def explain(band: str, x: Mapping[str, float], deltas: Mapping[str, float], medians: Mapping[str, float],
            top_k: int = 2) -> str:
    sign = -1.0 if band == "accept" else 1.0
    ranked = sorted((f for f in deltas if sign * deltas[f] > 0), key=lambda f: (-sign * deltas[f], f))[:top_k]
    if not ranked:
        return f"{VERDICT_TEXT[band]}: no single feature stands out from typical validation values."
    phrases = []
    for f in ranked:
        rel = "higher" if x[f] > medians[f] else "lower"
        phrases.append(f"{NAMES[f]} is {rel} than typical ({x[f]:.2f} vs {medians[f]:.2f})")
    return f"{VERDICT_TEXT[band]}: " + "; ".join(phrases) + "."
