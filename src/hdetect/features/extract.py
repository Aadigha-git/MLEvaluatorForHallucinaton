"""Compute feature rows (docs/PREREGISTRATION.md, "Features", F0–F11).

Two blocks per question:
- the evidence-free block (F6–F9), computed once per question and shared by every condition;
- the evidence block (F2–F5), computed per evidence condition from that condition's passages.

Each block is cached on disk under a key covering its input texts, model revisions, and version
strings, so a changed input or model can never reuse a stale value. Cached blocks keep the
timings measured when they were computed. Nothing here sees a label or reference answer.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from hdetect.cleaning import CLEAN_VERSION, clean_answer
from hdetect.features import EVIDENCE_FEATURES, EVIDENCE_FREE_FEATURES, RESAMPLING_FEATURES
from hdetect.features.access import Answers, QuestionInput
from hdetect.features.lexicon import LEXICON_VERSION
from hdetect.features.models import NLI, Embedder, cosine
from hdetect.features.text import hedge_rates, lex_overlap_max, log_length, specificity, split_sentences
from hdetect.labeling.fever import parse_verdict

FEATURE_VERSION = "features-v1"
TRUNCATED = "length"


class MissingFeatureValue(RuntimeError):
    pass


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class BlockCache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS blocks (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.hits = self.misses = 0

    @staticmethod
    def key(parts: dict) -> str:
        return sha256_text(json.dumps(parts, sort_keys=True, ensure_ascii=False))

    def get(self, key: str) -> dict | None:
        row = self.db.execute("SELECT value FROM blocks WHERE key = ?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def put(self, key: str, value: dict) -> None:
        self.db.execute("INSERT OR REPLACE INTO blocks VALUES (?, ?)", (key, json.dumps(value, sort_keys=True)))
        self.db.commit()


@dataclass(frozen=True)
class Versions:
    feature: str
    clean: str
    lexicon: str
    embedding: str
    nli: str

    @classmethod
    def current(cls, embedder_spec, nli_spec) -> Versions:
        return cls(
            feature=FEATURE_VERSION,
            clean=CLEAN_VERSION,
            lexicon=LEXICON_VERSION,
            embedding=f"{embedder_spec.model_id}@{embedder_spec.revision}",
            nli=f"{nli_spec.model_id}@{nli_spec.revision}",
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "feature_version": self.feature,
            "clean_version": self.clean,
            "lexicon_version": self.lexicon,
            "embedding_model": self.embedding,
            "nli_model": self.nli,
        }


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 3)


def usable_resamples(answers: Answers) -> list[str]:
    """F9: R = cached resamples that succeeded and are non-empty after cleaning."""
    cleaned = (clean_answer(r["text"]) for r in answers.resamples)
    return [r for r in cleaned if r]


def evidence_free_block(answer: str, resamples: Sequence[str], embedder: Embedder, nli: NLI) -> dict:
    """F6–F9 from the cleaned primary answer and the usable cleaned resamples."""
    out, timing = {}, {}

    t0 = time.perf_counter()
    out["hedge_rate"], out["booster_rate"] = hedge_rates(answer)
    timing["hedge"] = _ms(t0)

    t0 = time.perf_counter()
    out["log_length"], out["specificity"] = log_length(answer), specificity(answer)
    timing["length"] = _ms(t0)

    t0 = time.perf_counter()
    sentences = split_sentences(answer)
    out["self_contra"] = max(
        (nli.probs(si, sj)["contradiction"] for i, si in enumerate(sentences) for j, sj in enumerate(sentences) if i != j),
        default=0.0,
    )
    timing["self_contra"] = _ms(t0)

    if resamples:
        t0 = time.perf_counter()
        a = embedder.embed(answer)
        out["sc_embed"] = statistics.fmean(cosine(a, embedder.embed(r)) for r in resamples)
        timing["sc_embed"] = _ms(t0)
        t0 = time.perf_counter()
        probs = [nli.probs(r, answer) for r in resamples]
        out["sc_entail"] = statistics.fmean(p["entailment"] for p in probs)
        out["sc_contra"] = statistics.fmean(p["contradiction"] for p in probs)
        timing["sc_nli"] = _ms(t0)
    else:
        out.update(sc_embed=None, sc_entail=None, sc_contra=None)
        timing.update(sc_embed=0.0, sc_nli=0.0)
    return {"features": out, "time_ms": timing, "n_sentences": len(sentences)}


def evidence_block(answer: str, question: str, passages: Sequence[str], embedder: Embedder, nli: NLI) -> dict:
    """F2–F5: the maximum over passages of each evidence feature."""
    if not passages:
        raise ValueError("the evidence block needs at least one passage")
    out, timing = {}, {}

    t0 = time.perf_counter()
    a = embedder.embed(answer)
    out["sim_max"] = max(cosine(a, embedder.embed(p)) for p in passages)
    timing["sim"] = _ms(t0)

    t0 = time.perf_counter()
    probs = [nli.probs(p, answer) for p in passages]
    out["nli_entail_max"] = max(p["entailment"] for p in probs)
    out["nli_contra_max"] = max(p["contradiction"] for p in probs)
    timing["nli"] = _ms(t0)

    t0 = time.perf_counter()
    out["lex_overlap_max"] = lex_overlap_max(answer, question, passages)
    timing["lex"] = _ms(t0)
    return {"features": out, "time_ms": timing}


class FeatureExtractor:
    def __init__(self, embedder: Embedder, nli: NLI, cache: BlockCache, versions: Versions):
        self.embedder, self.nli, self.cache, self.versions = embedder, nli, cache, versions

    def _cached(self, kind: str, parts: dict, compute) -> dict:
        key = BlockCache.key({"kind": kind, **self.versions.as_dict(), **parts})
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        value = compute()
        self.cache.put(key, value)
        return value

    def free(self, answer: str, resamples: list[str]) -> dict:
        return self._cached(
            "evidence_free",
            {"answer": answer, "resamples": resamples},
            lambda: evidence_free_block(answer, resamples, self.embedder, self.nli),
        )

    def evidence(self, answer: str, question: str, passages: list[str]) -> dict:
        return self._cached(
            "evidence",
            {"answer": answer, "question": question, "passages": passages},
            lambda: evidence_block(answer, question, passages, self.embedder, self.nli),
        )

    def rows(self, q: QuestionInput, answers: Answers, passages_by_condition: dict[str, list[str]]) -> dict[str, dict]:
        """One row per evidence condition in `passages_by_condition` (None for no evidence)."""
        if len(answers.resamples) > 5:
            raise ValueError(f"{q.question_id}: more than 5 resamples")
        primary = answers.primary
        answer = clean_answer(primary["text"])
        resamples = usable_resamples(answers)
        free = self.free(answer, resamples)
        base = {
            "question_id": q.question_id,
            "split": q.split,
            "dataset": q.dataset,
            "generator": q.generator,
            "answer_id": primary["answer_id"],
            "meta_raw_sha256": sha256_text(primary["text"]),
            "meta_finish_reason": primary["finish_reason"],
            "meta_truncated_primary": primary["finish_reason"] == TRUNCATED,
            "meta_empty_primary": not answer,
            "meta_fever_verdict": parse_verdict(answer) if q.dataset == "fever" else None,
            "meta_n_resamples_cached": len(answers.resamples),
            "meta_n_resamples_usable": len(resamples),
            "meta_n_resamples_truncated": sum(r["finish_reason"] == TRUNCATED for r in answers.resamples),
            "meta_n_sentences": free["n_sentences"],
            **self.versions.as_dict(),
            **free["features"],
            **{f"time_{k}_ms": v for k, v in free["time_ms"].items()},
        }
        out = {}
        for condition, passages in passages_by_condition.items():
            row = {**base, "condition": condition, "meta_n_passages": len(passages or ())}
            if passages:
                ev = self.evidence(answer, q.query, list(passages))
                row.update(ev["features"])
                row.update({f"time_{k}_ms": v for k, v in ev["time_ms"].items()})
            check_complete(row, condition)
            out[condition] = row
        return out


def check_complete(row: dict, condition: str) -> None:
    """F10: every feature present and finite, except self-consistency when no resample is usable."""
    expected = (EVIDENCE_FEATURES if condition != "none" else ()) + EVIDENCE_FREE_FEATURES
    for name in expected:
        v = row.get(name)
        if v is None or not math.isfinite(v):
            raise MissingFeatureValue(f"{row['question_id']}/{condition}: {name} = {v}")
    for name in RESAMPLING_FEATURES:
        v = row.get(name)
        if row["meta_n_resamples_usable"] == 0:
            if v is not None:
                raise MissingFeatureValue(f"{row['question_id']}: {name} set without usable resamples")
        elif v is None or not math.isfinite(v):
            raise MissingFeatureValue(f"{row['question_id']}/{condition}: {name} = {v}")
    if condition == "none" and any(name in row for name in EVIDENCE_FEATURES):
        raise MissingFeatureValue(f"{row['question_id']}: evidence features in the no-evidence condition")
