"""Run retrieval and feature extraction for one split under the three evidence conditions.

Test splits go through `access.guard`: only ml/evaluation/ passes `allow_test_splits=True`.
No preprocessing, selection, or threshold is fitted here; tables hold raw feature values.
"""

from __future__ import annotations

import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from hdetect.data.io import read_jsonl
from hdetect.data.splits import SplitManifest
from hdetect.features import CONDITIONS, RESAMPLING_FEATURES, feature_columns
from hdetect.features.access import guard, load_answers, load_inputs
from hdetect.features.extract import BlockCache, FeatureExtractor, Versions
from hdetect.features.models import NLI, Embedder, ModelSpec, configure_torch
from hdetect.features.recall import recall
from hdetect.features.retrieval import passages_from_row, retrieval_row


@dataclass(frozen=True)
class FeatureConfig:
    raw: dict[str, Any]

    @property
    def embedding(self) -> ModelSpec:
        return ModelSpec(self.raw["embedding"]["model_id"], self.raw["embedding"]["revision"])

    @property
    def nli(self) -> ModelSpec:
        return ModelSpec(self.raw["nli"]["model_id"], self.raw["nli"]["revision"])

    @property
    def paths(self) -> dict[str, str]:
        return self.raw["paths"]

    @property
    def retrieval(self) -> dict[str, Any]:
        return self.raw["retrieval"]


def load_feature_config(path: str | Path) -> FeatureConfig:
    return FeatureConfig(yaml.safe_load(Path(path).read_text()))


def build_extractor(fcfg: FeatureConfig, cache_path: str | Path) -> FeatureExtractor:
    configure_torch(fcfg.raw["runtime"]["torch_threads"])
    embedder = Embedder(fcfg.embedding, fcfg.raw["embedding"]["max_seq_length"], fcfg.raw["runtime"]["device"])
    nli = NLI(fcfg.nli, fcfg.raw["nli"]["max_length"], fcfg.raw["runtime"]["device"])
    return FeatureExtractor(embedder, nli, BlockCache(cache_path), Versions.current(fcfg.embedding, fcfg.nli))


def build_retriever(fcfg: FeatureConfig):
    from hdetect.features.retrieval import BM25Retriever, configure_java

    r = fcfg.retrieval
    configure_java(r["jdk_glob"])
    retriever = BM25Retriever(r["index_dir"], r["k1"], r["b"], r["k"])
    if retriever.num_docs != r["index_documents"]:
        raise RuntimeError(f"index has {retriever.num_docs} documents, expected {r['index_documents']}")
    return retriever


def retrieve_split(questions, retriever, out_path: Path) -> dict[str, dict]:
    """Retrieve once per question and append to `out_path`; existing rows are reused."""
    done = {row["question_id"]: row for row in read_jsonl(out_path)} if out_path.exists() else {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        for q in questions:
            if q.question_id in done:
                if done[q.question_id]["query"] != q.query:
                    raise RuntimeError(f"{q.question_id}: cached retrieval used a different query")
                continue
            passages, seconds = retriever.search(q.query)
            row = retrieval_row(q.question_id, q.query, passages, seconds)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            done[q.question_id] = row
    return done


def _table(rows: list[dict], condition: str) -> pa.Table:
    feats = feature_columns(condition)
    columns = sorted({k for r in rows for k in r})
    ordered = [c for c in columns if not c.startswith(("time_",)) and c not in feats] + list(feats) + [
        c for c in columns if c.startswith("time_")
    ]
    return pa.Table.from_pylist([{c: r.get(c) for c in ordered} for r in rows])


def _summary_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    pick = lambda q: round(ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))], 3)  # noqa: E731
    return {"n": len(values), "mean": round(statistics.fmean(values), 3), "p50": pick(0.5), "p95": pick(0.95)}


def run_split(
    split: str,
    fcfg: FeatureConfig,
    *,
    examples_path: str | Path,
    manifest: SplitManifest,
    extractor: FeatureExtractor,
    retriever=None,
    out_dir: str | Path,
    sample: int | None = None,
    sample_seed: int = 20261008,
    allow_test_splits: bool = False,
) -> dict:
    guard(split, allow_test_splits)
    questions = load_inputs(split, examples_path, manifest, allow_test_splits=allow_test_splits)
    answers = load_answers(
        split, fcfg.paths["generations"], fcfg.paths["generation_manifest"], allow_test_splits=allow_test_splits
    )
    if sample is not None:
        questions = sorted(random.Random(sample_seed).sample(questions, sample), key=lambda q: q.question_id)
    out_dir = Path(out_dir)
    tag = f"{split}.sample{sample}" if sample is not None else split

    retrieval_path = Path(fcfg.paths["features"]) / "retrieval" / f"{split}.jsonl"
    if retriever is None:
        if not retrieval_path.exists():
            raise RuntimeError(f"no retrieval results at {retrieval_path} and no retriever")
        retrieved_rows = {row["question_id"]: row for row in read_jsonl(retrieval_path)}
    else:
        retrieved_rows = retrieve_split(questions, retriever, retrieval_path)
    retrieved = {qid: passages_from_row(row) for qid, row in retrieved_rows.items()}

    rows: dict[str, list[dict]] = {c: [] for c in CONDITIONS}
    skipped_no_answer = []
    t0 = time.perf_counter()
    for i, q in enumerate(questions, 1):
        if q.question_id not in answers:
            skipped_no_answer.append(q.question_id)
            continue
        conditions: dict[str, list[str] | None] = {
            "retrieved": [p.formatted() for p in retrieved[q.question_id]],
            "none": None,
        }
        if q.gold_passages:
            conditions["gold"] = list(q.gold_passages)
        for condition, row in extractor.rows(q, answers[q.question_id], conditions).items():
            row["meta_retrieval_s"] = retrieved_rows[q.question_id]["retrieval_s"]
            rows[condition].append(row)
        if i % 200 == 0:
            print(f"  {split}: {i}/{len(questions)} questions, {time.perf_counter() - t0:.0f}s", flush=True)

    written = {}
    for condition, crow in rows.items():
        if not crow:
            continue
        path = out_dir / condition / f"{tag}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(_table(crow, condition), path)
        written[condition] = str(path)

    time_cols = sorted({k for r in rows["retrieved"] for k in r if k.startswith("time_")})
    summary = {
        "split": split,
        "questions": len(questions),
        "skipped_no_primary_answer": len(skipped_no_answer),
        "rows": {c: len(r) for c, r in rows.items()},
        "files": written,
        "missing_values": {
            c: {
                f: sum(1 for r in crow if r.get(f) is None)
                for f in feature_columns(c)
            }
            for c, crow in rows.items() if crow
        },
        "no_usable_resamples": sum(1 for r in rows["none"] if r["meta_n_resamples_usable"] == 0),
        "truncated_primary": sum(1 for r in rows["none"] if r["meta_truncated_primary"]),
        "empty_primary": sum(1 for r in rows["none"] if r["meta_empty_primary"]),
        "time_ms": {c: _summary_stats([r[c] for r in rows["retrieved"] if c in r]) for c in time_cols},
        "retrieval_s": _summary_stats([retrieved_rows[q.question_id]["retrieval_s"] for q in questions]),
        "recall@5": recall(questions, retrieved),
        "wall_clock_s": round(time.perf_counter() - t0, 1),
        "cache": {"hits": extractor.cache.hits, "misses": extractor.cache.misses},
    }
    assert all(f in RESAMPLING_FEATURES for c in summary["missing_values"].values() for f, n in c.items() if n)
    return summary
