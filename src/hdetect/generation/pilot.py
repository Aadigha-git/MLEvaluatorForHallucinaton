"""The pre-freeze pilot: format and availability checks only. Outputs are never labeled."""

from __future__ import annotations

import random
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

from hdetect.data.splits import SplitManifest
from hdetect.labeling.fever import parse_verdict
from hdetect.schema import Example

PILOT_SOURCE_SPLITS = {"truthfulqa": "train", "halueval": "val_id", "fever": None}


class PilotSelectionError(ValueError):
    pass


def select_pilot(
    kept: Sequence[Example],
    cluster_of: Mapping[str, str],
    manifest: SplitManifest,
    counts: Mapping[str, int],
    seed: int,
) -> list[tuple[Example, str | None]]:
    """Pick pilot questions: TruthfulQA from `train`, HaluEval from `val_id`, and FEVER claims
    that are in no split and share no near-duplicate cluster with any split."""
    assigned_clusters = {a.cluster_id for a in manifest.assignments}
    split_of = {a.question_id: a.split for a in manifest.assignments}
    rng = random.Random(seed)
    chosen = []
    for dataset, n in sorted(counts.items()):
        if dataset not in PILOT_SOURCE_SPLITS:
            raise PilotSelectionError(f"no pilot source for {dataset}")
        split = PILOT_SOURCE_SPLITS[dataset]
        pool = [
            ex for ex in kept
            if ex.dataset == dataset
            and split_of.get(ex.question_id) == split
            and (split is not None or cluster_of[ex.question_id] not in assigned_clusters)
        ]
        if len(pool) < n:
            raise PilotSelectionError(f"{dataset}: {len(pool)} eligible questions, need {n}")
        pool.sort(key=lambda ex: ex.question_id)
        chosen += [(ex, split) for ex in rng.sample(pool, n)]
    return chosen


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def format_report(rows: Iterable[dict], prices: Mapping[str, tuple[float, float]]) -> dict:
    """Per generator and prompt version: availability, finish reasons, emptiness, truncation,
    verdict parseability for FEVER, greedy repeatability, latency, tokens, and cost.

    `prices` maps generator to USD per million (input, output) tokens."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["generator"], row["prompt_version"]), []).append(row)

    report = {}
    for (generator, version), group in sorted(groups.items()):
        ok = [r for r in group if r["status"] == "ok"]
        texts = {r["answer_id"]: r["text"] for r in ok}
        primaries = [r for r in ok if r["sample_index"] == 0 and r["repeat"] == 0]
        repeat_pairs = [
            (texts[r["answer_id"]], texts.get(f"{r['answer_id']}/repeat1"))
            for r in primaries
            if f"{r['answer_id']}/repeat1" in texts
        ]
        in_price, out_price = prices[generator]
        prompt_tokens = sum(r["prompt_tokens"] or 0 for r in ok)
        completion_tokens = sum(r["completion_tokens"] or 0 for r in ok)
        latencies = [r["latency_s"] for r in ok]
        entry = {
            "calls": len(group),
            "ok": len(ok),
            "errors": dict(Counter(r["error"].split(":")[0] for r in group if r["status"] != "ok")),
            "error_examples": sorted({r["error"] for r in group if r["status"] != "ok"})[:3],
            "models_returned": dict(Counter(r["model"] for r in ok)),
            "system_fingerprints": len({r["system_fingerprint"] for r in ok}),
            "finish_reasons": dict(Counter(r["finish_reason"] for r in ok)),
            "empty": sum(1 for r in ok if not r["text"].strip()),
            "think_tags": sum(1 for r in ok if "<think>" in r["text"]),
            "completion_tokens_mean": round(statistics.mean(r["completion_tokens"] or 0 for r in ok), 1) if ok else None,
            "completion_tokens_max": max((r["completion_tokens"] or 0 for r in ok), default=None),
            "greedy_repeat_identical": f"{sum(a == b for a, b in repeat_pairs)}/{len(repeat_pairs)}",
            "latency_s_p50": _quantile(latencies, 0.5),
            "latency_s_p95": _quantile(latencies, 0.95),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": round((prompt_tokens * in_price + completion_tokens * out_price) / 1e6, 8),
        }
        if version.startswith("fever-"):
            verdicts = Counter(parse_verdict(r["text"]) or "unparseable" for r in ok)
            entry["fever_verdicts"] = dict(verdicts)
        report[f"{generator}/{version}"] = entry
    return report
