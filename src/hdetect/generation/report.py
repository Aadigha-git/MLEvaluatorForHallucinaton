"""Generation counts, failures, spend, and latency per split. Reads no answer content beyond emptiness."""

from __future__ import annotations

import hashlib
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path

from hdetect.cleaning import clean_answer
from hdetect.generation.runner import OK, Request, attempts

TRUNCATED = "length"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"n": 0, "p50": None, "p95": None}
    ordered = sorted(values)
    pick = lambda q: round(ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))], 4)  # noqa: E731
    return {"n": len(ordered), "p50": pick(0.5), "p95": pick(0.95)}


def _kind(sample_index: int) -> str:
    return "primary" if sample_index == 0 else "resample"


def split_report(
    requests: Sequence[Request],
    rows: Sequence[dict],
    prices: Mapping[str, tuple[float, float]],
    max_attempts: int,
) -> dict:
    tried = attempts(rows)
    ok = {r["answer_id"]: r for r in rows if r["status"] == OK}
    planned = {r.answer_id: r for r in requests}

    def cost(row: dict) -> float:
        usd_in, usd_out = prices[row["generator"]]
        return ((row["prompt_tokens"] or 0) * usd_in + (row["completion_tokens"] or 0) * usd_out) / 1e6

    status: dict[str, Counter] = {"primary": Counter(), "resample": Counter()}
    for aid, req in planned.items():
        kind = _kind(req.sample_index)
        if aid in ok:
            row = ok[aid]
            status[kind]["ok"] += 1
            if not clean_answer(row["text"]):
                status[kind]["empty"] += 1
            if row["finish_reason"] == TRUNCATED:
                status[kind]["truncated"] += 1
            if row.get("prompt_tokens") is None or row.get("completion_tokens") is None:
                status[kind]["missing_usage"] += 1
        elif tried[aid] >= max_attempts:
            status[kind]["permanent_failure"] += 1
        elif tried[aid]:
            status[kind]["failed_retry_pending"] += 1
        else:
            status[kind]["not_attempted"] += 1

    ok_rows = list(ok.values())
    by_question: dict[str, list[dict]] = defaultdict(list)
    for row in ok_rows:
        if row["sample_index"] > 0:
            by_question[row["question_id"]].append(row)
    questions = {r.question_id for r in requests}
    usable = Counter(
        sum(1 for r in by_question.get(q, []) if clean_answer(r["text"])) for q in questions
    )
    full = [rs for rs in by_question.values() if len(rs) == 5]
    times = [datetime.fromisoformat(r["started_at"]) for r in rows] + [
        datetime.fromisoformat(r["finished_at"]) for r in rows
    ]

    return {
        "questions": len(questions),
        "planned_calls": len(planned),
        "calls_made": len(rows),
        "calls_errored": sum(1 for r in rows if r["status"] != OK),
        "status": {k: dict(sorted(v.items())) for k, v in status.items()},
        "usable_resamples_per_question": {str(k): usable[k] for k in sorted(usable)},
        "finish_reasons": dict(Counter(r["finish_reason"] for r in ok_rows)),
        "models_returned": dict(Counter(r["model"] for r in ok_rows)),
        "system_fingerprints": len({r["system_fingerprint"] for r in ok_rows}),
        "tokens": {
            "prompt": sum(r["prompt_tokens"] or 0 for r in ok_rows),
            "completion": sum(r["completion_tokens"] or 0 for r in ok_rows),
        },
        "cost_usd": {
            "complete": not any(v.get("missing_usage") for v in status.values()),
            "primary": round(sum(cost(r) for r in ok_rows if r["sample_index"] == 0), 6),
            "resamples": round(sum(cost(r) for r in ok_rows if r["sample_index"] > 0), 6),
            "total": round(sum(cost(r) for r in ok_rows), 6),
            "resamples_per_question_mean": round(
                statistics.mean(sum(cost(r) for r in rs) for rs in full), 8
            ) if full else None,
        },
        "latency_s": {
            "primary_call": _quantiles([r["latency_s"] for r in ok_rows if r["sample_index"] == 0]),
            "resample_call": _quantiles([r["latency_s"] for r in ok_rows if r["sample_index"] > 0]),
            "resamples_parallel_equivalent": _quantiles([max(r["latency_s"] for r in rs) for rs in full]),
            "resamples_sequential": _quantiles([sum(r["latency_s"] for r in rs) for rs in full]),
        },
        "wall_clock_s": round((max(times) - min(times)).total_seconds(), 1) if times else None,
    }


def complete(report: Mapping) -> bool:
    """Every planned call has an ok row or has used all its attempts."""
    return all(
        not counts.get("failed_retry_pending") and not counts.get("not_attempted")
        for counts in report["status"].values()
    )


def totals(reports: Iterable[Mapping]) -> dict:
    reports = list(reports)
    return {
        "planned_calls": sum(r["planned_calls"] for r in reports),
        "calls_made": sum(r["calls_made"] for r in reports),
        "calls_errored": sum(r["calls_errored"] for r in reports),
        "cost_usd": round(sum(r["cost_usd"]["total"] for r in reports), 6),
    }
