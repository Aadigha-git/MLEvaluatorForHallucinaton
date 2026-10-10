"""Score judge verdicts on labeled answers, select prompts, and measure run-to-run agreement.

Primary scoring: an answer without a valid verdict (a persistent API failure or any parse failure) is
scored as wrong, a false negative if it is hallucinated and a false positive if not. It is never
dropped from the denominator or counted as correct. F1 on parsed verdicts only is reported beside it,
with its n.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations

import numpy as np

from hdetect.judging.parse import OK as PARSE_OK
from hdetect.judging.runner import OK

NO_VERDICT_API = "api_failure"


def final_rows(rows: Iterable[dict]) -> dict[str, dict]:
    """The `ok` row per request if there is one, otherwise the latest error row."""
    latest: dict[str, dict] = {}
    for row in rows:
        if latest.get(row["request_id"], {}).get("status") != OK:
            latest[row["request_id"]] = row
    return latest


def outcome(row: dict | None) -> str:
    if row is None or row["status"] != OK:
        return NO_VERDICT_API
    return row["parse_outcome"]


def predicted(row: dict | None) -> int | None:
    if row is None or row["status"] != OK or row["parse_outcome"] != PARSE_OK:
        return None
    return int(row["verdict"] == "hallucinated")


def f1_counts(y: Sequence[int], pred: Sequence[int | None]) -> dict[str, float]:
    tp = fp = fn = 0
    for t, p in zip(y, pred):
        p = (1 - t) if p is None else p
        tp += t == 1 and p == 1
        fp += t == 0 and p == 1
        fn += t == 1 and p == 0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p95": None}
    return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95))}


def score_cell(rows: Mapping[str, dict], labels: Mapping[str, int], ids: Sequence[str]) -> dict:
    """`rows` maps answer_id to its final row for one judge, variant, condition, and run."""
    y = [labels[a] for a in ids]
    got = [rows.get(a) for a in ids]
    preds = [predicted(r) for r in got]
    outcomes = Counter(outcome(r) for r in got)
    parsed = [(t, p) for t, p in zip(y, preds) if p is not None]
    ok_rows = [r for r in got if r is not None and r["status"] == OK]
    costs = [r["cost_usd"] for r in ok_rows]
    lat_all = [r["latency_s"] for r in ok_rows]
    lat_failed = [r["latency_s"] for r, p in zip(got, preds) if p is None and r is not None and r["status"] == OK]
    # A transient retry's backoff is inside api_latency_s. `transient_retries` is None when the client
    # could not observe retries (Nebius via `openai`; Gemini calls made before throttling was added).
    lat_clean = [r["latency_s"] for r in ok_rows if r.get("transient_retries") == 0]
    return {
        "n": len(ids),
        "verdicts": sum(p is not None for p in preds),
        **{f"outcome_{k}": v for k, v in sorted(outcomes.items())},
        "parse_failures": sum(v for k, v in outcomes.items() if k not in (PARSE_OK, NO_VERDICT_API)),
        "api_failures": outcomes.get(NO_VERDICT_API, 0),
        **f1_counts(y, preds),
        "f1_parsed_only": f1_counts([t for t, _ in parsed], [p for _, p in parsed])["f1"],
        "n_parsed": len(parsed),
        "input_tokens": sum(r["input_tokens"] or 0 for r in ok_rows),
        "output_tokens": sum(r["output_tokens"] or 0 for r in ok_rows),
        "thinking_tokens": sum(r.get("thinking_tokens") or 0 for r in ok_rows),
        "cost_usd": sum(c for c in costs if c is not None),
        "cost_complete": all(c is not None for c in costs) and len(ok_rows) == len(ids),
        "cost_per_verdict_usd": (sum(c for c in costs if c is not None) / len(ids)) if ids else None,
        **{f"latency_{k}_s": v for k, v in _summary(lat_all).items()},
        **{f"latency_failed_{k}_s": v for k, v in _summary(lat_failed).items()},
        "latency_retried": sum((r.get("transient_retries") or 0) > 0 for r in ok_rows),
        "latency_retry_unknown": sum(r.get("transient_retries") is None for r in ok_rows),
        **{f"latency_no_retry_{k}_s": v for k, v in _summary(lat_clean).items()},
    }


def select_variant(scores: Mapping[str, dict], order: Sequence[str]) -> str:
    """Highest F1 (rounded to 12 decimals); ties go to fewer parse failures, then the earlier variant."""
    return min(order, key=lambda v: (-round(scores[v]["f1"], 12),
                                     scores[v]["parse_failures"] + scores[v]["api_failures"], order.index(v)))


def repeat_subset(labeled_ids: Iterable[str], size: int, seed: int) -> list[str]:
    ids = sorted(labeled_ids)
    return sorted(random.Random(seed).sample(ids, size))


def agreement(runs: Sequence[Mapping[str, dict]], ids: Sequence[str]) -> dict[str, float]:
    """Run-to-run variation. A failed or unparsed call is its own outcome, so it counts as disagreement
    with any verdict."""
    def label(row):
        p = predicted(row)
        return "failure" if p is None else p

    per_answer = [[label(run.get(a)) for run in runs] for a in ids]
    pairs = list(combinations(range(len(runs)), 2))
    pairwise = {f"agree_run{i}_run{j}": float(np.mean([v[i] == v[j] for v in per_answer])) for i, j in pairs}
    return {
        "n": len(ids),
        "all_runs_identical": float(np.mean([len(set(v)) == 1 for v in per_answer])),
        **pairwise,
        "verdict_flips": int(sum({0, 1} <= set(v) for v in per_answer)),
    }
