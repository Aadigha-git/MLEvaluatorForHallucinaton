"""Plan and run judge calls, appending one JSON row per call so runs resume and every call is counted.

Rows never hold the prompt (it contains answer text and evidence), only its SHA-256. A row whose HTTP
call succeeded is final, whatever its parse outcome: malformed output is never re-requested.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from hdetect.judging.clients import JudgeClient
from hdetect.judging.config import JudgeSpec
from hdetect.judging.parse import parse_verdict
from hdetect.judging.prompts import render

OK = "ok"
ERROR = "error"
# 4xx responses other than these are caused by the setup (key, billing, model name, request shape).
RETRYABLE_4XX = frozenset({408, 409, 429})
REQUEST_FIELDS = ("judge", "model_id", "variant", "condition", "run", "answer_id", "prompt_sha256",
                  "temperature", "top_p", "max_output_tokens", "thinking_level", "chat_template_kwargs")


class FatalApiError(RuntimeError):
    pass


class CacheMismatch(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class JudgeItem:
    """What the judge may see about one answer: the classifier's inputs, nothing about its label."""

    answer_id: str
    question_id: str
    question: str
    answer: str
    passages: tuple[str, ...]
    retrieval_s: float


@dataclass(frozen=True)
class JudgeRequest:
    request_id: str
    judge: str
    model_id: str
    variant: str
    condition: str
    run: int
    answer_id: str
    question_id: str
    prompt: str
    prompt_sha256: str
    retrieval_s: float
    temperature: float
    top_p: float | None
    max_output_tokens: int
    thinking_level: str | None
    chat_template_kwargs: dict


def request_id(judge: str, variant: str, condition: str, run: int, answer_id: str) -> str:
    return f"{judge}/{variant}/{condition}/run{run}/{answer_id}"


def plan(items: Iterable[JudgeItem], spec: JudgeSpec, variants: Sequence[str], conditions: Sequence[str],
         run: int = 0) -> list[JudgeRequest]:
    """Retrieval time counts toward judge latency only when the judge gets the retrieved passages."""
    out = []
    for item in items:
        for variant in variants:
            for condition in conditions:
                passages = list(item.passages) if condition == "retrieved" else None
                prompt = render(variant, item.question, item.answer, passages)
                out.append(JudgeRequest(
                    request_id=request_id(spec.name, variant, condition, run, item.answer_id),
                    judge=spec.name, model_id=spec.model_id, variant=variant, condition=condition, run=run,
                    answer_id=item.answer_id, question_id=item.question_id, prompt=prompt,
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    retrieval_s=item.retrieval_s if condition == "retrieved" else 0.0,
                    temperature=spec.temperature, top_p=spec.top_p, max_output_tokens=spec.max_output_tokens,
                    thinking_level=spec.thinking_level, chat_template_kwargs=dict(spec.chat_template_kwargs),
                ))
    ids = [r.request_id for r in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate request ids in plan")
    return out


def read_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def spent_usd(rows: Iterable[dict]) -> float:
    return sum(r.get("charged_usd", 0.0) for r in rows)


def check_cache(requests: Iterable[JudgeRequest], rows: Iterable[dict]) -> None:
    planned = {r.request_id: r for r in requests}
    for row in rows:
        req = planned.get(row["request_id"])
        if req is None:
            continue
        diff = [f for f in REQUEST_FIELDS if row.get(f) != getattr(req, f)]
        if diff:
            raise CacheMismatch(f"{row['request_id']}: stored {diff} differ from the plan")


class Budget:
    """Refuses a call when spend so far, plus reserved upper bounds of calls in flight, plus this call's
    upper bound would exceed the cap. Upper bound: prompt characters as input tokens (never fewer than
    the tokenizer count) plus the maximum output tokens, priced at the snapshot."""

    def __init__(self, cap_usd: float, specs: Mapping[str, JudgeSpec], spent: float = 0.0):
        self.cap = cap_usd
        self.specs = dict(specs)
        self.spent = spent
        self.reserved = 0.0
        self._lock = threading.Lock()

    def upper_bound(self, judge: str, prompt: str) -> float:
        spec = self.specs[judge]
        if spec.max_prompt_tokens_at_price is not None and len(prompt) > spec.max_prompt_tokens_at_price:
            raise BudgetExceeded(f"{judge}: prompt may exceed the {spec.max_prompt_tokens_at_price}-token price tier")
        return spec.cost_usd(len(prompt), spec.max_output_tokens)

    def reserve(self, judge: str, prompt: str) -> float:
        bound = self.upper_bound(judge, prompt)
        with self._lock:
            if self.spent + self.reserved + bound > self.cap:
                raise BudgetExceeded(f"spent ${self.spent:.4f} of the ${self.cap:.2f} cap; next call may cost "
                                     f"up to ${bound:.4f}")
            self.reserved += bound
        return bound

    def settle(self, bound: float, charged: float) -> None:
        with self._lock:
            self.reserved -= bound
            self.spent += charged


class RateLimiter:
    """Spaces call starts at least 60 / `per_minute` seconds apart across threads. The wait happens
    before the latency clock starts, so it is recorded as `throttle_wait_s`, never as latency."""

    def __init__(self, per_minute: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.interval = 60.0 / per_minute
        self._clock, self._sleep = clock, sleep
        self._next = float("-inf")
        self._lock = threading.Lock()

    def wait(self) -> float:
        with self._lock:
            now = self._clock()
            start = max(now, self._next)
            self._next = start + self.interval
        delay = start - now
        if delay > 0:
            self._sleep(delay)
        return delay


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _fatal(e: Exception) -> bool:
    code = getattr(e, "status_code", None)
    return isinstance(code, int) and 400 <= code < 500 and code not in RETRYABLE_4XX


def call(client: JudgeClient, spec: JudgeSpec, req: JudgeRequest, attempt: int, bound: float,
         throttle_wait_s: float = 0.0) -> dict:
    row = {k: v for k, v in asdict(req).items() if k != "prompt"}
    row.update(attempt=attempt, started_at=_now(), throttle_wait_s=round(throttle_wait_s, 4))
    t0 = time.perf_counter()
    try:
        c = client.complete(req.prompt)
    except Exception as e:
        if _fatal(e):
            raise FatalApiError(f"{type(e).__name__}: {e}"[:1000]) from e
        row.update(status=ERROR, error=f"{type(e).__name__}: {e}"[:1000], api_latency_s=round(time.perf_counter() - t0, 4),
                   transient_retries=getattr(e, "transient_retries", None), retry_wait_s=getattr(e, "retry_wait_s", None),
                   cost_usd=None, charged_usd=0.0, finished_at=_now())
        return row
    api_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    parsed = parse_verdict(c.text, blocked=c.blocked, reasoning=c.reasoning)
    parse_s = time.perf_counter() - t1
    known = c.input_tokens is not None and c.output_tokens is not None
    cost = spec.cost_usd(c.input_tokens, c.output_tokens) if known else None
    row.update(
        status=OK, error=None, text=c.text, reasoning_returned=bool(c.reasoning and c.reasoning.strip()),
        finish_reason=c.finish_reason, blocked=c.blocked, response_id=c.response_id, model_returned=c.model,
        input_tokens=c.input_tokens, output_tokens=c.output_tokens, thinking_tokens=c.thinking_tokens,
        transient_retries=c.transient_retries, retry_wait_s=c.retry_wait_s, parse_outcome=parsed.outcome, verdict=parsed.verdict,
        cost_usd=cost, charged_usd=cost if known else bound,
        api_latency_s=round(api_s, 4), parse_s=round(parse_s, 6),
        latency_s=round(req.retrieval_s + api_s + parse_s, 4), finished_at=_now(),
    )
    return row


def run_requests(requests: list[JudgeRequest], clients: Mapping[str, JudgeClient], specs: Mapping[str, JudgeSpec],
                 out_path: str | Path, budget: Budget, *, concurrency: int = 1, max_attempts: int = 5,
                 progress_every: int = 0, limiters: Mapping[str, RateLimiter] | None = None) -> dict[str, int]:
    """Make every planned call without an `ok` row (and with attempts left), appending results.

    A setup error or a budget stop halts the run: no further calls start, calls in flight finish and
    are recorded, then the exception is raised. Nothing is recorded for the call that stopped it.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_rows(out_path)
    check_cache(requests, rows)
    done = {r["request_id"] for r in rows if r["status"] == OK}
    tried = Counter(r["request_id"] for r in rows)
    todo = [r for r in requests if r.request_id not in done and tried[r.request_id] < max_attempts]
    counts = {"skipped": len(requests) - len(todo), OK: 0, ERROR: 0, "not_started": 0}
    lock, stop, halted = threading.Lock(), threading.Event(), []
    if not todo:
        return counts

    with out_path.open("a") as f:
        def work(req: JudgeRequest) -> None:
            if stop.is_set():
                with lock:
                    counts["not_started"] += 1
                return
            bound = 0.0
            limiter = (limiters or {}).get(req.judge)
            try:
                waited = limiter.wait() if limiter else 0.0
                if stop.is_set():
                    with lock:
                        counts["not_started"] += 1
                    return
                bound = budget.reserve(req.judge, req.prompt)
                row = call(clients[req.judge], specs[req.judge], req, tried[req.request_id] + 1, bound, waited)
            except (FatalApiError, BudgetExceeded) as e:
                budget.settle(bound, 0.0)
                with lock:
                    counts["not_started"] += 1
                    if not stop.is_set():
                        halted.append(e)
                    stop.set()
                return
            budget.settle(bound, row["charged_usd"])
            with lock:
                f.write(json.dumps(row, sort_keys=True) + "\n")
                f.flush()
                counts[row["status"]] += 1
                n = counts[OK] + counts[ERROR]
                if progress_every and n % progress_every == 0:
                    print(f"  {out_path.name}: {n}/{len(todo)} calls, ${budget.spent:.2f} spent", flush=True)

        if concurrency <= 1:
            for req in todo:
                work(req)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                list(pool.map(work, todo))
    if halted:
        raise halted[0]
    return counts
