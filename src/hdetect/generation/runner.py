"""Plan and run generation calls, appending one JSON row per call so runs can resume."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from hdetect.generation.client import ChatClient
from hdetect.generation.config import GenerationConfig
from hdetect.generation.prompts import messages, render
from hdetect.schema import Example

OK = "ok"
ERROR = "error"

# HTTP statuses caused by the setup (key, permissions, model name, request shape), not the question.
FATAL_STATUSES = frozenset({400, 401, 403, 404})
REQUEST_FIELDS = ("model_id", "prompt", "temperature", "top_p", "max_tokens", "seed", "prompt_version")


class FatalApiError(RuntimeError):
    pass


class CacheMismatch(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


def sample_seed(question_id: str, sample_index: int) -> int:
    return int(hashlib.sha256(f"{question_id}/{sample_index}".encode()).hexdigest()[:8], 16)


def answer_id(question_id: str, generator: str, sample_index: int, repeat: int = 0) -> str:
    base = f"{question_id}/{generator}/{sample_index}"
    return f"{base}/repeat{repeat}" if repeat else base


@dataclass(frozen=True)
class Request:
    answer_id: str
    question_id: str
    dataset: str
    split: str | None
    generator: str
    model_id: str
    sample_index: int
    repeat: int
    prompt_version: str
    prompt: str
    temperature: float
    top_p: float
    max_tokens: int
    seed: int | None


def plan_requests(
    examples: Iterable[Example],
    targets: Mapping[str, tuple[str | None, list[str]]],
    cfg: GenerationConfig,
    *,
    resamples: bool = True,
    repeat_primary: bool = False,
) -> list[Request]:
    """`targets` maps question_id to (split, generators). Questions not in `targets` are skipped."""
    out = []
    for ex in examples:
        if ex.question_id not in targets:
            continue
        split, generators = targets[ex.question_id]
        version = cfg.prompts[ex.dataset]
        prompt = render(version, ex)
        calls = [(0, 0)]
        if repeat_primary:
            calls.append((0, 1))
        if resamples:
            calls += [(i, 0) for i in range(1, cfg.resample_n + 1)]
        for generator in generators:
            spec = cfg.generators[generator]
            for sample_index, repeat in calls:
                sampling = cfg.primary if sample_index == 0 else cfg.resample
                out.append(Request(
                    answer_id=answer_id(ex.question_id, generator, sample_index, repeat),
                    question_id=ex.question_id,
                    dataset=ex.dataset,
                    split=split,
                    generator=generator,
                    model_id=spec.model_id,
                    sample_index=sample_index,
                    repeat=repeat,
                    prompt_version=version,
                    prompt=prompt,
                    temperature=sampling.temperature,
                    top_p=sampling.top_p,
                    max_tokens=cfg.max_tokens[version],
                    seed=sample_seed(ex.question_id, sample_index) if cfg.api.send_seed else None,
                ))
    ids = [r.answer_id for r in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate answer ids in plan")
    return out


def read_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_results(path: str | Path) -> dict[str, dict]:
    """The `ok` row per answer_id if there is one, otherwise the latest error row."""
    latest: dict[str, dict] = {}
    for row in read_rows(path):
        if latest.get(row["answer_id"], {}).get("status") != OK:
            latest[row["answer_id"]] = row
    return latest


def attempts(rows: Iterable[dict]) -> Counter:
    return Counter(row["answer_id"] for row in rows)


def check_cache(requests: Iterable[Request], rows: Iterable[dict]) -> None:
    """Raise CacheMismatch if any stored row was made with different request parameters."""
    planned = {r.answer_id: r for r in requests}
    for row in rows:
        req = planned.get(row["answer_id"])
        if req is None:
            raise CacheMismatch(f"{row['answer_id']} is stored but not planned")
        diff = [f for f in REQUEST_FIELDS if row.get(f) != getattr(req, f)]
        if diff:
            raise CacheMismatch(f"{row['answer_id']}: stored {diff} differ from the plan")


class Budget:
    """Refuses a call when spend so far plus reserved upper bounds would exceed the cap."""

    def __init__(self, cap_usd: float, prices: Mapping[str, tuple[float, float]], spent_usd: float = 0.0):
        """`prices` maps generator to USD per million (input, output) tokens."""
        self.cap = cap_usd
        self.prices = dict(prices)
        self.spent = spent_usd
        self.reserved = 0.0
        self._lock = threading.Lock()

    def cost(self, generator: str, prompt_tokens: int, completion_tokens: int) -> float:
        usd_in, usd_out = self.prices[generator]
        return (prompt_tokens * usd_in + completion_tokens * usd_out) / 1e6

    def upper_bound(self, req: Request) -> float:
        """Prompt characters stand in for input tokens (docs/PREREGISTRATION.md, "Budget guard")."""
        return self.cost(req.generator, len(req.prompt), req.max_tokens)

    def reserve(self, req: Request) -> float:
        bound = self.upper_bound(req)
        with self._lock:
            if self.spent + self.reserved + bound > self.cap:
                raise BudgetExceeded(f"spent ${self.spent:.4f} of the ${self.cap:.2f} cap")
            self.reserved += bound
        return bound

    def settle(
        self, bound: float, generator: str, prompt_tokens: int | None = 0, completion_tokens: int | None = 0
    ) -> None:
        """A call with unreported usage is charged its upper bound."""
        if prompt_tokens is None or completion_tokens is None:
            cost = bound
        else:
            cost = self.cost(generator, prompt_tokens, completion_tokens)
        with self._lock:
            self.reserved -= bound
            self.spent += cost


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _call(client: ChatClient, req: Request, attempt: int) -> dict:
    row = asdict(req)
    row["attempt"] = attempt
    row["started_at"] = _now()
    t0 = time.perf_counter()
    try:
        c = client.complete(
            model=req.model_id,
            messages=messages(req.prompt),
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            seed=req.seed,
        )
    except Exception as e:
        if getattr(e, "status_code", None) in FATAL_STATUSES:
            raise FatalApiError(f"{type(e).__name__}: {e}"[:1000]) from e
        row.update(status=ERROR, error=f"{type(e).__name__}: {e}"[:1000], prompt_tokens=0, completion_tokens=0)
    else:
        row.update(status=OK, error=None, **asdict(c))
    row["latency_s"] = round(time.perf_counter() - t0, 4)
    row["finished_at"] = _now()
    return row


def run_requests(
    requests: list[Request],
    client: ChatClient,
    out_path: str | Path,
    *,
    concurrency: int = 1,
    progress_every: int = 0,
    max_attempts: int | None = None,
    budget: Budget | None = None,
) -> dict[str, int]:
    """Make every call without an `ok` row in `out_path` (and with attempts left), appending results.

    Setup errors (FatalApiError) and budget stops halt the run: no further calls start, calls in
    flight finish and are recorded, then the exception is raised. Nothing is recorded for the
    call that hit the setup error.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_rows(out_path)
    check_cache(requests, rows)
    done = {r["answer_id"] for r in rows if r["status"] == OK}
    tried = attempts(rows)
    pending = [r for r in requests if r.answer_id not in done]
    todo = [r for r in pending if max_attempts is None or tried[r.answer_id] < max_attempts]
    lock = threading.Lock()
    stop = threading.Event()
    halted: list[BaseException] = []
    counts = {"skipped": len(requests) - len(todo), OK: 0, ERROR: 0, "not_started": 0}
    if not todo:
        return counts

    with out_path.open("a") as f:
        def work(req: Request) -> None:
            if stop.is_set():
                with lock:
                    counts["not_started"] += 1
                return
            bound = 0.0
            try:
                if budget is not None:
                    bound = budget.reserve(req)
                row = _call(client, req, tried[req.answer_id] + 1)
            except (FatalApiError, BudgetExceeded) as e:
                if budget is not None and bound:
                    budget.settle(bound, req.generator)
                with lock:
                    counts["not_started"] += 1
                    if not stop.is_set():
                        halted.append(e)
                    stop.set()
                return
            if budget is not None:
                budget.settle(bound, req.generator, row["prompt_tokens"], row["completion_tokens"])
            with lock:
                f.write(json.dumps(row, sort_keys=True) + "\n")
                f.flush()
                counts[row["status"]] += 1
                n = counts[OK] + counts[ERROR]
                if progress_every and n % progress_every == 0:
                    print(f"  {out_path.name}: {n}/{len(todo)} calls ({counts[ERROR]} errors)", flush=True)

        if concurrency <= 1:
            for req in todo:
                work(req)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                list(pool.map(work, todo))
    if halted:
        raise halted[0]
    return counts
