"""Plan and run generation calls, appending one JSON row per call so runs can resume."""

from __future__ import annotations

import hashlib
import json
import threading
import time
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


def load_results(path: str | Path) -> dict[str, dict]:
    """Latest row per answer_id; later rows supersede earlier ones (retries of failed calls)."""
    path = Path(path)
    latest: dict[str, dict] = {}
    if path.exists():
        with path.open() as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    latest[row["answer_id"]] = row
    return latest


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _call(client: ChatClient, req: Request) -> dict:
    row = asdict(req)
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
    except Exception as e:  # recorded, never silently dropped; the call is retried on the next run
        row.update(status=ERROR, error=f"{type(e).__name__}: {e}"[:1000])
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
) -> dict[str, int]:
    """Make every call without an `ok` row in `out_path` and append the results."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {k for k, row in load_results(out_path).items() if row["status"] == OK}
    todo = [r for r in requests if r.answer_id not in done]
    lock = threading.Lock()
    counts = {"skipped": len(requests) - len(todo), OK: 0, ERROR: 0}

    with out_path.open("a") as f:
        def work(req: Request) -> None:
            row = _call(client, req)
            with lock:
                f.write(json.dumps(row, sort_keys=True) + "\n")
                f.flush()
                counts[row["status"]] += 1
                n = counts[OK] + counts[ERROR]
                if progress_every and n % progress_every == 0:
                    print(f"  {n}/{len(todo)} calls ({counts[ERROR]} errors)", flush=True)

        if concurrency <= 1:
            for req in todo:
                work(req)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                list(pool.map(work, todo))
    return counts
