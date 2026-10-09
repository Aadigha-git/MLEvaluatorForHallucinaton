"""Safeguards for full generation: setup-error abort, attempt limit, cache check, budget, report."""

from __future__ import annotations

import copy
import json

import pytest
import yaml

from hdetect.config import ConfigError
from hdetect.generation.client import Completion
from hdetect.generation.config import parse_generation_config
from hdetect.generation.report import complete, split_report
from hdetect.generation.runner import (
    Budget,
    BudgetExceeded,
    CacheMismatch,
    FatalApiError,
    plan_requests,
    read_rows,
    run_requests,
)
from tests.conftest import ROOT, make_example

PRICES = {"model_a": (0.10, 0.30), "model_b": (0.10, 0.30)}


@pytest.fixture
def gen():
    return parse_generation_config(yaml.safe_load((ROOT / "configs" / "generation.yaml").read_text()))


class StatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class ScriptedClient:
    """Answers with `text`; raises `error` for prompts in `fail_on`."""

    def __init__(self, text="An answer.", error=None, fail_on=None, usage=(10, 5), finish="stop"):
        self.text, self.error, self.fail_on, self.usage, self.finish = text, error, fail_on, usage, finish
        self.calls = 0

    def complete(self, *, model, messages, temperature, top_p, max_tokens, seed):
        self.calls += 1
        if self.error is not None and (self.fail_on is None or messages[0]["content"] in self.fail_on):
            raise self.error
        return Completion(self.text, self.finish, "r", model, "fp", *self.usage)


def _plan(gen, n=2):
    exs = [make_example("truthfulqa", f"Question {i}?") for i in range(n)]
    return plan_requests(exs, {e.question_id: ("train", ["model_a"]) for e in exs}, gen)


def test_run_config_is_preregistered(gen):
    assert (gen.run.max_attempts, gen.run.budget_usd) == (5, 1.0)
    for key, value in (("max_attempts", 3), ("budget_usd", 2.0)):
        raw = yaml.safe_load((ROOT / "configs" / "generation.yaml").read_text())
        raw["run"][key] = value
        with pytest.raises(ConfigError):
            parse_generation_config(copy.deepcopy(raw))


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_setup_errors_halt_and_record_nothing(tmp_path, gen, status):
    out = tmp_path / "train.jsonl"
    client = ScriptedClient(error=StatusError(status))
    with pytest.raises(FatalApiError):
        run_requests(_plan(gen), client, out, concurrency=1)
    assert client.calls == 1
    assert read_rows(out) == []


def test_transient_errors_are_recorded_and_capped_at_max_attempts(tmp_path, gen):
    out = tmp_path / "train.jsonl"
    plan = _plan(gen, n=1)
    client = ScriptedClient(error=StatusError(503))
    for _ in range(7):
        run_requests(plan, client, out, max_attempts=5)
    rows = read_rows(out)
    assert len(rows) == 6 * 5
    assert {r["attempt"] for r in rows} == {1, 2, 3, 4, 5}
    report = split_report(plan, rows, PRICES, max_attempts=5)
    assert report["status"]["primary"] == {"permanent_failure": 1}
    assert report["status"]["resample"] == {"permanent_failure": 5}
    assert complete(report)


def test_cache_mismatch_aborts_before_any_call(tmp_path, gen):
    out = tmp_path / "train.jsonl"
    plan = _plan(gen, n=1)
    run_requests(plan, ScriptedClient(), out)
    rows = read_rows(out)
    rows[0]["temperature"] = 0.5
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    client = ScriptedClient()
    with pytest.raises(CacheMismatch):
        run_requests(plan, client, out)
    assert client.calls == 0


def test_ok_rows_are_never_requested_again(tmp_path, gen):
    out = tmp_path / "train.jsonl"
    plan = _plan(gen)
    run_requests(plan, ScriptedClient(), out)
    client = ScriptedClient()
    counts = run_requests(plan, client, out)
    assert client.calls == 0 and counts["skipped"] == len(plan)


def test_budget_stops_before_exceeding_cap(tmp_path, gen):
    out = tmp_path / "train.jsonl"
    plan = _plan(gen, n=3)
    budget = Budget(cap_usd=1.0, prices=PRICES)
    per_call = budget.upper_bound(plan[0])
    budget.spent = 1.0 - 2.5 * per_call
    with pytest.raises(BudgetExceeded):
        run_requests(plan, ScriptedClient(usage=(len(plan[0].prompt), plan[0].max_tokens)), out, budget=budget)
    assert len(read_rows(out)) == 2
    assert budget.spent <= 1.0 and budget.reserved == pytest.approx(0)


def test_budget_charges_upper_bound_when_usage_missing(gen):
    budget = Budget(cap_usd=1.0, prices=PRICES)
    req = _plan(gen, n=1)[0]
    bound = budget.reserve(req)
    budget.settle(bound, req.generator, None, None)
    assert budget.spent == pytest.approx(bound) and budget.reserved == pytest.approx(0)


def test_report_counts_empty_truncated_and_missing_usage(tmp_path, gen):
    out = tmp_path / "train.jsonl"
    plan = _plan(gen, n=2)
    run_requests(plan, ScriptedClient(text=" \n\n\t", finish="length", usage=(None, None)), out)
    report = split_report(plan, read_rows(out), PRICES, max_attempts=5)
    assert report["status"]["primary"] == {"ok": 2, "empty": 2, "truncated": 2, "missing_usage": 2}
    assert report["usable_resamples_per_question"] == {"0": 2}
    assert report["cost_usd"]["complete"] is False
    assert report["latency_s"]["resamples_parallel_equivalent"]["n"] == 2


def test_report_latency_and_cost_split(tmp_path, gen):
    out = tmp_path / "train.jsonl"
    plan = _plan(gen, n=2)
    run_requests(plan, ScriptedClient(usage=(100, 10)), out)
    report = split_report(plan, read_rows(out), PRICES, max_attempts=5)
    per_call = (100 * 0.10 + 10 * 0.30) / 1e6
    assert report["cost_usd"]["primary"] == pytest.approx(2 * per_call, abs=1e-6)
    assert report["cost_usd"]["resamples_per_question_mean"] == pytest.approx(5 * per_call)
    assert report["status"]["resample"] == {"ok": 10}
    lat = report["latency_s"]
    assert lat["resamples_sequential"]["p50"] >= lat["resamples_parallel_equivalent"]["p50"]
