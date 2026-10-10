"""Step 8 judges: config, prompts, strict parsing, clients, cost guard, runner, scoring, isolation. No network."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml

from hdetect.config import ConfigError
from hdetect.features import TEST_SPLITS
from hdetect.features.access import SplitAccessError
from hdetect.judging.clients import (
    ApiStatusError,
    GeminiJudgeClient,
    JudgeCompletion,
    NebiusJudgeClient,
    parse_gemini_response,
)
from hdetect.judging.config import load_judge_config, parse_judge_config
from hdetect.judging.inputs import load_judge_inputs
from hdetect.judging.parse import parse_verdict
from hdetect.judging.prompts import VARIANTS, render, template_hash
from hdetect.judging.runner import (
    Budget,
    BudgetExceeded,
    CacheMismatch,
    FatalApiError,
    JudgeItem,
    plan,
    read_rows,
    run_requests,
    spent_usd,
)
from hdetect.judging.scoring import agreement, f1_counts, repeat_subset, score_cell, select_variant

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "judges.yaml"
PASSAGES = tuple(f"Title {i}: passage text {i}" for i in range(1, 6))


def raw_config() -> dict:
    return yaml.safe_load(CONFIG.read_text())


@pytest.fixture
def cfg():
    return load_judge_config(CONFIG)


def item(i: int = 0) -> JudgeItem:
    return JudgeItem(f"q{i}/model_a/0", f"q{i}", f"Question {i}?", f"Answer {i}.", PASSAGES, retrieval_s=0.25)


# Config

def test_config_matches_the_amendment(cfg):
    assert cfg.status == "frozen"
    assert cfg.cap_usd == 35.0
    small, frontier = cfg.judges["small"], cfg.judges["frontier"]
    assert (small.provider, small.model_id, small.api_key_env) == (
        "nebius", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", "NEBIUS_API_KEY")
    assert (small.usd_per_million_input_tokens, small.usd_per_million_output_tokens) == (0.06, 0.24)
    assert (small.temperature, small.chat_template_kwargs) == (0.0, {"enable_thinking": False})
    assert (frontier.provider, frontier.model_id, frontier.api_key_env) == (
        "gemini", "gemini-3.1-pro-preview", "GEMINI_API_KEY")
    assert (frontier.usd_per_million_input_tokens, frontier.usd_per_million_output_tokens) == (2.0, 12.0)
    assert (frontier.temperature, frontier.thinking_level) == (1.0, "low")
    assert cfg.variants == tuple(VARIANTS) and cfg.conditions == ("retrieved", "none")
    assert (cfg.repeat_subset_size, cfg.repeat_extra_runs) == (200, 2)


@pytest.mark.parametrize("edit", [
    lambda r: r["judges"]["small"].update(temperature=0.7),
    lambda r: r["judges"]["small"].update(chat_template_kwargs={}),
    lambda r: r["judges"]["frontier"].update(temperature=0.0),
    lambda r: r["judges"]["frontier"].update(thinking_level="high"),
    lambda r: r["judges"]["small"].update(model_id="Qwen/Qwen3-30B-A3B-Instruct-2507"),
    lambda r: r["judges"]["frontier"].update(model_id="google/gemma-3-27b-it"),
    lambda r: r["prompts"].update(variants=["judge-v1a", "judge-v1b"]),
    lambda r: r["prompts"].update(conditions=["retrieved", "none", "gold"]),
    lambda r: r["selection"].update(metric="auroc"),
    lambda r: r["repeatability"].update(subset_size=100),
    lambda r: r.update(status="final"),
])
def test_config_rejects_changes_to_the_frozen_design(edit):
    raw = copy.deepcopy(raw_config())
    edit(raw)
    with pytest.raises(ConfigError):
        parse_judge_config(raw)


# Prompts

def test_variants_are_distinct_and_carry_the_classifier_inputs():
    retrieved = {v: render(v, "Q?", "A.", list(PASSAGES)) for v in VARIANTS}
    none = {v: render(v, "Q?", "A.", None) for v in VARIANTS}
    assert len(set(retrieved.values())) == len(VARIANTS) == 4
    for v in VARIANTS:
        assert "Question: Q?\nAnswer: A." in retrieved[v]
        assert all(f"[{i}] {p}" in retrieved[v] for i, p in enumerate(PASSAGES, 1))
        assert "Evidence" not in none[v] and "Title 1" not in none[v]
        assert retrieved[v].endswith('{"verdict": "not_hallucinated"}.')


def test_template_hashes_are_pinned():
    """Editing a variant in place breaks this test; add a new version instead."""
    pinned = json.loads((ROOT / "configs" / "judge_prompt_hashes.json").read_text())
    assert {v: template_hash(v) for v in VARIANTS} == pinned


# Parsing

@pytest.mark.parametrize("text,verdict", [
    ('{"verdict": "hallucinated"}', "hallucinated"),
    ('  {"verdict":"not_hallucinated"}\n', "not_hallucinated"),
    ('```json\n{"verdict": "hallucinated"}\n```', "hallucinated"),
    ('```\n{"verdict": "not_hallucinated"}\n```', "not_hallucinated"),
])
def test_parse_accepts_one_strict_verdict(text, verdict):
    p = parse_verdict(text)
    assert p.ok and p.verdict == verdict


@pytest.mark.parametrize("text,outcome", [
    ("", "empty"),
    ("   ", "empty"),
    ("hallucinated", "not_json"),
    ('Verdict: {"verdict": "hallucinated"}', "not_json"),
    ('{"verdict": "hallucinated"} because the date is wrong', "not_json"),
    ('{"verdict": "hallucinated"}{"verdict": "hallucinated"}', "not_json"),
    ('{"verdict": "hallucinated"', "not_json"),
    ('{"verdict": "hallucinated", "reason": "x"}', "wrong_schema"),
    ('{"verdict": "hallucinated", "verdict": "not_hallucinated"}', "wrong_schema"),
    ('["hallucinated"]', "wrong_schema"),
    ('{"answer": "hallucinated"}', "wrong_schema"),
    ('{"verdict": "Hallucinated"}', "invalid_verdict"),
    ('{"verdict": "unsure"}', "invalid_verdict"),
    ('{"verdict": true}', "invalid_verdict"),
    ('<think>hmm</think>{"verdict": "hallucinated"}', "reasoning_leak"),
])
def test_parse_failures_are_named_not_repaired(text, outcome):
    p = parse_verdict(text)
    assert not p.ok and p.verdict is None and p.outcome == outcome


def test_parse_blocked_and_separate_reasoning_are_failures():
    assert parse_verdict('{"verdict": "hallucinated"}', blocked=True).outcome == "blocked"
    assert parse_verdict('{"verdict": "hallucinated"}', reasoning="let me think").outcome == "reasoning_leak"
    assert parse_verdict('{"verdict": "hallucinated"}', reasoning="  ").ok


# Clients

class FakeSdk:
    def __init__(self, message=None, usage=None, finish="stop"):
        self.sent = []
        msg = message or SimpleNamespace(content='{"verdict": "hallucinated"}', reasoning_content=None, model_extra={})
        usage = usage or SimpleNamespace(prompt_tokens=100, completion_tokens=8, completion_tokens_details=None)
        resp = SimpleNamespace(id="r1", model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", usage=usage,
                               choices=[SimpleNamespace(message=msg, finish_reason=finish)])
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._resp = resp

    def _create(self, **kwargs):
        self.sent.append(kwargs)
        return self._resp


def test_nebius_request_disables_reasoning_at_temperature_zero_without_seed(cfg):
    sdk = FakeSdk()
    c = NebiusJudgeClient(cfg.judges["small"], cfg, sdk_client=sdk).complete("prompt")
    sent = sdk.sent[0]
    assert sent["temperature"] == 0.0 and sent["max_tokens"] == 64 and sent["top_p"] == 1.0
    assert sent["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "seed" not in sent and "logprobs" not in sent
    assert sent["messages"] == [{"role": "user", "content": "prompt"}]
    assert (c.input_tokens, c.output_tokens, c.reasoning, c.blocked) == (100, 8, None, False)


def test_nebius_reports_returned_reasoning(cfg):
    msg = SimpleNamespace(content='{"verdict": "hallucinated"}', reasoning_content="thinking...", model_extra={})
    c = NebiusJudgeClient(cfg.judges["small"], cfg, sdk_client=FakeSdk(message=msg)).complete("p")
    assert parse_verdict(c.text, reasoning=c.reasoning).outcome == "reasoning_leak"


GEMINI_OK = {
    "candidates": [{"content": {"parts": [{"text": "thought summary", "thought": True},
                                          {"text": '{"verdict": "not_hallucinated"}'}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 900, "candidatesTokenCount": 9, "thoughtsTokenCount": 150,
                      "totalTokenCount": 1059},
    "modelVersion": "gemini-3.1-pro-preview", "responseId": "abc",
}


def gemini(cfg, responses, sleeps=None):
    sent = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        status, body = queue.pop(0)
        return httpx.Response(status, json=body)

    client = GeminiJudgeClient(cfg.judges["frontier"], cfg, transport=httpx.MockTransport(handler), api_key="test-key",
                               sleep=(sleeps.append if sleeps is not None else lambda s: None))
    return client, sent


def test_gemini_request_settings_and_billed_output_includes_thinking(cfg):
    client, sent = gemini(cfg, [(200, GEMINI_OK)])
    c = client.complete("prompt")
    body = json.loads(sent[0].content)
    assert sent[0].url.path.endswith("/models/gemini-3.1-pro-preview:generateContent")
    assert sent[0].headers["x-goog-api-key"] == "test-key"
    assert body["generationConfig"] == {"temperature": 1.0, "maxOutputTokens": 2048,
                                        "thinkingConfig": {"thinkingLevel": "low"}}
    assert "seed" not in body["generationConfig"] and "responseLogprobs" not in body["generationConfig"]
    assert c.text == '{"verdict": "not_hallucinated"}'
    assert (c.input_tokens, c.output_tokens, c.thinking_tokens) == (900, 159, 150)
    assert c.model == "gemini-3.1-pro-preview" and not c.blocked


def test_gemini_retries_only_transient_errors(cfg):
    sleeps = []
    client, sent = gemini(cfg, [(503, {}), (429, {}), (200, GEMINI_OK)], sleeps)
    assert client.complete("p").output_tokens == 159
    assert len(sent) == 3 and len(sleeps) == 2

    client, sent = gemini(cfg, [(400, {"error": "bad request"})])
    with pytest.raises(ApiStatusError) as e:
        client.complete("p")
    assert e.value.status_code == 400 and len(sent) == 1

    client, sent = gemini(cfg, [(503, {})] * 6)
    with pytest.raises(ApiStatusError) as e:
        client.complete("p")
    assert e.value.status_code == 503 and len(sent) == cfg.max_retries + 1


def test_gemini_blocked_and_missing_usage():
    c = parse_gemini_response({"candidates": [{"finishReason": "SAFETY"}]})
    assert c.blocked and c.text == "" and c.input_tokens is None and c.output_tokens is None
    c = parse_gemini_response({"promptFeedback": {"blockReason": "OTHER"}, "usageMetadata": {"promptTokenCount": 5}})
    assert c.blocked and (c.input_tokens, c.output_tokens, c.thinking_tokens) == (5, 0, 0)


# Runner and cost guard

class FakeClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        out = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        if isinstance(out, Exception):
            raise out
        return out


def completion(text='{"verdict": "hallucinated"}', input_tokens=1000, output_tokens=10):
    return JudgeCompletion(text, "stop", False, None, "id", "m", input_tokens, output_tokens, None)


def test_parse_failure_is_final_and_rows_hold_no_prompt(cfg, tmp_path):
    spec = cfg.judges["small"]
    reqs = plan([item()], spec, ["judge-v1a"], ["retrieved", "none"])
    client = FakeClient([completion("Sure! hallucinated")])
    out = tmp_path / "small.jsonl"
    budget = Budget(20.0, cfg.judges)
    assert run_requests(reqs, {"small": client}, cfg.judges, out, budget)["ok"] == 2
    assert run_requests(reqs, {"small": client}, cfg.judges, out, budget)["skipped"] == 2
    rows = read_rows(out)
    assert client.calls == 2 and {r["parse_outcome"] for r in rows} == {"not_json"}
    assert all("prompt" not in r and "Answer 0." not in json.dumps(r) for r in rows)
    lat = {r["condition"]: (r["latency_s"], r["api_latency_s"], r["retrieval_s"]) for r in rows}
    assert lat["retrieved"][2] == 0.25 and lat["none"][2] == 0.0
    assert lat["retrieved"][0] >= 0.25


def test_api_errors_are_retried_across_passes_up_to_max_attempts(cfg, tmp_path):
    spec = cfg.judges["small"]
    reqs = plan([item()], spec, ["judge-v1a"], ["none"])
    client = FakeClient([TimeoutError("timed out")])
    out = tmp_path / "small.jsonl"
    for _ in range(7):
        run_requests(reqs, {"small": client}, cfg.judges, out, Budget(20.0, cfg.judges), max_attempts=5)
    rows = read_rows(out)
    assert client.calls == 5 and [r["status"] for r in rows] == ["error"] * 5
    assert all(r["charged_usd"] == 0.0 and r["cost_usd"] is None for r in rows)


def test_setup_errors_stop_the_run_without_a_row(cfg, tmp_path):
    spec = cfg.judges["small"]
    reqs = plan([item(0), item(1)], spec, ["judge-v1a"], ["none"])
    client = FakeClient([ApiStatusError(401, "unauthorized")])
    with pytest.raises(FatalApiError):
        run_requests(reqs, {"small": client}, cfg.judges, tmp_path / "s.jsonl", Budget(20.0, cfg.judges))
    assert read_rows(tmp_path / "s.jsonl") == []


def test_budget_refuses_a_call_whose_upper_bound_would_exceed_the_cap(cfg, tmp_path):
    spec = cfg.judges["frontier"]
    reqs = plan([item(0), item(1)], spec, ["judge-v1b"], ["retrieved"])
    bound = Budget(20.0, cfg.judges).upper_bound("frontier", reqs[0].prompt)
    assert bound == pytest.approx((len(reqs[0].prompt) * 2.0 + 2048 * 12.0) / 1e6)
    client = FakeClient([completion(input_tokens=len(reqs[0].prompt), output_tokens=2048)])
    budget = Budget(20.0, cfg.judges, spent=20.0 - bound * 1.5)
    with pytest.raises(BudgetExceeded):
        run_requests(reqs, {"frontier": client}, cfg.judges, tmp_path / "f.jsonl", budget)
    rows = read_rows(tmp_path / "f.jsonl")
    assert client.calls == 1 and len(rows) == 1
    assert rows[0]["cost_usd"] == pytest.approx(bound)
    assert budget.spent <= 20.0 and budget.reserved == pytest.approx(0.0)


def test_unreported_usage_is_charged_its_upper_bound_and_cost_left_unknown(cfg, tmp_path):
    spec = cfg.judges["small"]
    reqs = plan([item()], spec, ["judge-v1a"], ["none"])
    out = tmp_path / "s.jsonl"
    run_requests(reqs, {"small": FakeClient([completion(input_tokens=None, output_tokens=None)])}, cfg.judges, out,
                 Budget(20.0, cfg.judges))
    row = read_rows(out)[0]
    assert row["cost_usd"] is None
    assert row["charged_usd"] == pytest.approx(Budget(20.0, cfg.judges).upper_bound("small", reqs[0].prompt))
    assert spent_usd(read_rows(out)) == row["charged_usd"]


def test_cache_mismatch_aborts_before_any_call(cfg, tmp_path):
    spec = cfg.judges["small"]
    out = tmp_path / "s.jsonl"
    reqs = plan([item()], spec, ["judge-v1a"], ["none"])
    run_requests(reqs, {"small": FakeClient([completion()])}, cfg.judges, out, Budget(20.0, cfg.judges))
    changed = plan([JudgeItem("q0/model_a/0", "q0", "Question 0?", "Edited.", PASSAGES, 0.25)], spec,
                   ["judge-v1a"], ["none"])
    client = FakeClient([completion()])
    with pytest.raises(CacheMismatch):
        run_requests(changed, {"small": client}, cfg.judges, out, Budget(20.0, cfg.judges))
    assert client.calls == 0


# Scoring

def row(answer_id, verdict=None, outcome="ok", status="ok"):
    return {"request_id": answer_id, "answer_id": answer_id, "status": status, "parse_outcome": outcome,
            "verdict": verdict, "cost_usd": 0.001, "input_tokens": 10, "output_tokens": 2, "latency_s": 1.0}


def test_failures_are_scored_as_wrong_and_reported():
    labels = {"a": 1, "b": 0, "c": 1, "d": 0}
    rows = {"a": row("a", "hallucinated"), "b": row("b", None, "not_json"), "c": row("c", None, "empty"),
            "d": row("d", status="error")}
    s = score_cell(rows, labels, ["a", "b", "c", "d"])
    assert (s["tp"], s["fp"], s["fn"]) == (1, 2, 1)
    assert s["parse_failures"] == 2 and s["api_failures"] == 1 and s["verdicts"] == 1
    assert s["f1"] == pytest.approx(2 / 5)
    assert s["f1_parsed_only"] == 1.0 and s["n_parsed"] == 1
    assert not s["cost_complete"]


def test_f1_counts_matches_definition():
    assert f1_counts([1, 1, 0, 0], [1, 0, 1, 0])["f1"] == pytest.approx(0.5)


def test_selection_tie_breaks():
    order = ["judge-v1a", "judge-v1b", "judge-v1c", "judge-v1d"]
    base = {"parse_failures": 0, "api_failures": 0}
    scores = {"judge-v1a": {"f1": 0.7, **base}, "judge-v1b": {"f1": 0.8, "parse_failures": 3, "api_failures": 0},
              "judge-v1c": {"f1": 0.8, **base}, "judge-v1d": {"f1": 0.8, **base}}
    assert select_variant(scores, order) == "judge-v1c"
    scores["judge-v1a"]["f1"] = 0.9
    assert select_variant(scores, order) == "judge-v1a"


def test_repeat_subset_is_seeded_and_order_free():
    ids = [f"q{i}/model_a/0" for i in range(995)]
    a = repeat_subset(ids, 200, 20261012)
    assert a == repeat_subset(reversed(ids), 200, 20261012) and len(set(a)) == 200
    assert a != repeat_subset(ids, 200, 1)


def test_agreement_counts_failures_as_disagreement():
    runs = [{"a": row("a", "hallucinated"), "b": row("b", "hallucinated")},
            {"a": row("a", "hallucinated"), "b": row("b", None, "not_json")},
            {"a": row("a", "hallucinated"), "b": row("b", "not_hallucinated")}]
    out = agreement(runs, ["a", "b"])
    assert out["all_runs_identical"] == 0.5 and out["agree_run0_run1"] == 0.5 and out["verdict_flips"] == 1


def test_frozen_repeat_subset_matches_the_seeded_draw():
    path = ROOT / "configs" / "judge_repeat_subset.txt"
    ids = [line for line in path.read_text().splitlines() if line]
    assert len(ids) == len(set(ids)) == 200 and ids == sorted(ids)
    assert all(i.endswith("/model_a/0") for i in ids)


# Isolation

JUDGE_SOURCES = sorted((ROOT / "src" / "hdetect" / "judging").glob("*.py")) + [
    ROOT / "scripts" / "run_judges.py", ROOT / "scripts" / "judge_smoke.py"]


@pytest.mark.parametrize("path", JUDGE_SOURCES, ids=lambda p: p.name)
def test_judge_code_names_no_test_split_and_never_authorizes_one(path):
    tree = ast.parse(path.read_text())
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not any(s in TEST_SPLITS for s in strings), f"{path.name} names a test split"
    kws = {kw.arg for n in ast.walk(tree) if isinstance(n, ast.Call) for kw in n.keywords}
    assert "allow_test_splits" not in kws
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not any(m.startswith("ml") for m in mods)


@pytest.mark.parametrize("split", sorted(TEST_SPLITS) + ["train"])
def test_judge_inputs_refuse_everything_but_val_id(split):
    with pytest.raises(SplitAccessError):
        load_judge_inputs(None, None, {}, split=split)


def test_no_judge_source_reads_an_api_key_value_into_output():
    for path in JUDGE_SOURCES:
        text = path.read_text()
        assert "print(key" not in text and "print(os.environ" not in text
