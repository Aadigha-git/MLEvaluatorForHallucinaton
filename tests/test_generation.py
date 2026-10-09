from __future__ import annotations

import copy
import json

import pytest
import yaml

from hdetect.config import ConfigError
from hdetect.data.splits import Assignment, SplitManifest
from hdetect.generation.client import Completion, MissingApiKey, OpenAICompatibleClient
from hdetect.generation.config import parse_generation_config
from hdetect.generation.pilot import PilotSelectionError, format_report, select_pilot
from hdetect.generation.prompts import messages, render
from hdetect.generation.runner import answer_id, load_results, plan_requests, run_requests, sample_seed
from tests.conftest import ROOT, make_example


@pytest.fixture(scope="module")
def raw_gen():
    return yaml.safe_load((ROOT / "configs" / "generation.yaml").read_text())


@pytest.fixture
def gen(raw_gen):
    return parse_generation_config(copy.deepcopy(raw_gen))


class FakeClient:
    def __init__(self, fail_on: set[str] = frozenset()):
        self.calls = []
        self.fail_on = fail_on

    def complete(self, *, model, messages, temperature, top_p, max_tokens, seed):
        self.calls.append(dict(model=model, messages=messages, temperature=temperature, seed=seed))
        if messages[0]["content"] in self.fail_on:
            raise RuntimeError("boom")
        return Completion(f"answer t={temperature}", "stop", "resp-1", model, "fp", 10, 5)


# --- config ---


def test_committed_generation_config_matches_preregistration(gen):
    assert gen.generators["model_a"].model_id == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert gen.generators["model_b"].model_id == "google/gemma-3-27b-it"
    assert gen.prompts == {"truthfulqa": "qa-v1", "halueval": "qa-v1", "fever": "fever-v1"}
    assert gen.max_tokens == {"qa-v1": 128, "fever-v1": 96}
    assert (gen.primary.temperature, gen.primary.top_p) == (0.0, 1.0)
    assert (gen.resample.temperature, gen.resample.top_p, gen.resample_n) == (0.7, 1.0, 5)
    assert sum(gen.pilot.questions.values()) == 20


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["primary"].update(temperature=0.2),
        lambda r: r["resample"].update(n=3),
        lambda r: r["resample"].update(temperature=0),
        lambda r: r["generators"]["model_b"].update(model_id=r["generators"]["model_a"]["model_id"]),
        lambda r: r["prompts"].update(fever="fever-v9"),
        lambda r: r["prompts"].pop("halueval"),
        lambda r: r["max_tokens"].pop("fever-v1"),
        lambda r: r["generators"].pop("model_b"),
        lambda r: r.pop("api"),
    ],
)
def test_generation_config_rejects_deviations(raw_gen, mutate):
    raw = copy.deepcopy(raw_gen)
    mutate(raw)
    with pytest.raises(ConfigError):
        parse_generation_config(raw)


# --- prompts ---


def test_qa_prompt_is_one_user_message():
    ex = make_example("truthfulqa", "What happens if you crack your knuckles?")
    text = render("qa-v1", ex)
    assert text == "Answer the following question in one or two sentences.\n\nQuestion: What happens if you crack your knuckles?"
    assert messages(text) == [{"role": "user", "content": text}]


def test_fever_prompt_uses_claim():
    ex = make_example("fever", "Paris is the capital of France.")
    text = render("fever-v1", ex)
    assert text.startswith("Claim: Paris is the capital of France.\n")
    assert "SUPPORTED, REFUTED, or NOT ENOUGH INFO" in text


# --- runner ---


def test_seed_is_deterministic_and_per_sample():
    assert sample_seed("q1", 0) == sample_seed("q1", 0)
    assert sample_seed("q1", 0) != sample_seed("q1", 1)
    assert 0 <= sample_seed("q1", 3) < 2**32


def test_plan_counts_and_sampling(gen):
    a = make_example("truthfulqa", "Q one?")
    b = make_example("fever", "A claim.")
    plan = plan_requests([a, b], {a.question_id: ("train", ["model_a"]), b.question_id: (None, ["model_a", "model_b"])}, gen)
    assert len(plan) == 3 * 6
    primaries = [r for r in plan if r.sample_index == 0]
    assert all(r.temperature == 0.0 for r in primaries)
    assert all(r.temperature == 0.7 for r in plan if r.sample_index > 0)
    assert {r.max_tokens for r in plan if r.dataset == "fever"} == {96}
    assert {r.max_tokens for r in plan if r.dataset == "truthfulqa"} == {128}
    assert {r.model_id for r in plan if r.generator == "model_b"} == {"google/gemma-3-27b-it"}


def test_plan_skips_untargeted_and_adds_repeat(gen):
    a = make_example("halueval", "Q two?")
    c = make_example("halueval", "Q three?")
    plan = plan_requests([a, c], {a.question_id: ("val_id", ["model_a"])}, gen, resamples=False, repeat_primary=True)
    assert [r.answer_id for r in plan] == [answer_id(a.question_id, "model_a", 0), answer_id(a.question_id, "model_a", 0, 1)]
    assert plan[0].seed == plan[1].seed


def test_run_records_errors_and_resumes(tmp_path, gen):
    a = make_example("truthfulqa", "Q one?")
    b = make_example("truthfulqa", "Q two?")
    plan = plan_requests([a, b], {a.question_id: ("train", ["model_a"]), b.question_id: ("train", ["model_a"])}, gen)
    out = tmp_path / "out.jsonl"
    failing = FakeClient(fail_on={render("qa-v1", b)})
    counts = run_requests(plan, failing, out)
    assert counts == {"skipped": 0, "ok": 6, "error": 6}
    rows = load_results(out)
    assert {r["status"] for r in rows.values()} == {"ok", "error"}
    assert all(r["error"].startswith("RuntimeError") for r in rows.values() if r["status"] == "error")

    retry = FakeClient()
    counts = run_requests(plan, retry, out, concurrency=4)
    assert counts == {"skipped": 6, "ok": 6, "error": 0}
    assert len(retry.calls) == 6
    rows = load_results(out)
    assert all(r["status"] == "ok" for r in rows.values())
    assert len(out.read_text().splitlines()) == 18
    row = rows[answer_id(a.question_id, "model_a", 0)]
    for field in ("prompt", "seed", "response_id", "system_fingerprint", "finish_reason", "latency_s", "started_at", "text"):
        assert field in row


def test_client_requires_key(monkeypatch, gen):
    monkeypatch.delenv(gen.api.api_key_env, raising=False)
    with pytest.raises(MissingApiKey):
        OpenAICompatibleClient(gen.api)


def test_client_omits_seed_when_none(gen):
    sent = {}

    class Sdk:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    sent.update(kwargs)

                    class R:
                        id = "r"
                        model = kwargs["model"]
                        system_fingerprint = None
                        choices = [type("C", (), {"finish_reason": "stop", "message": type("M", (), {"content": "x"})()})()]
                        usage = type("U", (), {"prompt_tokens": 3, "completion_tokens": 1})()

                    return R()

    client = OpenAICompatibleClient(gen.api, sdk_client=Sdk())
    c = client.complete(model="m", messages=[], temperature=0, top_p=1, max_tokens=5, seed=None)
    assert "seed" not in sent and "top_k" not in sent
    assert (c.text, c.prompt_tokens, c.completion_tokens) == ("x", 3, 1)


# --- pilot ---


def test_pilot_selection_respects_sources():
    tqa = [make_example("truthfulqa", f"T{i}?") for i in range(4)]
    hal = [make_example("halueval", f"H{i}?") for i in range(4)]
    fev_free = [make_example("fever", f"Free claim {i}.") for i in range(3)]
    fev_tied = make_example("fever", "Tied claim.")
    fev_assigned = make_example("fever", "Assigned claim.")
    kept = tqa + hal + fev_free + [fev_tied, fev_assigned]
    cluster_of = {e.question_id: f"c-{e.question_id}" for e in kept}
    cluster_of[fev_tied.question_id] = cluster_of[tqa[0].question_id]
    manifest = SplitManifest(
        [Assignment(e.question_id, e.dataset, "train", "model_a", cluster_of[e.question_id]) for e in tqa[:2]]
        + [Assignment(e.question_id, e.dataset, "val_id", "model_a", cluster_of[e.question_id]) for e in hal[:2]]
        + [Assignment(fev_assigned.question_id, "fever", "test_heldout", "model_b", cluster_of[fev_assigned.question_id])]
    )
    chosen = select_pilot(kept, cluster_of, manifest, {"truthfulqa": 2, "halueval": 2, "fever": 3}, seed=1)
    ids = {ex.question_id: split for ex, split in chosen}
    assert {q for q, s in ids.items() if s == "train"} == {e.question_id for e in tqa[:2]}
    assert {q for q, s in ids.items() if s == "val_id"} == {e.question_id for e in hal[:2]}
    assert {q for q, s in ids.items() if s is None} == {e.question_id for e in fev_free}
    assert select_pilot(kept, cluster_of, manifest, {"truthfulqa": 2, "halueval": 2, "fever": 3}, seed=1) == chosen
    with pytest.raises(PilotSelectionError):
        select_pilot(kept, cluster_of, manifest, {"fever": 4}, seed=1)


def test_format_report_counts(tmp_path, gen):
    ex = make_example("fever", "A claim.")
    plan = plan_requests([ex], {ex.question_id: (None, ["model_a"])}, gen, repeat_primary=True)
    out = tmp_path / "o.jsonl"
    run_requests(plan, FakeClient(), out)
    report = format_report(load_results(out).values(), {"model_a": (0.1, 0.3)})
    entry = report["model_a/fever-v1"]
    assert entry["calls"] == entry["ok"] == 7
    assert entry["greedy_repeat_identical"] == "1/1"
    assert entry["fever_verdicts"] == {"unparseable": 7}
    assert entry["cost_usd"] == pytest.approx((70 * 0.1 + 35 * 0.3) / 1e6)
    json.dumps(report)
