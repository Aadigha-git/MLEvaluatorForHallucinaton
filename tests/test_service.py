"""Demo scoring service: event schema, explanation generator, and policy bands (no network, no Kafka)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from hdetect.service.explain import contributions, explain
from hdetect.service.schema import SCHEMA_VERSION, VERDICT_FIELDS, SchemaError, encode, parse_question, verdict_event
from hdetect.service.scorer import band_of
from tests.conftest import ROOT


def test_question_events_need_the_schema_version_and_fields():
    q = {"schema_version": SCHEMA_VERSION, "question_id": "q1", "question": "?", "source": "trap",
         "produced_at": "2026-10-23T00:00:00Z", "extra": 1}
    assert parse_question(json.dumps(q))["question_id"] == "q1"
    with pytest.raises(SchemaError):
        parse_question(json.dumps({**q, "schema_version": "demo-event-v0"}))
    with pytest.raises(SchemaError):
        parse_question(json.dumps({k: v for k, v in q.items() if k != "question"}))


def test_verdict_events_carry_every_field_and_no_nan():
    fields = {f: None for f in VERDICT_FIELDS if f != "schema_version"}
    event = verdict_event(**fields)
    assert event["schema_version"] == SCHEMA_VERSION
    with pytest.raises(SchemaError):
        verdict_event(**{k: v for k, v in fields.items() if k != "band"})
    with pytest.raises(ValueError):
        encode({**event, "p_hallucinated": float("nan")})


def test_contributions_replace_one_feature_at_a_time():
    w = {"a": 2.0, "b": -1.0}
    score = lambda M: M @ np.array([w["a"], w["b"]])
    d = contributions({"a": 3.0, "b": 1.0}, ("a", "b"), {"a": 1.0, "b": 4.0}, score)
    assert d == pytest.approx({"a": 4.0, "b": 3.0})


def test_explanation_names_features_in_the_verdict_direction():
    a, b, c = "lex_overlap_max", "hedge_rate", "log_length"
    x, med = {a: 0.9, b: 0.1, c: 0.5}, {a: 0.5, b: 0.5, c: 0.5}
    deltas = {a: 0.3, b: -0.2, c: 0.1}
    flag = explain("flag", x, deltas, med)
    assert flag.startswith("Flagged: word overlap with the evidence is higher than typical (0.90 vs 0.50)")
    assert "answer length" in flag and "hedging" not in flag
    accept = explain("accept", x, deltas, med)
    assert accept == "Accepted: hedging is lower than typical (0.10 vs 0.50)."
    assert "no single feature" in explain("escalate", x, {k: 0.0 for k in deltas}, med)


def test_policy_bands_follow_the_frozen_rule():
    assert band_of(0.1, 0.3, 0.6) == "accept"
    assert band_of(0.3, 0.3, 0.6) == "escalate"
    assert band_of(0.6, 0.3, 0.6) == "flag"


def test_demo_config_uses_a_frozen_artifact_and_a_budget():
    d = yaml.safe_load((ROOT / "configs" / "demo.yaml").read_text())
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["classifier"] == {"tier": "single_shot", "condition": "retrieved", "seed": 20261008,
                               "artifact": "data/models/step7/single_shot__retrieved__seed20261008.joblib"}
    assert 0 < d["chatbot"]["budget_usd"] <= 0.5
    traps = [json.loads(line) for line in (ROOT / d["questions"]["traps"]).read_text().splitlines()]
    assert len({t["question_id"] for t in traps}) == len(traps) and all(t["reference_answer"] for t in traps)


def test_demo_outputs_are_not_committed():
    ignored = (ROOT / ".gitignore").read_text().split()
    assert {"data/demo/", "demo/app/target/"} <= set(ignored)
