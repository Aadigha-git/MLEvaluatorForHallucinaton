"""Versioned demo events (`demo-event-v1`; amendment 2026-10-23). JSON on Kafka, one object per message.

A question event is produced by the Spring Boot producer and consumed by the scoring service. A
verdict event is produced by the scoring service and consumed by the dashboard, which adds the
`displayed` timestamp itself. Unknown fields are ignored on read; missing required fields or a
different schema version are errors.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = "demo-event-v1"
QUESTION_FIELDS = ("schema_version", "question_id", "question", "source", "produced_at")
VERDICT_FIELDS = (
    "schema_version", "question_id", "question", "source", "reference_answer", "answer", "generator", "tier",
    "condition", "classifier", "calibration", "policy", "p_hallucinated", "score", "predicted_hallucinated",
    "band", "explanation", "features", "evidence", "cost_usd", "latency_s", "timestamps", "error",
)


class SchemaError(ValueError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _check(event: dict[str, Any], fields: tuple[str, ...], kind: str) -> dict[str, Any]:
    if event.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(f"{kind} event has schema {event.get('schema_version')!r}, expected {SCHEMA_VERSION!r}")
    missing = [f for f in fields if f not in event]
    if missing:
        raise SchemaError(f"{kind} event is missing {missing}")
    return event


def parse_question(raw: bytes | str) -> dict[str, Any]:
    return _check(json.loads(raw), QUESTION_FIELDS, "question")


def verdict_event(**fields: Any) -> dict[str, Any]:
    event = {"schema_version": SCHEMA_VERSION, **fields}
    return _check(event, VERDICT_FIELDS, "verdict")


def encode(event: dict[str, Any]) -> bytes:
    return json.dumps(event, ensure_ascii=False, allow_nan=False).encode()
