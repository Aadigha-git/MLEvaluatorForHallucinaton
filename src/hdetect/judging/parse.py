"""Strict verdict parsing. Anything other than one well-formed verdict is a parse failure.

Accepted: a JSON object with exactly one key, "verdict", whose value is "hallucinated" or
"not_hallucinated", optionally wrapped in one Markdown code fence (``` or ```json) and surrounding
whitespace. Nothing else is repaired, and a failure is never retried, excluded, or counted as correct.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from hdetect.judging import VERDICTS

OK = "ok"
EMPTY = "empty"
BLOCKED = "blocked"
REASONING_LEAK = "reasoning_leak"
NOT_JSON = "not_json"
WRONG_SCHEMA = "wrong_schema"
INVALID_VERDICT = "invalid_verdict"
OUTCOMES = (OK, EMPTY, BLOCKED, REASONING_LEAK, NOT_JSON, WRONG_SCHEMA, INVALID_VERDICT)

_FENCE = re.compile(r"\A```(?:json)?[ \t]*\n(.*?)\n?```\Z", re.DOTALL)
_THINK = re.compile(r"</?think>", re.IGNORECASE)


@dataclass(frozen=True)
class Parsed:
    verdict: str | None
    outcome: str

    @property
    def ok(self) -> bool:
        return self.outcome == OK


class _DuplicateKey(ValueError):
    pass


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict:
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise _DuplicateKey("duplicate keys")
    return dict(pairs)


def parse_verdict(text: str | None, *, blocked: bool = False, reasoning: str | None = None) -> Parsed:
    """`blocked`: the provider stopped the response (safety, recitation, or similar).
    `reasoning`: reasoning text the provider returned separately; any is a leak for a judge run with
    reasoning off."""
    if blocked:
        return Parsed(None, BLOCKED)
    if reasoning and reasoning.strip():
        return Parsed(None, REASONING_LEAK)
    body = (text or "").strip()
    if not body:
        return Parsed(None, EMPTY)
    if _THINK.search(body):
        return Parsed(None, REASONING_LEAK)
    fenced = _FENCE.match(body)
    if fenced:
        body = fenced.group(1).strip()
    try:
        obj = json.loads(body, object_pairs_hook=_reject_duplicates)
    except _DuplicateKey:
        return Parsed(None, WRONG_SCHEMA)
    except ValueError:
        return Parsed(None, NOT_JSON)
    if not isinstance(obj, dict) or set(obj) != {"verdict"}:
        return Parsed(None, WRONG_SCHEMA)
    if obj["verdict"] not in VERDICTS:
        return Parsed(None, INVALID_VERDICT)
    return Parsed(obj["verdict"], OK)
