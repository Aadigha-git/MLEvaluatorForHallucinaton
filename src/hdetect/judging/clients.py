"""Judge API clients. Neither ever logs or returns the API key.

Nebius (small judge): OpenAI-compatible chat completions through the `openai` package, which retries
rate limits, timeouts, connection errors, and 5xx responses with backoff.

Gemini (frontier judge): the native `generateContent` REST endpoint, because it reports thinking
tokens (`thoughtsTokenCount`) separately and they are billed as output. Retries follow Google's
troubleshooting guide: only 408, 429, and 5xx responses and transport errors, with exponential
backoff and jitter. Other 4xx responses are setup errors and stop the run.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol

import httpx

from hdetect.judging.config import JudgeConfig, JudgeSpec
from hdetect.judging.prompts import messages

GEMINI_TRANSIENT = frozenset({408, 429, 500, 502, 503, 504})
GEMINI_BLOCKED = frozenset({"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"})


@dataclass(frozen=True)
class JudgeCompletion:
    text: str
    finish_reason: str | None
    blocked: bool
    reasoning: str | None
    response_id: str | None
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    thinking_tokens: int | None
    # Transient errors retried inside this call and the backoff slept for them. None when the client
    # cannot observe its retries (the `openai` package retries internally).
    transient_retries: int | None = None
    retry_wait_s: float | None = None


class JudgeClient(Protocol):
    def complete(self, prompt: str) -> JudgeCompletion: ...


class MissingApiKey(RuntimeError):
    pass


class ApiStatusError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(f"HTTP {status_code}: {message}"[:1000])
        self.status_code = status_code


def load_dotenv(path: str | os.PathLike) -> None:
    """Set variables from a .env file without overriding the environment; values are never printed."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _key(spec: JudgeSpec) -> str:
    key = os.environ.get(spec.api_key_env)
    if not key:
        raise MissingApiKey(f"set {spec.api_key_env} to call {spec.base_url}")
    return key


class NebiusJudgeClient:
    def __init__(self, spec: JudgeSpec, cfg: JudgeConfig, sdk_client: Any = None):
        if sdk_client is None:
            from openai import OpenAI

            sdk_client = OpenAI(base_url=spec.base_url, api_key=_key(spec), timeout=cfg.timeout_s,
                                max_retries=cfg.max_retries)
        self.spec = spec
        self._client = sdk_client

    def request_kwargs(self, prompt: str) -> dict[str, Any]:
        """No seed is sent: judge determinism is measured, not assumed."""
        kwargs: dict[str, Any] = dict(model=self.spec.model_id, messages=messages(prompt),
                                      temperature=self.spec.temperature, max_tokens=self.spec.max_output_tokens)
        if self.spec.top_p is not None:
            kwargs["top_p"] = self.spec.top_p
        if self.spec.chat_template_kwargs:
            kwargs["extra_body"] = {"chat_template_kwargs": dict(self.spec.chat_template_kwargs)}
        return kwargs

    def complete(self, prompt: str) -> JudgeCompletion:
        response = self._client.chat.completions.create(**self.request_kwargs(prompt))
        choice = response.choices[0]
        msg = choice.message
        reasoning = getattr(msg, "reasoning_content", None) or (getattr(msg, "model_extra", None) or {}).get(
            "reasoning_content")
        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None)
        return JudgeCompletion(
            text=msg.content or "",
            finish_reason=choice.finish_reason,
            blocked=choice.finish_reason == "content_filter",
            reasoning=reasoning,
            response_id=getattr(response, "id", None),
            model=getattr(response, "model", None),
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            thinking_tokens=getattr(details, "reasoning_tokens", None),
        )


class GeminiJudgeClient:
    def __init__(self, spec: JudgeSpec, cfg: JudgeConfig, *, transport: httpx.BaseTransport | None = None,
                 api_key: str | None = None, sleep: Callable[[float], None] = time.sleep,
                 rng: random.Random | None = None):
        self.spec = spec
        self.max_retries = cfg.max_retries
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._http = httpx.Client(
            base_url=spec.base_url, timeout=cfg.timeout_s, transport=transport,
            headers={"x-goog-api-key": api_key if api_key is not None else _key(spec),
                     "Content-Type": "application/json"},
        )

    def request_body(self, prompt: str) -> dict[str, Any]:
        """No seed is sent: judge determinism is measured, not assumed."""
        return {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.spec.temperature,
                "maxOutputTokens": self.spec.max_output_tokens,
                "thinkingConfig": {"thinkingLevel": self.spec.thinking_level},
            },
        }

    def _post(self, body: dict[str, Any]) -> tuple[dict[str, Any], int, float]:
        """Returns the response, the number of transient retries, and the backoff slept. An error raised
        after retrying carries the same two counts as `transient_retries` and `retry_wait_s`."""
        path = f"models/{self.spec.model_id}:generateContent"
        waited = 0.0
        for attempt in range(self.max_retries + 1):
            error: Exception
            try:
                r = self._http.post(path, json=body)
            except httpx.TransportError as e:
                error, transient = e, True
            else:
                if r.status_code == 200:
                    return r.json(), attempt, waited
                error, transient = ApiStatusError(r.status_code, r.text), r.status_code in GEMINI_TRANSIENT
            if not transient or attempt == self.max_retries:
                error.transient_retries, error.retry_wait_s = attempt, round(waited, 4)  # type: ignore[attr-defined]
                raise error
            delay = min(60.0, 2.0 ** attempt) + self._rng.uniform(0, 1)
            self._sleep(delay)
            waited += delay
        raise AssertionError("unreachable")  # pragma: no cover

    def complete(self, prompt: str) -> JudgeCompletion:
        data, retries, waited = self._post(self.request_body(prompt))
        return replace(parse_gemini_response(data), transient_retries=retries, retry_wait_s=round(waited, 4))


def parse_gemini_response(data: dict[str, Any]) -> JudgeCompletion:
    usage = data.get("usageMetadata")
    candidates = data.get("candidates") or []
    cand = candidates[0] if candidates else {}
    finish = cand.get("finishReason")
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    blocked = bool((data.get("promptFeedback") or {}).get("blockReason")) or finish in GEMINI_BLOCKED
    if usage is None:
        input_tokens = output_tokens = thinking = None
    else:
        thinking = int(usage.get("thoughtsTokenCount", 0))
        input_tokens = int(usage.get("promptTokenCount", 0))
        output_tokens = int(usage.get("candidatesTokenCount", 0)) + thinking
    return JudgeCompletion(
        text=text,
        finish_reason=finish,
        blocked=blocked,
        reasoning=None,
        response_id=data.get("responseId"),
        model=data.get("modelVersion"),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thinking_tokens=thinking,
    )


def make_client(spec: JudgeSpec, cfg: JudgeConfig) -> JudgeClient:
    if spec.provider == "nebius":
        return NebiusJudgeClient(spec, cfg)
    if spec.provider == "gemini":
        return GeminiJudgeClient(spec, cfg)
    raise ValueError(f"unknown provider {spec.provider!r}")
