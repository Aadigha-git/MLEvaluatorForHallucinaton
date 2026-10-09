from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

from hdetect.generation.config import ApiConfig


@dataclass(frozen=True)
class Completion:
    text: str
    finish_reason: str | None
    response_id: str | None
    model: str | None
    system_fingerprint: str | None
    prompt_tokens: int | None
    completion_tokens: int | None


class ChatClient(Protocol):
    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int | None,
    ) -> Completion: ...


class MissingApiKey(RuntimeError):
    pass


class OpenAICompatibleClient:
    """Chat completions against an OpenAI-compatible endpoint (Nebius Token Factory).

    Rate limits, timeouts, and 5xx errors are retried with backoff by the openai package.
    """

    def __init__(self, api: ApiConfig, sdk_client: Any = None):
        if sdk_client is None:
            key = os.environ.get(api.api_key_env)
            if not key:
                raise MissingApiKey(f"set {api.api_key_env} to call {api.base_url}")
            from openai import OpenAI

            sdk_client = OpenAI(
                base_url=api.base_url, api_key=key, timeout=api.timeout_s, max_retries=api.max_retries
            )
        self._client = sdk_client

    def complete(self, *, model, messages, temperature, top_p, max_tokens, seed) -> Completion:
        kwargs = dict(model=model, messages=messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        if seed is not None:
            kwargs["seed"] = seed
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        usage = response.usage
        return Completion(
            text=choice.message.content or "",
            finish_reason=choice.finish_reason,
            response_id=getattr(response, "id", None),
            model=getattr(response, "model", None),
            system_fingerprint=getattr(response, "system_fingerprint", None),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )
