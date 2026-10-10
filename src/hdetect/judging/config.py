from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hdetect.config import ConfigError, config_hash
from hdetect.judging import CONDITIONS
from hdetect.judging.prompts import VARIANTS

PROVIDERS = ("nebius", "gemini")
STATUSES = ("provisional", "frozen")
# Generator families (Model A Qwen, Model B Gemma) that no judge may come from.
EXCLUDED_FAMILIES = ("qwen", "gemma")


@dataclass(frozen=True)
class JudgeSpec:
    name: str
    provider: str
    model_id: str
    base_url: str
    api_key_env: str
    usd_per_million_input_tokens: float
    usd_per_million_output_tokens: float
    temperature: float
    max_output_tokens: int
    top_p: float | None = None
    region: str | None = None
    quantization: str | None = None
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    thinking_level: str | None = None
    max_prompt_tokens_at_price: int | None = None
    max_requests_per_minute: float | None = None
    # Reduced plan for one judge: its own spend cap (inside the global cap), tuning on the first
    # `tune_first_n` labeled val_id answers in answer-ID order, and repeat runs on the first
    # `repeat_first_n` of those instead of the shared repeat subset.
    cap_usd: float | None = None
    tune_first_n: int | None = None
    repeat_first_n: int | None = None

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.usd_per_million_input_tokens
                + output_tokens * self.usd_per_million_output_tokens) / 1e6


@dataclass(frozen=True)
class JudgeConfig:
    status: str
    price_snapshot: str
    cap_usd: float
    max_attempts: int
    max_retries: int
    timeout_s: float
    concurrency: int
    judges: dict[str, JudgeSpec]
    variants: tuple[str, ...]
    conditions: tuple[str, ...]
    evidence_passages: int
    repeat_subset_file: str
    repeat_subset_size: int
    repeat_subset_seed: int
    repeat_extra_runs: int
    paths: dict[str, str]
    hash: str

    @property
    def frozen(self) -> bool:
        return self.status == "frozen"


def load_judge_config(path: str | Path) -> JudgeConfig:
    return parse_judge_config(yaml.safe_load(Path(path).read_text()))


def parse_judge_config(raw: dict[str, Any]) -> JudgeConfig:
    try:
        cfg = JudgeConfig(
            status=str(raw["status"]),
            price_snapshot=str(raw["price_snapshot"]),
            cap_usd=float(raw["budget"]["cap_usd"]),
            max_attempts=int(raw["max_attempts"]),
            max_retries=int(raw["max_retries"]),
            timeout_s=float(raw["timeout_s"]),
            concurrency=int(raw["concurrency"]),
            judges={k: JudgeSpec(name=k, **v) for k, v in raw["judges"].items()},
            variants=tuple(raw["prompts"]["variants"]),
            conditions=tuple(raw["prompts"]["conditions"]),
            evidence_passages=int(raw["prompts"]["evidence_passages"]),
            repeat_subset_file=str(raw["repeatability"]["subset_file"]),
            repeat_subset_size=int(raw["repeatability"]["subset_size"]),
            repeat_subset_seed=int(raw["repeatability"]["subset_seed"]),
            repeat_extra_runs=int(raw["repeatability"]["extra_runs"]),
            paths=dict(raw["paths"]),
            hash=config_hash(raw),
        )
    except (KeyError, TypeError) as e:
        raise ConfigError(f"malformed judge config: {e!r}") from e
    _validate(cfg, raw)
    return cfg


def _validate(cfg: JudgeConfig, raw: dict[str, Any]) -> None:
    if cfg.status not in STATUSES:
        raise ConfigError(f"status must be one of {STATUSES}")
    if set(cfg.judges) != {"small", "frontier"}:
        raise ConfigError("exactly two judges are pre-registered: small and frontier")
    for j in cfg.judges.values():
        if j.provider not in PROVIDERS:
            raise ConfigError(f"{j.name}: unknown provider {j.provider!r}")
        if any(f in j.model_id.lower() for f in EXCLUDED_FAMILIES):
            raise ConfigError(f"{j.name}: judges must come from a family other than Qwen and Gemma")
        if j.max_output_tokens < 1 or j.usd_per_million_input_tokens <= 0 or j.usd_per_million_output_tokens <= 0:
            raise ConfigError(f"{j.name}: max_output_tokens and prices must be positive")
        if j.max_requests_per_minute is not None and j.max_requests_per_minute <= 0:
            raise ConfigError(f"{j.name}: max_requests_per_minute must be positive")
        if j.cap_usd is not None and not 0 < j.cap_usd <= cfg.cap_usd:
            raise ConfigError(f"{j.name}: a judge's cap must be positive and within the global cap")
        if (j.tune_first_n is None) != (j.repeat_first_n is None):
            raise ConfigError(f"{j.name}: a reduced plan sets both tune_first_n and repeat_first_n")
        if j.tune_first_n is not None and not 0 < j.repeat_first_n <= j.tune_first_n:
            raise ConfigError(f"{j.name}: repeat_first_n must be positive and at most tune_first_n")
    small, frontier = cfg.judges["small"], cfg.judges["frontier"]
    if small.temperature != 0:
        raise ConfigError("the small judge runs at temperature 0")
    if small.provider == "nebius" and small.chat_template_kwargs != {"enable_thinking": False}:
        raise ConfigError("the small judge runs with reasoning disabled (enable_thinking: false)")
    if frontier.temperature != 1.0 or frontier.thinking_level != "low":
        raise ConfigError("the frontier judge's temperature (1.0) and thinking level (low) are fixed by amendment")
    if cfg.variants != tuple(VARIANTS):
        raise ConfigError(f"prompt variants must be exactly {tuple(VARIANTS)}")
    if cfg.conditions != CONDITIONS:
        raise ConfigError(f"evidence conditions must be exactly {CONDITIONS}; gold is not part of Step 8")
    if cfg.evidence_passages != 5:
        raise ConfigError("the judge gets the classifier's evidence: the top 5 retrieved passages")
    if raw["selection"] != {"split": "val_id", "metric": "f1_hallucinated", "per": ["judge", "condition"]}:
        raise ConfigError("prompt selection is pre-specified: val_id, hallucinated-class F1, per judge and condition")
    if (cfg.repeat_subset_size, cfg.repeat_extra_runs) != (200, 2):
        raise ConfigError("repeatability is pre-specified: 200 answers, 2 extra runs")
    if cfg.max_attempts != 5 or cfg.max_retries != 5:
        raise ConfigError("retries follow the generation protocol: 5 retries per attempt, 5 attempts")
    if cfg.cap_usd <= 0 or cfg.concurrency < 1:
        raise ConfigError("budget cap and concurrency must be positive")
