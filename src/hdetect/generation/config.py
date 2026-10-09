from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from hdetect.config import PREREGISTERED_RESAMPLE_N, SPLIT_ROLES, ConfigError, config_hash
from hdetect.generation.prompts import PROMPTS
from hdetect.schema import DATASETS


@dataclass(frozen=True)
class ApiConfig:
    base_url: str
    api_key_env: str
    timeout_s: float
    max_retries: int
    concurrency: int
    send_seed: bool


@dataclass(frozen=True)
class GeneratorSpec:
    name: str
    model_id: str
    hf_revision: str
    region: str
    quantization: str
    usd_per_million_input_tokens: float
    usd_per_million_output_tokens: float

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.usd_per_million_input_tokens
            + completion_tokens * self.usd_per_million_output_tokens
        ) / 1e6


@dataclass(frozen=True)
class Sampling:
    temperature: float
    top_p: float


@dataclass(frozen=True)
class PilotConfig:
    seed: int
    questions: dict[str, int]
    repeat_primary: bool


@dataclass(frozen=True)
class GenerationConfig:
    api: ApiConfig
    catalog_snapshot: str
    generators: dict[str, GeneratorSpec]
    prompts: dict[str, str]
    max_tokens: dict[str, int]
    primary: Sampling
    resample: Sampling
    resample_n: int
    pilot: PilotConfig
    hash: str


def load_generation_config(path: str | Path) -> GenerationConfig:
    return parse_generation_config(yaml.safe_load(Path(path).read_text()))


def parse_generation_config(raw: dict[str, Any]) -> GenerationConfig:
    try:
        cfg = GenerationConfig(
            api=ApiConfig(**raw["api"]),
            catalog_snapshot=str(raw["catalog_snapshot"]),
            generators={k: GeneratorSpec(name=k, **v) for k, v in raw["generators"].items()},
            prompts=dict(raw["prompts"]),
            max_tokens={k: int(v) for k, v in raw["max_tokens"].items()},
            primary=Sampling(**raw["primary"]),
            resample=Sampling(**{k: v for k, v in raw["resample"].items() if k != "n"}),
            resample_n=int(raw["resample"]["n"]),
            pilot=PilotConfig(**raw["pilot"]),
            hash=config_hash(raw),
        )
    except (KeyError, TypeError) as e:
        raise ConfigError(f"malformed generation config: {e!r}") from e
    _validate(cfg)
    return cfg


def _validate(cfg: GenerationConfig) -> None:
    roles = {generator for _, generator in SPLIT_ROLES.values()}
    if set(cfg.generators) != roles:
        raise ConfigError(f"generators must be exactly {sorted(roles)}")
    if len({g.model_id for g in cfg.generators.values()}) != len(cfg.generators):
        raise ConfigError("Model A and Model B must be different models")
    if set(cfg.prompts) != set(DATASETS):
        raise ConfigError(f"prompts must map exactly {sorted(DATASETS)}")
    unknown = set(cfg.prompts.values()) - set(PROMPTS)
    if unknown:
        raise ConfigError(f"unknown prompt versions: {sorted(unknown)}")
    if set(cfg.max_tokens) != set(cfg.prompts.values()):
        raise ConfigError("max_tokens must be set for exactly the prompt versions in use")
    if cfg.primary.temperature != 0:
        raise ConfigError("primary answers are pre-registered as greedy (temperature 0)")
    if cfg.resample.temperature <= 0:
        raise ConfigError("resamples need a nonzero temperature")
    if cfg.resample_n != PREREGISTERED_RESAMPLE_N:
        raise ConfigError(f"resample n is pre-registered as {PREREGISTERED_RESAMPLE_N}")
    if cfg.api.concurrency < 1 or cfg.api.max_retries < 0:
        raise ConfigError("api.concurrency must be >= 1 and api.max_retries >= 0")
    if set(cfg.pilot.questions) - set(DATASETS):
        raise ConfigError("pilot.questions has an unknown dataset")
