from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from hdetect.schema import DATASETS

PREREGISTERED_RESAMPLE_N = 5

# Split roles from docs/SPEC.md ("Data sources, split for generalization testing").
RULE_VERSIONS = frozenset({"1", "2"})

SPLIT_ROLES: dict[str, tuple[str, str]] = {
    "train": ("truthfulqa", "model_a"),
    "val_id": ("halueval", "model_a"),
    "test_id": ("halueval", "model_a"),
    "test_ood": ("truthfulqa", "model_b"),
    "test_heldout": ("fever", "model_b"),
}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetSource:
    repo: str
    revision: str
    file: str


@dataclass(frozen=True)
class SplitSpec:
    name: str
    dataset: str
    generator: str
    fraction: float | None = None
    n: int | None = None
    stratify: str | None = None


@dataclass(frozen=True)
class DedupConfig:
    ngram_range: tuple[int, int]
    threshold: float
    keep_priority: tuple[str, ...]


@dataclass(frozen=True)
class TruthfulQALabeling:
    min_match: float
    margin: float


@dataclass(frozen=True)
class HaluEvalLabeling:
    f1_threshold: float
    unresolved_floor: float


@dataclass(frozen=True)
class LabelingConfig:
    rule_version: str
    truthfulqa: TruthfulQALabeling
    halueval: HaluEvalLabeling


@dataclass(frozen=True)
class AuditConfig:
    sample_size: int
    bootstrap_samples: int


@dataclass(frozen=True)
class Config:
    seed: int
    resample_n: int
    paths: dict[str, Path]
    datasets: dict[str, DatasetSource]
    splits: dict[str, SplitSpec]
    dedup: DedupConfig
    labeling: LabelingConfig
    audit: AuditConfig
    hash: str

    def splits_for(self, dataset: str) -> list[SplitSpec]:
        return [s for s in self.splits.values() if s.dataset == dataset]


def config_hash(raw: dict[str, Any]) -> str:
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> Config:
    try:
        cfg = Config(
            seed=int(raw["seed"]),
            resample_n=int(raw["resample_n"]),
            paths={k: Path(v) for k, v in raw["paths"].items()},
            datasets={k: DatasetSource(**v) for k, v in raw["datasets"].items()},
            splits={k: SplitSpec(name=k, **v) for k, v in raw["splits"].items()},
            dedup=DedupConfig(
                ngram_range=tuple(raw["dedup"]["ngram_range"]),
                threshold=float(raw["dedup"]["threshold"]),
                keep_priority=tuple(raw["dedup"]["keep_priority"]),
            ),
            labeling=LabelingConfig(
                rule_version=str(raw["labeling"]["rule_version"]),
                truthfulqa=TruthfulQALabeling(**raw["labeling"]["truthfulqa"]),
                halueval=HaluEvalLabeling(**raw["labeling"]["halueval"]),
            ),
            audit=AuditConfig(**raw["audit"]),
            hash=config_hash(raw),
        )
    except (KeyError, TypeError) as e:
        raise ConfigError(f"malformed config: {e!r}") from e
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if cfg.resample_n != PREREGISTERED_RESAMPLE_N:
        raise ConfigError(f"resample_n is pre-registered as {PREREGISTERED_RESAMPLE_N}")

    if set(cfg.datasets) != set(DATASETS):
        raise ConfigError(f"datasets must be exactly {sorted(DATASETS)}")

    if set(cfg.splits) != set(SPLIT_ROLES):
        raise ConfigError(f"splits must be exactly {sorted(SPLIT_ROLES)}")
    for name, spec in cfg.splits.items():
        if (spec.dataset, spec.generator) != SPLIT_ROLES[name]:
            raise ConfigError(
                f"split {name!r} must use dataset/generator {SPLIT_ROLES[name]}, "
                f"got {(spec.dataset, spec.generator)}"
            )

    for dataset in DATASETS:
        specs = cfg.splits_for(dataset)
        if any(s.fraction is not None and s.n is not None for s in specs):
            raise ConfigError(f"{dataset}: a split takes `fraction` or `n`, not both")
        if any(s.fraction is not None for s in specs):
            if len(specs) < 2 or any(s.fraction is None for s in specs):
                raise ConfigError(f"{dataset}: `fraction` is for dividing a dataset among all its splits")
            if any(not 0 < s.fraction < 1 for s in specs):
                raise ConfigError(f"{dataset} split fractions must be in (0, 1)")
            if abs(sum(s.fraction for s in specs) - 1.0) > 1e-9:
                raise ConfigError(f"{dataset} split fractions must sum to 1")
        else:
            if any(s.n is not None and s.n <= 0 for s in specs):
                raise ConfigError(f"{dataset} sample sizes must be positive")
            if len(specs) > 1 and any(s.n is None for s in specs):
                raise ConfigError(f"{dataset} feeds several sampled splits; each needs `n`")

    d = cfg.dedup
    if not 0 < d.threshold <= 1:
        raise ConfigError("dedup.threshold must be in (0, 1]")
    if len(d.ngram_range) != 2 or not 1 <= d.ngram_range[0] <= d.ngram_range[1]:
        raise ConfigError("dedup.ngram_range must be [min, max] with 1 <= min <= max")
    if sorted(d.keep_priority) != sorted(DATASETS):
        raise ConfigError("dedup.keep_priority must list every dataset exactly once")

    if cfg.labeling.rule_version not in RULE_VERSIONS:
        raise ConfigError(f"labeling.rule_version must be one of {sorted(RULE_VERSIONS)}")
    t = cfg.labeling.truthfulqa
    if not (0 < t.min_match <= 1 and 0 <= t.margin <= 1):
        raise ConfigError("labeling.truthfulqa thresholds must be in [0, 1]")
    h = cfg.labeling.halueval
    if not 0 <= h.unresolved_floor <= h.f1_threshold <= 1:
        raise ConfigError("labeling.halueval needs 0 <= unresolved_floor <= f1_threshold <= 1")

    if cfg.audit.sample_size <= 0 or cfg.audit.bootstrap_samples <= 0:
        raise ConfigError("audit sizes must be positive")
