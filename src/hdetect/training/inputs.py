"""Load train and val_id exactly as Step 6 does, with the same alignment and version checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hdetect.config import Config
from hdetect.data.splits import SplitManifest
from hdetect.features.pipeline import FeatureConfig
from hdetect.labels import FinalLabels
from hdetect.training.data import (
    DevData,
    load_dev,
    load_labels,
    manifest_clusters_disjoint,
    read_hand,
    read_manifest,
    verify_generations,
)


@dataclass
class DevInputs:
    train: dict[str, DevData]
    val: dict[str, DevData]
    labels: dict[str, tuple[FinalLabels, list[dict]]]
    manifest: SplitManifest
    versions: dict[str, str]
    checks: list[dict[str, Any]]


def feature_versions(fcfg: FeatureConfig) -> dict[str, str]:
    return {"feature_version": fcfg.raw["feature_version"], "clean_version": str(fcfg.raw["clean_version"]),
            "lexicon_version": fcfg.raw["lexicon_version"],
            "embedding_model": f"{fcfg.embedding.model_id}@{fcfg.embedding.revision}",
            "nli_model": f"{fcfg.nli.model_id}@{fcfg.nli.revision}"}


def load_dev_inputs(cfg: Config, fcfg: FeatureConfig, paths: Mapping[str, Any],
                    val_conditions: tuple[str, ...] = ("retrieved", "none", "gold")) -> DevInputs:
    checks = [{"check": "generation files match manifest", "ok": bool(verify_generations(
        fcfg.paths["generations"], fcfg.paths["generation_manifest"]))}]
    manifest = read_manifest(Path(cfg.paths["splits"]) / "split_manifest.jsonl")
    checks.append({"check": "no cluster spans two splits", "ok": manifest_clusters_disjoint(manifest)})
    if not all(c["ok"] for c in checks):
        raise SystemExit(f"data checks failed: {checks}")

    hand = read_hand(paths["hand_labels"])
    labels = {s: load_labels(s, paths["labels"], hand, cfg.labeling.rule_version) for s in ("train", "val_id")}
    versions = feature_versions(fcfg)

    def load(split: str, cond: str) -> DevData:
        return load_dev(split, cond, features_dir=paths["features"], labels=labels[split][0],
                        auto_rows=labels[split][1], manifest=manifest, expected_versions=versions)

    train = {c: load("train", c) for c in ("retrieved", "none")}
    val = {c: load("val_id", c) for c in val_conditions}
    checks.append({"check": "feature rows align with manifest questions, labels, and raw-answer hashes", "ok": True})
    checks.append({"check": "train and val_id share no cluster",
                   "ok": not (set(train["none"].groups) & set(val["none"].groups))})
    return DevInputs(train, val, labels, manifest, versions, checks)
