"""Aligned feature rows, final labels, and per-question timing for one split and evidence condition.

The same alignment checks as Step 6 (`hdetect.training.data.load_dev`): one row per manifest
question, answer IDs and raw-answer hashes equal to the labels, the frozen feature versions, and
finite features. Test splits need `allow_test_splits=True`, which only ml/evaluation/ passes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from hdetect.config import SPLIT_ROLES
from hdetect.data.splits import SplitManifest
from hdetect.features import feature_columns
from hdetect.features.access import guard
from hdetect.labels import FinalLabels, auto_path, final_labels
from hdetect.training.data import ACCESS_LOG, AlignmentError

# The per-question timing column that covers each feature (features-v1). Several features share one.
FEATURE_TIME = {
    "sim_max": "time_sim_ms", "nli_entail_max": "time_nli_ms", "nli_contra_max": "time_nli_ms",
    "lex_overlap_max": "time_lex_ms", "hedge_rate": "time_hedge_ms", "booster_rate": "time_hedge_ms",
    "log_length": "time_length_ms", "specificity": "time_length_ms", "self_contra": "time_self_contra_ms",
    "sc_embed": "time_sc_embed_ms", "sc_entail": "time_sc_nli_ms", "sc_contra": "time_sc_nli_ms",
}
META = ["answer_id", "question_id", "meta_truncated_primary", "meta_n_resamples_usable", "meta_retrieval_s"]


@dataclass
class SplitData:
    split: str
    condition: str
    X: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    meta: pd.DataFrame
    times_ms: pd.DataFrame
    counts: dict = field(default_factory=dict)

    def subset(self, features: tuple[str, ...]) -> np.ndarray:
        """C-ordered, as in Steps 6 and 7: memory layout changes BLAS rounding, and the frozen F1
        thresholds are exact val_id scores, so a one-ulp shift moves answers across them."""
        return np.ascontiguousarray(self.X.loc[:, list(features)].to_numpy(dtype=float))

    def feature_seconds(self, features: tuple[str, ...]) -> np.ndarray:
        """Local feature time per question: each timing column counted once, whatever shares it."""
        cols = sorted({FEATURE_TIME[f] for f in features})
        return self.times_ms[cols].to_numpy(dtype=float).sum(axis=1) / 1000.0


def _open(path: Path, mode: str = "r"):
    ACCESS_LOG.append(path.as_posix())
    return open(path, mode, **({} if "b" in mode else {"encoding": "utf-8"}))


def read_auto(labels_dir: str | Path, split: str, *, allow_test_splits: bool = False) -> list[dict]:
    guard(split, allow_test_splits)
    with _open(auto_path(split, labels_dir)) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_labels(split: str, labels_dir: str | Path, hand: Mapping[str, str], rule_version: str, *,
                allow_test_splits: bool = False) -> tuple[FinalLabels, list[dict]]:
    auto = read_auto(labels_dir, split, allow_test_splits=allow_test_splits)
    versions = {r["rule_version"] for r in auto}
    if versions != {rule_version}:
        raise AlignmentError(f"{split}: labels have rule versions {sorted(versions)}, expected {rule_version!r}")
    return final_labels(split, auto, hand), auto


def load_split(split: str, condition: str, *, features_dir: str | Path, labels: FinalLabels, auto_rows: list[dict],
               manifest: SplitManifest, expected_versions: Mapping[str, str],
               allow_test_splits: bool = False) -> SplitData:
    guard(split, allow_test_splits)
    with _open(Path(features_dir) / "tables" / condition / f"{split}.parquet", "rb") as f:
        df = pd.read_parquet(f)
    if df["question_id"].duplicated().any() or set(df["question_id"]) != set(manifest.question_ids(split)):
        raise AlignmentError(f"{split}/{condition}: feature rows do not match the split manifest's questions")
    dataset, generator = SPLIT_ROLES[split]
    if set(df["dataset"]) != {dataset} or set(df["generator"]) != {generator} or set(df["split"]) != {split}:
        raise AlignmentError(f"{split}/{condition}: dataset, generator, or split columns disagree with the manifest")
    for col, value in expected_versions.items():
        if set(df[col]) != {value}:
            raise AlignmentError(f"{split}/{condition}: {col} is {sorted(set(df[col]))}, expected {value!r}")
    auto = {r["answer_id"]: r for r in auto_rows}
    if set(df["answer_id"]) != set(auto):
        raise AlignmentError(f"{split}/{condition}: answer IDs differ from the labeled primary answers")
    if any(auto[a]["raw_sha256"] != h for a, h in zip(df["answer_id"], df["meta_raw_sha256"])):
        raise AlignmentError(f"{split}/{condition}: raw-answer hashes differ between features and labels")
    feats = list(feature_columns(condition))
    if not np.isfinite(df[feats].to_numpy(dtype=float)).all():
        raise AlignmentError(f"{split}/{condition}: non-finite feature values (features-v1 never imputes)")

    cluster = {a.question_id: a.cluster_id for a in manifest.assignments if a.split == split}
    keep = df["answer_id"].isin(labels.y).to_numpy()
    df = df.loc[keep].sort_values("question_id").reset_index(drop=True)
    meta = df[META].copy()
    meta["abstained"] = [bool(auto[a]["abstained"]) for a in df["answer_id"]]
    time_cols = sorted({FEATURE_TIME[f] for f in feats})
    return SplitData(
        split=split, condition=condition,
        X=df[feats].astype(float).reset_index(drop=True),
        y=np.array([labels.y[a] for a in df["answer_id"]], dtype=int),
        groups=np.array([cluster[q] for q in df["question_id"]]),
        meta=meta, times_ms=df[time_cols].astype(float).reset_index(drop=True),
        counts={"rows": int(len(df)), "excluded_undecidable": int((~keep).sum())},
    )
