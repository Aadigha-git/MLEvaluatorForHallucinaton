"""Load aligned feature tables and final labels for train and val_id only.

There is no switch for test splits: `load_dev` refuses anything but train and val_id. Every file
read goes through `_open`, which records it, so a run can prove it never touched a test split.
Alignment checks: one row per manifest question, answer IDs and raw-answer hashes equal to the
labels (which come from manifest-verified generation files), and the frozen feature versions.
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
from hdetect.features import DEV_SPLITS, feature_columns
from hdetect.features.access import SplitAccessError, verify_generation_file
from hdetect.labeling.audit import read_hand_labels
from hdetect.labels import FinalLabels, auto_path, final_labels

ACCESS_LOG: list[str] = []


class AlignmentError(RuntimeError):
    pass


def _check_split(split: str) -> None:
    if split not in DEV_SPLITS:
        raise SplitAccessError(f"training code reads only {DEV_SPLITS}, not {split!r}")


def _open(path: str | Path, mode: str = "r"):
    ACCESS_LOG.append(Path(path).as_posix())
    return open(path, mode, **({} if "b" in mode else {"encoding": "utf-8"}))


def read_table(features_dir: str | Path, condition: str, split: str) -> pd.DataFrame:
    _check_split(split)
    with _open(Path(features_dir) / "tables" / condition / f"{split}.parquet", "rb") as f:
        return pd.read_parquet(f)


def read_auto(labels_dir: str | Path, split: str) -> list[dict]:
    _check_split(split)
    with _open(auto_path(split, labels_dir)) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_hand(path: str | Path) -> dict[str, str]:
    ACCESS_LOG.append(Path(path).as_posix())
    return read_hand_labels(path, allow_undecidable=True)


def load_labels(
    split: str, labels_dir: str | Path, hand: Mapping[str, str], rule_version: str
) -> tuple[FinalLabels, list[dict]]:
    auto = read_auto(labels_dir, split)
    versions = {r["rule_version"] for r in auto}
    if versions != {rule_version}:
        raise AlignmentError(f"{split}: labels have rule versions {sorted(versions)}, expected {rule_version!r}")
    return final_labels(split, auto, hand), auto


@dataclass
class DevData:
    split: str
    condition: str
    X: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    meta: pd.DataFrame
    counts: dict = field(default_factory=dict)

    def subset(self, features: tuple[str, ...]) -> np.ndarray:
        return self.X.loc[:, list(features)].to_numpy(dtype=float)


def load_dev(
    split: str,
    condition: str,
    *,
    features_dir: str | Path,
    labels: FinalLabels,
    auto_rows: list[dict],
    manifest: SplitManifest,
    expected_versions: Mapping[str, str],
) -> DevData:
    """Feature rows for `split` under `condition`, joined with final labels and cluster IDs."""
    _check_split(split)
    df = read_table(features_dir, condition, split)
    qids = manifest.question_ids(split)
    if df["question_id"].duplicated().any() or set(df["question_id"]) != set(qids):
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
    meta = df[["answer_id", "question_id", "meta_truncated_primary", "meta_n_resamples_usable"]].copy()
    meta["abstained"] = [bool(auto[a]["abstained"]) for a in df["answer_id"]]
    return DevData(
        split=split,
        condition=condition,
        X=df[feats].astype(float).reset_index(drop=True),
        y=np.array([labels.y[a] for a in df["answer_id"]], dtype=int),
        groups=np.array([cluster[q] for q in df["question_id"]]),
        meta=meta,
        counts={"rows": int(len(df)), "excluded_undecidable": int((~keep).sum())},
    )


def resampling_rows(data: DevData) -> np.ndarray:
    """F9: questions with no usable resample are left out of resampling-tier results (and counted)."""
    return data.meta["meta_n_resamples_usable"].to_numpy() > 0


def verify_generations(generations_dir: str | Path, generation_manifest: str | Path) -> list[str]:
    for split in DEV_SPLITS:
        ACCESS_LOG.append(verify_generation_file(split, generations_dir, generation_manifest).as_posix())
    return list(DEV_SPLITS)


def manifest_clusters_disjoint(manifest: SplitManifest) -> bool:
    by_split: dict[str, set[str]] = {}
    for a in manifest.assignments:
        by_split.setdefault(a.split, set()).add(a.cluster_id)
    splits = sorted(by_split)
    return all(not (by_split[a] & by_split[b]) for i, a in enumerate(splits) for b in splits[i + 1:])


def read_manifest(path: str | Path) -> SplitManifest:
    ACCESS_LOG.append(Path(path).as_posix())
    return SplitManifest.read(path)
