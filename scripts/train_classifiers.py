"""Step 6: train on train, select and validate on val_id, ablations and redundancy; write reports/step6/.

Refuses to run until every unresolved train and val_id answer has a hand decision. Reads no test
split: every file opened under data/ is recorded (Python audit hook plus the loader's own log) and
the run fails if any belongs to a test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from hdetect.config import load_config
from hdetect.features import TEST_SPLITS
from hdetect.features.pipeline import load_feature_config
from hdetect.training.data import (
    ACCESS_LOG,
    load_dev,
    load_labels,
    manifest_clusters_disjoint,
    read_hand,
    read_manifest,
    verify_generations,
)
from hdetect.training.report import write_readme
from hdetect.training.run import Inputs, Step6, write_tables

ROOT = Path(__file__).resolve().parents[1]
OPENED: list[str] = []


def _record_open(event: str, args: tuple) -> None:
    if event == "open" and args and isinstance(args[0], (str, Path)):
        OPENED.append(str(args[0]))


def is_test_path(path: str) -> bool:
    name = Path(path).name
    return any(name.startswith(s) or f"/{s}/" in path for s in TEST_SPLITS)


def data_access_table() -> pd.DataFrame:
    paths = set()
    for p in OPENED + ACCESS_LOG:
        resolved = Path(p).resolve() if not Path(p).is_absolute() else Path(p)
        try:
            rel = resolved.relative_to(ROOT)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] == "data":
            paths.add(rel.as_posix())
    rows = [{"path": p, "test_split": is_test_path(p)} for p in sorted(paths)]
    return pd.DataFrame(rows, columns=["path", "test_split"])


def notes(fcfg_paths: dict, train: dict, val: dict) -> dict[str, str]:
    report = json.loads(Path(fcfg_paths["features"], "report.train-val_id.full.json").read_text())
    recall = report["splits"]["val_id"]["recall@5"]
    times = report["splits"]
    ft = "; ".join(f"{s}: " + ", ".join(f"{k[5:-3]} {v['mean']:.1f} ms" for k, v in sorted(times[s]["time_ms"].items()))
                   for s in ("train", "val_id"))
    return {
        "retrieval": (
            "BM25 top-5 over Wiki-DPR. `val_id` (HaluEval) knowledge-sentence recall@5 = "
            f"{recall['halueval']['knowledge_sentence_recall@5']:.3f} (n = {recall['halueval']['n']}). `train` "
            "(TruthfulQA) has no gold evidence, so its recall is not measurable. A retrieved-evidence feature can only "
            "help where retrieval found the relevant passage; low recall caps the evidence group, and this is a "
            "retrieval limitation rather than a classifier one."
        ),
        "datasets": (
            "`train` is TruthfulQA only and `val_id` is HaluEval only, so every validation number is already a "
            "cross-dataset result (TruthfulQA to HaluEval, same generator) and per-dataset results equal per-split "
            "results; there is nothing to pool. `train` CV average precision is TruthfulQA in-distribution. "
            "Evidence-condition effects are compared within `val_id` (same answers, same labels): retrieved versus "
            "no evidence, and gold as an exploratory oracle. Gold-evidence results are not realistic retrieval "
            "performance: HaluEval's gold passage contains the reference answer, which the labeling rule also uses."
        ),
        "feature_times": "Mean feature computation time per answer (Step 5 reports) — " + ft,
        "sc_entail": "",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--feature-config", default="configs/features.yaml")
    p.add_argument("--training-config", default="configs/training.yaml")
    p.add_argument("--n-jobs", type=int, default=None)
    args = p.parse_args()
    sys.addaudithook(_record_open)

    cfg = load_config(args.config)
    fcfg = load_feature_config(args.feature_config)
    tcfg = yaml.safe_load(Path(args.training_config).read_text())
    paths = tcfg["paths"]

    checks = [{"check": "generation files match manifest", "ok": bool(verify_generations(
        fcfg.paths["generations"], fcfg.paths["generation_manifest"]))}]
    manifest = read_manifest(Path(cfg.paths["splits"]) / "split_manifest.jsonl")
    checks.append({"check": "no cluster spans two splits", "ok": manifest_clusters_disjoint(manifest)})
    if not all(c["ok"] for c in checks):
        raise SystemExit(f"data checks failed: {checks}")

    hand = read_hand(paths["hand_labels"])
    labels = {s: load_labels(s, paths["labels"], hand, cfg.labeling.rule_version) for s in ("train", "val_id")}
    versions = {"feature_version": fcfg.raw["feature_version"], "clean_version": str(fcfg.raw["clean_version"]),
                "lexicon_version": fcfg.raw["lexicon_version"],
                "embedding_model": f"{fcfg.embedding.model_id}@{fcfg.embedding.revision}",
                "nli_model": f"{fcfg.nli.model_id}@{fcfg.nli.revision}"}
    load = lambda split, cond: load_dev(split, cond, features_dir=paths["features"], labels=labels[split][0],
                                        auto_rows=labels[split][1], manifest=manifest, expected_versions=versions)
    train = {c: load("train", c) for c in ("retrieved", "none")}
    val = {c: load("val_id", c) for c in ("retrieved", "none", "gold")}
    checks.append({"check": "feature rows align with manifest questions, labels, and raw-answer hashes", "ok": True})
    checks.append({"check": "train and val_id share no cluster",
                   "ok": not (set(train["none"].groups) & set(val["none"].groups))})

    inputs = Inputs(train, val, [labels[s][0].summary() for s in ("train", "val_id")], checks,
                    notes(fcfg.paths, train, val))
    step = Step6(tcfg, inputs, args.n_jobs)
    tables = step.run()
    tables["data_access"] = data_access_table()
    if tables["data_access"]["test_split"].any():
        raise SystemExit("a test-split file was opened; reports not written")

    out = Path(paths["reports"])
    write_tables({k: v for k, v in tables.items()}, out)
    (out / "README.md").write_text(write_readme(tables, inputs.notes, tcfg) + "\n")
    print(f"reports -> {out}")


if __name__ == "__main__":
    main()
