"""Extract feature tables for the development splits (train, val_id) under all evidence conditions.

Test splits are refused here; they go through ml/evaluation/extract_test_features.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from hdetect.config import load_config
from hdetect.data.splits import SplitManifest
from hdetect.features import DEV_SPLITS
from hdetect.features.pipeline import build_extractor, build_retriever, load_feature_config, run_split


def run(splits, args, *, allow_test_splits: bool = False) -> dict:
    cfg = load_config(args.config)
    fcfg = load_feature_config(args.feature_config)
    manifest = SplitManifest.read(Path(cfg.paths["splits"]) / "split_manifest.jsonl")
    out_dir = Path(fcfg.paths["features"]) / "tables"
    extractor = build_extractor(fcfg, Path(fcfg.paths["features"]) / "cache.sqlite")
    retriever = None if args.no_retrieval else build_retriever(fcfg)
    summaries = {}
    for split in splits:
        print(f"{split}: extracting", flush=True)
        summaries[split] = run_split(
            split, fcfg,
            examples_path=Path(cfg.paths["processed"]) / "examples.jsonl",
            manifest=manifest, extractor=extractor, retriever=retriever, out_dir=out_dir,
            sample=args.sample, allow_test_splits=allow_test_splits,
        )
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_config": fcfg.raw,
        "versions": extractor.versions.as_dict(),
        "index_documents": getattr(retriever, "num_docs", None),
        "splits": summaries,
    }
    tag = f"sample{args.sample}" if args.sample else "full"
    report = Path(fcfg.paths["features"]) / f"report.{'-'.join(splits)}.{tag}.json"
    report.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({s: {k: v for k, v in summ.items() if k != "files"} for s, summ in summaries.items()}, indent=2))
    return record


def parser(description: str, choices) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--splits", nargs="+", choices=choices, default=list(choices))
    p.add_argument("--sample", type=int, default=None, help="a seeded random sample of questions per split")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--feature-config", default="configs/features.yaml")
    p.add_argument("--no-retrieval", action="store_true", help="reuse stored retrieval results only")
    return p


def main() -> None:
    args = parser(__doc__, DEV_SPLITS).parse_args()
    run(args.splits, args)


if __name__ == "__main__":
    sys.exit(main())
