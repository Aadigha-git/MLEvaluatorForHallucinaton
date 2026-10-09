"""Label the development splits' primary answers and export their unresolved queue for hand labeling.

Uses the rule version in the config. Writes data/labels/v{version}/auto/{train,val_id}.jsonl and
the blind queue data/audit/unresolved_dev[_v{version}]_blind.csv (no automatic labels shown). Test
splits are refused here; they go through ml/evaluation/label_test_answers.py. An existing queue
CSV is never overwritten, so hand labels already entered are safe.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from hdetect.config import load_config
from hdetect.data.io import read_jsonl
from hdetect.data.splits import SplitManifest
from hdetect.features import DEV_SPLITS
from hdetect.labeling.audit import unresolved_queue, write_blind_csv
from hdetect.labels import audit_file, label_split, version_dir, write_auto_labels
from hdetect.schema import Example


def labels_dir(args, cfg) -> Path:
    return Path(args.labels_dir) if args.labels_dir else version_dir(cfg.labeling.rule_version)


def label_splits(splits, args, *, allow_test_splits: bool = False) -> dict[str, list[dict]]:
    cfg = load_config(args.config)
    examples = {e.question_id: e for e in map(Example.from_dict, read_jsonl(Path(cfg.paths["processed"]) / "examples.jsonl"))}
    manifest = SplitManifest.read(Path(cfg.paths["splits"]) / "split_manifest.jsonl")
    out = {}
    for split in splits:
        rows = label_split(
            split, examples, manifest, args.generations, args.generation_manifest, cfg.labeling, cfg.hash,
            allow_test_splits=allow_test_splits,
        )
        write_auto_labels(rows, split, labels_dir(args, cfg))
        out[split] = rows
    return out


def parser(description: str, choices) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", nargs="+", choices=choices, default=list(choices))
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--generations", default="data/generations")
    p.add_argument("--generation-manifest", default="data/manifests/generations.json")
    p.add_argument("--labels-dir", default=None, help="default: data/labels/v{rule_version}")
    return p


def main() -> None:
    args = parser(__doc__, DEV_SPLITS).parse_args()
    labeled = label_splits(args.splits, args)
    cfg = load_config(args.config)
    summary = {
        split: {"rule_version": cfg.labeling.rule_version, "answers": len(rows), **Counter(r["label"] for r in rows),
                "abstained": sum(r["abstained"] for r in rows)}
        for split, rows in labeled.items()
    }
    print(json.dumps(summary, indent=2))

    queue_path = Path(cfg.paths["audit"]) / audit_file("unresolved_dev", cfg.labeling.rule_version, "_blind.csv")
    queue = unresolved_queue([r for rows in labeled.values() for r in rows])
    if queue_path.exists():
        print(f"{queue_path} exists; not overwritten ({len(queue)} unresolved answers)")
        return
    examples = {e.question_id: e for e in map(Example.from_dict, read_jsonl(Path(cfg.paths["processed"]) / "examples.jsonl"))}
    write_blind_csv(queue, examples, queue_path)
    print(f"unresolved queue: {len(queue)} answers -> {queue_path}")


if __name__ == "__main__":
    main()
