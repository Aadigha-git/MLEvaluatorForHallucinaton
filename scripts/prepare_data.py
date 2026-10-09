"""Download the pinned datasets, deduplicate questions, and write the frozen split manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from hdetect.config import load_config
from hdetect.data import fetch_all
from hdetect.data.dedup import deduplicate, question_text
from hdetect.data.io import write_jsonl
from hdetect.data.splits import SplitManifest, assign_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    loaded = fetch_all(cfg)
    examples = [ex for exs, _ in loaded.values() for ex in exs]
    by_id = {ex.question_id: ex for ex in examples}

    dedup = deduplicate(examples, cfg.dedup)
    manifest = SplitManifest(assign_splits(dedup.kept, dedup.cluster_of, cfg))
    assigned = {a.question_id for a in manifest.assignments}

    processed = Path(cfg.paths["processed"])
    splits_dir = Path(cfg.paths["splits"])
    write_jsonl(processed / "examples.jsonl", (ex.to_dict() for ex in dedup.kept if ex.question_id in assigned))
    manifest.write(splits_dir / "split_manifest.jsonl")
    write_jsonl(
        splits_dir / "dedup_pairs.jsonl",
        (
            {
                "a": p.a,
                "b": p.b,
                "similarity": p.similarity,
                "text_a": question_text(by_id[p.a]),
                "text_b": question_text(by_id[p.b]),
            }
            for p in sorted(dedup.pairs, key=lambda p: (-p.similarity, p.a, p.b))
        ),
    )
    write_jsonl(splits_dir / "dedup_dropped.jsonl", dedup.dropped)

    strata: dict[str, Counter] = {}
    for a in manifest.assignments:
        strata.setdefault(a.split, Counter())[a.stratum or "-"] += 1
    summary = {
        "config_hash": cfg.hash,
        "seed": cfg.seed,
        "revisions": {name: src.revision for name, src in cfg.datasets.items()},
        "loaded": {name: stats for name, (_, stats) in loaded.items()},
        "dedup": {
            "near_duplicate_pairs": len(dedup.pairs),
            "dropped": dict(Counter(d["reason"] for d in dedup.dropped)),
            "clusters_with_more_than_one_question": sum(
                1 for n in Counter(dedup.cluster_of.values()).values() if n > 1
            ),
        },
        "split_counts": manifest.counts(),
        "split_strata": {k: dict(sorted(v.items())) for k, v in sorted(strata.items())},
    }
    (splits_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
