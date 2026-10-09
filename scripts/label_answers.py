"""Label generated answers against reference answers or gold evidence.

Input rows need answer_id, question_id, generator, and answer. Output rows add the
dataset, split, automatic label, reason, scores, rule version, and config hash.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from hdetect.config import load_config
from hdetect.data.io import read_jsonl, write_jsonl
from hdetect.data.splits import SplitManifest
from hdetect.labeling import label
from hdetect.schema import Example


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("answers")
    parser.add_argument("output")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    examples = {
        ex.question_id: ex
        for ex in map(Example.from_dict, read_jsonl(f"{cfg.paths['processed']}/examples.jsonl"))
    }
    manifest = SplitManifest.read(f"{cfg.paths['splits']}/split_manifest.jsonl")

    out = []
    for row in manifest.attach(read_jsonl(args.answers)):
        ex = examples[row["question_id"]]
        if row["generator"] != manifest.generator_of(ex.question_id):
            raise ValueError(f"{row['answer_id']}: generator {row['generator']} does not match split {row['split']}")
        result = label(ex, row["answer"], cfg.labeling)
        out.append({**row, "dataset": ex.dataset, **result.to_dict(), "config_hash": cfg.hash})
    write_jsonl(args.output, out)

    counts = Counter((r["dataset"], r["label"]) for r in out)
    abstained = Counter(r["dataset"] for r in out if r["abstained"])
    print(json.dumps({"labels": {f"{d}/{l}": n for (d, l), n in sorted(counts.items())},
                      "abstained": dict(sorted(abstained.items()))}, indent=2))


if __name__ == "__main__":
    main()
