"""Hand-audit workflow.

  export: write a blind stratified sample (for kappa) and the unresolved queue as CSVs.
  score:  compare a filled-in sample CSV with the automatic labels and report Cohen's kappa.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from hdetect.config import load_config
from hdetect.data.io import read_jsonl, write_jsonl
from hdetect.labeling.audit import agreement, audit_sample, read_hand_labels, unresolved_queue, write_blind_csv
from hdetect.schema import Example


def export(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    records = read_jsonl(args.labeled)
    examples = {
        ex.question_id: ex
        for ex in map(Example.from_dict, read_jsonl(f"{cfg.paths['processed']}/examples.jsonl"))
    }
    out = Path(cfg.paths["audit"])
    sample = audit_sample(records, cfg.audit.sample_size, cfg.seed)
    write_blind_csv(sample, examples, out / "audit_sample_blind.csv")
    write_jsonl(out / "audit_sample_key.jsonl", ({k: r[k] for k in ("answer_id", "question_id", "dataset", "label")} for r in sample))
    queue = unresolved_queue(records)
    write_blind_csv(queue, examples, out / "unresolved_blind.csv")
    print(f"audit sample: {len(sample)} answers; unresolved queue: {len(queue)} answers -> {out}")


def score(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    key = read_jsonl(Path(cfg.paths["audit"]) / "audit_sample_key.jsonl")
    hand = read_hand_labels(args.hand_csv)
    groups = {"all": key}
    for r in key:
        groups.setdefault(r["dataset"], []).append(r)
    report = {}
    for name, rows in groups.items():
        if not any(r["answer_id"] in hand for r in rows):
            continue
        report[name] = asdict(
            agreement(
                auto={r["answer_id"]: r["label"] for r in rows},
                hand=hand,
                question_of={r["answer_id"]: r["question_id"] for r in rows},
                n_boot=cfg.audit.bootstrap_samples,
                seed=cfg.seed,
            )
        )
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("export")
    p.add_argument("labeled", help="output of scripts/label_answers.py")
    p.set_defaults(func=export)
    p = sub.add_parser("score")
    p.add_argument("hand_csv", help="audit_sample_blind.csv with hand_label filled in")
    p.set_defaults(func=score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
