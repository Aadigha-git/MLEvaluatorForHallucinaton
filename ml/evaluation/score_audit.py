"""Score the filled-in agreement sample: raw agreement and Cohen's kappa, overall and per dataset.

The key holds automatic labels from every split, including test splits, so scoring lives under
ml/evaluation/. The minimum acceptable kappa is 0.6 overall and per dataset
(docs/PREREGISTRATION.md, "Labeling commitments").
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from hdetect.config import load_config
from hdetect.data.io import read_jsonl
from hdetect.labeling.audit import agreement, read_hand_labels

MIN_KAPPA = 0.6


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("hand_csv", help="audit_sample_blind.csv with hand_label filled in")
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    key = read_jsonl(Path(cfg.paths["audit"]) / "audit_sample_key.jsonl")
    hand = read_hand_labels(args.hand_csv)
    missing = [r["answer_id"] for r in key if r["answer_id"] not in hand]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(key)} sample answers have no hand label")
    groups = {"all": key}
    for r in key:
        groups.setdefault(r["dataset"], []).append(r)
    report = {}
    for name, rows in groups.items():
        result = asdict(
            agreement(
                auto={r["answer_id"]: r["label"] for r in rows},
                hand=hand,
                question_of={r["answer_id"]: r["question_id"] for r in rows},
                n_boot=cfg.audit.bootstrap_samples,
                seed=cfg.seed,
            )
        )
        report[name] = {**result, "passes_min_kappa": result["kappa"] >= MIN_KAPPA}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
