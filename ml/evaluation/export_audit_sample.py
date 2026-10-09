"""Export the blind 200-answer agreement sample, drawn from automatically resolved answers in all splits.

docs/PREREGISTRATION.md, "Labeling commitments": stratified by dataset x generator x automatic
label, from all splits including the test splits. Reads the automatic labels of every split, so it
lives under ml/evaluation/. Writes data/audit/audit_sample_blind.csv (no automatic labels shown)
and the key data/audit/audit_sample_key.jsonl. Prints the sample size only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hdetect.config import SPLIT_ROLES, load_config
from hdetect.data.io import read_jsonl, write_jsonl
from hdetect.labeling.audit import audit_sample, write_blind_csv
from hdetect.labels import LABELS_DIR, read_auto_labels
from hdetect.schema import Example

KEY_FIELDS = ("answer_id", "question_id", "dataset", "split", "label")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--labels-dir", default=str(LABELS_DIR))
    args = p.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg.paths["audit"])
    blind, key = out / "audit_sample_blind.csv", out / "audit_sample_key.jsonl"
    if blind.exists():
        raise SystemExit(f"{blind} exists; the audit sample is drawn once")
    records = [r for split in SPLIT_ROLES for r in read_auto_labels(split, args.labels_dir, allow_test_splits=True)]
    sample = audit_sample(records, cfg.audit.sample_size, cfg.seed)
    examples = {e.question_id: e for e in map(Example.from_dict, read_jsonl(Path(cfg.paths["processed"]) / "examples.jsonl"))}
    write_blind_csv(sample, examples, blind)
    write_jsonl(key, ({k: r[k] for k in KEY_FIELDS} for r in sample))
    print(f"audit sample: {len(sample)} answers -> {blind}")


if __name__ == "__main__":
    main()
