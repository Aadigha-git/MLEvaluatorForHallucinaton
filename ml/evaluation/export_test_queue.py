"""Export the test splits' unresolved answers for blind hand labeling (Step 9).

docs/PREREGISTRATION.md, "Labeling commitments": unresolved answers are hand-labeled, and answers
the hand labeler cannot decide (`undecidable`) are excluded and counted. Reads the configured rule
version's automatic test labels, so it lives under ml/evaluation/. Writes
data/audit/unresolved_test[_v{version}]_blind.csv with the same columns as the development queue
and no automatic labels. Rows are sorted by answer ID, as in the development queue. An existing
queue is never overwritten, so hand labels already entered are safe. Prints the total queue size
only: no per-split counts, label counts, or rates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hdetect.config import load_config
from hdetect.data.io import read_jsonl
from hdetect.features import TEST_SPLITS
from hdetect.labeling.audit import unresolved_queue, write_blind_csv
from hdetect.labels import audit_file, read_auto_labels, version_dir
from hdetect.schema import Example

TEST_SPLIT_ORDER = ("test_id", "test_ood", "test_heldout")


def main() -> None:
    assert set(TEST_SPLIT_ORDER) == TEST_SPLITS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    version = cfg.labeling.rule_version
    queue_path = Path(cfg.paths["audit"]) / audit_file("unresolved_test", version, "_blind.csv")
    if queue_path.exists():
        raise SystemExit(f"{queue_path} exists; not overwritten")
    records = [r for split in TEST_SPLIT_ORDER
               for r in read_auto_labels(split, version_dir(version), allow_test_splits=True)]
    if any(r["rule_version"] != version for r in records):
        raise SystemExit(f"labels in {version_dir(version)} are not all rule version {version}")
    queue = unresolved_queue(records)
    examples = {e.question_id: e for e in map(Example.from_dict, read_jsonl(Path(cfg.paths["processed"]) / "examples.jsonl"))}
    write_blind_csv(queue, examples, queue_path)
    print(f"test unresolved queue (rule version {version}): {len(queue)} answers -> {queue_path}")


if __name__ == "__main__":
    main()
