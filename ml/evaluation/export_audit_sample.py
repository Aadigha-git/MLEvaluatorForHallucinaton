"""Export the blind 200-answer agreement sample for the configured rule version.

docs/PREREGISTRATION.md, "Labeling commitments": automatically resolved answers from all splits,
including the test splits, stratified by dataset x generator x automatic label. Each audit round
is drawn once, with the seed fixed in AUDIT_ROUNDS, and excludes every answer audited in earlier
rounds (read from their blind CSVs, which hold no labels). Reads the automatic labels of every
split, so it lives under ml/evaluation/. Writes data/audit/audit_sample[_v{version}]_blind.csv
(no automatic labels shown) and its key. Prints the sample size only.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from hdetect.config import SPLIT_ROLES, load_config
from hdetect.data.io import read_jsonl, write_jsonl
from hdetect.labeling.audit import audit_sample, write_blind_csv
from hdetect.labels import audit_file, read_auto_labels, version_dir
from hdetect.schema import Example

KEY_FIELDS = ("answer_id", "question_id", "dataset", "split", "label", "rule_version")

# Rule version -> seed and earlier rounds to exclude. Fixed before each round is drawn.
AUDIT_ROUNDS = {
    "1": {"seed": 20261008, "exclude_rounds": ()},
    "2": {"seed": 20261011, "exclude_rounds": ("1",)},
}


def audited_answer_ids(audit_dir: Path, versions) -> set[str]:
    ids = set()
    for v in versions:
        with open(audit_dir / audit_file("audit_sample", v, "_blind.csv"), newline="", encoding="utf-8") as f:
            ids |= {row["answer_id"] for row in csv.DictReader(f)}
    return ids


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    version = cfg.labeling.rule_version
    round_cfg = AUDIT_ROUNDS[version]
    out = Path(cfg.paths["audit"])
    blind = out / audit_file("audit_sample", version, "_blind.csv")
    key = out / audit_file("audit_sample", version, "_key.jsonl")
    if blind.exists():
        raise SystemExit(f"{blind} exists; the audit sample is drawn once")
    excluded = audited_answer_ids(out, round_cfg["exclude_rounds"])
    records = [r for split in SPLIT_ROLES
               for r in read_auto_labels(split, version_dir(version), allow_test_splits=True)
               if r["answer_id"] not in excluded]
    if any(r["rule_version"] != version for r in records):
        raise SystemExit(f"labels in {version_dir(version)} are not all rule version {version}")
    sample = audit_sample(records, cfg.audit.sample_size, round_cfg["seed"])
    examples = {e.question_id: e for e in map(Example.from_dict, read_jsonl(Path(cfg.paths["processed"]) / "examples.jsonl"))}
    write_blind_csv(sample, examples, blind)
    write_jsonl(key, ({k: r[k] for k in KEY_FIELDS} for r in sample))
    print(f"audit sample (rule version {version}): {len(sample)} answers, {len(excluded)} previously audited "
          f"answers excluded -> {blind}")


if __name__ == "__main__":
    main()
