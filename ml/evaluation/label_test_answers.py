"""The authorized entry point for automatically labeling the test splits' primary answers.

Writes data/labels/v{version}/auto/{test_id,test_ood,test_heldout}.jsonl with the frozen rules of
the configured version. It prints row counts only: no label counts, rates, or answers. The files
feed the blind audit sample (ml/evaluation/export_audit_sample.py) and, at Step 9, the test
evaluation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from label_answers import label_splits, labels_dir, parser  # noqa: E402

from hdetect.config import load_config  # noqa: E402
from hdetect.features import TEST_SPLITS  # noqa: E402

TEST_SPLIT_ORDER = ("test_id", "test_ood", "test_heldout")


def main() -> None:
    assert set(TEST_SPLIT_ORDER) == TEST_SPLITS
    args = parser(__doc__, TEST_SPLIT_ORDER).parse_args()
    labeled = label_splits(args.splits, args, allow_test_splits=True)
    out = labels_dir(args, load_config(args.config))
    for split, rows in labeled.items():
        print(f"{split}: {len(rows)} answers labeled -> {out}/auto/{split}.jsonl")


if __name__ == "__main__":
    main()
