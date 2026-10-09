"""The authorized entry point for reading test splits during feature extraction.

Computes feature tables for test_id, test_ood, and test_heldout with the same frozen pipeline as
the development splits. It reads questions, gold evidence, and stored answers, never labels, and
fits nothing. Run it only after the pipeline has been validated on train and val_id.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from extract_features import parser, run  # noqa: E402

from hdetect.features import TEST_SPLITS  # noqa: E402

TEST_SPLIT_ORDER = ("test_id", "test_ood", "test_heldout")


def main() -> None:
    assert set(TEST_SPLIT_ORDER) == TEST_SPLITS
    args = parser(__doc__, TEST_SPLIT_ORDER).parse_args()
    run(args.splits, args, allow_test_splits=True)


if __name__ == "__main__":
    main()
