"""Label store: immutability, dev/test entry points, and blind exports."""

from __future__ import annotations

import csv
import subprocess
import sys

import pytest

from hdetect.features import TEST_SPLITS
from hdetect.features.access import SplitAccessError
from hdetect.labeling.audit import BLIND_COLUMNS
from hdetect.labels import LabelFileMismatch, read_auto_labels, write_auto_labels
from tests.conftest import ROOT


def test_auto_labels_are_written_once(tmp_path):
    rows = [{"answer_id": "a", "label": "hallucinated", "scores": {"c": 0.5}}]
    path = write_auto_labels(rows, "train", tmp_path)
    assert write_auto_labels(rows, "train", tmp_path) == path
    with pytest.raises(LabelFileMismatch):
        write_auto_labels([{**rows[0], "label": "not_hallucinated"}], "train", tmp_path)


@pytest.mark.parametrize("split", sorted(TEST_SPLITS))
def test_test_split_labels_need_authorization(split, tmp_path):
    with pytest.raises(SplitAccessError):
        read_auto_labels(split, tmp_path)


def test_dev_labeling_entry_point_rejects_test_splits():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "label_answers.py"), "--splits", "test_id"],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode != 0 and "invalid choice" in out.stderr


def test_blind_exports_never_show_automatic_labels():
    assert "label" not in BLIND_COLUMNS
    for name in ("unresolved_dev_blind.csv", "audit_sample_blind.csv"):
        path = ROOT / "data" / "audit" / name
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        assert tuple(header) == BLIND_COLUMNS
