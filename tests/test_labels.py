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


def test_rule_versions_never_share_files():
    from hdetect.labels import audit_file, version_dir

    assert version_dir("1").as_posix() == "data/labels/v1" and version_dir("2").as_posix() == "data/labels/v2"
    assert audit_file("audit_sample", "1", "_blind.csv") == "audit_sample_blind.csv"
    assert audit_file("audit_sample", "2", "_key.jsonl") == "audit_sample_v2_key.jsonl"
    assert audit_file("unresolved_dev", "2", "_blind.csv") == "unresolved_dev_v2_blind.csv"


def test_audit_rounds_are_fixed_before_drawing():
    import importlib.util

    spec = importlib.util.spec_from_file_location("export_audit_sample", ROOT / "ml" / "evaluation" / "export_audit_sample.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.AUDIT_ROUNDS == {"1": {"seed": 20261008, "exclude_rounds": ()},
                                "2": {"seed": 20261011, "exclude_rounds": ("1",)}}


def test_blind_exports_never_show_automatic_labels():
    assert "label" not in BLIND_COLUMNS
    for name in ("unresolved_dev_blind.csv", "audit_sample_blind.csv", "unresolved_dev_v2_blind.csv",
                 "audit_sample_v2_blind.csv"):
        path = ROOT / "data" / "audit" / name
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        assert tuple(header) == BLIND_COLUMNS
