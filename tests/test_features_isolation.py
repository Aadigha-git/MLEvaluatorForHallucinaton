"""Feature extraction never sees labels, and test splits are read only through ml/evaluation/."""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path

import pytest

from hdetect.data.splits import SplitManifest
from hdetect.features import TEST_SPLITS
from hdetect.features.access import (
    ManifestMismatch,
    QuestionInput,
    SplitAccessError,
    load_answers,
    load_inputs,
    to_input,
)
from tests.conftest import ROOT, make_example

FEATURES_DIR = ROOT / "src" / "hdetect" / "features"
LABEL_FIELDS = {"gold_label", "correct_refs", "incorrect_refs", "best_answer", "label", "hand_label"}
ALLOWED_LABELING_IMPORTS = {"hdetect.labeling.text", "hdetect.labeling.fever"}
SOURCES = sorted(FEATURES_DIR.glob("*.py"))


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_feature_code_reads_no_label_fields(path):
    tree = ast.parse(path.read_text())
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    keys = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not (attrs & LABEL_FIELDS), f"{path.name} reads {attrs & LABEL_FIELDS}"
    assert not (keys & LABEL_FIELDS), f"{path.name} names {keys & LABEL_FIELDS}"


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_feature_code_imports_only_text_utilities_from_labeling(path):
    tree = ast.parse(path.read_text())
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    labeling = {m for m in mods if m.startswith("hdetect.labeling")}
    assert labeling <= ALLOWED_LABELING_IMPORTS
    if "hdetect.labeling.fever" in labeling:
        names = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                 and n.module == "hdetect.labeling.fever" for a in n.names}
        assert names == {"parse_verdict"}, "only the answer-derived verdict parser may be used"


def test_question_input_carries_no_labels():
    fields = {f.name for f in dataclasses.fields(QuestionInput)}
    assert not fields & LABEL_FIELDS
    ex = make_example("fever", "A claim.", gold_label="refuted", gold_evidence=("T: s.",),
                      correct_refs=("yes",), incorrect_refs=("no",), meta={"evidence_pages": [["T", "0"]]})
    q = to_input(ex, "train", "model_b")
    flat = json.dumps(dataclasses.asdict(q))
    assert "refuted" not in flat and '"yes"' not in flat and '"no"' not in flat
    assert q.query == "A claim." and q.gold_passages == ("T: s.",) and q.gold_pages == ("T",)


def test_truthfulqa_has_no_gold_passages():
    ex = make_example("truthfulqa", "Q?", correct_refs=("A",))
    assert to_input(ex, "train", "model_a").gold_passages == ()


@pytest.mark.parametrize("split", sorted(TEST_SPLITS))
def test_test_splits_are_refused_without_authorization(split, tmp_path):
    with pytest.raises(SplitAccessError):
        load_inputs(split, tmp_path / "x.jsonl", SplitManifest([]))
    with pytest.raises(SplitAccessError):
        load_answers(split, tmp_path, tmp_path / "m.json")


def _authorizes_test_splits(path: Path) -> bool:
    """A call passing allow_test_splits=True as a literal."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "allow_test_splits" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    return True
    return False


def test_only_ml_evaluation_authorizes_test_splits():
    paths = list(ROOT.glob("src/**/*.py")) + list(ROOT.glob("scripts/**/*.py")) + list(ROOT.glob("ml/**/*.py"))
    offenders = [str(p.relative_to(ROOT)) for p in paths
                 if _authorizes_test_splits(p) and "ml/evaluation" not in p.as_posix()]
    assert offenders == []
    assert _authorizes_test_splits(ROOT / "ml" / "evaluation" / "extract_test_features.py")


def test_dev_entry_point_rejects_test_splits():
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "extract_features.py"), "--splits", "test_id", "--no-retrieval"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert out.returncode != 0 and "invalid choice" in out.stderr


def test_generation_hashes_are_verified(tmp_path):
    gen = tmp_path / "train.jsonl"
    gen.write_text(json.dumps({"question_id": "q", "sample_index": 0, "status": "ok", "text": "x"}) + "\n")
    from hdetect.generation.report import sha256_file

    manifest = tmp_path / "generations.json"
    manifest.write_text(json.dumps({"files": {"train": {"path": str(gen), "sha256": sha256_file(gen)}}}))
    assert set(load_answers("train", tmp_path, manifest)) == {"q"}
    gen.write_text(gen.read_text().replace('"x"', '"y"'))
    with pytest.raises(ManifestMismatch):
        load_answers("train", tmp_path, manifest)


def test_committed_generation_manifest_matches_files():
    manifest = ROOT / "data" / "manifests" / "generations.json"
    gen_dir = ROOT / "data" / "generations"
    if not (gen_dir / "train.jsonl").exists():
        pytest.skip("generation files not present on this machine")
    from hdetect.features.access import verify_generation_file

    for split in ("train", "val_id"):
        verify_generation_file(split, gen_dir, manifest)


def test_feature_tables_are_not_committed():
    ignored = (ROOT / ".gitignore").read_text().split()
    assert {"data/features/", "data/generations/", "data/indexes/"} <= set(ignored)


def test_test_split_constants_match_split_roles():
    from hdetect.config import SPLIT_ROLES

    assert TEST_SPLITS == {s for s in SPLIT_ROLES if s.startswith("test")}
    assert Path(ROOT / "ml" / "evaluation").is_dir()
