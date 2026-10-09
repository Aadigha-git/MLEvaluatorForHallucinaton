"""The labeling path must not touch the LLM judge or the models used as features."""

import ast
import re
import sys
from pathlib import Path

import pytest

LABELING_DIR = Path(__file__).resolve().parents[1] / "src" / "hdetect" / "labeling"
ALLOWED_THIRD_PARTY = {"numpy", "sklearn.metrics"}
ALLOWED_INTERNAL = {"hdetect.schema", "hdetect.config", "hdetect.data.fever"}
FORBIDDEN_STRINGS = re.compile(
    r"transformers|torch|mnli|deberta|roberta|bart-large|nli-|openai|anthropic|nebius|embedding", re.IGNORECASE
)


def _imports(tree: ast.AST) -> list[str]:
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.append(node.module)
    return names


def _allowed(module: str) -> bool:
    top = module.split(".")[0]
    if top in sys.stdlib_module_names or module == "__future__":
        return True
    if module.startswith("hdetect.labeling") or module == "hdetect.labeling":
        return True
    return module in ALLOWED_INTERNAL or module in ALLOWED_THIRD_PARTY


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


SOURCES = sorted(LABELING_DIR.glob("*.py"))


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_labeling_imports_are_allowlisted(path):
    tree = ast.parse(path.read_text())
    bad = [m for m in _imports(tree) if not _allowed(m)]
    assert bad == [], f"{path.name} imports {bad}"


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_labeling_code_names_no_model(path):
    tree = ast.parse(path.read_text())
    docstrings = _docstring_nodes(tree)
    hits = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
        and FORBIDDEN_STRINGS.search(node.value)
    ]
    assert hits == []


def test_scan_covers_every_labeler():
    assert {p.name for p in SOURCES} >= {"__init__.py", "text.py", "truthfulqa.py", "halueval.py", "fever.py"}
