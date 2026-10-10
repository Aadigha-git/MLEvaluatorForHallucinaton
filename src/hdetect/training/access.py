"""Record every file a run opens under data/, so a run can prove it never read a test split."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from hdetect.features import TEST_SPLITS
from hdetect.training.data import ACCESS_LOG

OPENED: list[str] = []


def _record_open(event: str, args: tuple) -> None:
    if event == "open" and args and isinstance(args[0], (str, Path)):
        OPENED.append(str(args[0]))


def install_audit_hook() -> None:
    sys.addaudithook(_record_open)


def is_test_path(path: str) -> bool:
    name = Path(path).name
    return any(name.startswith(s) or f"/{s}/" in path for s in TEST_SPLITS)


def data_access_table(root: Path) -> pd.DataFrame:
    paths = set()
    for p in OPENED + ACCESS_LOG:
        resolved = Path(p).resolve() if not Path(p).is_absolute() else Path(p)
        try:
            rel = resolved.relative_to(root)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] == "data":
            paths.add(rel.as_posix())
    rows = [{"path": p, "test_split": is_test_path(p)} for p in sorted(paths)]
    return pd.DataFrame(rows, columns=["path", "test_split"])
