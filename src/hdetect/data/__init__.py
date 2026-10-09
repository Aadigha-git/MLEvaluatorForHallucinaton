from __future__ import annotations

from pathlib import Path

from hdetect.config import Config
from hdetect.data import fever, halueval, truthfulqa
from hdetect.data.io import download, read_rows
from hdetect.schema import Example


def load_dataset(name: str, path: str | Path) -> tuple[list[Example], dict[str, int]]:
    rows = read_rows(path)
    if name == "truthfulqa":
        examples = truthfulqa.examples_from_rows(rows)
        return examples, {"rows_in": len(rows), "examples_out": len(examples)}
    if name == "halueval":
        examples = halueval.examples_from_rows(rows)
        return examples, {"rows_in": len(rows), "examples_out": len(examples)}
    if name == "fever":
        return fever.examples_from_rows(rows)
    raise ValueError(f"unknown dataset: {name}")


def fetch_all(cfg: Config) -> dict[str, tuple[list[Example], dict[str, int]]]:
    return {
        name: load_dataset(name, download(source, cfg.paths["raw"]))
        for name, source in cfg.datasets.items()
    }
