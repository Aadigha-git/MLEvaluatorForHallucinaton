from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from hdetect.config import Config, parse_config
from hdetect.schema import Example, make_question_id

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


@pytest.fixture(scope="session")
def raw_config() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())


@pytest.fixture
def raw_cfg(raw_config) -> dict[str, Any]:
    return copy.deepcopy(raw_config)


@pytest.fixture(scope="session")
def cfg(raw_config) -> Config:
    return parse_config(copy.deepcopy(raw_config))


@pytest.fixture(scope="session")
def fixture_examples() -> dict[str, list[Example]]:
    from hdetect.data import load_dataset

    return {name: load_dataset(name, FIXTURES / f"{name}.jsonl")[0] for name in ("truthfulqa", "halueval", "fever")}


def make_example(dataset: str, text: str, **kwargs: Any) -> Example:
    meta = kwargs.pop("meta", {})
    if dataset == "fever":
        meta = {"claim": text, **meta}
    return Example(
        question_id=make_question_id(dataset, text),
        dataset=dataset,
        source_id=text,
        prompt_text=text,
        meta=meta,
        **kwargs,
    )
