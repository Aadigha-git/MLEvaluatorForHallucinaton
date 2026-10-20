"""Step 10 perturbation rules and failure sampling, on synthetic text only."""

from __future__ import annotations

import subprocess
import sys

import pytest
import yaml

from hdetect.evaluation.perturb import (
    HEDGE_PREFIX,
    PAD_SENTENCE,
    entities,
    entity_pool,
    entity_swap,
    hedge_insert,
    hedge_strip,
    length_pad,
    perturb,
    truncate,
)
from hdetect.features.text import hedge_rates
from tests.conftest import ROOT

sys.path.insert(0, str(ROOT / "ml" / "evaluation"))


def test_hedge_strip_removes_every_hedge_and_nothing_else():
    a = "It is possible that Paris is probably the capital, I think."
    out = hedge_strip(a)
    assert hedge_rates(out)[0] == 0.0
    assert "Paris" in out and "capital" in out and ",." not in out
    assert hedge_strip("Paris is the capital.") is None


def test_hedge_insert_and_pad_always_apply():
    assert hedge_insert("Paris.") == HEDGE_PREFIX + "Paris."
    assert hedge_rates(hedge_insert("Paris is the capital."))[0] > 0
    assert length_pad("Paris.") == f"Paris. {PAD_SENTENCE}"
    assert hedge_insert("") is None and length_pad("") is None


def test_truncate_keeps_the_first_sentence():
    assert truncate("Paris is the capital. It is in France.") == "Paris is the capital."
    assert truncate("Paris is the capital.") is None


def test_entities_skip_sentence_starts_and_question_words():
    a = "The capital is Paris. Marie Curie was born in Warsaw."
    assert entities(a, "What is the capital?") == ["Paris", "Warsaw"]
    assert entities(a, "Where is Paris?") == ["Warsaw"]


def test_entity_swap_is_seeded_by_answer_key():
    pool = entity_pool([("It is in Berlin.", "q"), ("It is in Madrid, near Toledo.", "q")])
    a = "The capital is Paris, and Paris is large."
    one = entity_swap(a, "What is the capital?", pool, "ans-1", 7)
    assert one == entity_swap(a, "What is the capital?", pool, "ans-1", 7)
    assert "Paris" not in one and one.count(one.split()[3].rstrip(",")) == 2
    assert entity_swap("no entities here.", "q", pool, "k", 7) is None


def test_perturb_dispatches_every_configured_rule():
    cfg = yaml.safe_load((ROOT / "configs" / "step10.yaml").read_text())
    for name in cfg["perturbation"]["perturbations"]:
        perturb(name, "The capital is Paris. It is large.", "What is it?", ["Rome"], "k", 1)


def test_allocation_is_equal_with_shortfalls_filled():
    from step10 import allocate

    order = ["test_id", "test_ood", "test_heldout"]
    assert allocate({"test_id": 100, "test_ood": 100, "test_heldout": 100}, 50, order) == \
        {"test_id": 17, "test_ood": 17, "test_heldout": 16}
    assert allocate({"test_id": 3, "test_ood": 1, "test_heldout": 0}, 50, order) == \
        {"test_id": 3, "test_ood": 1, "test_heldout": 0}
    assert allocate({"test_id": 5, "test_ood": 100, "test_heldout": 100}, 50, order) == \
        {"test_id": 5, "test_ood": 23, "test_heldout": 22}


def test_majority_of_three_seeds():
    import numpy as np
    from step10 import majority

    assert majority([np.array([1, 0, 1]), np.array([1, 0, 0]), np.array([0, 0, 0])]).tolist() == [1, 0, 0]


def test_step10_config_categories_are_fixed():
    cfg = yaml.safe_load((ROOT / "configs" / "step10.yaml").read_text())
    fa = cfg["failure_analysis"]
    assert fa["tier"] == "resampling" and fa["condition"] == "retrieved" and fa["max_per_error_type"] == 50
    assert "label_doubt" in fa["categories"] and "other" in fa["categories"]
    assert cfg["paths"]["stand_in_reports"].startswith("data/")


@pytest.mark.skipif(yaml.safe_load((ROOT / "configs" / "step10.yaml").read_text())["status"] == "frozen",
                    reason="step 10 config is frozen")
def test_step10_refuses_test_splits_while_provisional():
    out = subprocess.run([sys.executable, str(ROOT / "ml" / "evaluation" / "step10.py"), "perturb"],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode != 0 and "provisional" in out.stderr


def test_step10_outputs_are_not_committed():
    ignored = (ROOT / ".gitignore").read_text().split()
    assert "data/step10_stand_in/" in ignored and "data/features/" in ignored and "data/audit/*" in ignored
