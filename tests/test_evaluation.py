"""Step 9 statistics and guards, on synthetic data only. No test split is read here."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from hdetect.evaluation.data import SplitData, load_split, read_auto
from hdetect.evaluation.stats import (
    MIXED,
    NO,
    YES,
    Scored,
    all_hallucinated_f1,
    band_rates,
    fixed_predictions,
    interval,
    judge_predictions,
    point_f1,
    replicate,
    rq1_split,
    rq2_split,
    rq_outcome,
    seed_point,
)
from hdetect.features import TEST_SPLITS
from hdetect.features.access import SplitAccessError
from hdetect.training.metrics import Threshold
from tests.conftest import ROOT

R, CHUNK = 400, 100


def toy(n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, n), np.repeat(np.arange(n // 3), 3)


def scored(name: str, scores, value: float = 0.5) -> Scored:
    scores = [np.asarray(s, dtype=float) for s in scores]
    return Scored(name, scores, [Threshold(value, float("nan"))] * len(scores))


def test_missing_judge_verdict_is_scored_wrong():
    y = np.array([1, 0, 1, 0])
    assert judge_predictions([1, None, None, 0], y).tolist() == [1, 1, 0, 0]


def test_all_hallucinated_f1_is_2p_over_1_plus_p():
    y = np.array([1, 1, 0, 0, 0])
    assert all_hallucinated_f1(y) == pytest.approx(2 * 0.4 / 1.4)
    assert all_hallucinated_f1(np.zeros(4, dtype=int)) == 0.0


def test_flag_everything_classifier_never_beats_the_trivial_baseline():
    y, g = toy()
    clf = scored("clf", [np.ones(len(y))] * 3)
    rep = replicate([clf], y, g, R, 7, CHUNK)
    assert np.array_equal(rep.f1["clf"], rep.all_hallucinated)
    point = {"clf": seed_point(clf, y)}
    r = rq2_split(rep, rep, point["clf"], point["clf"], "clf", 0.05, y)
    assert r["vs_trivial"] == 0.0 and r["vs_trivial_ci_low"] == 0.0 and not r["beats_trivial"]


def test_seed_average_is_taken_within_each_replicate():
    y, g = toy()
    rng = np.random.default_rng(1)
    a, b = rng.random(len(y)), rng.random(len(y))
    both = replicate([scored("ab", [a, b]), scored("a", [a]), scored("b", [b])], y, g, R, 3, CHUNK)
    np.testing.assert_allclose(both.f1["ab"], (both.f1["a"] + both.f1["b"]) / 2, atol=1e-11)
    p = seed_point(scored("ab", [a, b]), y)
    assert p["n_seeds"] == 2 and p["f1_seed_min"] <= p["f1"] <= p["f1_seed_max"]


def test_paired_replicates_identical_evaluators_differ_by_zero():
    y, g = toy()
    s = np.random.default_rng(2).random(len(y))
    rep = replicate([scored("x", [s]), scored("y", [s.copy()])], y, g, R, 5, CHUNK)
    point = {k: seed_point(scored(k, [s]), y) for k in ("x", "y")}
    r = rq1_split(rep, point, "x", "y")
    assert r["diff"] == 0.0 and r["diff_ci_low"] == 0.0 and not r["passes"]


def test_rq1_passes_when_the_classifier_is_clearly_better():
    y, g = toy(600)
    perfect = scored("clf", [y.astype(float)] * 3)
    noisy = scored("single", [np.where(np.random.default_rng(3).random(len(y)) < 0.6, y, 1 - y).astype(float)])
    rep = replicate([perfect, noisy], y, g, R, 11, CHUNK)
    point = {e.name: seed_point(e, y) for e in (perfect, noisy)}
    r = rq1_split(rep, point, "clf", "single")
    assert r["passes"] and r["diff_ci_low"] > 0 and r["degenerate_replicates"] == 0
    auc = rq1_split(rep, point, "clf", "single", "auroc")
    assert auc["metric"] == "auroc" and auc["passes"]


def test_judges_are_fixed_predictions_without_auroc():
    y, g = toy()
    j = fixed_predictions("judge", judge_predictions([None] * len(y), y))
    rep = replicate([j], y, g, R, 1, CHUNK)
    assert np.isnan(rep.auroc["judge"]).all() and (rep.f1["judge"] == 0).all()
    assert np.isnan(seed_point(j, y)["auroc"])


def test_splits_use_independent_draws_for_rq2():
    y, g = toy()
    clf = scored("clf", [np.random.default_rng(4).random(len(y))])
    a = replicate([clf], y, g, R, 100, CHUNK)
    b = replicate([clf], y, g, R, 100 + 10_000_000, CHUNK)
    assert not np.array_equal(a.f1["clf"], b.f1["clf"])
    again = replicate([clf], y, g, R, 100, CHUNK)
    assert np.array_equal(a.f1["clf"], again.f1["clf"])


def test_rq2_margin_and_trivial_baseline_rules():
    y, g = toy(600)
    good = scored("clf", [y.astype(float)])
    rep = replicate([good], y, g, R, 9, CHUNK)
    pt = seed_point(good, y)
    ref_point = dict(pt, f1=pt["f1"] + 0.05)
    assert rq2_split(rep, rep, ref_point, pt, "clf", 0.05, y)["passes"]
    ref_point = dict(pt, f1=pt["f1"] + 0.0501)
    r = rq2_split(rep, rep, ref_point, pt, "clf", 0.05, y)
    assert not r["drop_within_margin"] and r["beats_trivial"] and not r["passes"]
    assert rq2_split(rep, rep, pt, pt, "clf", 0.05, y, "auroc")["trivial_baseline"] == 0.5


def test_rq_outcome():
    assert rq_outcome([True, True]) == YES
    assert rq_outcome([True, False]) == MIXED
    assert rq_outcome([False, False]) == NO


def test_interval_skips_nan_replicates():
    ci = interval(np.array([np.nan, 0.0, 1.0, 0.5]))
    assert ci["skipped_replicates"] == 1 and 0.0 <= ci["ci_low"] <= ci["ci_high"] <= 1.0


def test_point_f1_degenerate_is_zero():
    assert point_f1(np.zeros(3), np.zeros(3, dtype=int)) == 0.0


def test_band_rates():
    P = np.array([0.1, 0.2, 0.5, 0.9, 0.95])
    y = np.array([0, 0, 1, 1, 1])
    r = band_rates(P, y, 0.3, 0.8)
    assert (r["accept_rate"], r["escalate_rate"], r["flag_rate"]) == (0.4, 0.2, 0.4)
    assert r["accept_hallucinated_share"] == 0.0 and r["flag_hallucinated_share"] == 1.0


def test_feature_matrix_is_c_ordered():
    X = pd.DataFrame(np.asfortranarray(np.arange(12, dtype=float).reshape(4, 3)), columns=list("abc"))
    d = SplitData("val_id", "none", X, np.zeros(4, dtype=int), np.arange(4), pd.DataFrame(), pd.DataFrame())
    assert d.subset(("a", "c")).flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("split", sorted(TEST_SPLITS))
def test_test_splits_need_explicit_authorization(split, tmp_path):
    with pytest.raises(SplitAccessError):
        read_auto(tmp_path, split)
    with pytest.raises(SplitAccessError):
        load_split(split, "retrieved", features_dir=tmp_path, labels=None, auto_rows=[], manifest=None,
                   expected_versions={})


def test_evaluation_config_matches_the_preregistration():
    e = yaml.safe_load((ROOT / "configs" / "evaluation.yaml").read_text())
    assert e["rq1_splits"] == e["rq2_splits"] == ["test_ood", "test_heldout"]
    assert e["rq2_reference"] == "test_id" and e["rq2_margin"] == 0.05
    assert e["bootstrap"]["replicates"] == 10000 and e["bootstrap"]["seed"] == 20261008
    assert e["primary_condition"] == "retrieved" and e["secondary_condition"] == "none"
    assert set(e["splits"]) == TEST_SPLITS and e["gold_splits"] == ["test_id", "test_heldout"]


def _provisional() -> bool:
    return yaml.safe_load((ROOT / "configs" / "evaluation.yaml").read_text())["status"] != "frozen"


@pytest.mark.skipif(not _provisional(), reason="evaluation config is frozen")
@pytest.mark.parametrize("script", ["evaluate.py", "run_test_judges.py"])
def test_test_entry_points_refuse_a_provisional_config(script):
    out = subprocess.run([sys.executable, str(ROOT / "ml" / "evaluation" / script)],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode != 0 and "provisional" in out.stderr


def test_stand_in_outputs_are_not_committed():
    assert "data/step9_stand_in/" in (ROOT / ".gitignore").read_text().split()
