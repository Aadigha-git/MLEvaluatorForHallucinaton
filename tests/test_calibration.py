"""Step 7 calibration and decision policy, plus the Step 6 report annotations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.linear_model import LogisticRegression

from hdetect.training.calibration import (
    METHODS,
    calibrated,
    choose_method,
    cross_fit,
    ece,
    fit_calibrator,
    policy_thresholds,
    reliability,
    summarize_ece,
)
from hdetect.training.metrics import Evaluator, Threshold, bootstrap
from hdetect.training.report import FLAG_COLUMN, annotate_tables, write_readme

ROOT = Path(__file__).resolve().parents[1]


# ECE and reliability

def test_ece_is_zero_when_bins_match_observed_rates():
    p = np.array([0.25] * 4 + [0.75] * 4)
    y = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    assert ece(p, y) == pytest.approx(0.0)


def test_ece_weights_bins_by_size_and_puts_one_in_last_bin():
    p = np.array([0.05, 0.05, 1.0, 1.0])
    y = np.array([0, 0, 0, 1])
    assert ece(p, y) == pytest.approx(0.5 * 0.05 + 0.5 * 0.5)
    rel = reliability(p, y)
    assert len(rel) == 10 and rel.loc[9, "n"] == 2 and rel.loc[0, "n"] == 2


# Method choice: lower mean ECE wins, a difference under 0.005 goes to Platt

@pytest.mark.parametrize("platt, iso, expected", [
    (0.050, 0.046, "platt"),
    (0.050, 0.045, "isotonic"),
    (0.040, 0.050, "platt"),
    (0.050, 0.050, "platt"),
])
def test_choose_method_margin(platt, iso, expected):
    assert choose_method({"platt": platt, "isotonic": iso}, 0.005) == expected


def test_summarize_ece_averages_over_folds_and_seeds():
    rows = [{"tier": "t", "condition": "c", "seed": s, "fold": f, "method": m, "n": 1,
             "ece": {"platt": 0.10, "isotonic": 0.02}[m] + 0.01 * s}
            for s in (0, 1) for f in range(5) for m in METHODS]
    out = summarize_ece(pd.DataFrame(rows), ["tier", "condition"], 0.005)
    assert out["ece_platt"].item() == pytest.approx(0.105)
    assert out["ece_isotonic"].item() == pytest.approx(0.025)
    assert out["chosen_method"].item() == "isotonic"


# Calibrators: fitted on val data around a frozen classifier

def _toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] + rng.normal(scale=1.0, size=n) > 0).astype(int)
    groups = np.array([f"c{i // 3}" for i in range(n)])
    return X, y, groups


def test_calibrator_leaves_the_classifier_and_its_ranking_untouched():
    X, y, _ = _toy()
    est = LogisticRegression().fit(X, y)
    coef = est.coef_.copy()
    before = est.predict_proba(X)[:, 1]
    for method in METHODS:
        cal = fit_calibrator(est, method, X, y)
        np.testing.assert_array_equal(est.coef_, coef)
        np.testing.assert_array_equal(est.predict_proba(X)[:, 1], before)
        p = calibrated(cal, X)
        order = np.argsort(before)
        assert np.all(np.diff(p[order]) >= -1e-12)


def test_cross_fit_measures_every_fold_out_of_sample_and_is_deterministic():
    X, y, groups = _toy()
    est = LogisticRegression().fit(X, y)
    a, oof = cross_fit(est, X, y, groups, seed=7, n_splits=5, n_bins=10)
    b, _ = cross_fit(est, X, y, groups, seed=7, n_splits=5, n_bins=10)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 5 * len(METHODS) and a.groupby("method")["n"].sum().eq(len(y)).all()
    assert all(np.isfinite(v).all() for v in oof.values())


# Decision policy

def test_policy_accepts_up_to_five_percent_hallucinated_exactly():
    p = np.arange(40) / 40
    y = np.zeros(40, dtype=int)
    y[[0, 25, 30, 35]] = 1
    pol = policy_thresholds(p, y, 0.05, 0.20)
    accepted = p < pol.t_low
    assert pol.feasible and y[accepted].sum() / accepted.sum() <= 0.05
    assert pol.t_low == pytest.approx(25 / 40)
    assert pol.accept_hallucinated_share == pytest.approx(1 / 25)


def test_policy_escalates_at_most_twenty_percent_and_flags_the_rest():
    p = np.arange(40) / 40
    y = np.zeros(40, dtype=int)
    y[[0, 25, 30, 35]] = 1
    pol = policy_thresholds(p, y, 0.05, 0.20)
    bands = pol.band(p)
    assert (bands == "escalate").mean() <= 0.20
    assert pol.t_high == pytest.approx((25 + 8) / 40)
    assert pol.accept_rate + pol.escalate_rate + pol.flag_rate == pytest.approx(1.0)
    assert set(bands[p >= pol.t_high]) == {"flag"}


def test_policy_is_infeasible_when_no_accept_band_meets_the_goal():
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y = np.array([1, 1, 0, 1, 0])
    pol = policy_thresholds(p, y, 0.05, 0.20)
    assert not pol.feasible and pol.accept_rate == 0.0
    assert pol.t_low == pytest.approx(0.1)
    assert pol.escalate_rate <= 0.20


def test_policy_reports_plateaus():
    p = np.array([0.1] * 6 + [0.9] * 4)
    y = np.array([0] * 6 + [1] * 4)
    pol = policy_thresholds(p, y, 0.05, 0.20)
    assert pol.n_distinct == 2 and pol.t_low == pytest.approx(0.9)
    assert pol.plateau_share_t_low == pytest.approx(0.4)


def test_calibration_config_matches_preregistration():
    c = yaml.safe_load((ROOT / "configs" / "calibration.yaml").read_text())
    assert c["calibration"] == {"methods": ["platt", "isotonic"], "ece_bins": 10, "n_splits": 5, "platt_margin": 0.005}
    assert c["policy"] == {"accept_max_hallucinated": 0.05, "escalate_max_share": 0.20}
    assert "data/models/" in (ROOT / ".gitignore").read_text().split()


# Bootstrap: degenerate and skipped counts for difference intervals

def test_bootstrap_counts_degenerate_replicates_for_differences():
    y = np.array([1, 0, 1, 0, 1, 0])
    groups = np.array(["a", "b", "c", "d", "e", "f"])
    a = Evaluator("a", [np.arange(6.0)], [Threshold(100.0, 0.0)])
    b = Evaluator("b", [np.arange(6.0)], [Threshold(0.0, 0.0)])
    out = bootstrap([a, b], y, groups, 200, 3, 50, pairs=[("a", "b")])
    diff = out["a - b"]
    assert {"f1_degenerate_replicates", "auroc_skipped_replicates", "auprc_skipped_replicates"} <= set(diff)
    assert diff["f1_degenerate_replicates"] >= out["a"]["f1_degenerate_replicates"]
    assert diff["auroc_skipped_replicates"] == out["a"]["auroc_skipped_replicates"]


# Step 6 report annotations

def _tables():
    labels = pd.DataFrame({"split": ["train", "val_id"], "exclusion_rate": [0.092, 0.005],
                           "flagged_over_5pct": [True, False]})
    vs = pd.DataFrame({
        "tier": ["single_shot"] * 4, "eval_condition": ["none"] * 4,
        "evaluator": ["logistic_regression", "feature:log_length", "all_hallucinated",
                      "logistic_regression - feature:log_length"],
        "f1_degenerate_replicates": [0, 0, np.nan, np.nan], "auroc_skipped_replicates": [0, 0, np.nan, np.nan],
    })
    logo = pd.DataFrame({
        "tier": ["single_shot"] * 3, "train_condition": ["retrieved"] * 3, "eval_condition": ["retrieved"] * 3,
        "subset": ["all", "-evidence_dependent", "only:evidence_free"],
    })
    corr = pd.DataFrame({"split": ["train", "val_id"], "r": [0.1, 0.2]})
    return {"labels_summary": labels, "validation_summary": vs, "ablation_leave_one_group_out": logo,
            "correlations": corr, "cv_results": pd.DataFrame({"x": [1]}), "single_features": pd.DataFrame({"x": [1]}),
            "data_access": pd.DataFrame({"path": ["data/x"], "test_split": [False]})}


def test_annotation_flags_only_rows_that_use_the_flagged_split():
    t = annotate_tables(_tables())
    vs = t["validation_summary"].set_index("evaluator")[FLAG_COLUMN]
    assert vs["logistic_regression"] == "train"
    assert vs["logistic_regression - feature:log_length"] == "train"
    assert vs["feature:log_length"] == "" and vs["all_hallucinated"] == ""
    assert list(t["correlations"][FLAG_COLUMN]) == ["train", ""]
    assert list(t["cv_results"][FLAG_COLUMN]) == ["train"]
    assert list(t["single_features"][FLAG_COLUMN]) == [""]
    assert FLAG_COLUMN not in t["data_access"]


def test_annotation_backfills_difference_counts_only_when_exact():
    vs = annotate_tables(_tables())["validation_summary"].set_index("evaluator")
    assert vs.loc["logistic_regression - feature:log_length", "f1_degenerate_replicates"] == 0
    assert pd.isna(vs.loc["all_hallucinated", "f1_degenerate_replicates"])


def test_annotation_names_identical_two_group_ablation_rows():
    logo = annotate_tables(_tables())["ablation_leave_one_group_out"].set_index("subset")["same_features_as"]
    assert logo["-evidence_dependent"] == "only:evidence_free"
    assert logo["only:evidence_free"] == "-evidence_dependent"
    assert logo["all"] == ""


def test_step6_readme_carries_flag_kappas_and_counts(tmp_path):
    import importlib.util

    from hdetect.training.run import Step6

    spec = importlib.util.spec_from_file_location("step6_fixtures", ROOT / "tests" / "test_training.py")
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    t = fixtures.small_config()
    inputs = fixtures.synthetic_inputs()
    inputs.label_summaries = [{"split": "train", "answers": 120, "labeled": 110, "exclusion_rate": 0.0917,
                               "flagged_over_5pct": True},
                              {"split": "val_id", "answers": 100, "labeled": 100, "exclusion_rate": 0.0,
                               "flagged_over_5pct": False}]
    tables = annotate_tables(dict(Step6(t, inputs).run(), data_access=pd.DataFrame({"path": ["data/x"], "test_split": [False]})))
    text = write_readme(tables, inputs.notes, t)
    assert "0.7117" in text and "0.578" in text
    assert text.count("**Exclusion flag.**") >= 5
    assert "F1 degenerate" in text and "AUROC skipped" in text and "same_features_as" in text
    vs = tables["validation_summary"]
    diffs = vs[vs["evaluator"].str.contains(" - ")]
    assert diffs["f1_degenerate_replicates"].notna().all() and diffs["auroc_skipped_replicates"].notna().all()
    assert "sensitivity" not in text.lower()
