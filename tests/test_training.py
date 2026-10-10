"""Step 6: split isolation, group separation, fold-local preprocessing, tier separation, determinism,
metrics, labels, and report completeness. Synthetic data only; no real labels are needed."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from hdetect.data.splits import Assignment, SplitManifest
from hdetect.features import EVIDENCE_FEATURES, RESAMPLING_FEATURES, TEST_SPLITS, feature_columns
from hdetect.features.access import SplitAccessError
from hdetect.labeling.audit import read_hand_labels
from hdetect.labels import IncompleteLabels, LabelFileMismatch, final_labels
from hdetect.training import TIERS, tier_features, training_condition
from hdetect.training.data import AlignmentError, DevData, load_dev, read_auto, read_table
from hdetect.training.metrics import (
    Threshold,
    best_oriented_threshold,
    best_threshold,
    bootstrap,
    cluster_weights,
    Evaluator,
    weighted_f1,
    weighted_ranking,
)
from hdetect.training.models import Config, Subset, cv_results, fit, folds, grid, make_estimator, select
from hdetect.training.report import write_readme
from hdetect.training.run import REPORT_FILES, Inputs, Step6, write_tables
from tests.conftest import ROOT

TRAINING_CONFIG = yaml.safe_load((ROOT / "configs" / "training.yaml").read_text())


def small_config() -> dict:
    t = copy.deepcopy(TRAINING_CONFIG)
    t["logistic_regression"].update(C=[0.1, 1], class_weight=[None, "balanced"])
    t["hist_gradient_boosting"].update(max_depth=[2], learning_rate=[0.1], max_iter=[10, 20])
    t["bootstrap"].update(replicates=60, chunk=25)
    t["permutation_importance"]["n_repeats"] = 3
    t["runtime"].update(n_jobs=1, latency_rows=5)
    return t


def make_dev(split: str, condition: str, n: int, seed: int, signal: float = 1.5) -> DevData:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    feats = feature_columns(condition)
    X = pd.DataFrame({f: rng.normal(size=n) + (signal * y if i % 3 == 0 else 0) for i, f in enumerate(feats)})
    groups = np.array([f"{split}-c{i // 2}" for i in range(n)])
    meta = pd.DataFrame({"answer_id": [f"{split}-q{i}/m/0" for i in range(n)], "question_id": [f"{split}-q{i}" for i in range(n)],
                         "meta_truncated_primary": False, "meta_n_resamples_usable": 5, "abstained": False})
    return DevData(split, condition, X, y, groups, meta, {"rows": n, "excluded_undecidable": 0})


def synthetic_inputs() -> Inputs:
    train = {c: make_dev("train", c, 120, 1) for c in ("retrieved", "none")}
    val = {c: make_dev("val_id", c, 100, 2) for c in ("retrieved", "none", "gold")}
    labels = [{"split": "train", "answers": 120, "labeled": 120}, {"split": "val_id", "answers": 100, "labeled": 100}]
    return Inputs(train, val, labels, [{"check": "synthetic", "ok": True}],
                  {"retrieval": "r", "datasets": "d", "feature_times": "t"})


# Tiers and conditions

def test_single_shot_tier_never_uses_resampling_features():
    for cond in ("retrieved", "none", "gold"):
        assert not set(tier_features("single_shot", cond)) & set(RESAMPLING_FEATURES)
        assert set(RESAMPLING_FEATURES) <= set(tier_features("resampling", cond))
        assert len(tier_features("resampling", cond)) == len(tier_features("single_shot", cond)) + 3
    assert len(tier_features("single_shot", "retrieved")) == 9


def test_no_evidence_condition_has_no_evidence_features():
    for tier in TIERS:
        assert not set(tier_features(tier, "none")) & set(EVIDENCE_FEATURES)
    assert tier_features("single_shot", "gold") == tier_features("single_shot", "retrieved")
    assert training_condition("gold") == "retrieved"


# Split isolation

@pytest.mark.parametrize("split", sorted(TEST_SPLITS))
def test_training_loaders_refuse_test_splits(split, tmp_path):
    with pytest.raises(SplitAccessError):
        read_table(tmp_path, "retrieved", split)
    with pytest.raises(SplitAccessError):
        read_auto(tmp_path, split)


TRAINING_SOURCES = sorted((ROOT / "src" / "hdetect" / "training").glob("*.py")) + [
    ROOT / "scripts" / "train_classifiers.py", ROOT / "scripts" / "calibrate.py"]


@pytest.mark.parametrize("path", TRAINING_SOURCES, ids=lambda p: p.name)
def test_training_code_names_no_test_split_and_never_authorizes_one(path):
    tree = ast.parse(path.read_text())
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not any(s in TEST_SPLITS for s in strings), f"{path.name} names a test split"
    kws = {kw.arg for n in ast.walk(tree) if isinstance(n, ast.Call) for kw in n.keywords}
    assert "allow_test_splits" not in kws
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not any(m.startswith("ml") for m in mods)


def test_access_audit_flags_test_split_paths():
    import importlib.util

    spec = importlib.util.spec_from_file_location("train_classifiers", ROOT / "scripts" / "train_classifiers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.is_test_path("data/features/tables/retrieved/test_id.parquet")
    assert mod.is_test_path("data/labels/auto/test_heldout.jsonl")
    assert not mod.is_test_path("data/features/tables/retrieved/train.parquet")
    assert not mod.is_test_path("data/labels/auto/val_id.jsonl")


# Alignment of features, labels, and the manifest

def _write_split(tmp_path, n=6, tamper=None):
    qids = [f"truthfulqa-{i:04d}" for i in range(n)]
    manifest = SplitManifest([Assignment(q, "truthfulqa", "train", "model_a", f"c{i}") for i, q in enumerate(qids)])
    rows = []
    for i, q in enumerate(qids):
        row = {"question_id": q, "answer_id": f"{q}/model_a/0", "dataset": "truthfulqa", "generator": "model_a",
               "split": "train", "meta_raw_sha256": f"h{i}", "meta_truncated_primary": False, "meta_n_resamples_usable": 5,
               "feature_version": "features-v1"}
        row.update({f: float(i) for f in feature_columns("none")})
        rows.append(row)
    df = pd.DataFrame(rows)
    if tamper:
        tamper(df)
    (tmp_path / "tables" / "none").mkdir(parents=True)
    df.to_parquet(tmp_path / "tables" / "none" / "train.parquet")
    auto = [{"answer_id": f"{q}/model_a/0", "question_id": q, "label": "hallucinated" if i % 2 else "not_hallucinated",
             "raw_sha256": f"h{i}", "abstained": False} for i, q in enumerate(qids)]
    labels = final_labels("train", auto, {})
    return dict(features_dir=tmp_path, labels=labels, auto_rows=auto, manifest=manifest,
                expected_versions={"feature_version": "features-v1"})


def test_load_dev_aligns_rows_labels_and_clusters(tmp_path):
    data = load_dev("train", "none", **_write_split(tmp_path))
    assert list(data.y) == [0, 1, 0, 1, 0, 1]
    assert list(data.groups) == [f"c{i}" for i in range(6)]
    assert list(data.X.columns) == list(feature_columns("none"))


@pytest.mark.parametrize("tamper", [
    lambda df: df.drop(index=0, inplace=True),
    lambda df: df.__setitem__("meta_raw_sha256", ["x"] * len(df)),
    lambda df: df.__setitem__("feature_version", ["features-v0"] * len(df)),
    lambda df: df.__setitem__("hedge_rate", [np.nan] * len(df)),
], ids=["missing_question", "hash_mismatch", "version_mismatch", "non_finite"])
def test_load_dev_rejects_misaligned_tables(tmp_path, tamper):
    with pytest.raises(AlignmentError):
        load_dev("train", "none", **_write_split(tmp_path, tamper=tamper))


# Labels

def _auto(labels):
    return [{"answer_id": f"a{i}", "label": lab, "abstained": False} for i, lab in enumerate(labels)]


def test_training_requires_a_hand_decision_for_every_unresolved_answer():
    auto = _auto(["hallucinated", "unresolved", "not_hallucinated", "unresolved"])
    with pytest.raises(IncompleteLabels):
        final_labels("train", auto, {"a1": "hallucinated"})
    labels = final_labels("train", auto, {"a1": "hallucinated", "a3": "undecidable"})
    assert labels.y == {"a0": 1, "a1": 1, "a2": 0}
    assert labels.excluded == ("a3",)
    assert labels.flagged and labels.counts["hand_labeled"] == 1


def test_hand_labels_cannot_override_automatic_labels():
    with pytest.raises(LabelFileMismatch):
        final_labels("train", _auto(["hallucinated"]), {"a0": "not_hallucinated"})


def test_undecidable_only_allowed_in_the_unresolved_queue(tmp_path):
    path = tmp_path / "q.csv"
    path.write_text("answer_id,hand_label\na1,undecidable\na2,\n")
    with pytest.raises(ValueError):
        read_hand_labels(path)
    assert read_hand_labels(path, allow_undecidable=True) == {"a1": "undecidable"}


# Folds and preprocessing

def test_folds_keep_every_cluster_on_one_side():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(60), 3)
    y = np.repeat(rng.integers(0, 2, 60), 3)
    for seed in TRAINING_CONFIG["seeds"]:
        fs = folds(y, groups, seed, 5)
        assert len(fs) == 5
        assert sorted(np.concatenate([te for _, te in fs])) == list(range(len(y)))
        for tr, te in fs:
            assert not set(groups[tr]) & set(groups[te])


def test_folds_depend_on_the_seed():
    groups = np.arange(100)
    y = np.arange(100) % 2
    a, b = folds(y, groups, 1, 5), folds(y, groups, 2, 5)
    assert any(not np.array_equal(x[1], z[1]) for x, z in zip(a, b))


def test_scaler_is_fitted_on_the_training_fold_only():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    X[50:] += 10
    y = np.arange(100) % 2
    tr = np.arange(50)
    est = fit(Config("logistic_regression", (("C", 1), ("class_weight", None))), 1, X[tr], y[tr], TRAINING_CONFIG)
    np.testing.assert_allclose(est.named_steps["scale"].mean_, X[tr].mean(axis=0))
    assert not np.allclose(est.named_steps["scale"].mean_, X.mean(axis=0))


def test_estimators_match_the_preregistered_grids():
    lr = grid("logistic_regression", TRAINING_CONFIG)
    hgb = grid("hist_gradient_boosting", TRAINING_CONFIG)
    assert sorted({c.p["C"] for c in lr}) == [0.01, 0.1, 1, 10] and len(lr) == 8
    assert {c.p["class_weight"] for c in lr} == {None, "balanced"}
    assert len(hgb) == 8
    est = make_estimator(hgb[0], 7, TRAINING_CONFIG)
    assert est.min_samples_leaf == 20 and est.early_stopping is False and est.random_state == 7
    pipe = make_estimator(lr[0], 7, TRAINING_CONFIG)
    assert pipe.named_steps["clf"].solver == "lbfgs" and pipe.named_steps["clf"].max_iter == 1000
    assert TRAINING_CONFIG["seeds"] == [20261008, 20261009, 20261010]
    assert TRAINING_CONFIG["cv"] == {"n_splits": 5, "scoring": "average_precision"}


def test_fit_refuses_non_finite_features():
    X = np.array([[0.0], [np.nan], [1.0], [2.0]])
    with pytest.raises(ValueError):
        fit(grid("logistic_regression", TRAINING_CONFIG)[0], 1, X, np.array([0, 1, 0, 1]), TRAINING_CONFIG)


def test_selection_ties_go_to_the_simpler_setting():
    rows = []
    for c in grid("logistic_regression", TRAINING_CONFIG):
        for fold in range(5):
            rows.append({"subset": "all", "family": "logistic_regression", "config": c.key, "seed": 1, "fold": fold,
                         "average_precision": 0.7 if c.p["C"] >= 0.1 else 0.6})
    sel = select(pd.DataFrame(rows), TRAINING_CONFIG)
    assert sel.iloc[0]["config"] == "C=0.1,class_weight=None"


def test_cv_is_deterministic_and_independent_of_parallelism():
    data = make_dev("train", "none", 120, 3)
    t = small_config()
    args = (data.X, data.y, data.groups, [Subset("all", feature_columns("none"))], ["logistic_regression", "hist_gradient_boosting"], [1, 2], t)
    a = cv_results(*args, n_jobs=1)
    b = cv_results(*args, n_jobs=1)
    c = cv_results(*args, n_jobs=2)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(a, c)


# Thresholds and metrics

def test_threshold_maximizes_f1_with_ties_to_the_lowest():
    s = np.array([0.1, 0.2, 0.3, 0.4])
    y = np.array([0, 1, 0, 1])
    th = best_threshold(s, y)
    assert th.value == 0.2 and th.f1 == pytest.approx(0.8)
    tie = best_threshold(np.array([0.1, 0.9]), np.array([1, 0]))
    assert tie.value == 0.1


def test_single_feature_direction_is_chosen_with_the_threshold():
    s = np.array([0.9, 0.8, 0.2, 0.1])
    y = np.array([0, 0, 1, 1])
    th = best_oriented_threshold(s, y)
    assert th.direction == "low" and th.f1 == 1.0
    assert list(th.predict(s)) == [0, 0, 1, 1]
    assert best_oriented_threshold(-s, y).direction == "high"


def test_weighted_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    s = np.round(rng.random(80), 1)
    y = rng.integers(0, 2, 80)
    w = rng.integers(0, 4, 80).astype(float)
    auroc, ap = weighted_ranking(w[None, :], s, y)
    assert auroc[0] == pytest.approx(roc_auc_score(y, s, sample_weight=w))
    assert ap[0] == pytest.approx(average_precision_score(y, s, sample_weight=w))
    pred = (s >= 0.5).astype(int)
    f1, _ = weighted_f1(w[None, :], pred, y)
    tp, fp, fn = (w * pred * y).sum(), (w * pred * (1 - y)).sum(), (w * (1 - pred) * y).sum()
    assert f1[0] == pytest.approx(2 * tp / (2 * tp + fp + fn))


def test_bootstrap_keeps_clusters_together_and_is_seeded():
    groups = np.array(["a", "a", "b", "c", "c", "c"])
    W = cluster_weights(groups, 50, 3)
    assert (W[:, 0] == W[:, 1]).all() and (W[:, 3] == W[:, 5]).all()
    assert (W.sum(axis=1) > 0).all()
    np.testing.assert_array_equal(W, cluster_weights(groups, 50, 3))
    y = np.array([1, 1, 0, 0, 1, 0])
    ev = Evaluator("e", [np.arange(6.0)], [Threshold(2.0, 0.0)])
    assert bootstrap([ev], y, groups, 40, 1, 15) == bootstrap([ev], y, groups, 40, 1, 15)


# End-to-end on synthetic data: determinism and report completeness

@pytest.fixture(scope="module")
def synthetic_run():
    t = small_config()
    return t, Step6(t, synthetic_inputs()).run()


def test_step6_is_deterministic(synthetic_run):
    t, tables = synthetic_run
    again = Step6(t, synthetic_inputs()).run()
    for name, df in tables.items():
        if name == "runtime":
            continue
        pd.testing.assert_frame_equal(df, again[name], obj=name)


def test_step6_reports_are_complete(synthetic_run, tmp_path):
    t, tables = synthetic_run
    tables = dict(tables, data_access=pd.DataFrame({"path": ["data/x"], "test_split": [False]}))
    write_tables(tables, tmp_path)
    (tmp_path / "README.md").write_text(write_readme(tables, synthetic_inputs().notes, t))
    for name in REPORT_FILES:
        path = tmp_path / name
        assert path.exists() and path.stat().st_size > 0, name
    vps = pd.read_csv(tmp_path / "validation_per_seed.csv")
    assert set(vps["seed"]) == set(t["seeds"])
    assert set(vps["family"]) == {"logistic_regression", "hist_gradient_boosting"}
    assert set(vps["eval_condition"]) == {"retrieved", "none", "gold"}
    assert set(vps[vps["eval_condition"] == "gold"]["train_condition"]) == {"retrieved"}
    assert {"f1", "auroc", "auprc", "threshold"} <= set(vps.columns)
    vs = pd.read_csv(tmp_path / "validation_summary.csv")
    assert {"f1_ci_low", "f1_ci_high", "auroc_ci_low", "auprc_ci_high"} <= set(vs.columns)
    lofo = pd.read_csv(tmp_path / "ablation_leave_one_feature_out.csv")
    res = lofo[(lofo["tier"] == "resampling") & (lofo["train_condition"] == "retrieved")]
    assert set(RESAMPLING_FEATURES) <= set(res["removed"].dropna())
    logo = pd.read_csv(tmp_path / "ablation_leave_one_group_out.csv")
    assert {"-evidence_dependent", "-evidence_free", "-self_consistency"} <= set(logo["subset"])
    fwd = pd.read_csv(tmp_path / "ablation_forward.csv")
    assert (fwd["step"] == 0).any()
    single = pd.read_csv(tmp_path / "single_features.csv")
    ss = single[single["tier"] == "single_shot"]
    assert not set(ss["feature"]) & set(RESAMPLING_FEATURES)
    imp = pd.read_csv(tmp_path / "importance_stability.csv")
    assert "spearman_rho" in imp.columns
    assert "sc_entail" in (tmp_path / "README.md").read_text()


def test_single_shot_models_are_fitted_without_resampling_features(synthetic_run):
    _, tables = synthetic_run
    imp = tables["permutation_importance"]
    assert not set(imp[imp["tier"] == "single_shot"]["feature"]) & set(RESAMPLING_FEATURES)
    assert set(RESAMPLING_FEATURES) <= set(imp[imp["tier"] == "resampling"]["feature"])


def test_committed_step6_reports_are_complete():
    out = ROOT / "reports" / "step6"
    if not out.exists():
        pytest.skip("Step 6 reports not generated yet")
    for name in REPORT_FILES:
        assert (out / name).exists(), name
    access = pd.read_csv(out / "data_access.csv")
    assert not access["test_split"].any()
    assert not any(s in p for p in access["path"] for s in TEST_SPLITS)


def test_label_and_report_outputs_are_ignored_or_safe():
    ignored = (ROOT / ".gitignore").read_text().split()
    assert "data/labels/" in ignored and "data/features/" in ignored
    assert json.loads(json.dumps(REPORT_FILES))
