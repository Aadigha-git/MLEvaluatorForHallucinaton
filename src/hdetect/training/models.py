"""Classifier grids, fold-local pipelines, and cross-validated hyperparameter selection on train.

docs/PREREGISTRATION.md, "Fitting and selection": StratifiedGroupKFold(5, shuffle=True,
random_state=seed) grouped by cluster_id, scored by average precision; ties go to the simpler
setting. Scaling lives inside the pipeline, so it is fitted on each training fold only. Nothing is
imputed (features-v1 has no missing values; non-finite input is an error).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

TIE_DECIMALS = 12


@dataclass(frozen=True)
class Config:
    family: str
    params: tuple[tuple[str, Any], ...]

    @property
    def p(self) -> dict[str, Any]:
        return dict(self.params)

    @property
    def key(self) -> str:
        return ",".join(f"{k}={v}" for k, v in self.params)

    def simplicity(self) -> tuple:
        """Smaller is simpler. LR: C, then no class weighting. HGB: depth, iterations, learning rate."""
        p = self.p
        if self.family == "logistic_regression":
            return (p["C"], p["class_weight"] is not None)
        return (p["max_depth"], p["max_iter"], p["learning_rate"])


def grid(family: str, tcfg: Mapping[str, Any]) -> list[Config]:
    g = tcfg[family]
    if family == "logistic_regression":
        return [Config(family, (("C", c), ("class_weight", w))) for c, w in product(g["C"], g["class_weight"])]
    if family == "hist_gradient_boosting":
        return [
            Config(family, (("max_depth", d), ("learning_rate", lr), ("max_iter", it)))
            for d, lr, it in product(g["max_depth"], g["learning_rate"], g["max_iter"])
        ]
    raise ValueError(f"unknown family {family!r}")


def make_estimator(config: Config, seed: int, tcfg: Mapping[str, Any]):
    p = config.p
    if config.family == "logistic_regression":
        lr = tcfg["logistic_regression"]
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=p["C"], class_weight=p["class_weight"], solver=lr["solver"], max_iter=lr["max_iter"])),
        ])
    hgb = tcfg["hist_gradient_boosting"]
    return HistGradientBoostingClassifier(
        max_depth=p["max_depth"], learning_rate=p["learning_rate"], max_iter=p["max_iter"],
        min_samples_leaf=hgb["min_samples_leaf"], early_stopping=False, random_state=seed,
    )


def folds(y: np.ndarray, groups: np.ndarray, seed: int, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in cv.split(np.zeros(len(y)), y, groups)]


def score(estimator, X: np.ndarray) -> np.ndarray:
    return estimator.predict_proba(X)[:, 1]


def fit(config: Config, seed: int, X: np.ndarray, y: np.ndarray, tcfg: Mapping[str, Any]):
    if not np.isfinite(X).all():
        raise ValueError("non-finite feature values")
    with threadpool_limits(1):
        return make_estimator(config, seed, tcfg).fit(X, y)


def _fold_ap(config, seed, X, y, tr, te, tcfg) -> float:
    est = fit(config, seed, X[tr], y[tr], tcfg)
    with threadpool_limits(1):
        return float(average_precision_score(y[te], score(est, X[te])))


@dataclass(frozen=True)
class Subset:
    name: str
    features: tuple[str, ...]


def cv_results(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    subsets: Sequence[Subset],
    families: Sequence[str],
    seeds: Sequence[int],
    tcfg: Mapping[str, Any],
    n_jobs: int,
) -> pd.DataFrame:
    """Fold-level average precision for every subset x family x config x seed x fold."""
    n_splits = tcfg["cv"]["n_splits"]
    fold_sets = {seed: folds(y, groups, seed, n_splits) for seed in seeds}
    tasks, keys = [], []
    for sub in subsets:
        Xs = X.loc[:, list(sub.features)].to_numpy(dtype=float)
        for family in families:
            for config in grid(family, tcfg):
                for seed in seeds:
                    for k, (tr, te) in enumerate(fold_sets[seed]):
                        tasks.append(delayed(_fold_ap)(config, seed, Xs, y, tr, te, tcfg))
                        keys.append((sub.name, family, config.key, seed, k))
    values = Parallel(n_jobs=n_jobs)(tasks) if n_jobs != 1 else [t[0](*t[1], **t[2]) for t in tasks]
    df = pd.DataFrame(keys, columns=["subset", "family", "config", "seed", "fold"])
    df["average_precision"] = values
    return df


def select(cv: pd.DataFrame, tcfg: Mapping[str, Any]) -> pd.DataFrame:
    """Per subset x family x seed: the config with the highest mean fold AP; ties to the simpler one."""
    configs = {c.key: c for fam in cv["family"].unique() for c in grid(fam, tcfg)}
    agg = (
        cv.groupby(["subset", "family", "seed", "config"], sort=True)["average_precision"]
        .agg(cv_ap_mean="mean", cv_ap_sd="std")
        .reset_index()
    )
    agg["rounded"] = agg["cv_ap_mean"].round(TIE_DECIMALS)
    agg["simplicity"] = [configs[k].simplicity() for k in agg["config"]]
    rows = []
    for _, g in agg.groupby(["subset", "family", "seed"], sort=True):
        best = sorted(g.itertuples(index=False), key=lambda r: (-r.rounded, r.simplicity))[0]
        rows.append({"subset": best.subset, "family": best.family, "seed": best.seed, "config": best.config,
                     "cv_ap_mean": best.cv_ap_mean, "cv_ap_sd": best.cv_ap_sd})
    return pd.DataFrame(rows)


def config_from_key(family: str, key: str, tcfg: Mapping[str, Any]) -> Config:
    return {c.key: c for c in grid(family, tcfg)}[key]
