"""Step 6 analyses on train and val_id, written as CSV and Markdown under reports/step6/.

Every number here comes from train (fitting, cross-validation) or val_id (model choice, decision
thresholds, validation metrics). Validation F1 uses the threshold chosen on val_id itself, so it is
optimistic in the same way for every evaluator; AUROC and AUPRC do not depend on thresholds.
"""

from __future__ import annotations

import os
import platform
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits

from hdetect.features import feature_columns
from hdetect.training import FAMILIES, TIERS, TRAIN_CONDITIONS, present_groups, tier_features
from hdetect.training.data import DevData, resampling_rows
from hdetect.training.metrics import (
    Evaluator,
    all_hallucinated_f1,
    best_oriented_threshold,
    best_threshold,
    bootstrap,
    point_metrics,
)
from hdetect.training.models import Subset, config_from_key, cv_results, fit, score, select

ROUND = 6
TIE_F1 = 12
NEAR_ZERO = 0.01
METRICS = ("f1", "auroc", "auprc")

REPORT_FILES = (
    "labels_summary.csv",
    "data_checks.csv",
    "cv_results.csv",
    "selected_hyperparameters.csv",
    "validation_per_seed.csv",
    "validation_summary.csv",
    "family_selection.csv",
    "single_features.csv",
    "ablation_forward.csv",
    "ablation_forward_candidates.csv",
    "ablation_leave_one_feature_out.csv",
    "ablation_leave_one_group_out.csv",
    "correlations.csv",
    "vif.csv",
    "permutation_importance.csv",
    "importance_stability.csv",
    "feature_distributions.csv",
    "missingness.csv",
    "runtime.csv",
    "data_access.csv",
    "README.md",
)


@dataclass
class Inputs:
    train: dict[str, DevData]
    val: dict[str, DevData]
    label_summaries: list[dict]
    checks: list[dict]
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Fitted:
    tier: str
    condition: str
    family: str
    seed: int
    config: str
    estimator: Any


def _tier_rows(data: DevData, tier: str) -> np.ndarray:
    return resampling_rows(data) if tier == "resampling" else np.ones(len(data.y), dtype=bool)


def _xy(data: DevData, tier: str, features: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _tier_rows(data, tier)
    return data.subset(tuple(features))[rows], data.y[rows], data.groups[rows]


def _eval_conditions(condition: str, val: Mapping[str, DevData]) -> list[str]:
    return [condition] + (["gold"] if condition == "retrieved" and "gold" in val else [])


def _seed_summary(df: pd.DataFrame, keys: list[str], metrics: Sequence[str]) -> pd.DataFrame:
    g = df.groupby(keys, sort=False)
    out = g[list(metrics)].agg(["mean", "min", "max", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    return out.reset_index()


class Step6:
    def __init__(self, tcfg: Mapping[str, Any], inputs: Inputs, n_jobs: int | None = None):
        self.tcfg = tcfg
        self.inputs = inputs
        self.seeds = list(tcfg["seeds"])
        self.n_jobs = n_jobs if n_jobs is not None else tcfg["runtime"]["n_jobs"]
        self.timings: list[dict] = []
        self.tables: dict[str, pd.DataFrame] = {}
        self.fitted: list[Fitted] = []
        self.selected_family: dict[tuple[str, str], str] = {}

    def _timed(self, stage: str, fn, *args, **kwargs):
        start = time.perf_counter()
        out = fn(*args, **kwargs)
        self.timings.append({"stage": stage, "seconds": round(time.perf_counter() - start, 3)})
        return out

    # Model selection and validation

    def _fit_and_validate(self, subset_feats, tier, condition, families, cv) -> tuple[pd.DataFrame, list[Fitted]]:
        """Refit each seed's selected config on all of train; threshold and metrics on val_id."""
        train = self.inputs.train[condition]
        Xtr, ytr, _ = _xy(train, tier, subset_feats)
        sel = select(cv, self.tcfg)
        rows, fitted = [], []
        for family in families:
            for seed in self.seeds:
                s = sel[(sel["family"] == family) & (sel["seed"] == seed)].iloc[0]
                config = config_from_key(family, s["config"], self.tcfg)
                est = fit(config, seed, Xtr, ytr, self.tcfg)
                fitted.append(Fitted(tier, condition, family, seed, config.key, est))
                for ev in _eval_conditions(condition, self.inputs.val):
                    Xv, yv, _ = _xy(self.inputs.val[ev], tier, subset_feats)
                    with threadpool_limits(1):
                        sv = score(est, Xv)
                    th = best_threshold(sv, yv)
                    rows.append({
                        "tier": tier, "train_condition": condition, "eval_condition": ev, "family": family,
                        "seed": seed, "config": config.key, "cv_ap_mean": s["cv_ap_mean"], "cv_ap_sd": s["cv_ap_sd"],
                        "n_train": len(ytr), "n_val": len(yv), **point_metrics(sv, yv, th),
                    })
        return pd.DataFrame(rows), fitted

    def selection(self) -> None:
        cv_all, per_seed, sel_rows = [], [], []
        for tier in TIERS:
            for condition in TRAIN_CONDITIONS:
                feats = tier_features(tier, condition)
                train = self.inputs.train[condition]
                rows = _tier_rows(train, tier)
                X = train.X.loc[rows].reset_index(drop=True)
                cv = self._timed(f"cv {tier}/{condition}", cv_results, X, train.y[rows], train.groups[rows],
                                 [Subset("all", feats)], FAMILIES, self.seeds, self.tcfg, self.n_jobs)
                cv.insert(0, "tier", tier)
                cv.insert(1, "condition", condition)
                cv_all.append(cv)
                sel = select(cv.drop(columns=["tier", "condition"]), self.tcfg)
                sel.insert(0, "tier", tier)
                sel.insert(1, "condition", condition)
                sel_rows.append(sel)
                res, fitted = self._timed(f"refit {tier}/{condition}", self._fit_and_validate, feats, tier, condition,
                                          FAMILIES, cv.drop(columns=["tier", "condition"]))
                per_seed.append(res)
                self.fitted.extend(fitted)
        self.tables["cv_results"] = pd.concat(cv_all, ignore_index=True)
        self.tables["selected_hyperparameters"] = pd.concat(sel_rows, ignore_index=True).drop(columns=["subset"])
        self.tables["validation_per_seed"] = pd.concat(per_seed, ignore_index=True)

        fam_rows = []
        vps = self.tables["validation_per_seed"]
        for tier in TIERS:
            for condition in TRAIN_CONDITIONS:
                own = vps[(vps["tier"] == tier) & (vps["train_condition"] == condition) & (vps["eval_condition"] == condition)]
                means = own.groupby("family")["f1"].mean()
                best = max(FAMILIES, key=lambda f: (round(means[f], TIE_F1), f == "logistic_regression"))
                self.selected_family[(tier, condition)] = best
                for f in FAMILIES:
                    fam_rows.append({"tier": tier, "condition": condition, "family": f,
                                     "mean_val_f1_over_seeds": means[f], "selected": f == best})
        self.tables["family_selection"] = pd.DataFrame(fam_rows)

    # Single-feature baselines

    def single_features(self) -> None:
        rows = []
        for ev, data in self.inputs.val.items():
            for tier in TIERS:
                for feat in tier_features(tier, ev):
                    rows_mask = _tier_rows(data, tier)
                    s, y = data.X[feat].to_numpy(dtype=float)[rows_mask], data.y[rows_mask]
                    th = best_oriented_threshold(s, y)
                    rows.append({"tier": tier, "eval_condition": ev, "feature": feat, "n_val": len(y),
                                 **point_metrics(s, y, th)})
        df = pd.DataFrame(rows)
        df["best_in_tier_condition"] = False
        for (tier, ev), g in df.groupby(["tier", "eval_condition"]):
            df.loc[g["f1"].idxmax(), "best_in_tier_condition"] = True
        self.tables["single_features"] = df

    # Uncertainty

    def validation_summary(self) -> None:
        vps = self.tables["validation_per_seed"]
        singles = self.tables["single_features"]
        bcfg = self.tcfg["bootstrap"]
        rows = []
        for tier in TIERS:
            for ev, data in self.inputs.val.items():
                train_cond = "retrieved" if ev == "gold" else ev
                mask = _tier_rows(data, tier)
                y, groups = data.y[mask], data.groups[mask]
                feats = tier_features(tier, ev)
                fits = {(f.family, f.seed): f for f in self.fitted if f.tier == tier and f.condition == train_cond}
                evaluators, point = [], {}
                for family in FAMILIES:
                    own = vps[(vps["tier"] == tier) & (vps["eval_condition"] == ev) & (vps["family"] == family)]
                    scores, ths = [], []
                    for seed in self.seeds:
                        r = own[own["seed"] == seed].iloc[0]
                        with threadpool_limits(1):
                            scores.append(score(fits[(family, seed)].estimator, data.subset(feats)[mask]))
                        ths.append(best_threshold(scores[-1], y))
                        assert abs(ths[-1].value - r["threshold"]) < 1e-12
                    evaluators.append(Evaluator(family, scores, ths))
                    point[family] = {m: own[m].mean() for m in METRICS} | {f"{m}_seed_{s}": own[m].agg(s) for m in METRICS for s in ("min", "max", "std")}
                st = singles[(singles["tier"] == tier) & (singles["eval_condition"] == ev)]
                for r in st.itertuples(index=False):
                    s = data.X[r.feature].to_numpy(dtype=float)[mask]
                    evaluators.append(Evaluator(f"feature:{r.feature}", [s], [best_oriented_threshold(s, y)]))
                    point[f"feature:{r.feature}"] = {"f1": r.f1, "auroc": r.auroc, "auprc": r.auprc}
                best_single = "feature:" + st.loc[st["f1"].idxmax(), "feature"]
                chosen = self.selected_family[(tier, train_cond)]
                boot = self._timed(f"bootstrap {tier}/{ev}", bootstrap, evaluators, y, groups,
                                   bcfg["replicates"], bcfg["seed"], bcfg["chunk"], pairs=[(chosen, best_single)])
                prevalence = float(y.mean())
                for name, p in point.items():
                    rows.append({"tier": tier, "eval_condition": ev, "train_condition": train_cond, "evaluator": name,
                                 "selected_family": name == chosen, "best_single_feature": name == best_single,
                                 "n_val": len(y), "prevalence": prevalence, **p, **boot[name]})
                rows.append({"tier": tier, "eval_condition": ev, "train_condition": train_cond, "evaluator": "all_hallucinated",
                             "n_val": len(y), "prevalence": prevalence, "f1": all_hallucinated_f1(y), **boot["all_hallucinated"]})
                diff = boot[f"{chosen} - {best_single}"]
                rows.append({"tier": tier, "eval_condition": ev, "train_condition": train_cond,
                             "evaluator": f"{chosen} - {best_single}", "n_val": len(y), "prevalence": prevalence,
                             **{m: point[chosen][m] - point[best_single][m] for m in METRICS}, **diff})
        self.tables["validation_summary"] = pd.DataFrame(rows)

    # Ablations

    def _subset_scores(self, tier, condition, family, subsets: Sequence[Subset]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """CV selection, refit, and val_id metrics for each subset with one family; returns (cv summary, val rows)."""
        train = self.inputs.train[condition]
        rows_mask = _tier_rows(train, tier)
        X = train.X.loc[rows_mask].reset_index(drop=True)
        cv = cv_results(X, train.y[rows_mask], train.groups[rows_mask], subsets, [family], self.seeds, self.tcfg, self.n_jobs)
        sel = select(cv, self.tcfg)
        val_rows = []
        for sub in subsets:
            Xtr, ytr, _ = _xy(train, tier, sub.features)
            for seed in self.seeds:
                s = sel[(sel["subset"] == sub.name) & (sel["seed"] == seed)].iloc[0]
                est = fit(config_from_key(family, s["config"], self.tcfg), seed, Xtr, ytr, self.tcfg)
                for ev in _eval_conditions(condition, self.inputs.val):
                    Xv, yv, _ = _xy(self.inputs.val[ev], tier, sub.features)
                    with threadpool_limits(1):
                        sv = score(est, Xv)
                    m = point_metrics(sv, yv, best_threshold(sv, yv))
                    val_rows.append({"subset": sub.name, "eval_condition": ev, "seed": seed, "config": s["config"],
                                     "cv_ap_mean": s["cv_ap_mean"], **{k: m[k] for k in METRICS}})
        return sel, pd.DataFrame(val_rows)

    def _summarize_subsets(self, val_rows: pd.DataFrame, extra: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
        summ = _seed_summary(val_rows, ["subset", "eval_condition"], ("cv_ap_mean",) + METRICS)
        for col in ("features", "n_features", "step", "added", "removed", "group"):
            if any(col in e for e in extra.values()):
                summ[col] = [extra[s].get(col) for s in summ["subset"]]
        return summ

    def ablations(self) -> None:
        fwd, cand_rows, lofo, logo = [], [], [], []
        for tier in TIERS:
            for condition in TRAIN_CONDITIONS:
                family = self.selected_family[(tier, condition)]
                feats = tier_features(tier, condition)
                tag = {"tier": tier, "train_condition": condition, "family": family}
                start = time.perf_counter()

                chosen: list[str] = []
                extra = {}
                steps = []
                for step in range(1, len(feats) + 1):
                    remaining = [f for f in feats if f not in chosen]
                    subsets = [Subset("+".join(chosen + [f]), tuple(chosen + [f])) for f in remaining]
                    sel, _ = self._subset_scores_cv_only(tier, condition, family, subsets)
                    mean_ap = sel.groupby("subset", sort=False)["cv_ap_mean"].mean()
                    ranked = sorted(remaining, key=lambda f: (-round(mean_ap["+".join(chosen + [f])], 12), feats.index(f)))
                    for f in remaining:
                        cand_rows.append({**tag, "step": step, "candidate": f, "selected": f == ranked[0],
                                          "cv_ap_mean_over_seeds": mean_ap["+".join(chosen + [f])]})
                    chosen.append(ranked[0])
                    name = "+".join(chosen)
                    steps.append(Subset(name, tuple(chosen)))
                    extra[name] = {"step": step, "added": ranked[0], "n_features": len(chosen), "features": " ".join(chosen)}
                _, val_rows = self._subset_scores(tier, condition, family, steps)
                f = self._summarize_subsets(val_rows, extra)
                fwd.append(self._baseline_rows(tier, condition, family) )
                fwd.append(f.assign(**tag))

                all_sub = Subset("all", feats)
                subsets = [all_sub] + [Subset(f"-{f}", tuple(x for x in feats if x != f)) for f in feats]
                extra = {"all": {"removed": "", "n_features": len(feats)}} | {
                    f"-{f}": {"removed": f, "n_features": len(feats) - 1} for f in feats}
                _, val_rows = self._subset_scores(tier, condition, family, subsets)
                lofo.append(self._with_deltas(self._summarize_subsets(val_rows, extra), "all").assign(**tag))

                groups = present_groups(feats)
                subsets = [all_sub] + [Subset(f"-{g}", tuple(x for x in feats if x not in gf))
                                       for g, gf in groups.items() if len(gf) < len(feats)]
                subsets += [Subset(f"only:{g}", gf) for g, gf in groups.items() if len(gf) < len(feats)]
                extra = {"all": {"group": ""}} | {s.name: {"group": s.name, "n_features": len(s.features)} for s in subsets[1:]}
                _, val_rows = self._subset_scores(tier, condition, family, subsets)
                logo.append(self._with_deltas(self._summarize_subsets(val_rows, extra), "all").assign(**tag))
                self.timings.append({"stage": f"ablations {tier}/{condition}", "seconds": round(time.perf_counter() - start, 3)})
        self.tables["ablation_forward"] = pd.concat(fwd, ignore_index=True)
        self.tables["ablation_forward_candidates"] = pd.DataFrame(cand_rows)
        self.tables["ablation_leave_one_feature_out"] = pd.concat(lofo, ignore_index=True)
        self.tables["ablation_leave_one_group_out"] = pd.concat(logo, ignore_index=True)

    def _subset_scores_cv_only(self, tier, condition, family, subsets):
        train = self.inputs.train[condition]
        rows_mask = _tier_rows(train, tier)
        X = train.X.loc[rows_mask].reset_index(drop=True)
        cv = cv_results(X, train.y[rows_mask], train.groups[rows_mask], subsets, [family], self.seeds, self.tcfg, self.n_jobs)
        return select(cv, self.tcfg), cv

    def _baseline_rows(self, tier, condition, family) -> pd.DataFrame:
        """Step 0: the bare (intercept-only) baseline. AUROC 0.5, AUPRC = prevalence, F1 = all-hallucinated."""
        train = self.inputs.train[condition]
        ytr = train.y[_tier_rows(train, tier)]
        rows = []
        for ev in _eval_conditions(condition, self.inputs.val):
            yv = self.inputs.val[ev].y[_tier_rows(self.inputs.val[ev], tier)]
            rows.append({"subset": "(bare)", "eval_condition": ev, "step": 0, "added": "", "n_features": 0, "features": "",
                         "cv_ap_mean_mean": float(ytr.mean()), "f1_mean": all_hallucinated_f1(yv), "auroc_mean": 0.5,
                         "auprc_mean": float(yv.mean()), "tier": tier, "train_condition": condition, "family": family})
        return pd.DataFrame(rows)

    @staticmethod
    def _with_deltas(summ: pd.DataFrame, ref: str) -> pd.DataFrame:
        out = []
        for ev, g in summ.groupby("eval_condition", sort=False):
            base = g[g["subset"] == ref].iloc[0]
            g = g.copy()
            for m in ("cv_ap_mean",) + METRICS:
                g[f"delta_{m}_mean"] = g[f"{m}_mean"] - base[f"{m}_mean"]
            out.append(g)
        return pd.concat(out, ignore_index=True)

    # Permutation importance

    def importance(self) -> None:
        pcfg = self.tcfg["permutation_importance"]
        rows = []
        for tier in TIERS:
            for condition in TRAIN_CONDITIONS:
                family = self.selected_family[(tier, condition)]
                feats = tier_features(tier, condition)
                for ev in _eval_conditions(condition, self.inputs.val):
                    Xv, yv, _ = _xy(self.inputs.val[ev], tier, feats)
                    for f in (f for f in self.fitted if f.tier == tier and f.condition == condition and f.family == family):
                        with threadpool_limits(1):
                            pi = permutation_importance(f.estimator, Xv, yv, scoring=pcfg["scoring"],
                                                        n_repeats=pcfg["n_repeats"], random_state=f.seed, n_jobs=1)
                        for i, feat in enumerate(feats):
                            rows.append({"tier": tier, "train_condition": condition, "eval_condition": ev, "family": family,
                                         "seed": f.seed, "feature": feat, "importance_mean": pi.importances_mean[i],
                                         "importance_sd": pi.importances_std[i]})
        df = pd.DataFrame(rows)
        df["rank"] = df.groupby(["tier", "eval_condition", "seed"])["importance_mean"].rank(ascending=False, method="min")
        self.tables["permutation_importance"] = df

        stab = []
        for (tier, ev), g in df.groupby(["tier", "eval_condition"], sort=False):
            wide = g.pivot(index="feature", columns="seed", values="importance_mean")
            top = {s: set(wide[s].nlargest(3).index) for s in wide.columns}
            for a, b in combinations(wide.columns, 2):
                rho = spearmanr(wide[a], wide[b]).statistic
                stab.append({"tier": tier, "eval_condition": ev, "seed_a": a, "seed_b": b, "spearman_rho": rho,
                             "top3_jaccard": len(top[a] & top[b]) / len(top[a] | top[b])})
            summary = wide.agg(["mean", "std", "min", "max"], axis=1)
            for feat, r in summary.iterrows():
                stab.append({"tier": tier, "eval_condition": ev, "feature": feat, "importance_mean_over_seeds": r["mean"],
                             "importance_sd_over_seeds": r["std"], "importance_min": r["min"], "importance_max": r["max"],
                             "rank_min": g[g["feature"] == feat]["rank"].min(), "rank_max": g[g["feature"] == feat]["rank"].max()})
        self.tables["importance_stability"] = pd.DataFrame(stab)

    # Redundancy, distributions, missingness

    def redundancy(self) -> None:
        corr_rows, vif_rows = [], []
        for split, datasets in (("train", self.inputs.train), ("val_id", self.inputs.val)):
            for cond, data in datasets.items():
                feats = list(feature_columns(cond))
                X = data.X[feats]
                for method in ("pearson", "spearman"):
                    c = X.corr(method=method)
                    for a, b in combinations(feats, 2):
                        corr_rows.append({"split": split, "condition": cond, "method": method, "feature_a": a,
                                          "feature_b": b, "r": c.loc[a, b]})
                for tier in TIERS:
                    tf = list(tier_features(tier, cond))
                    R = X[tf].corr().to_numpy()
                    try:
                        vif = np.diag(np.linalg.inv(R))
                    except np.linalg.LinAlgError:
                        vif = np.full(len(tf), np.inf)
                    for f, v in zip(tf, vif):
                        vif_rows.append({"split": split, "condition": cond, "tier": tier, "feature": f, "vif": v})
        self.tables["correlations"] = pd.DataFrame(corr_rows)
        self.tables["vif"] = pd.DataFrame(vif_rows)

    def distributions(self) -> None:
        rows, miss = [], []
        for split, datasets in (("train", self.inputs.train), ("val_id", self.inputs.val)):
            for cond, data in datasets.items():
                for feat in feature_columns(cond):
                    v = data.X[feat]
                    miss.append({"split": split, "condition": cond, "feature": feat, "rows": len(v),
                                 "missing": int(v.isna().sum()), "non_finite": int((~np.isfinite(v)).sum())})
                    for label, mask in (("all", np.ones(len(v), bool)), ("hallucinated", data.y == 1), ("not_hallucinated", data.y == 0)):
                        x = v[mask]
                        q = x.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_numpy()
                        rows.append({"split": split, "condition": cond, "feature": feat, "label": label, "n": len(x),
                                     "mean": x.mean(), "sd": x.std(), "min": x.min(), "p05": q[0], "p25": q[1],
                                     "median": q[2], "p75": q[3], "p95": q[4], "max": x.max(),
                                     "share_zero": float((x == 0).mean()), f"share_below_{NEAR_ZERO}": float((x < NEAR_ZERO).mean())})
                miss.append({"split": split, "condition": cond, "feature": "(rows)", "rows": data.counts["rows"],
                             "excluded_undecidable": data.counts["excluded_undecidable"],
                             "no_usable_resamples": int((~resampling_rows(data)).sum()),
                             "truncated_primary": int(data.meta["meta_truncated_primary"].sum()),
                             "abstained": int(data.meta["abstained"].sum())})
        self.tables["feature_distributions"] = pd.DataFrame(rows)
        self.tables["missingness"] = pd.DataFrame(miss)

    # Runtime

    def runtime(self) -> None:
        n = self.tcfg["runtime"]["latency_rows"]
        rows = [dict(t, kind="stage") for t in self.timings]
        for tier in TIERS:
            for condition in TRAIN_CONDITIONS:
                family = self.selected_family[(tier, condition)]
                f = next(f for f in self.fitted if f.tier == tier and f.condition == condition and f.family == family)
                Xv = self.inputs.val[condition].subset(tier_features(tier, condition))[:n]
                lat = []
                with threadpool_limits(1):
                    score(f.estimator, Xv[:1])
                    for i in range(len(Xv)):
                        t0 = time.perf_counter()
                        score(f.estimator, Xv[i:i + 1])
                        lat.append((time.perf_counter() - t0) * 1000)
                rows.append({"kind": "classifier_latency_ms", "stage": f"{tier}/{condition}/{family}/seed{f.seed}",
                             "median_ms": float(np.median(lat)), "p95_ms": float(np.percentile(lat, 95)), "n": len(lat)})
        pages = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") if hasattr(os, "sysconf") else None
        rows.append({"kind": "hardware", "stage": f"{platform.platform()} {platform.machine()} cpus={os.cpu_count()} "
                     f"memory_gb={round(pages / 1e9, 1) if pages else 'unknown'} sklearn={sklearn.__version__} n_jobs={self.n_jobs}"})
        self.tables["runtime"] = pd.DataFrame(rows)

    def run(self) -> dict[str, pd.DataFrame]:
        self.tables["labels_summary"] = pd.DataFrame(self.inputs.label_summaries)
        self.tables["data_checks"] = pd.DataFrame(self.inputs.checks)
        self._timed("selection", self.selection)
        self._timed("single features", self.single_features)
        self._timed("validation summary", self.validation_summary)
        self._timed("ablations", self.ablations)
        self._timed("permutation importance", self.importance)
        self._timed("redundancy", self.redundancy)
        self._timed("distributions", self.distributions)
        self.runtime()
        return self.tables


def write_tables(tables: Mapping[str, pd.DataFrame], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.round(ROUND).to_csv(out / f"{name}.csv", index=False)
