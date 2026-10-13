"""Step 9: score the frozen classifiers, single-feature baselines, and LLM judges on the test splits.

Everything scored here was frozen earlier: the Step 7 artifacts (classifier, F1 threshold,
calibrator, policy), the best single feature per tier and evidence condition with its val_id
threshold (Step 6), and the Step 8 judge prompts. Nothing is fitted or selected on a test split.
Implements docs/PREREGISTRATION.md ("Metrics and uncertainty", "Hypothesis tiers", RQ1, RQ2,
"Decision policy", "Cost and latency") and the Step 9 amendment.

`--stand-in` is a dry run: every test split is replaced by val_id (with its own bootstrap seed), the
Step 8 val_id judge rows stand in for the test judge rows, and output goes to the git-ignored
stand-in directory. No test split is opened in that mode; the run fails if one is. Without
`--stand-in`, configs/evaluation.yaml must say `status: frozen`, the hand labels for every
unresolved test answer must be present, the test judge runs must be complete, and an existing
reports/step9/ is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import yaml
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hdetect.cleaning import clean_answer  # noqa: E402
from hdetect.config import load_config  # noqa: E402
from hdetect.evaluation.data import SplitData, load_labels, load_split  # noqa: E402
from hdetect.evaluation.stats import (  # noqa: E402
    Scored,
    fixed_predictions,
    interval,
    judge_predictions,
    replicate,
    rq1_split,
    rq2_split,
    rq_outcome,
    seed_point,
    band_rates,
    all_hallucinated_f1,
)
from hdetect.features import EVIDENCE_FEATURES, RESAMPLING_FEATURES, TEST_SPLITS  # noqa: E402
from hdetect.features.access import load_answers  # noqa: E402
from hdetect.features.pipeline import load_feature_config  # noqa: E402
from hdetect.generation.config import load_generation_config  # noqa: E402
from hdetect.judging.runner import read_rows  # noqa: E402
from hdetect.judging.scoring import final_rows, predicted, score_cell  # noqa: E402
from hdetect.labeling.audit import read_hand_labels  # noqa: E402
from hdetect.training import TIERS, tier_features, training_condition  # noqa: E402
from hdetect.training.access import OPENED, data_access_table, install_audit_hook, is_test_path  # noqa: E402
from hdetect.training.calibration import calibrated, ece, reliability  # noqa: E402
from hdetect.training.data import ACCESS_LOG, read_manifest, resampling_rows  # noqa: E402
from hdetect.training.inputs import feature_versions, load_dev_inputs  # noqa: E402
from hdetect.training.metrics import Threshold, best_oriented_threshold, best_threshold  # noqa: E402
from hdetect.training.models import score  # noqa: E402
from hdetect.training.report import md_table  # noqa: E402
from hdetect.training.run import ROUND  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("retrieved", "none")
SUBSETS = ("all", "no_abstentions", "no_truncated")
JUDGES = ("small", "frontier")
METRICS = ("f1", "auroc")
EXCLUSION_FLAG_RATE = 0.05


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT).stdout.strip())
    return {"commit": commit, "dirty": dirty}


def hypothesis_tier(condition: str, subset: str, metric: str, split: str | None = None,
                    reference: str = "test_id") -> str:
    if condition == "gold" or split == "pooled":
        return "exploratory"
    if condition == "retrieved" and subset == "all" and metric == "f1" and split != reference:
        return "primary"
    return "secondary"


def subset_mask(d: SplitData, tier: str, subset: str) -> np.ndarray:
    m = resampling_rows(d) if tier == "resampling" else np.ones(len(d.y), dtype=bool)
    if subset == "no_abstentions":
        m &= ~d.meta["abstained"].to_numpy()
    elif subset == "no_truncated":
        m &= ~d.meta["meta_truncated_primary"].to_numpy(dtype=bool)
    return m


class Step9:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.stand_in = args.stand_in
        self.ecfg = yaml.safe_load((ROOT / args.evaluation_config).read_text())
        if not self.stand_in and self.ecfg["status"] != "frozen":
            sys.exit("configs/evaluation.yaml is provisional: only `--stand-in` may run")
        self.out = ROOT / self.ecfg["paths"]["stand_in_reports" if self.stand_in else "reports"]
        if not self.stand_in and self.out.exists() and any(self.out.iterdir()):
            sys.exit(f"{self.out} exists; Step 9 test results are written once and never overwritten")
        self.cfg = load_config(ROOT / args.config)
        self.fcfg = load_feature_config(ROOT / args.feature_config)
        self.tcfg = yaml.safe_load((ROOT / args.training_config).read_text())
        self.gcfg = load_generation_config(ROOT / args.generation_config)
        self.splits = list(self.ecfg["splits"])
        assert set(self.splits) == TEST_SPLITS
        b = self.ecfg["bootstrap"]
        self.seeds = {s: b["seed"] + k * b["seed_stride"] for k, s in enumerate(self.splits)}
        self.seeds["pooled"] = b["seed"] + len(self.splits) * b["seed_stride"]
        self.tables: dict[str, pd.DataFrame] = {}
        self.timings: list[dict] = []

    def source(self, split: str) -> str:
        return "val_id" if self.stand_in else split

    def timed(self, name, fn, *a, **kw):
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        self.timings.append({"step": name, "seconds": round(time.perf_counter() - t0, 2)})
        print(f"  {name}: {self.timings[-1]['seconds']} s", flush=True)
        return out

    # Frozen inputs

    def load_frozen(self) -> None:
        paths = self.tcfg["paths"]
        self.dev = load_dev_inputs(self.cfg, self.fcfg, paths)
        self.versions = feature_versions(self.fcfg)
        models = ROOT / self.ecfg["paths"]["models"]
        manifest = json.loads((models / "manifest.json").read_text())
        if manifest["feature_versions"] != self.versions or manifest["rule_version"] != self.cfg.labeling.rule_version:
            sys.exit("Step 7 artifacts were built with different feature or label versions")
        self.step7_manifest = manifest
        self.artifacts: dict[tuple[str, str], list[dict]] = {}
        for entry in manifest["artifacts"]:
            path = ROOT / entry["path"]
            if sha256(path) != entry["sha256"]:
                sys.exit(f"{entry['path']} does not match the Step 7 manifest")
            art = joblib.load(path)
            if tuple(art["features"]) != tier_features(art["tier"], art["condition"]):
                sys.exit(f"{entry['path']}: features differ from the tier's feature set")
            self.artifacts.setdefault((art["tier"], art["condition"]), []).append(art)
        for key in self.artifacts:
            self.artifacts[key].sort(key=lambda a: a["seed"])
        self.check_frozen_thresholds()

    def check_frozen_thresholds(self) -> None:
        """Rebuild every val_id threshold this step uses and check it against Steps 6 and 7."""
        s6 = ROOT / self.ecfg["paths"]["step6_reports"]
        summary = pd.read_csv(s6 / "validation_summary.csv")
        singles_csv = pd.read_csv(s6 / "single_features.csv")
        per_seed = pd.read_csv(s6 / "validation_per_seed.csv")
        rows = []
        self.single: dict[tuple[str, str], tuple[str, Threshold]] = {}
        self.gold_thresholds: dict[str, list[Threshold]] = {}
        for tier in TIERS:
            for cond in (*CONDITIONS, "gold"):
                val = self.dev.val[cond]
                mask = resampling_rows(val) if tier == "resampling" else np.ones(len(val.y), dtype=bool)
                y = val.y[mask]
                best = summary[(summary["tier"] == tier) & (summary["eval_condition"] == cond)
                               & (summary["best_single_feature"] == True)]  # noqa: E712
                feat = best["evaluator"].iloc[0].removeprefix("feature:")
                th = best_oriented_threshold(val.X[feat].to_numpy(dtype=float)[mask], y)
                ref = singles_csv[(singles_csv["tier"] == tier) & (singles_csv["eval_condition"] == cond)
                                  & (singles_csv["feature"] == feat)].iloc[0]
                ok = abs(round(th.value, ROUND) - ref["threshold"]) < 1e-9 and th.direction == ref["direction"]
                rows.append({"tier": tier, "condition": cond, "evaluator": f"feature:{feat}", "seed": "",
                             "val_threshold": th.value, "direction": th.direction, "reference": ref["threshold"],
                             "reference_source": "reports/step6/single_features.csv", "reproduced": ok})
                self.single[(tier, cond)] = (feat, th)
                arts = self.artifacts[(tier, training_condition(cond))]
                ths = []
                for art in arts:
                    s = score(art["estimator"], np.ascontiguousarray(val.subset(tuple(art["features"]))[mask]))
                    t = best_threshold(s, y)
                    if cond == "gold":
                        ref_v = per_seed[(per_seed["tier"] == tier) & (per_seed["eval_condition"] == cond)
                                         & (per_seed["family"] == art["family"]) & (per_seed["seed"] == art["seed"])]
                        reference, src = float(ref_v["threshold"].iloc[0]), "reports/step6/validation_per_seed.csv"
                        ok = abs(round(t.value, ROUND) - reference) < 1e-9
                    else:
                        reference, src = art["f1_threshold"]["value"], "Step 7 artifact"
                        ok = t.value == reference and art["f1_threshold"]["direction"] == "high"
                    ths.append(t)
                    rows.append({"tier": tier, "condition": cond, "evaluator": art["family"], "seed": art["seed"],
                                 "val_threshold": t.value, "direction": "high", "reference": reference,
                                 "reference_source": src, "reproduced": ok})
                if cond == "gold":
                    self.gold_thresholds[tier] = ths
        df = pd.DataFrame(rows)
        self.tables["threshold_check"] = df
        if not df["reproduced"].all():
            sys.exit(f"frozen thresholds not reproduced on val_id:\n{df[~df['reproduced']].to_string()}")

    # Test data

    def load_test(self) -> None:
        paths = self.tcfg["paths"]
        hand_path = paths["hand_labels"] if self.stand_in else self.ecfg["hand_labels"]
        ACCESS_LOG.append(Path(hand_path).as_posix())
        hand = read_hand_labels(ROOT / hand_path, allow_undecidable=True)
        manifest = read_manifest(Path(self.cfg.paths["splits"]) / "split_manifest.jsonl")
        self.data: dict[str, dict[str, SplitData]] = {}
        self.answer_rows: dict[str, dict] = {}
        exclusions = []
        for split in self.splits:
            src = self.source(split)
            labels, auto = load_labels(src, paths["labels"], hand, self.cfg.labeling.rule_version,
                                       allow_test_splits=True)
            conds = (*CONDITIONS, "gold") if split in self.ecfg["gold_splits"] else CONDITIONS
            self.data[split] = {c: load_split(src, c, features_dir=paths["features"], labels=labels, auto_rows=auto,
                                              manifest=manifest, expected_versions=self.versions,
                                              allow_test_splits=True) for c in conds}
            answers = load_answers(src, self.fcfg.paths["generations"], self.fcfg.paths["generation_manifest"],
                                   allow_test_splits=True)
            ACCESS_LOG.append((Path(self.fcfg.paths["generations"]) / f"{src}.jsonl").as_posix())
            self.answer_rows[split] = answers
            planned = len(manifest.question_ids(src))
            api_failure = planned - len(answers)
            empty = sum(not clean_answer(a.primary["text"]) for a in answers.values())
            d = self.data[split]["retrieved"]
            no_resample = int((~resampling_rows(d)).sum())
            excluded = api_failure + empty + len(labels.excluded)
            exclusions.append({"split": split, "source": src, "planned": planned, "primary_api_failure": api_failure,
                               "empty": empty, **{k: v for k, v in labels.counts.items()},
                               "excluded_undecidable": len(labels.excluded),
                               "excluded_total": excluded, "exclusion_rate": round(excluded / planned, 4),
                               "flagged_over_5pct": excluded / planned > EXCLUSION_FLAG_RATE,
                               "resampling_tier_no_usable_resample": no_resample,
                               "truncated_primary": int(d.meta["meta_truncated_primary"].astype(bool).sum())})
            for c, dc in self.data[split].items():
                if not (dc.meta["answer_id"].to_numpy() == d.meta["answer_id"].to_numpy()).all():
                    sys.exit(f"{split}: {c} rows are not aligned with retrieved rows")
        self.tables["exclusions"] = pd.DataFrame(exclusions)
        self.flagged = set(self.tables["exclusions"].loc[self.tables["exclusions"]["flagged_over_5pct"], "split"])

    def load_judges(self) -> None:
        """answer_id -> final row, per judge, split, and condition, for the selected Step 8 prompts."""
        selected = json.loads((ROOT / self.ecfg["paths"]["selected_prompts"]).read_text())["selected"]
        plan = self.ecfg["judges"]["plan"]
        self.judge_rows: dict[tuple[str, str, str], dict[str, dict]] = {}
        missing = []
        for j in JUDGES:
            for split in plan[j]["splits"]:
                path = (ROOT / "data" / "judges" / f"{j}.jsonl" if self.stand_in
                        else ROOT / self.ecfg["paths"]["judge_outputs"] / j / f"{split}.jsonl")
                ACCESS_LOG.append(path.relative_to(ROOT).as_posix())
                rows = read_rows(path)
                for cond in plan[j]["conditions"]:
                    variant = selected[j][cond]
                    own = [r for r in rows if r["variant"] == variant and r["condition"] == cond and r["run"] == 0]
                    by_answer = {r["answer_id"]: r for r in final_rows(own).values()}
                    self.judge_rows[(j, split, cond)] = by_answer
                    need = set(self.data[split][cond].meta["answer_id"])
                    missing.append({"judge": j, "split": split, "condition": cond, "variant": variant,
                                    "evaluated_answers": len(need), "missing_rows": len(need - set(by_answer))})
        self.tables["judge_coverage"] = pd.DataFrame(missing)
        if not self.stand_in and any(m["missing_rows"] for m in missing):
            sys.exit("test judge runs are incomplete; finish ml/evaluation/run_test_judges.py first")

    # Evaluators

    def classifier(self, tier: str, cond: str, d: SplitData, mask: np.ndarray) -> Scored:
        arts = self.artifacts[(tier, training_condition(cond))]
        scores = [score(a["estimator"], d.subset(tuple(a["features"]))[mask]) for a in arts]
        if cond == "gold":
            ths = self.gold_thresholds[tier]
        else:
            ths = [Threshold(a["f1_threshold"]["value"], float("nan"), "high") for a in arts]
        return Scored("classifier", scores, ths)

    def single_feature(self, tier: str, cond: str, d: SplitData, mask: np.ndarray) -> Scored:
        feat, th = self.single[(tier, cond)]
        return Scored(f"feature:{feat}", [d.X[feat].to_numpy(dtype=float)[mask]], [th])

    def judges_for(self, split: str, cond: str, d: SplitData, mask: np.ndarray) -> list[Scored]:
        out = []
        y = d.y[mask]
        ids = d.meta["answer_id"].to_numpy()[mask]
        for j in JUDGES:
            rows = self.judge_rows.get((j, split, cond))
            if rows is not None:
                out.append(fixed_predictions(f"judge:{j}", judge_predictions([predicted(rows.get(a)) for a in ids], y)))
        return out

    def cell(self, split: str, cond: str, tier: str, subset: str, seed_key: str | None = None,
             with_judges: bool = True) -> dict:
        d = self.data[split][cond]
        mask = subset_mask(d, tier, subset)
        evs = [self.classifier(tier, cond, d, mask), self.single_feature(tier, cond, d, mask)]
        if with_judges and cond != "gold":
            evs += self.judges_for(split, cond, d, mask)
        return self.score_cell(evs, d.y[mask], d.groups[mask], self.seeds[seed_key or split])

    def score_cell(self, evs: list[Scored], y: np.ndarray, groups: np.ndarray, seed: int) -> dict:
        b = self.ecfg["bootstrap"]
        rep = replicate(evs, y, groups, b["replicates"], seed, b["chunk"])
        point = {e.name: seed_point(e, y) for e in evs}
        return {"evaluators": evs, "y": y, "groups": groups, "rep": rep, "point": point,
                "single": evs[1].name, "seed": seed}

    def evaluator_rows(self, c: dict, keys: dict) -> list[dict]:
        rows = []
        y, rep = c["y"], c["rep"]
        for e in c["evaluators"]:
            p = c["point"][e.name]
            f1_ci, auc_ci = interval(rep.f1[e.name]), interval(rep.auroc[e.name])
            rows.append({**keys, "evaluator": e.name, "n": len(y), "prevalence": float(y.mean()) if len(y) else np.nan,
                         **p, "f1_ci_low": f1_ci["ci_low"], "f1_ci_high": f1_ci["ci_high"],
                         "f1_degenerate_replicates": int(rep.degenerate[e.name].sum()),
                         "auroc_ci_low": auc_ci["ci_low"], "auroc_ci_high": auc_ci["ci_high"],
                         "auroc_skipped_replicates": auc_ci["skipped_replicates"]})
        base = interval(rep.all_hallucinated)
        rows.append({**keys, "evaluator": "all_hallucinated", "n": len(y), "prevalence": float(y.mean()),
                     "f1": all_hallucinated_f1(y), "auroc": 0.5, "f1_ci_low": base["ci_low"],
                     "f1_ci_high": base["ci_high"]})
        return rows

    # Analyses

    def main_cells(self) -> None:
        self.cells: dict[tuple[str, str, str, str], dict] = {}
        for split in self.splits:
            for cond in CONDITIONS:
                for tier in TIERS:
                    for subset in SUBSETS:
                        self.cells[(split, cond, tier, subset)] = self.cell(split, cond, tier, subset)
        rows = [r for (split, cond, tier, subset), c in self.cells.items()
                for r in self.evaluator_rows(c, {"split": split, "condition": cond, "tier": tier, "subset": subset})]
        self.tables["evaluators"] = pd.DataFrame(rows)

    def research_questions(self) -> None:
        rq1, rq2, outcomes = [], [], []
        ref = self.ecfg["rq2_reference"]
        for cond in CONDITIONS:
            for tier in TIERS:
                for subset in SUBSETS:
                    for metric in METRICS:
                        passes1 = []
                        for split in self.splits:
                            c = self.cells[(split, cond, tier, subset)]
                            r = rq1_split(c["rep"], c["point"], "classifier", c["single"], metric)
                            rq1.append({"tier": tier, "condition": cond, "subset": subset, "split": split,
                                        "best_single": c["single"], "n": len(c["y"]),
                                        "hypothesis_tier": hypothesis_tier(cond, subset, metric, split, ref), **r})
                            if split in self.ecfg["rq1_splits"]:
                                passes1.append(r["passes"])
                        passes2 = []
                        rc = self.cells[(ref, cond, tier, subset)]
                        for split in self.ecfg["rq2_splits"]:
                            c = self.cells[(split, cond, tier, subset)]
                            r = rq2_split(rc["rep"], c["rep"], rc["point"]["classifier"], c["point"]["classifier"],
                                          "classifier", self.ecfg["rq2_margin"], c["y"], metric)
                            rq2.append({"tier": tier, "condition": cond, "subset": subset, "split": split,
                                        "n_reference": len(rc["y"]), "n": len(c["y"]),
                                        "hypothesis_tier": hypothesis_tier(cond, subset, metric, split, ref), **r})
                            passes2.append(r["passes"])
                        level = hypothesis_tier(cond, subset, metric)
                        for name, passes in (("RQ1", passes1), ("RQ2", passes2)):
                            outcomes.append({"question": name, "tier": tier, "condition": cond, "subset": subset,
                                             "metric": metric, "hypothesis_tier": level,
                                             "splits_passing": int(sum(passes)), "outcome": rq_outcome(passes)})
        self.tables["rq1"] = pd.DataFrame(rq1)
        self.tables["rq2"] = pd.DataFrame(rq2)
        self.tables["rq_outcomes"] = pd.DataFrame(outcomes)

    def judge_tables(self) -> None:
        stats, comps = [], []
        for (j, split, cond), rows in self.judge_rows.items():
            d = self.data[split][cond]
            ids = list(d.meta["answer_id"])
            labels = dict(zip(ids, d.y.tolist()))
            stats.append({"judge": j, "split": split, "condition": cond, **score_cell(rows, labels, ids)})
            for tier in TIERS:
                c = self.cells[(split, cond, tier, "all")]
                name = f"judge:{j}"
                diff = interval(c["rep"].f1["classifier"] - c["rep"].f1[name])
                comps.append({"judge": j, "split": split, "condition": cond, "tier": tier, "n": len(c["y"]),
                              "f1_classifier": c["point"]["classifier"]["f1"], "f1_judge": c["point"][name]["f1"],
                              "diff_classifier_minus_judge": c["point"]["classifier"]["f1"] - c["point"][name]["f1"],
                              "diff_ci_low": diff["ci_low"], "diff_ci_high": diff["ci_high"]})
        self.tables["judges"] = pd.DataFrame(stats)
        self.tables["judge_comparisons"] = pd.DataFrame(comps)

    def gold(self) -> None:
        rows, diffs = [], []
        for split in self.ecfg["gold_splits"]:
            for tier in TIERS:
                c = self.cell(split, "gold", tier, "all")
                rows += self.evaluator_rows(c, {"split": split, "condition": "gold", "tier": tier, "subset": "all"})
                for metric in METRICS:
                    diffs.append({"split": split, "tier": tier, "best_single": c["single"],
                                  "hypothesis_tier": "exploratory",
                                  **rq1_split(c["rep"], c["point"], "classifier", c["single"], metric)})
        self.tables["gold"] = pd.DataFrame(rows)
        self.tables["gold_differences"] = pd.DataFrame(diffs)

    def pooled(self) -> None:
        """Exploratory: the three test splits as one sample (clusters never span splits)."""
        rows = []
        pooled_judges = [j for j in JUDGES if set(self.ecfg["judges"]["plan"][j]["splits"]) == set(self.splits)]
        for cond in CONDITIONS:
            for tier in TIERS:
                parts = [self.cell_parts(split, cond, tier, pooled_judges) for split in self.splits]
                evs = []
                for k, e in enumerate(parts[0][0]):
                    evs.append(Scored(e.name, [np.concatenate([p[0][k].scores[i] for p in parts])
                                               for i in range(len(e.scores))], e.thresholds, e.ranking))
                y = np.concatenate([p[1] for p in parts])
                groups = np.concatenate([p[2] for p in parts])
                c = self.score_cell(evs, y, groups, self.seeds["pooled"])
                rows += self.evaluator_rows(c, {"split": "pooled", "condition": cond, "tier": tier, "subset": "all"})
        self.tables["pooled"] = pd.DataFrame(rows)

    def cell_parts(self, split: str, cond: str, tier: str, judges: list[str]):
        d = self.data[split][cond]
        mask = subset_mask(d, tier, "all")
        evs = [self.classifier(tier, cond, d, mask), self.single_feature(tier, cond, d, mask)]
        evs += [e for e in self.judges_for(split, cond, d, mask) if e.name.removeprefix("judge:") in judges]
        return evs, d.y[mask], np.array([f"{split}:{g}" for g in d.groups[mask]])

    def calibration(self) -> None:
        n_bins = self.ecfg["ece_bins"]
        ece_rows, rel_rows, drift = [], [], []
        for split in self.splits:
            for cond in CONDITIONS:
                d = self.data[split][cond]
                for tier in TIERS:
                    mask = subset_mask(d, tier, "all")
                    y = d.y[mask]
                    for art in self.artifacts[(tier, cond)]:
                        P = calibrated(art["calibrator"], d.subset(tuple(art["features"]))[mask])
                        keys = {"split": split, "condition": cond, "tier": tier, "seed": art["seed"],
                                "method": art["calibration_method"]}
                        ece_rows.append({**keys, "n": len(y), "ece": ece(P, y, n_bins), "mean_p": float(P.mean()),
                                         "prevalence": float(y.mean())})
                        rel_rows += [{**keys, **r} for r in reliability(P, y, n_bins).to_dict("records")]
                        pol = art["policy"]
                        val = self.dev.val[cond]
                        vmask = subset_mask(val, tier, "all")
                        vP = calibrated(art["calibrator"], np.ascontiguousarray(val.subset(tuple(art["features"]))[vmask]))
                        v = band_rates(vP, val.y[vmask], pol["t_low"], pol["t_high"])
                        drift.append({**keys, "t_low": pol["t_low"], "t_high": pol["t_high"],
                                      "feasible_on_val": pol["feasible"],
                                      **{f"val_{k}": v[k] for k in ("accept_rate", "escalate_rate", "flag_rate",
                                                                   "accept_hallucinated_share",
                                                                   "flag_hallucinated_share")},
                                      **band_rates(P, y, pol["t_low"], pol["t_high"])})
        self.tables["calibration"] = pd.DataFrame(ece_rows)
        self.tables["reliability"] = pd.DataFrame(rel_rows)
        self.tables["policy_drift"] = pd.DataFrame(drift)

    # Cost and latency

    def classifier_seconds(self, art: dict, X: np.ndarray, with_policy: bool) -> np.ndarray:
        """Classifier (+ calibration and policy band) per question at batch size 1, warm."""
        out = np.empty(len(X))
        with threadpool_limits(1):
            score(art["estimator"], X[:1])
            for i in range(len(X)):
                x = X[i:i + 1]
                t0 = time.perf_counter()
                score(art["estimator"], x)
                if with_policy:
                    p = art["calibrator"].predict_proba(x)[:, 1]
                    np.where(p < art["policy"]["t_low"], 0, np.where(p < art["policy"]["t_high"], 1, 2))
                out[i] = time.perf_counter() - t0
        return out

    def resampling_cost(self, split: str, d: SplitData, mask: np.ndarray) -> dict:
        answers = self.answer_rows[split]
        spec = self.gcfg.generators[d.meta["answer_id"].iloc[0].split("/")[1]]
        costs, par, seq, complete, prim = [], [], [], True, []
        for qid in d.meta["question_id"].to_numpy()[mask]:
            a = answers[qid]
            res = a.resamples
            complete &= len(res) == self.gcfg.resample_n and all(r.get("prompt_tokens") is not None for r in res)
            costs.append(sum(spec.cost_usd(r["prompt_tokens"] or 0, r["completion_tokens"] or 0) for r in res))
            lat = [r["latency_s"] for r in res]
            par.append(max(lat) if lat else 0.0)
            seq.append(sum(lat))
            prim.append(spec.cost_usd(a.primary["prompt_tokens"] or 0, a.primary["completion_tokens"] or 0))
        return {"cost": np.array(costs), "parallel": np.array(par), "sequential": np.array(seq),
                "complete": bool(complete), "primary_cost": float(np.mean(prim)) if prim else np.nan}

    def cost_latency(self) -> None:
        rows = []
        q = lambda v: {"latency_median_s": float(np.median(v)), "latency_p95_s": float(np.percentile(v, 95))}
        for split in self.splits:
            for cond in CONDITIONS:
                d = self.data[split][cond]
                retrieval = d.meta["meta_retrieval_s"].to_numpy(dtype=float) if cond == "retrieved" else np.zeros(len(d.y))
                for tier in TIERS:
                    mask = subset_mask(d, tier, "all")
                    feats = tier_features(tier, cond)
                    art = self.artifacts[(tier, cond)][0]
                    clf = self.classifier_seconds(art, d.subset(tuple(art["features"]))[mask], with_policy=True)
                    local = retrieval[mask] + d.feature_seconds(feats)[mask] + clf
                    keys = {"split": split, "condition": cond, "tier": tier, "n": int(mask.sum())}
                    if tier == "resampling":
                        rs = self.resampling_cost(split, d, mask)
                        extra = {"cost_per_verdict_usd": float(rs["cost"].mean()), "cost_complete": rs["complete"],
                                 "primary_cost_usd_context": rs["primary_cost"]}
                        rows.append({**keys, "evaluator": "classifier", "latency_basis": "parallel_equivalent",
                                     **extra, **q(local + rs["parallel"])})
                        rows.append({**keys, "evaluator": "classifier", "latency_basis": "sequential",
                                     **extra, **q(local + rs["sequential"])})
                    else:
                        rows.append({**keys, "evaluator": "classifier", "latency_basis": "local",
                                     "cost_per_verdict_usd": 0.0, "cost_complete": True, **q(local)})
                    feat, _ = self.single[(tier, cond)]
                    s_local = d.feature_seconds((feat,))[mask] + (retrieval[mask] if feat in EVIDENCE_FEATURES else 0.0)
                    if feat in RESAMPLING_FEATURES:
                        rs = self.resampling_cost(split, d, mask)
                        rows.append({**keys, "evaluator": f"feature:{feat}", "latency_basis": "parallel_equivalent",
                                     "cost_per_verdict_usd": float(rs["cost"].mean()), "cost_complete": rs["complete"],
                                     **q(s_local + rs["parallel"])})
                    else:
                        rows.append({**keys, "evaluator": f"feature:{feat}", "latency_basis": "local",
                                     "cost_per_verdict_usd": 0.0, "cost_complete": True, **q(s_local)})
                for j in JUDGES:
                    jr = self.judge_rows.get((j, split, cond))
                    if jr is None:
                        continue
                    ids = list(d.meta["answer_id"])
                    sc = score_cell(jr, dict(zip(ids, d.y.tolist())), ids)
                    rows.append({"split": split, "condition": cond, "tier": "", "n": len(ids), "evaluator": f"judge:{j}",
                                 "latency_basis": "api_concurrency_8", "cost_per_verdict_usd": sc["cost_per_verdict_usd"],
                                 "cost_complete": sc["cost_complete"], "latency_median_s": sc["latency_median_s"],
                                 "latency_p95_s": sc["latency_p95_s"]})
        self.tables["cost_latency"] = pd.DataFrame(rows)

    def pareto(self) -> None:
        cl, ev = self.tables["cost_latency"], self.tables["evaluators"]
        splits = self.ecfg["rq1_splits"]
        fig, axes = plt.subplots(1, len(splits), figsize=(5.5 * len(splits), 4.2), sharey=True)
        for ax, split in zip(np.atleast_1d(axes), splits):
            e = ev[(ev["split"] == split) & (ev["condition"] == "retrieved") & (ev["subset"] == "all")]
            c = cl[(cl["split"] == split) & (cl["condition"] == "retrieved")
                   & cl["latency_basis"].isin(["local", "parallel_equivalent", "api_concurrency_8"])]
            seen = set()
            for r in c.itertuples(index=False):
                label = f"{r.tier} classifier" if r.evaluator == "classifier" else r.evaluator
                if label in seen:
                    continue
                seen.add(label)
                pick = e[(e["evaluator"] == r.evaluator) & ((e["tier"] == r.tier) | (r.tier == ""))].iloc[0]
                ax.errorbar(r.cost_per_verdict_usd, pick["f1"], yerr=[[pick["f1"] - pick["f1_ci_low"]],
                            [pick["f1_ci_high"] - pick["f1"]]], fmt="o", capsize=3, label=label)
            base = e[e["evaluator"] == "all_hallucinated"]["f1"].iloc[0]
            ax.axhline(base, ls="--", c="grey", lw=1, label="all hallucinated")
            ax.set_xscale("symlog", linthresh=1e-5)
            ax.set_xlim(-2e-6, max(1e-4, float(c["cost_per_verdict_usd"].max())) * 3)
            ax.set_title(f"{split}, retrieved evidence")
            ax.set_xlabel("incremental cost per verdict (USD)")
        np.atleast_1d(axes)[0].set_ylabel("hallucinated-class F1 (95% CI)")
        np.atleast_1d(axes)[-1].legend(fontsize=7, loc="center right")
        fig.tight_layout()
        fig.savefig(self.out / "pareto_f1_cost.png", dpi=150)
        plt.close(fig)

    # Output

    def manifest(self) -> dict:
        data_files = sorted({p for p in data_access_table(ROOT)["path"]})
        return {
            "stand_in": self.stand_in, "git": git_state(), "python": platform.python_version(),
            "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
            "platform": platform.platform(), "evaluation_config_sha256": sha256(ROOT / self.args.evaluation_config),
            "evaluation_status": self.ecfg["status"], "bootstrap_seeds": self.seeds,
            "replicates": self.ecfg["bootstrap"]["replicates"], "flagged_splits": sorted(self.flagged),
            "step7_artifacts": {a["path"]: a["sha256"] for a in self.step7_manifest["artifacts"]},
            "inputs_sha256": {p: sha256(ROOT / p) for p in data_files if (ROOT / p).is_file()},
            "timings": self.timings,
        }

    def readme(self) -> str:
        out = self.tables["rq_outcomes"]
        prim = out[out["hypothesis_tier"] == "primary"]
        lines = ["# Step 9: test evaluation" + (" (STAND-IN DRY RUN on val_id; not test results)" if self.stand_in else ""), ""]
        if self.stand_in:
            lines += ["Every test split below is val_id under its own bootstrap seed, and the judge rows are",
                      "the Step 8 val_id rows. These numbers exercise the code only.", ""]
        if self.flagged:
            lines += [f"**Exclusion flag.** Over 5% of answers excluded in: {', '.join(sorted(self.flagged))}.", ""]
        lines += ["## Primary research-question outcomes (retrieved evidence, F1)", "",
                  md_table(prim, ["question", "tier", "splits_passing", "outcome"]), "",
                  "## RQ1 per split", "",
                  md_table(self.tables["rq1"].query("subset == 'all' and metric == 'f1'"),
                           ["tier", "condition", "split", "hypothesis_tier", "best_single", "classifier_value",
                            "best_single_value", "diff", "diff_ci_low", "diff_ci_high", "passes"]), "",
                  "## RQ2 per split", "",
                  md_table(self.tables["rq2"].query("subset == 'all' and metric == 'f1'"),
                           ["tier", "condition", "split", "hypothesis_tier", "reference_value", "split_value", "drop",
                            "drop_within_margin", "vs_trivial", "vs_trivial_ci_low", "passes"]), "",
                  "## Exclusions", "", md_table(self.tables["exclusions"],
                                               ["split", "planned", "primary_api_failure", "empty", "labeled",
                                                "excluded_undecidable", "exclusion_rate", "flagged_over_5pct"]), "",
                  "All tables are CSV files in this directory; `data_access.csv` lists every data file read."]
        return "\n".join(lines) + "\n"

    def write(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        for name, df in self.tables.items():
            df = df.copy()
            if "split" in df.columns and name != "exclusions":
                df["flagged_label_split"] = df["split"].isin(self.flagged) | ((df["split"] == "pooled") & bool(self.flagged))
            df.round(ROUND).to_csv(self.out / f"{name}.csv", index=False)
        self.pareto()
        access = data_access_table(ROOT)
        if self.stand_in and access["test_split"].any():
            sys.exit(f"stand-in run opened test-split files: {list(access.loc[access['test_split'], 'path'])}")
        access.to_csv(self.out / "data_access.csv", index=False)
        (self.out / "manifest.json").write_text(json.dumps(self.manifest(), indent=1) + "\n")
        (self.out / "README.md").write_text(self.readme())

    def run(self) -> None:
        self.timed("frozen inputs and threshold checks", self.load_frozen)
        self.timed("test data", self.load_test)
        self.timed("judge rows", self.load_judges)
        self.timed("bootstrap: main cells", self.main_cells)
        self.timed("research questions", self.research_questions)
        self.timed("judges", self.judge_tables)
        self.timed("gold (exploratory)", self.gold)
        self.timed("pooled (exploratory)", self.pooled)
        self.timed("calibration and policy drift", self.calibration)
        self.timed("cost and latency", self.cost_latency)
        self.write()
        if self.stand_in and any(is_test_path(p) for p in OPENED + ACCESS_LOG):
            sys.exit("stand-in run touched a test-split path")
        print(f"wrote {self.out.relative_to(ROOT)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--feature-config", default="configs/features.yaml")
    p.add_argument("--training-config", default="configs/training.yaml")
    p.add_argument("--generation-config", default="configs/generation.yaml")
    p.add_argument("--evaluation-config", default="configs/evaluation.yaml")
    p.add_argument("--stand-in", action="store_true", help="dry run with val_id in place of every test split")
    p.add_argument("--replicates", type=int, help="stand-in only: fewer bootstrap replicates for a quick check")
    args = p.parse_args()
    if args.replicates and not args.stand_in:
        sys.exit("--replicates is only for the stand-in dry run")
    install_audit_hook()
    run = Step9(args)
    if args.replicates:
        run.ecfg["bootstrap"]["replicates"] = args.replicates
    run.run()


if __name__ == "__main__":
    main()
