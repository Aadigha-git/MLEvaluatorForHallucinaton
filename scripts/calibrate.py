"""Step 7: calibrate the Step 6 classifiers and set the decision policy on val_id; save frozen artifacts.

Refits each selected classifier deterministically from reports/step6/ (family, hyperparameters,
seed) on train, and stops unless every refit reproduces its Step 6 F1 threshold exactly. Then, on
val_id only: cross-fitted Platt-versus-isotonic comparison, the chosen calibrator refitted on all
of val_id, and the accept/escalate/flag thresholds. Writes reports/step7/ and one artifact per tier,
evidence condition, and seed under data/models/step7/ (git-ignored). Reads no test split: every
file opened under data/ is recorded and the run fails if any belongs to a test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hdetect.config import load_config  # noqa: E402
from hdetect.features.pipeline import load_feature_config  # noqa: E402
from hdetect.training import TIERS, TRAIN_CONDITIONS, tier_features  # noqa: E402
from hdetect.training.access import data_access_table, install_audit_hook  # noqa: E402
from hdetect.training.calibration import (  # noqa: E402
    METHODS,
    calibrated,
    cross_fit,
    fit_calibrator,
    policy_thresholds,
    reliability,
    summarize_ece,
)
from hdetect.training.data import resampling_rows  # noqa: E402
from hdetect.training.inputs import load_dev_inputs  # noqa: E402
from hdetect.training.metrics import Evaluator, Threshold, best_threshold, bootstrap  # noqa: E402
from hdetect.training.models import config_from_key, fit, folds, score  # noqa: E402
from hdetect.training.report import flagged_splits, md_table  # noqa: E402
from hdetect.training.run import ROUND  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_VERSION = "step7-v1"
KAPPAS = "Cohen's kappa 0.7117 in the first round and 0.578 in the fresh round under rule version 2"


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state() -> dict[str, str | bool]:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                    capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": True}
    return {"commit": head, "dirty": dirty}


def rows_for(data, tier: str) -> np.ndarray:
    return resampling_rows(data) if tier == "resampling" else np.ones(len(data.y), dtype=bool)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--feature-config", default="configs/features.yaml")
    p.add_argument("--training-config", default="configs/training.yaml")
    p.add_argument("--calibration-config", default="configs/calibration.yaml")
    args = p.parse_args()
    install_audit_hook()

    cfg = load_config(args.config)
    fcfg = load_feature_config(args.feature_config)
    tcfg = yaml.safe_load(Path(args.training_config).read_text())
    ccfg = yaml.safe_load(Path(args.calibration_config).read_text())
    cal, pol, cpaths = ccfg["calibration"], ccfg["policy"], ccfg["paths"]
    if tuple(cal["methods"]) != METHODS:
        raise SystemExit(f"calibration methods must be {METHODS}")

    step6 = Path(cpaths["step6_reports"])
    family_sel = pd.read_csv(step6 / "family_selection.csv")
    hyper = pd.read_csv(step6 / "selected_hyperparameters.csv")
    vps = pd.read_csv(step6 / "validation_per_seed.csv")
    labels_summary = pd.read_csv(step6 / "labels_summary.csv")
    step6_summary = pd.read_csv(step6 / "validation_summary.csv")

    dev = load_dev_inputs(cfg, fcfg, tcfg["paths"], val_conditions=TRAIN_CONDITIONS)
    seeds = list(tcfg["seeds"])
    models_dir, out = Path(cpaths["models"]), Path(cpaths["reports"])
    models_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    refit_rows, ece_rows, rel_rows, policy_rows, base_rows, diag_rows, artifacts = [], [], [], [], [], [], []
    selection_parts = []
    for tier in TIERS:
        for cond in TRAIN_CONDITIONS:
            sel = family_sel[(family_sel["tier"] == tier) & (family_sel["condition"] == cond)
                             & family_sel["selected"].astype(bool)]
            if len(sel) != 1:
                raise SystemExit(f"{tier}/{cond}: Step 6 selected {len(sel)} families")
            family = sel["family"].item()
            feats = tier_features(tier, cond)
            tr, va = dev.train[cond], dev.val[cond]
            tmask, vmask = rows_for(tr, tier), rows_for(va, tier)
            Xtr, ytr, gtr = tr.subset(feats)[tmask], tr.y[tmask], tr.groups[tmask]
            Xv, yv, gv = va.subset(feats)[vmask], va.y[vmask], va.groups[vmask]

            fitted, val_scores, thresholds = {}, {}, {}
            for seed in seeds:
                key = hyper[(hyper["tier"] == tier) & (hyper["condition"] == cond) & (hyper["family"] == family)
                            & (hyper["seed"] == seed)]["config"].item()
                config = config_from_key(family, key, tcfg)
                est = fit(config, seed, Xtr, ytr, tcfg)
                sv = score(est, Xv)
                th = best_threshold(sv, yv)
                ref = vps[(vps["tier"] == tier) & (vps["eval_condition"] == cond) & (vps["family"] == family)
                          & (vps["seed"] == seed)].iloc[0]
                # Step 6 tables are rounded to ROUND decimals, so the refit must match to that precision.
                match = abs(round(th.value, ROUND) - ref["threshold"]) < 10 ** -ROUND and ref["config"] == key
                refit_rows.append({"tier": tier, "condition": cond, "family": family, "seed": seed, "config": key,
                                   "step6_threshold": ref["threshold"], "refit_threshold": th.value,
                                   "step6_f1": ref["f1"], "refit_f1": th.f1, "reproduced": bool(match)})
                if not match:
                    raise SystemExit(f"{tier}/{cond}/{family}/{seed}: refit does not reproduce Step 6")
                fitted[seed], val_scores[seed], thresholds[seed] = (est, key), sv, th

                oof = np.zeros(len(ytr))
                for a, b in folds(ytr, gtr, seed, tcfg["cv"]["n_splits"]):
                    oof[b] = score(fit(config, seed, Xtr[a], ytr[a], tcfg), Xtr[b])
                diag_rows.append({"tier": tier, "condition": cond, "family": family, "seed": seed,
                                  "classes": ",".join(map(str, est.classes_)),
                                  "train_prevalence": float(ytr.mean()), "val_prevalence": float(yv.mean()),
                                  "train_out_of_fold_auroc": roc_auc_score(ytr, oof),
                                  "val_auroc": roc_auc_score(yv, sv),
                                  "val_predicted_hallucinated_share": float((sv >= th.value).mean()),
                                  "threshold_is_lowest_val_score": bool(th.value == sv.min()),
                                  "train_mean_score": float(score(est, Xtr).mean()), "val_mean_score": float(sv.mean())})

                folds_df, oof_p = cross_fit(est, Xv, yv, gv, seed, cal["n_splits"], cal["ece_bins"])
                ece_rows.append(folds_df.assign(tier=tier, condition=cond, family=family))
                for method in METHODS:
                    rel_rows.append(reliability(oof_p[method], yv, cal["ece_bins"])
                                    .assign(tier=tier, condition=cond, method=method, seed=seed, source="cross_fitted"))

            ece_df = pd.concat([r for r in ece_rows if (r["tier"].iloc[0], r["condition"].iloc[0]) == (tier, cond)])
            summary = summarize_ece(ece_df, ["tier", "condition", "family"], cal["platt_margin"])
            selection_parts.append(summary)
            method = summary["chosen_method"].item()

            for seed in seeds:
                est, key = fitted[seed]
                calibrator = fit_calibrator(est, method, Xv, yv)
                P = calibrated(calibrator, Xv)
                policy = policy_thresholds(P, yv, pol["accept_max_hallucinated"], pol["escalate_max_share"])
                policy_rows.append({"tier": tier, "condition": cond, "family": family, "seed": seed, "method": method,
                                    "accept_goal_max_hallucinated": pol["accept_max_hallucinated"],
                                    "escalate_goal_max_share": pol["escalate_max_share"], **policy.to_dict()})
                rel_rows.append(reliability(P, yv, cal["ece_bins"])
                                .assign(tier=tier, condition=cond, method=method, seed=seed, source="final_in_sample"))

                path = models_dir / f"{tier}__{cond}__seed{seed}.joblib"
                artifact = {
                    "artifact_version": ARTIFACT_VERSION, "tier": tier, "condition": cond, "family": family,
                    "config": key, "seed": seed, "features": list(feats), "estimator": est,
                    "f1_threshold": {"value": thresholds[seed].value, "direction": "high",
                                     "rule": "predict hallucinated when uncalibrated score >= value"},
                    "calibration_method": method, "calibrator": calibrator,
                    "policy": {"t_low": policy.t_low, "t_high": policy.t_high, "feasible": policy.feasible,
                               "rule": "accept if P < t_low; escalate if t_low <= P < t_high; flag if P >= t_high"},
                    "positive_class": 1, "positive_label": "hallucinated",
                }
                joblib.dump(artifact, path)
                back = joblib.load(path)
                same = (np.array_equal(score(back["estimator"], Xv), val_scores[seed])
                        and np.array_equal(calibrated(back["calibrator"], Xv), P))
                if not same:
                    raise SystemExit(f"{path}: reloaded artifact does not reproduce its scores")
                artifacts.append({"path": path.as_posix(), "sha256": sha256(path), "tier": tier, "condition": cond,
                                  "seed": seed, "family": family, "config": key, "features": list(feats),
                                  "f1_threshold": thresholds[seed].value, "calibration_method": method,
                                  "t_low": policy.t_low, "t_high": policy.t_high, "policy_feasible": policy.feasible,
                                  "reload_reproduces_scores": True})

            ev = Evaluator(family, [val_scores[s] for s in seeds], [thresholds[s] for s in seeds])
            allpos = Evaluator("predict_all_hallucinated", [np.ones(len(yv))], [Threshold(1.0, 0.0)])
            b = tcfg["bootstrap"]
            boot = bootstrap([ev, allpos], yv, gv, b["replicates"], b["seed"], b["chunk"],
                             pairs=[(family, "predict_all_hallucinated")])
            d = boot[f"{family} - predict_all_hallucinated"]
            s6 = step6_summary[(step6_summary["tier"] == tier) & (step6_summary["eval_condition"] == cond)
                               & step6_summary["evaluator"].str.contains(" - ")].iloc[0]
            base_rows.append({
                "tier": tier, "condition": cond, "family": family, "n_val": len(yv),
                "f1_classifier": np.mean([thresholds[s].f1 for s in seeds]),
                "f1_all_hallucinated": 2 * yv.mean() / (1 + yv.mean()),
                "f1_diff_vs_all_hallucinated": d["f1_diff_mean"], "f1_diff_ci_low": d["f1_diff_ci_low"],
                "f1_diff_ci_high": d["f1_diff_ci_high"], "f1_degenerate_replicates": d["f1_degenerate_replicates"],
                "beats_all_hallucinated_on_val": bool(d["f1_diff_ci_low"] > 0),
                "best_single_feature": s6["evaluator"].split(" - ")[1],
                "f1_diff_vs_best_single": s6["f1"], "f1_diff_vs_best_single_ci_low": s6["f1_diff_ci_low"],
                "f1_diff_vs_best_single_ci_high": s6["f1_diff_ci_high"],
                "beats_best_single_on_val": bool(s6["f1_diff_ci_low"] > 0),
            })

    feat_rows = []
    for feat in tier_features("resampling", "retrieved"):
        xt, xv = dev.train["retrieved"].X[feat], dev.val["retrieved"].X[feat]
        feat_rows.append({"feature": feat, "auroc_train": roc_auc_score(dev.train["retrieved"].y, xt),
                          "auroc_val": roc_auc_score(dev.val["retrieved"].y, xv),
                          "mean_train": xt.mean(), "mean_val": xv.mean(),
                          "standardized_mean_difference": (xv.mean() - xt.mean()) / np.sqrt((xt.var() + xv.var()) / 2)})

    tables = {
        "refit_check": pd.DataFrame(refit_rows),
        "calibration_ece_folds": pd.concat(ece_rows, ignore_index=True),
        "calibration_selection": pd.concat(selection_parts, ignore_index=True),
        "reliability": pd.concat(rel_rows, ignore_index=True),
        "policy": pd.DataFrame(policy_rows),
        "baseline_check": pd.DataFrame(base_rows),
        "diagnostics_models": pd.DataFrame(diag_rows),
        "diagnostics_features": pd.DataFrame(feat_rows),
    }
    tables["data_access"] = data_access_table(ROOT)
    if tables["data_access"]["test_split"].any():
        raise SystemExit("a test-split file was opened; reports not written")

    flagged = sorted(flagged_splits(labels_summary))
    for name in ("refit_check", "calibration_ece_folds", "calibration_selection", "reliability", "policy",
                 "baseline_check", "diagnostics_models"):
        tables[name]["flagged_label_splits"] = ", ".join(flagged)
    tables["diagnostics_features"]["flagged_label_splits"] = ", ".join(flagged)

    for name, df in tables.items():
        df.round(ROUND).to_csv(out / f"{name}.csv", index=False)
    plot_reliability(tables["reliability"], out)

    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "git": git_state(),
        "python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "joblib": joblib.__version__, "platform": platform.platform(),
        "seeds": seeds, "rule_version": cfg.labeling.rule_version, "config_hash": cfg.hash,
        "feature_versions": dev.versions,
        "label_exclusion_flagged_splits": flagged,
        "inputs_sha256": {p: sha256(p) for p in sorted(tables["data_access"]["path"])
                          if not p.startswith(models_dir.as_posix())},
        "configs_sha256": {p: sha256(p) for p in (args.config, args.feature_config, args.training_config,
                                                   args.calibration_config)},
        "step6_tables_sha256": {f.name: sha256(f) for f in sorted(step6.glob("*.csv"))},
        "artifacts": artifacts,
    }
    (models_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    (out / "README.md").write_text(readme(tables, manifest, ccfg, labels_summary) + "\n")
    print(f"reports -> {out}; artifacts -> {models_dir}")


def plot_reliability(rel: pd.DataFrame, out: Path) -> None:
    cf = rel[rel["source"] == "cross_fitted"]
    for (tier, cond), g in cf.groupby(["tier", "condition"], sort=False):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
        for ax, method in zip(axes, METHODS):
            ax.plot([0, 1], [0, 1], color="0.7", lw=1)
            for seed, s in g[g["method"] == method].groupby("seed"):
                s = s[s["n"] > 0]
                ax.plot(s["mean_predicted"], s["observed_hallucinated"], marker="o", lw=1, label=str(seed))
            ax.set_title(f"{method} (cross-fitted)")
            ax.set_xlabel("mean calibrated P(hallucinated)")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        axes[0].set_ylabel("observed hallucinated share")
        axes[1].legend(title="seed", fontsize=8)
        fig.suptitle(f"val_id reliability: {tier}, {cond} evidence")
        fig.tight_layout()
        fig.savefig(out / f"reliability_{tier}_{cond}.png", dpi=120)
        plt.close(fig)


def readme(t: dict[str, pd.DataFrame], manifest: dict, ccfg: dict, labels_summary: pd.DataFrame) -> str:
    flagged = manifest["label_exclusion_flagged_splits"]
    ls = labels_summary.set_index("split")
    flag_text = ", ".join(f"`{s}` ({float(ls.loc[s, 'exclusion_rate']):.1%} of answers excluded)" for s in flagged)
    pol = t["policy"]
    base = t["baseline_check"]
    out = [
        "# Step 7: calibration and decision policy (val_id only)", "",
        "Every number here comes from `train` (refitting the Step 6 classifiers) or `val_id` (calibration and the",
        "policy). No test split was read (see `data_access.csv`). These are not test results.", "",
        f"**Label limitation.** The HaluEval labels failed the hand-audit agreement gate ({KAPPAS}, against a",
        "minimum of 0.6). Calibration, the policy thresholds, and every rate below rest on those `val_id` labels",
        "(amendment 2026-10-09).", "",
    ]
    if flagged:
        out += [f"**Exclusion flag.** Every classifier here was fitted on {flag_text}, above the 5% limit; every",
                "table carries `flagged_label_splits`.", ""]
    n_val = int(pol["n"].iloc[0])
    val_excluded = int(ls.loc["val_id", "excluded"]) if "excluded" in ls.columns else 0
    infeasible = pol[~pol["feasible"].astype(bool)]
    feasible = pol[pol["feasible"].astype(bool)]
    max_accepted = int(round((pol["accept_rate"] * pol["n"]).max()))
    out += [
        "## Summary (validation only; no research-question criterion is judged here)", "",
        f"- **Baselines.** {int(base['beats_all_hallucinated_on_val'].sum())} of {len(base)} selected classifiers beat "
        "predicting every answer hallucinated on `val_id` (lower bound of the paired 95% F1 difference interval",
        f"  above 0), and {int(base['beats_best_single_on_val'].sum())} of {len(base)} beat the best single feature.",
        f"- **Policies.** {len(infeasible)} of {len(pol)} seed-level policies are infeasible (no accept band meets the 5%",
        "  goal); their escalate and flag split is shown for diagnosis only, and their constraints are not met. The",
        f"  {len(feasible)} feasible ones are nearly degenerate: at most {max_accepted} of {n_val} answers accepted. Flag",
        f"  precision is {pol['flag_hallucinated_share'].min():.3f}–{pol['flag_hallucinated_share'].max():.3f}, about the "
        f"`val_id` prevalence of {float(t['diagnostics_models']['val_prevalence'].iloc[0]):.3f}.",
        "- **Seeds.** No deployment seed is selected; all three seeds keep their own calibrator and policy, and",
        "  seed-level variation is reported throughout.",
        "- **Gold evidence.** No calibration or operating-point policy is defined for the gold-evidence condition.",
        f"- **Denominators.** Rates use the {n_val} labeled `val_id` answers; {val_excluded} answers the hand labeler",
        "  marked undecidable are excluded.",
        "- Implementation choices are recorded in docs/PREREGISTRATION.md (amendment 2026-10-09, Step 7",
        "  implementation details).", "",
    ]
    out += [
        "## Refit check", "",
        "Each selected classifier was refitted from its Step 6 family, hyperparameters, and seed; all reproduce",
        "their Step 6 F1 thresholds:", "",
        md_table(t["refit_check"], ["tier", "condition", "family", "seed", "config", "step6_threshold",
                                    "refit_threshold", "reproduced"]), "",
        "## Diagnostics: why every selected classifier flags nearly everything", "",
        md_table(t["diagnostics_models"], ["tier", "condition", "family", "seed", "classes", "train_prevalence",
                                           "val_prevalence", "train_out_of_fold_auroc", "val_auroc",
                                           "val_predicted_hallucinated_share", "threshold_is_lowest_val_score"]), "",
        "Single-feature AUROC with high values read as hallucinated (retrieved condition, resampling tier features):", "",
        md_table(t["diagnostics_features"], ["feature", "auroc_train", "auroc_val", "mean_train", "mean_val",
                                             "standardized_mean_difference"]), "",
        "## Calibration method", "",
        "Mean ECE over 5 cross-fitted `val_id` folds (grouped by `cluster_id`) and the three seeds; isotonic must",
        f"beat Platt by at least {ccfg['calibration']['platt_margin']}. Per-fold values in `calibration_ece_folds.csv`;",
        "reliability diagrams in `reliability_*.png` and `reliability.csv`.", "",
        md_table(t["calibration_selection"], ["tier", "condition", "family", "ece_platt", "ece_isotonic",
                                              "platt_minus_isotonic", "chosen_method"]), "",
        "## Decision policy", "",
        f"Accept if P < t_low (at most {ccfg['policy']['accept_max_hallucinated']:.0%} of accepted answers hallucinated);",
        f"escalate if t_low <= P < t_high (at most {ccfg['policy']['escalate_max_share']:.0%} of traffic); flag otherwise.",
        "P is the chosen calibrator refitted on all of `val_id`. Infeasible means no t_low meets the accept goal, so",
        "the accept band is empty; the escalate and flag columns of infeasible rows are diagnostic only. Plateau",
        "shares are the `val_id` answers whose P equals the threshold.", "",
        md_table(pol, ["tier", "condition", "seed", "method", "n_distinct", "feasible", "t_low", "t_high",
                       "accept_rate", "accept_hallucinated_share", "escalate_rate", "flag_rate",
                       "flag_hallucinated_share", "plateau_share_t_low", "plateau_share_t_high"]), "",
        "## Baseline checks on val_id (diagnostic, not a research-question verdict)", "",
        "The classifier versus predicting every answer hallucinated (paired cluster bootstrap, 10,000 replicates,",
        "thresholds fixed, seed-averaged), and versus the best single feature (from Step 6):", "",
        md_table(base, ["tier", "condition", "family", "f1_classifier", "f1_all_hallucinated",
                        "f1_diff_vs_all_hallucinated", "f1_diff_ci_low", "f1_diff_ci_high",
                        "beats_all_hallucinated_on_val", "best_single_feature", "f1_diff_vs_best_single",
                        "beats_best_single_on_val"]), "",
        f"Classifiers beating all-hallucinated on `val_id`: {int(base['beats_all_hallucinated_on_val'].sum())} of "
        f"{len(base)}; beating the best single feature: {int(base['beats_best_single_on_val'].sum())} of {len(base)}.", "",
        "## Not implemented", "",
        "The explanation generator (docs/SPEC.md, build step 7) is deferred until a separate specification is",
        "written and recorded in the pre-registration.", "",
        "## Artifacts", "",
        f"{len(manifest['artifacts'])} files under `data/models/step7/` (git-ignored), listed with SHA-256 in",
        "`data/models/step7/manifest.json`, together with library versions, the git commit, config and input hashes.",
        "Each holds the fitted classifier (with its scaler for logistic regression), feature order, seed, F1",
        "threshold on the uncalibrated score, calibrator, and policy thresholds. Each was reloaded and reproduced",
        "its `val_id` scores exactly.", "",
        "## Data access", "",
        f"{int((~t['data_access']['path'].str.startswith('data/models/')).sum())} input files read, plus the "
        f"artifacts written here (reopened to check them); test-split files read: "
        f"{int(t['data_access']['test_split'].sum())}.",
    ]
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
