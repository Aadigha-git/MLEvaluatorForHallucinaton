"""reports/step6/README.md: a readable summary of the Step 6 CSV tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def _fmt(v: Any) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else ""
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{v:.3f}"
    return str(v)


def md_table(df: pd.DataFrame, cols: Sequence[str], headers: Sequence[str] | None = None) -> str:
    pairs = [(c, h) for c, h in zip(cols, headers or cols) if c in df.columns]
    cols, headers = [c for c, _ in pairs], [h for _, h in pairs]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(cols)]
    for r in df[list(cols)].itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v) for v in r) + " |")
    return "\n".join(lines)


# Splits each table's rows draw on: "all" rows use both, "per_row" tables name the split in a column,
# and validation_summary rows use train only through a fitted classifier.
_ALL_ROWS = {
    "cv_results": ("train",), "selected_hyperparameters": ("train",), "ablation_forward_candidates": ("train",),
    "family_selection": ("train", "val_id"), "validation_per_seed": ("train", "val_id"),
    "ablation_forward": ("train", "val_id"), "ablation_leave_one_feature_out": ("train", "val_id"),
    "ablation_leave_one_group_out": ("train", "val_id"), "permutation_importance": ("train", "val_id"),
    "importance_stability": ("train", "val_id"), "single_features": ("val_id",),
}
_PER_ROW = ("correlations", "vif", "feature_distributions", "missingness")
FLAG_COLUMN = "flagged_label_splits"


def flagged_splits(labels_summary: pd.DataFrame) -> set[str]:
    if "flagged_over_5pct" not in labels_summary:
        return set()
    flags = labels_summary["flagged_over_5pct"].astype(str).str.lower().isin(("true", "1", "yes"))
    return set(labels_summary.loc[flags, "split"])


def _validation_summary_splits(df: pd.DataFrame) -> list[tuple[str, ...]]:
    def uses(evaluator: str) -> tuple[str, ...]:
        parts = evaluator.split(" - ")
        fitted = any(not p.startswith("feature:") and p != "all_hallucinated" for p in parts)
        return ("train", "val_id") if fitted else ("val_id",)
    return [uses(e) for e in df["evaluator"]]


def _logo_twins(df: pd.DataFrame) -> list[str]:
    """For tiers with exactly two feature groups, "-A" and "only:B" fit the same features."""
    from hdetect.training import present_groups, tier_features

    def feats(r) -> frozenset[str]:
        all_f = tier_features(r["tier"], r["train_condition"])
        groups = present_groups(all_f)
        name = r["subset"]
        if name.startswith("-"):
            return frozenset(all_f) - frozenset(groups[name[1:]])
        if name.startswith("only:"):
            return frozenset(groups[name[5:]])
        return frozenset(all_f)

    sets = [feats(r) for r in df.to_dict("records")]
    keys = list(zip(df["tier"], df["train_condition"], df["eval_condition"]))
    out = []
    for i, (k, s) in enumerate(zip(keys, sets)):
        twins = [df["subset"].iloc[j] for j, (k2, s2) in enumerate(zip(keys, sets)) if j != i and k2 == k and s2 == s]
        out.append(", ".join(twins))
    return out


_COUNTS = ("f1_degenerate_replicates", "f1_skipped_replicates", "auroc_skipped_replicates", "auprc_skipped_replicates")


def _backfill_diff_counts(vs: pd.DataFrame) -> pd.DataFrame:
    """Difference rows from tables written before the bootstrap counted them: a difference replicate is
    degenerate or skipped only if one of its two evaluators is, so two zero counts give exactly zero.
    Anything else stays missing rather than guessed."""
    vs = vs.copy()
    for col in _COUNTS:
        if col not in vs:
            vs[col] = np.nan
    for i, r in vs[vs["evaluator"].str.contains(" - ")].iterrows():
        a, b = r["evaluator"].split(" - ")
        same = (vs["tier"] == r["tier"]) & (vs["eval_condition"] == r["eval_condition"])
        for col in _COUNTS:
            if pd.isna(r[col]):
                comp = vs.loc[same & vs["evaluator"].isin((a, b)), col]
                if len(comp) == 2 and (comp == 0).all():
                    vs.loc[i, col] = 0
    return vs


def annotate_tables(tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Mark every result row that rests on a split with more than 5% of answers excluded, and mark
    leave-one-group-out rows that fit the same features as another row."""
    flagged = flagged_splits(tables["labels_summary"])
    out = dict(tables)
    for name, df in tables.items():
        if name in _ALL_ROWS:
            uses = [_ALL_ROWS[name]] * len(df)
        elif name in _PER_ROW:
            uses = [(s,) for s in df["split"]]
        elif name == "validation_summary":
            uses = _validation_summary_splits(df)
        else:
            continue
        df = df.copy()
        df[FLAG_COLUMN] = [", ".join(s for s in u if s in flagged) for u in uses]
        out[name] = df
    out["validation_summary"] = _backfill_diff_counts(out["validation_summary"])
    logo = out["ablation_leave_one_group_out"].copy()
    logo["same_features_as"] = _logo_twins(logo)
    out["ablation_leave_one_group_out"] = logo
    return out


def _flag_note(t: Mapping[str, pd.DataFrame], what: str) -> list[str]:
    ls = t["labels_summary"]
    flagged = sorted(flagged_splits(ls))
    if not flagged:
        return []
    rates = ", ".join(f"`{s}` {float(ls.loc[ls['split'] == s, 'exclusion_rate'].iloc[0]):.1%}" for s in flagged)
    return [f"> **Exclusion flag.** {what} rest on {', '.join(f'`{s}`' for s in flagged)}, which has more than 5% of its "
            f"answers excluded as undecidable ({rates}); rows are marked in `{FLAG_COLUMN}`.", ""]


def _ci(row, m: str) -> str:
    lo, hi = row.get(f"{m}_ci_low"), row.get(f"{m}_ci_high")
    if lo is None or pd.isna(lo):
        return _fmt(row.get(m))
    return f"{row[m]:.3f} [{lo:.3f}, {hi:.3f}]"


def write_readme(tables: Mapping[str, pd.DataFrame], notes: Mapping[str, Any], tcfg: Mapping[str, Any]) -> str:
    t = tables
    out = ["# Step 6: classifier training, ablations, and feature redundancy (train and val_id only)", ""]
    out += [
        "Fitting and cross-validation use `train` (TruthfulQA, Model A). Model choice, F1 decision thresholds, and",
        "every metric below use `val_id` (HaluEval, Model A). No test split was read (see `data_access.csv`).",
        "Validation F1 uses the threshold chosen on `val_id` itself, so it is optimistic by the same rule for every",
        "evaluator; AUROC and AUPRC are threshold-free and computed on uncalibrated scores. These are not test results",
        "and do not answer RQ1 or RQ2. Analysis rules the pre-registration leaves open are listed in its amendment log",
        "(2026-10-09, Step 6 analysis details).",
        "",
        "**Label limitation.** The HaluEval labels failed the hand-audit agreement gate (Cohen's kappa 0.7117 in the",
        "first round and 0.578 in the fresh round under rule version 2, against a minimum of 0.6), so every `val_id`",
        "number below rests on labels below the pre-registered agreement threshold (amendment 2026-10-09).",
        "",
        "## Labels",
        "",
        md_table(t["labels_summary"], ["split", "answers", "labeled", "hallucinated", "not_hallucinated", "auto_labeled",
                                       "hand_labeled", "excluded", "exclusion_rate", "flagged_over_5pct", "abstained"]),
        "",
        "## Selected models",
        "",
        *_flag_note(t, "Hyperparameters, family choice, and every classifier result in this report"),
        f"Seeds {', '.join(map(str, tcfg['seeds']))}. Hyperparameters by 5-fold StratifiedGroupKFold on `train` (grouped by",
        "`cluster_id`, average precision); the family by mean `val_id` F1 over seeds, per tier and evidence condition.",
        "The gold-evidence rows reuse the retrieved-evidence classifier.",
        "",
        md_table(t["family_selection"], ["tier", "condition", "family", "mean_val_f1_over_seeds", "selected"]),
        "",
        md_table(t["selected_hyperparameters"], ["tier", "condition", "family", "seed", "config", "cv_ap_mean", "cv_ap_sd"]),
        "",
        "## Validation results",
        "",
        "Point estimates are means over the three seeds; brackets are 95% paired cluster-bootstrap intervals",
        f"({tcfg['bootstrap']['replicates']:,} replicates over `val_id` clusters, thresholds held fixed, seed-averaged",
        "within each replicate). The difference row is the selected classifier minus the best single feature.",
        "Degenerate replicates (pre-registered: F1 is 0 with no true positives, false positives, or false negatives;",
        "AUROC and AUPRC are skipped for one-class replicates) are counted for every interval.",
        "",
        *_flag_note(t, "Classifier rows and difference rows (single-feature and all-hallucinated rows use `val_id` only)"),
    ]
    vs = t["validation_summary"].copy()
    keep = vs["selected_family"].fillna(False).astype(bool) | vs["best_single_feature"].fillna(False).astype(bool) \
        | vs["evaluator"].eq("all_hallucinated") | vs["evaluator"].str.contains(" - ")
    vs = vs[keep]
    rows = []
    for r in vs.to_dict("records"):
        if " - " in r["evaluator"]:
            rows.append({**r, "F1": f"{r['f1']:+.3f} [{r['f1_diff_ci_low']:+.3f}, {r['f1_diff_ci_high']:+.3f}]",
                         "AUROC": f"{r['auroc']:+.3f} [{r['auroc_diff_ci_low']:+.3f}, {r['auroc_diff_ci_high']:+.3f}]",
                         "AUPRC": f"{r['auprc']:+.3f} [{r['auprc_diff_ci_low']:+.3f}, {r['auprc_diff_ci_high']:+.3f}]"})
        else:
            rows.append({**r, "F1": _ci(r, "f1"), "AUROC": _ci(r, "auroc"), "AUPRC": _ci(r, "auprc")})
    out += [md_table(pd.DataFrame(rows), ["tier", "eval_condition", "evaluator", "n_val", "F1", "AUROC", "AUPRC",
                                          "f1_degenerate_replicates", "auroc_skipped_replicates",
                                          "auprc_skipped_replicates", FLAG_COLUMN],
                     ["tier", "eval_condition", "evaluator", "n_val", "F1", "AUROC", "AUPRC", "F1 degenerate",
                      "AUROC skipped", "AUPRC skipped", "flagged splits"]), ""]
    diffs = vs[vs["evaluator"].str.contains(" - ")]
    if len(diffs) and "f1_diff_ci_low" in diffs:
        wins = int((diffs["f1_diff_ci_low"] > 0).sum())
        out += [f"Selected classifiers beating the best single feature on `val_id` (lower bound of the F1 difference",
                f"interval above 0): {wins} of {len(diffs)} tier and evidence-condition rows. The paired comparison with",
                "predicting every answer hallucinated is in `reports/step7/baseline_check.csv`. These are validation",
                "results, not research-question verdicts.", ""]

    vps = t["validation_per_seed"].sort_values(["tier", "eval_condition", "family", "seed"], kind="stable")
    out += ["Per-seed validation metrics (both families):", "",
            md_table(vps, ["tier", "eval_condition", "family", "seed", "config", "threshold", "f1", "precision", "recall",
                           "auroc", "auprc"]), ""]

    sf = t["single_features"]
    out += ["## Single features alone (direction and threshold chosen on `val_id`)", "",
            md_table(sf[sf["tier"] == "resampling"].sort_values(["eval_condition", "f1"], ascending=[True, False]),
                     ["eval_condition", "feature", "direction", "f1", "auroc", "auprc", "best_in_tier_condition"]),
            "", "The single-shot tier's best feature is the best of the rows above without `sc_*`.", ""]

    fwd = t["ablation_forward"]
    out += ["## Ablations", "",
            "All ablations use the selected family for each tier and training condition. Every subset gets its own",
            "cross-validated hyperparameters per seed on `train`; values are means over seeds on `val_id`. Forward",
            "selection adds, at each step, the feature with the highest mean `train` CV average precision; `val_id` never",
            "chooses the order.", "",
            *_flag_note(t, "All ablation results"),
            "### Add one feature at a time from the bare baseline", "",
            md_table(fwd, ["tier", "train_condition", "eval_condition", "step", "added", "cv_ap_mean_mean", "f1_mean",
                           "auroc_mean", "auprc_mean"]), ""]
    lofo = t["ablation_leave_one_feature_out"]
    out += ["### Leave one feature out (change from all features)", "",
            md_table(lofo[lofo["subset"] != "all"], ["tier", "train_condition", "eval_condition", "removed",
                                                    "delta_cv_ap_mean_mean", "delta_f1_mean", "delta_auroc_mean",
                                                    "delta_auprc_mean"]), ""]
    logo = t["ablation_leave_one_group_out"]
    out += ["### Leave one group out, and each group alone (change from all features)", "",
            "Where a tier and condition has only two feature groups (the single-shot tier, and the resampling tier",
            "without evidence), dropping one group leaves exactly the other, so \"-A\" and \"only:B\" fit the same",
            "features and their rows are identical by construction. `same_features_as` names the twin row.", "",
            md_table(logo[logo["subset"] != "all"], ["tier", "train_condition", "eval_condition", "subset",
                                                    "delta_cv_ap_mean_mean", "delta_f1_mean", "delta_auroc_mean",
                                                    "delta_auprc_mean", "same_features_as"]), ""]

    st = t["importance_stability"]
    pairs = st[st["seed_a"].notna()] if "seed_a" in st else st.iloc[0:0]
    feats = st[st["feature"].notna()] if "feature" in st else st.iloc[0:0]
    out += ["## Permutation importance and stability across seeds", "",
            f"Drop in `val_id` average precision when a feature is permuted ({tcfg['permutation_importance']['n_repeats']}",
            "repeats), for each seed's selected classifier.", "",
            *_flag_note(t, "All importance results"),
            md_table(feats, ["tier", "eval_condition", "feature", "importance_mean_over_seeds", "importance_sd_over_seeds",
                             "rank_min", "rank_max"]), "",
            md_table(pairs.groupby(["tier", "eval_condition"], sort=False)[["spearman_rho", "top3_jaccard"]].agg(["min", "mean"])
                     .pipe(lambda d: d.set_axis([f"{a}_{b}" for a, b in d.columns], axis=1)).reset_index(),
                     ["tier", "eval_condition", "spearman_rho_min", "spearman_rho_mean", "top3_jaccard_min", "top3_jaccard_mean"]),
            ""]

    corr = t["correlations"]
    strong = corr[(corr["method"] == "spearman") & (corr["r"].abs() >= 0.7)]
    vif = t["vif"]
    out += ["## Redundancy", "",
            *_flag_note(t, "Rows for `train` here and in the missingness and `sc_entail` tables below"),
            "Feature pairs with |Spearman r| >= 0.7 (full matrices in `correlations.csv`):", "",
            md_table(strong, ["split", "condition", "feature_a", "feature_b", "r"]) if len(strong) else "None.", "",
            "VIF >= 5 (all values in `vif.csv`):", "",
            md_table(vif[vif["vif"] >= 5], ["split", "condition", "tier", "feature", "vif"]) if (vif["vif"] >= 5).any() else "None.",
            ""]

    miss = t["missingness"]
    rows_info = miss[miss["feature"] == "(rows)"]
    out += ["## Missingness and data checks", "",
            f"Missing or non-finite feature values: {int(miss['missing'].fillna(0).sum())} and "
            f"{int(miss['non_finite'].fillna(0).sum())} (features-v1 never imputes).", "",
            md_table(rows_info, ["split", "condition", "rows", "excluded_undecidable", "no_usable_resamples",
                                 "truncated_primary", "abstained"]), "",
            md_table(t["data_checks"], list(t["data_checks"].columns)), ""]

    d = t["feature_distributions"]
    sc = d[(d["feature"] == "sc_entail") & (d["condition"] == "none")]
    out += ["## Note on `sc_entail`", "",
            "`sc_entail` is kept in the resampling tier exactly as frozen. The pinned NLI model labels near-paraphrased",
            "resamples as neutral, so the feature sits close to zero:", "",
            md_table(sc, ["split", "label", "n", "median", "p95", "max", "share_below_0.01"]), "",
            "Whether it carries signal is left to the pre-registered analyses: removing it (leave one feature out),",
            "its permutation importance, and its value alone:", "",
            md_table(lofo[lofo["removed"] == "sc_entail"], ["train_condition", "eval_condition", "delta_cv_ap_mean_mean",
                                                            "delta_f1_mean", "delta_auroc_mean", "delta_auprc_mean"]), "",
            md_table(feats[feats["feature"] == "sc_entail"], ["eval_condition", "importance_mean_over_seeds",
                                                             "importance_sd_over_seeds", "rank_min", "rank_max"]), "",
            md_table(sf[sf["feature"] == "sc_entail"], ["eval_condition", "direction", "f1", "auroc", "auprc"]), ""]
    out += ["## Retrieval recall", "", notes.get("retrieval", ""), ""]
    out += ["## Dataset and evidence-condition effects", "", notes.get("datasets", ""), ""]

    rt = t["runtime"]
    out += ["## Runtime", "", md_table(rt[rt["kind"] == "stage"], ["stage", "seconds"]), "",
            md_table(rt[rt["kind"] == "classifier_latency_ms"], ["stage", "median_ms", "p95_ms", "n"]), "",
            "Hardware: " + "; ".join(rt[rt["kind"] == "hardware"]["stage"]), "",
            notes.get("feature_times", ""), ""]
    acc = t["data_access"]
    out += ["## Data access", "", f"{len(acc)} files read; test-split files read: "
            f"{int(acc['test_split'].sum())}. Full list in `data_access.csv`.", ""]
    return "\n".join(out)
