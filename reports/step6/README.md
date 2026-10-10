# Step 6: classifier training, ablations, and feature redundancy (train and val_id only)

Fitting and cross-validation use `train` (TruthfulQA, Model A). Model choice, F1 decision thresholds, and
every metric below use `val_id` (HaluEval, Model A). No test split was read (see `data_access.csv`).
Validation F1 uses the threshold chosen on `val_id` itself, so it is optimistic by the same rule for every
evaluator; AUROC and AUPRC are threshold-free and computed on uncalibrated scores. These are not test results
and do not answer RQ1 or RQ2. Analysis rules the pre-registration leaves open are listed in its amendment log
(2026-10-09, Step 6 analysis details).

**Label limitation.** The HaluEval labels failed the hand-audit agreement gate (Cohen's kappa 0.7117 in the
first round and 0.578 in the fresh round under rule version 2, against a minimum of 0.6), so every `val_id`
number below rests on labels below the pre-registered agreement threshold (amendment 2026-10-09).

## Labels

| split | answers | labeled | hallucinated | not_hallucinated | auto_labeled | hand_labeled | excluded | exclusion_rate | flagged_over_5pct | abstained |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 575 | 522 | 196 | 326 | 245 | 277 | 53 | 0.092 | yes | 4 |
| val_id | 1000 | 995 | 646 | 349 | 905 | 90 | 5 | 0.005 |  | 0 |

## Selected models

> **Exclusion flag.** Hyperparameters, family choice, and every classifier result in this report rest on `train`, which has more than 5% of its answers excluded as undecidable (`train` 9.2%); rows are marked in `flagged_label_splits`.

Seeds 20261008, 20261009, 20261010. Hyperparameters by 5-fold StratifiedGroupKFold on `train` (grouped by
`cluster_id`, average precision); the family by mean `val_id` F1 over seeds, per tier and evidence condition.
The gold-evidence rows reuse the retrieved-evidence classifier.

| tier | condition | family | mean_val_f1_over_seeds | selected |
|---|---|---|---|---|
| single_shot | retrieved | logistic_regression | 0.787 |  |
| single_shot | retrieved | hist_gradient_boosting | 0.788 | yes |
| single_shot | none | logistic_regression | 0.787 | yes |
| single_shot | none | hist_gradient_boosting | 0.787 |  |
| resampling | retrieved | logistic_regression | 0.788 | yes |
| resampling | retrieved | hist_gradient_boosting | 0.787 |  |
| resampling | none | logistic_regression | 0.788 | yes |
| resampling | none | hist_gradient_boosting | 0.787 |  |

| tier | condition | family | seed | config | cv_ap_mean | cv_ap_sd |
|---|---|---|---|---|---|---|
| single_shot | retrieved | hist_gradient_boosting | 20261008 | max_depth=2,learning_rate=0.05,max_iter=100 | 0.552 | 0.058 |
| single_shot | retrieved | hist_gradient_boosting | 20261009 | max_depth=2,learning_rate=0.05,max_iter=100 | 0.561 | 0.046 |
| single_shot | retrieved | hist_gradient_boosting | 20261010 | max_depth=3,learning_rate=0.05,max_iter=100 | 0.542 | 0.041 |
| single_shot | retrieved | logistic_regression | 20261008 | C=0.01,class_weight=None | 0.532 | 0.059 |
| single_shot | retrieved | logistic_regression | 20261009 | C=0.01,class_weight=None | 0.533 | 0.060 |
| single_shot | retrieved | logistic_regression | 20261010 | C=0.01,class_weight=None | 0.530 | 0.033 |
| single_shot | none | hist_gradient_boosting | 20261008 | max_depth=3,learning_rate=0.05,max_iter=100 | 0.590 | 0.047 |
| single_shot | none | hist_gradient_boosting | 20261009 | max_depth=3,learning_rate=0.05,max_iter=100 | 0.578 | 0.084 |
| single_shot | none | hist_gradient_boosting | 20261010 | max_depth=3,learning_rate=0.1,max_iter=100 | 0.572 | 0.066 |
| single_shot | none | logistic_regression | 20261008 | C=0.01,class_weight=balanced | 0.541 | 0.041 |
| single_shot | none | logistic_regression | 20261009 | C=0.01,class_weight=None | 0.537 | 0.051 |
| single_shot | none | logistic_regression | 20261010 | C=0.01,class_weight=balanced | 0.540 | 0.045 |
| resampling | retrieved | hist_gradient_boosting | 20261008 | max_depth=2,learning_rate=0.1,max_iter=100 | 0.563 | 0.059 |
| resampling | retrieved | hist_gradient_boosting | 20261009 | max_depth=2,learning_rate=0.05,max_iter=100 | 0.554 | 0.046 |
| resampling | retrieved | hist_gradient_boosting | 20261010 | max_depth=3,learning_rate=0.05,max_iter=100 | 0.556 | 0.063 |
| resampling | retrieved | logistic_regression | 20261008 | C=0.01,class_weight=None | 0.529 | 0.050 |
| resampling | retrieved | logistic_regression | 20261009 | C=0.01,class_weight=balanced | 0.521 | 0.062 |
| resampling | retrieved | logistic_regression | 20261010 | C=0.01,class_weight=None | 0.520 | 0.043 |
| resampling | none | hist_gradient_boosting | 20261008 | max_depth=2,learning_rate=0.05,max_iter=300 | 0.587 | 0.059 |
| resampling | none | hist_gradient_boosting | 20261009 | max_depth=2,learning_rate=0.05,max_iter=300 | 0.531 | 0.047 |
| resampling | none | hist_gradient_boosting | 20261010 | max_depth=2,learning_rate=0.1,max_iter=100 | 0.568 | 0.080 |
| resampling | none | logistic_regression | 20261008 | C=0.01,class_weight=balanced | 0.532 | 0.035 |
| resampling | none | logistic_regression | 20261009 | C=0.01,class_weight=None | 0.523 | 0.059 |
| resampling | none | logistic_regression | 20261010 | C=0.01,class_weight=balanced | 0.526 | 0.049 |

## Validation results

Point estimates are means over the three seeds; brackets are 95% paired cluster-bootstrap intervals
(10,000 replicates over `val_id` clusters, thresholds held fixed, seed-averaged
within each replicate). The difference row is the selected classifier minus the best single feature.
Degenerate replicates (pre-registered: F1 is 0 with no true positives, false positives, or false negatives;
AUROC and AUPRC are skipped for one-class replicates) are counted for every interval.

> **Exclusion flag.** Classifier rows and difference rows (single-feature and all-hallucinated rows use `val_id` only) rest on `train`, which has more than 5% of its answers excluded as undecidable (`train` 9.2%); rows are marked in `flagged_label_splits`.

| tier | eval_condition | evaluator | n_val | F1 | AUROC | AUPRC | F1 degenerate | AUROC skipped | AUPRC skipped | flagged splits |
|---|---|---|---|---|---|---|---|---|---|---|
| single_shot | retrieved | hist_gradient_boosting | 995 | 0.788 [0.766, 0.810] | 0.361 [0.323, 0.399] | 0.546 [0.512, 0.584] | 0.000 | 0.000 | 0.000 | train |
| single_shot | retrieved | feature:lex_overlap_max | 995 | 0.806 [0.783, 0.827] | 0.725 [0.692, 0.757] | 0.795 [0.759, 0.829] | 0.000 | 0.000 | 0.000 |  |
| single_shot | retrieved | all_hallucinated | 995 | 0.787 [0.766, 0.809] |  |  | 0.000 |  |  |  |
| single_shot | retrieved | hist_gradient_boosting - feature:lex_overlap_max | 995 | -0.018 [-0.034, -0.002] | -0.364 [-0.426, -0.301] | -0.249 [-0.288, -0.208] | 0.000 | 0.000 | 0.000 | train |
| single_shot | none | logistic_regression | 995 | 0.787 [0.766, 0.809] | 0.477 [0.440, 0.516] | 0.630 [0.591, 0.674] | 0.000 | 0.000 | 0.000 | train |
| single_shot | none | feature:log_length | 995 | 0.791 [0.769, 0.812] | 0.537 [0.499, 0.575] | 0.685 [0.646, 0.726] | 0.000 | 0.000 | 0.000 |  |
| single_shot | none | all_hallucinated | 995 | 0.787 [0.766, 0.809] |  |  | 0.000 |  |  |  |
| single_shot | none | logistic_regression - feature:log_length | 995 | -0.003 [-0.008, +0.002] | -0.060 [-0.131, +0.012] | -0.055 [-0.104, -0.003] | 0.000 | 0.000 | 0.000 | train |
| single_shot | gold | hist_gradient_boosting | 995 | 0.787 [0.766, 0.809] | 0.314 [0.276, 0.352] | 0.521 [0.489, 0.558] | 0.000 | 0.000 | 0.000 | train |
| single_shot | gold | feature:lex_overlap_max | 995 | 0.839 [0.818, 0.859] | 0.827 [0.799, 0.853] | 0.877 [0.850, 0.902] | 0.000 | 0.000 | 0.000 |  |
| single_shot | gold | all_hallucinated | 995 | 0.787 [0.766, 0.809] |  |  | 0.000 |  |  |  |
| single_shot | gold | hist_gradient_boosting - feature:lex_overlap_max | 995 | -0.052 [-0.068, -0.035] | -0.513 [-0.571, -0.455] | -0.356 [-0.390, -0.317] | 0.000 | 0.000 | 0.000 | train |
| resampling | retrieved | logistic_regression | 995 | 0.788 [0.766, 0.809] | 0.461 [0.424, 0.500] | 0.613 [0.576, 0.657] | 0.000 | 0.000 | 0.000 | train |
| resampling | retrieved | feature:lex_overlap_max | 995 | 0.806 [0.783, 0.827] | 0.725 [0.692, 0.757] | 0.795 [0.759, 0.829] | 0.000 | 0.000 | 0.000 |  |
| resampling | retrieved | all_hallucinated | 995 | 0.787 [0.766, 0.809] |  |  | 0.000 |  |  |  |
| resampling | retrieved | logistic_regression - feature:lex_overlap_max | 995 | -0.018 [-0.034, -0.002] | -0.264 [-0.326, -0.199] | -0.182 [-0.225, -0.133] | 0.000 | 0.000 | 0.000 | train |
| resampling | none | logistic_regression | 995 | 0.788 [0.766, 0.809] | 0.577 [0.541, 0.614] | 0.720 [0.681, 0.760] | 0.000 | 0.000 | 0.000 | train |
| resampling | none | feature:log_length | 995 | 0.791 [0.769, 0.812] | 0.537 [0.499, 0.575] | 0.685 [0.646, 0.726] | 0.000 | 0.000 | 0.000 |  |
| resampling | none | all_hallucinated | 995 | 0.787 [0.766, 0.809] |  |  | 0.000 |  |  |  |
| resampling | none | logistic_regression - feature:log_length | 995 | -0.003 [-0.009, +0.003] | +0.040 [-0.028, +0.109] | +0.034 [-0.016, +0.086] | 0.000 | 0.000 | 0.000 | train |
| resampling | gold | logistic_regression | 995 | 0.788 [0.766, 0.809] | 0.405 [0.367, 0.444] | 0.570 [0.535, 0.611] | 0.000 | 0.000 | 0.000 | train |
| resampling | gold | feature:lex_overlap_max | 995 | 0.839 [0.818, 0.859] | 0.827 [0.799, 0.853] | 0.877 [0.850, 0.902] | 0.000 | 0.000 | 0.000 |  |
| resampling | gold | all_hallucinated | 995 | 0.787 [0.766, 0.809] |  |  | 0.000 |  |  |  |
| resampling | gold | logistic_regression - feature:lex_overlap_max | 995 | -0.051 [-0.068, -0.035] | -0.422 [-0.479, -0.362] | -0.307 [-0.345, -0.263] | 0.000 | 0.000 | 0.000 | train |

Selected classifiers beating the best single feature on `val_id` (lower bound of the F1 difference
interval above 0): 0 of 6 tier and evidence-condition rows. The paired comparison with
predicting every answer hallucinated is in `reports/step7/baseline_check.csv`. These are validation
results, not research-question verdicts.

Per-seed validation metrics (both families):

| tier | eval_condition | family | seed | config | threshold | f1 | precision | recall | auroc | auprc |
|---|---|---|---|---|---|---|---|---|---|---|
| resampling | gold | hist_gradient_boosting | 20261008 | max_depth=2,learning_rate=0.1,max_iter=100 | 0.062 | 0.787 | 0.649 | 1.000 | 0.392 | 0.563 |
| resampling | gold | hist_gradient_boosting | 20261009 | max_depth=2,learning_rate=0.05,max_iter=100 | 0.126 | 0.787 | 0.649 | 1.000 | 0.353 | 0.541 |
| resampling | gold | hist_gradient_boosting | 20261010 | max_depth=3,learning_rate=0.05,max_iter=100 | 0.067 | 0.787 | 0.649 | 1.000 | 0.350 | 0.538 |
| resampling | gold | logistic_regression | 20261008 | C=0.01,class_weight=None | 0.274 | 0.788 | 0.650 | 1.000 | 0.398 | 0.566 |
| resampling | gold | logistic_regression | 20261009 | C=0.01,class_weight=balanced | 0.381 | 0.788 | 0.650 | 1.000 | 0.417 | 0.579 |
| resampling | gold | logistic_regression | 20261010 | C=0.01,class_weight=None | 0.274 | 0.788 | 0.650 | 1.000 | 0.398 | 0.566 |
| resampling | none | hist_gradient_boosting | 20261008 | max_depth=2,learning_rate=0.05,max_iter=300 | 0.072 | 0.787 | 0.649 | 1.000 | 0.649 | 0.786 |
| resampling | none | hist_gradient_boosting | 20261009 | max_depth=2,learning_rate=0.05,max_iter=300 | 0.072 | 0.787 | 0.649 | 1.000 | 0.649 | 0.786 |
| resampling | none | hist_gradient_boosting | 20261010 | max_depth=2,learning_rate=0.1,max_iter=100 | 0.101 | 0.787 | 0.649 | 1.000 | 0.636 | 0.781 |
| resampling | none | logistic_regression | 20261008 | C=0.01,class_weight=balanced | 0.406 | 0.788 | 0.651 | 0.995 | 0.576 | 0.719 |
| resampling | none | logistic_regression | 20261009 | C=0.01,class_weight=None | 0.295 | 0.788 | 0.651 | 0.995 | 0.579 | 0.721 |
| resampling | none | logistic_regression | 20261010 | C=0.01,class_weight=balanced | 0.406 | 0.788 | 0.651 | 0.995 | 0.576 | 0.719 |
| resampling | retrieved | hist_gradient_boosting | 20261008 | max_depth=2,learning_rate=0.1,max_iter=100 | 0.066 | 0.787 | 0.649 | 1.000 | 0.442 | 0.603 |
| resampling | retrieved | hist_gradient_boosting | 20261009 | max_depth=2,learning_rate=0.05,max_iter=100 | 0.157 | 0.787 | 0.649 | 1.000 | 0.414 | 0.581 |
| resampling | retrieved | hist_gradient_boosting | 20261010 | max_depth=3,learning_rate=0.05,max_iter=100 | 0.085 | 0.787 | 0.649 | 1.000 | 0.413 | 0.577 |
| resampling | retrieved | logistic_regression | 20261008 | C=0.01,class_weight=None | 0.273 | 0.788 | 0.650 | 1.000 | 0.458 | 0.610 |
| resampling | retrieved | logistic_regression | 20261009 | C=0.01,class_weight=balanced | 0.378 | 0.788 | 0.650 | 1.000 | 0.468 | 0.619 |
| resampling | retrieved | logistic_regression | 20261010 | C=0.01,class_weight=None | 0.273 | 0.788 | 0.650 | 1.000 | 0.458 | 0.610 |
| single_shot | gold | hist_gradient_boosting | 20261008 | max_depth=2,learning_rate=0.05,max_iter=100 | 0.118 | 0.787 | 0.649 | 1.000 | 0.313 | 0.519 |
| single_shot | gold | hist_gradient_boosting | 20261009 | max_depth=2,learning_rate=0.05,max_iter=100 | 0.118 | 0.787 | 0.649 | 1.000 | 0.313 | 0.519 |
| single_shot | gold | hist_gradient_boosting | 20261010 | max_depth=3,learning_rate=0.05,max_iter=100 | 0.104 | 0.787 | 0.649 | 1.000 | 0.317 | 0.525 |
| single_shot | gold | logistic_regression | 20261008 | C=0.01,class_weight=None | 0.254 | 0.787 | 0.649 | 1.000 | 0.315 | 0.521 |
| single_shot | gold | logistic_regression | 20261009 | C=0.01,class_weight=None | 0.254 | 0.787 | 0.649 | 1.000 | 0.315 | 0.521 |
| single_shot | gold | logistic_regression | 20261010 | C=0.01,class_weight=None | 0.254 | 0.787 | 0.649 | 1.000 | 0.315 | 0.521 |
| single_shot | none | hist_gradient_boosting | 20261008 | max_depth=3,learning_rate=0.05,max_iter=100 | 0.163 | 0.787 | 0.649 | 1.000 | 0.508 | 0.672 |
| single_shot | none | hist_gradient_boosting | 20261009 | max_depth=3,learning_rate=0.05,max_iter=100 | 0.163 | 0.787 | 0.649 | 1.000 | 0.508 | 0.672 |
| single_shot | none | hist_gradient_boosting | 20261010 | max_depth=3,learning_rate=0.1,max_iter=100 | 0.109 | 0.787 | 0.649 | 1.000 | 0.515 | 0.676 |
| single_shot | none | logistic_regression | 20261008 | C=0.01,class_weight=balanced | 0.375 | 0.787 | 0.649 | 1.000 | 0.477 | 0.630 |
| single_shot | none | logistic_regression | 20261009 | C=0.01,class_weight=None | 0.269 | 0.787 | 0.649 | 1.000 | 0.477 | 0.630 |
| single_shot | none | logistic_regression | 20261010 | C=0.01,class_weight=balanced | 0.375 | 0.787 | 0.649 | 1.000 | 0.477 | 0.630 |
| single_shot | retrieved | hist_gradient_boosting | 20261008 | max_depth=2,learning_rate=0.05,max_iter=100 | 0.201 | 0.789 | 0.651 | 1.000 | 0.358 | 0.543 |
| single_shot | retrieved | hist_gradient_boosting | 20261009 | max_depth=2,learning_rate=0.05,max_iter=100 | 0.201 | 0.789 | 0.651 | 1.000 | 0.358 | 0.543 |
| single_shot | retrieved | hist_gradient_boosting | 20261010 | max_depth=3,learning_rate=0.05,max_iter=100 | 0.104 | 0.787 | 0.649 | 1.000 | 0.368 | 0.552 |
| single_shot | retrieved | logistic_regression | 20261008 | C=0.01,class_weight=None | 0.260 | 0.787 | 0.649 | 1.000 | 0.365 | 0.550 |
| single_shot | retrieved | logistic_regression | 20261009 | C=0.01,class_weight=None | 0.260 | 0.787 | 0.649 | 1.000 | 0.365 | 0.550 |
| single_shot | retrieved | logistic_regression | 20261010 | C=0.01,class_weight=None | 0.260 | 0.787 | 0.649 | 1.000 | 0.365 | 0.550 |

## Single features alone (direction and threshold chosen on `val_id`)

| eval_condition | feature | direction | f1 | auroc | auprc | best_in_tier_condition |
|---|---|---|---|---|---|---|
| gold | lex_overlap_max | low | 0.839 | 0.827 | 0.877 | yes |
| gold | nli_entail_max | low | 0.820 | 0.765 | 0.855 |  |
| gold | nli_contra_max | high | 0.806 | 0.750 | 0.837 |  |
| gold | sim_max | low | 0.801 | 0.732 | 0.828 |  |
| gold | log_length | high | 0.791 | 0.537 | 0.685 |  |
| gold | sc_entail | low | 0.789 | 0.729 | 0.840 |  |
| gold | hedge_rate | low | 0.788 | 0.471 | 0.636 |  |
| gold | self_contra | low | 0.788 | 0.490 | 0.643 |  |
| gold | sc_embed | low | 0.788 | 0.734 | 0.838 |  |
| gold | sc_contra | high | 0.788 | 0.730 | 0.849 |  |
| gold | booster_rate | high | 0.787 | 0.502 | 0.650 |  |
| gold | specificity | high | 0.787 | 0.492 | 0.648 |  |
| none | log_length | high | 0.791 | 0.537 | 0.685 | yes |
| none | sc_entail | low | 0.789 | 0.729 | 0.840 |  |
| none | hedge_rate | low | 0.788 | 0.471 | 0.636 |  |
| none | self_contra | low | 0.788 | 0.490 | 0.643 |  |
| none | sc_embed | low | 0.788 | 0.734 | 0.838 |  |
| none | sc_contra | high | 0.788 | 0.730 | 0.849 |  |
| none | booster_rate | high | 0.787 | 0.502 | 0.650 |  |
| none | specificity | high | 0.787 | 0.492 | 0.648 |  |
| retrieved | lex_overlap_max | low | 0.806 | 0.725 | 0.795 | yes |
| retrieved | nli_entail_max | low | 0.795 | 0.630 | 0.736 |  |
| retrieved | log_length | high | 0.791 | 0.537 | 0.685 |  |
| retrieved | sim_max | low | 0.790 | 0.655 | 0.760 |  |
| retrieved | sc_entail | low | 0.789 | 0.729 | 0.840 |  |
| retrieved | hedge_rate | low | 0.788 | 0.471 | 0.636 |  |
| retrieved | self_contra | low | 0.788 | 0.490 | 0.643 |  |
| retrieved | sc_embed | low | 0.788 | 0.734 | 0.838 |  |
| retrieved | sc_contra | high | 0.788 | 0.730 | 0.849 |  |
| retrieved | nli_contra_max | high | 0.787 | 0.603 | 0.727 |  |
| retrieved | booster_rate | high | 0.787 | 0.502 | 0.650 |  |
| retrieved | specificity | high | 0.787 | 0.492 | 0.648 |  |

The single-shot tier's best feature is the best of the rows above without `sc_*`.

## Ablations

All ablations use the selected family for each tier and training condition. Every subset gets its own
cross-validated hyperparameters per seed on `train`; values are means over seeds on `val_id`. Forward
selection adds, at each step, the feature with the highest mean `train` CV average precision; `val_id` never
chooses the order.

> **Exclusion flag.** All ablation results rest on `train`, which has more than 5% of its answers excluded as undecidable (`train` 9.2%); rows are marked in `flagged_label_splits`.

### Add one feature at a time from the bare baseline

| tier | train_condition | eval_condition | step | added | cv_ap_mean_mean | f1_mean | auroc_mean | auprc_mean |
|---|---|---|---|---|---|---|---|---|
| single_shot | retrieved | retrieved | 0 |  | 0.375 | 0.787 | 0.500 | 0.649 |
| single_shot | retrieved | gold | 0 |  | 0.375 | 0.787 | 0.500 | 0.649 |
| single_shot | retrieved | retrieved | 1 | log_length | 0.518 | 0.787 | 0.462 | 0.626 |
| single_shot | retrieved | gold | 1 | log_length | 0.518 | 0.787 | 0.462 | 0.626 |
| single_shot | retrieved | retrieved | 2 | specificity | 0.568 | 0.788 | 0.479 | 0.636 |
| single_shot | retrieved | gold | 2 | specificity | 0.568 | 0.788 | 0.479 | 0.636 |
| single_shot | retrieved | retrieved | 3 | self_contra | 0.579 | 0.788 | 0.496 | 0.653 |
| single_shot | retrieved | gold | 3 | self_contra | 0.579 | 0.788 | 0.496 | 0.653 |
| single_shot | retrieved | retrieved | 4 | booster_rate | 0.579 | 0.788 | 0.502 | 0.659 |
| single_shot | retrieved | gold | 4 | booster_rate | 0.579 | 0.788 | 0.502 | 0.659 |
| single_shot | retrieved | retrieved | 5 | hedge_rate | 0.580 | 0.787 | 0.510 | 0.673 |
| single_shot | retrieved | gold | 5 | hedge_rate | 0.580 | 0.787 | 0.510 | 0.673 |
| single_shot | retrieved | retrieved | 6 | lex_overlap_max | 0.576 | 0.787 | 0.408 | 0.571 |
| single_shot | retrieved | gold | 6 | lex_overlap_max | 0.576 | 0.787 | 0.345 | 0.536 |
| single_shot | retrieved | retrieved | 7 | nli_contra_max | 0.578 | 0.789 | 0.358 | 0.546 |
| single_shot | retrieved | gold | 7 | nli_contra_max | 0.578 | 0.788 | 0.311 | 0.521 |
| single_shot | retrieved | retrieved | 8 | nli_entail_max | 0.568 | 0.788 | 0.360 | 0.547 |
| single_shot | retrieved | gold | 8 | nli_entail_max | 0.568 | 0.787 | 0.325 | 0.527 |
| single_shot | retrieved | retrieved | 9 | sim_max | 0.552 | 0.788 | 0.361 | 0.546 |
| single_shot | retrieved | gold | 9 | sim_max | 0.552 | 0.787 | 0.314 | 0.521 |
| single_shot | none | none | 0 |  | 0.375 | 0.787 | 0.500 | 0.649 |
| single_shot | none | none | 1 | log_length | 0.532 | 0.787 | 0.463 | 0.622 |
| single_shot | none | none | 2 | self_contra | 0.543 | 0.787 | 0.466 | 0.621 |
| single_shot | none | none | 3 | booster_rate | 0.543 | 0.787 | 0.465 | 0.620 |
| single_shot | none | none | 4 | hedge_rate | 0.542 | 0.787 | 0.475 | 0.625 |
| single_shot | none | none | 5 | specificity | 0.539 | 0.787 | 0.477 | 0.630 |
| resampling | retrieved | retrieved | 0 |  | 0.375 | 0.787 | 0.500 | 0.649 |
| resampling | retrieved | gold | 0 |  | 0.375 | 0.787 | 0.500 | 0.649 |
| resampling | retrieved | retrieved | 1 | log_length | 0.532 | 0.787 | 0.463 | 0.622 |
| resampling | retrieved | gold | 1 | log_length | 0.532 | 0.787 | 0.463 | 0.622 |
| resampling | retrieved | retrieved | 2 | nli_contra_max | 0.544 | 0.787 | 0.436 | 0.600 |
| resampling | retrieved | gold | 2 | nli_contra_max | 0.544 | 0.787 | 0.385 | 0.568 |
| resampling | retrieved | retrieved | 3 | self_contra | 0.546 | 0.787 | 0.447 | 0.604 |
| resampling | retrieved | gold | 3 | self_contra | 0.546 | 0.787 | 0.409 | 0.577 |
| resampling | retrieved | retrieved | 4 | booster_rate | 0.545 | 0.787 | 0.446 | 0.604 |
| resampling | retrieved | gold | 4 | booster_rate | 0.545 | 0.787 | 0.408 | 0.576 |
| resampling | retrieved | retrieved | 5 | sim_max | 0.544 | 0.787 | 0.421 | 0.588 |
| resampling | retrieved | gold | 5 | sim_max | 0.544 | 0.787 | 0.375 | 0.559 |
| resampling | retrieved | retrieved | 6 | hedge_rate | 0.542 | 0.787 | 0.435 | 0.596 |
| resampling | retrieved | gold | 6 | hedge_rate | 0.542 | 0.787 | 0.393 | 0.570 |
| resampling | retrieved | retrieved | 7 | lex_overlap_max | 0.539 | 0.787 | 0.340 | 0.534 |
| resampling | retrieved | gold | 7 | lex_overlap_max | 0.539 | 0.787 | 0.273 | 0.503 |
| resampling | retrieved | retrieved | 8 | sc_embed | 0.538 | 0.787 | 0.398 | 0.562 |
| resampling | retrieved | gold | 8 | sc_embed | 0.538 | 0.787 | 0.317 | 0.519 |
| resampling | retrieved | retrieved | 9 | nli_entail_max | 0.537 | 0.787 | 0.408 | 0.569 |
| resampling | retrieved | gold | 9 | nli_entail_max | 0.537 | 0.787 | 0.348 | 0.534 |
| resampling | retrieved | retrieved | 10 | sc_contra | 0.532 | 0.788 | 0.475 | 0.624 |
| resampling | retrieved | gold | 10 | sc_contra | 0.532 | 0.787 | 0.417 | 0.580 |
| resampling | retrieved | retrieved | 11 | specificity | 0.528 | 0.788 | 0.467 | 0.618 |
| resampling | retrieved | gold | 11 | specificity | 0.528 | 0.788 | 0.407 | 0.572 |
| resampling | retrieved | retrieved | 12 | sc_entail | 0.523 | 0.788 | 0.461 | 0.613 |
| resampling | retrieved | gold | 12 | sc_entail | 0.523 | 0.788 | 0.405 | 0.570 |
| resampling | none | none | 0 |  | 0.375 | 0.787 | 0.500 | 0.649 |
| resampling | none | none | 1 | log_length | 0.532 | 0.787 | 0.463 | 0.622 |
| resampling | none | none | 2 | self_contra | 0.543 | 0.787 | 0.466 | 0.621 |
| resampling | none | none | 3 | booster_rate | 0.543 | 0.787 | 0.465 | 0.620 |
| resampling | none | none | 4 | hedge_rate | 0.542 | 0.787 | 0.475 | 0.625 |
| resampling | none | none | 5 | specificity | 0.539 | 0.787 | 0.477 | 0.630 |
| resampling | none | none | 6 | sc_contra | 0.534 | 0.787 | 0.552 | 0.698 |
| resampling | none | none | 7 | sc_embed | 0.532 | 0.787 | 0.588 | 0.729 |
| resampling | none | none | 8 | sc_entail | 0.527 | 0.788 | 0.577 | 0.720 |

### Leave one feature out (change from all features)

| tier | train_condition | eval_condition | removed | delta_cv_ap_mean_mean | delta_f1_mean | delta_auroc_mean | delta_auprc_mean |
|---|---|---|---|---|---|---|---|
| single_shot | retrieved | retrieved | sim_max | 0.016 | -0.000 | -0.001 | 0.002 |
| single_shot | retrieved | retrieved | nli_entail_max | 0.014 | 0.000 | -0.002 | -0.001 |
| single_shot | retrieved | retrieved | nli_contra_max | 0.003 | -0.001 | 0.035 | 0.016 |
| single_shot | retrieved | retrieved | lex_overlap_max | -0.004 | -0.001 | 0.083 | 0.067 |
| single_shot | retrieved | retrieved | hedge_rate | -0.002 | -0.001 | -0.009 | -0.006 |
| single_shot | retrieved | retrieved | booster_rate | 0.001 | 0.000 | 0.001 | 0.000 |
| single_shot | retrieved | retrieved | log_length | -0.016 | -0.000 | 0.012 | 0.004 |
| single_shot | retrieved | retrieved | specificity | -0.019 | -0.001 | 0.003 | 0.002 |
| single_shot | retrieved | retrieved | self_contra | 0.000 | 0.000 | -0.024 | -0.006 |
| single_shot | retrieved | gold | sim_max | 0.016 | 0.000 | 0.011 | 0.006 |
| single_shot | retrieved | gold | nli_entail_max | 0.014 | 0.000 | -0.008 | -0.004 |
| single_shot | retrieved | gold | nli_contra_max | 0.003 | 0.001 | 0.010 | 0.006 |
| single_shot | retrieved | gold | lex_overlap_max | -0.004 | 0.000 | 0.126 | 0.091 |
| single_shot | retrieved | gold | hedge_rate | -0.002 | 0.000 | -0.010 | -0.005 |
| single_shot | retrieved | gold | booster_rate | 0.001 | 0.000 | -0.004 | -0.002 |
| single_shot | retrieved | gold | log_length | -0.016 | 0.001 | 0.034 | 0.014 |
| single_shot | retrieved | gold | specificity | -0.019 | 0.000 | 0.000 | -0.001 |
| single_shot | retrieved | gold | self_contra | 0.000 | 0.000 | -0.029 | -0.010 |
| single_shot | none | none | hedge_rate | -0.004 | 0.000 | -0.008 | -0.005 |
| single_shot | none | none | booster_rate | 0.001 | 0.000 | 0.001 | 0.000 |
| single_shot | none | none | log_length | -0.029 | 0.001 | 0.016 | 0.014 |
| single_shot | none | none | specificity | 0.002 | 0.000 | -0.002 | -0.005 |
| single_shot | none | none | self_contra | -0.002 | 0.000 | 0.003 | 0.004 |
| resampling | retrieved | retrieved | sim_max | 0.001 | 0.000 | -0.002 | -0.002 |
| resampling | retrieved | retrieved | nli_entail_max | 0.003 | -0.000 | -0.011 | -0.011 |
| resampling | retrieved | retrieved | nli_contra_max | 0.000 | 0.000 | 0.005 | 0.003 |
| resampling | retrieved | retrieved | lex_overlap_max | -0.006 | -0.000 | 0.107 | 0.099 |
| resampling | retrieved | retrieved | hedge_rate | -0.003 | 0.000 | -0.008 | -0.003 |
| resampling | retrieved | retrieved | booster_rate | 0.002 | 0.000 | -0.006 | -0.005 |
| resampling | retrieved | retrieved | log_length | -0.018 | 0.001 | -0.023 | -0.016 |
| resampling | retrieved | retrieved | specificity | 0.003 | 0.000 | 0.012 | 0.008 |
| resampling | retrieved | retrieved | self_contra | -0.006 | 0.000 | -0.017 | -0.009 |
| resampling | retrieved | retrieved | sc_embed | -0.000 | -0.000 | -0.040 | -0.029 |
| resampling | retrieved | retrieved | sc_entail | 0.004 | 0.000 | 0.006 | 0.005 |
| resampling | retrieved | retrieved | sc_contra | 0.003 | 0.000 | -0.047 | -0.037 |
| resampling | retrieved | gold | sim_max | 0.001 | 0.000 | -0.002 | -0.003 |
| resampling | retrieved | gold | nli_entail_max | 0.003 | -0.000 | -0.035 | -0.022 |
| resampling | retrieved | gold | nli_contra_max | 0.000 | 0.000 | 0.016 | 0.012 |
| resampling | retrieved | gold | lex_overlap_max | -0.006 | -0.000 | 0.160 | 0.138 |
| resampling | retrieved | gold | hedge_rate | -0.003 | -0.000 | -0.013 | -0.007 |
| resampling | retrieved | gold | booster_rate | 0.002 | 0.000 | -0.011 | -0.008 |
| resampling | retrieved | gold | log_length | -0.018 | 0.000 | -0.024 | -0.014 |
| resampling | retrieved | gold | specificity | 0.003 | -0.000 | 0.015 | 0.011 |
| resampling | retrieved | gold | self_contra | -0.006 | -0.000 | -0.034 | -0.017 |
| resampling | retrieved | gold | sc_embed | -0.000 | -0.000 | -0.039 | -0.024 |
| resampling | retrieved | gold | sc_entail | 0.004 | 0.000 | 0.002 | 0.002 |
| resampling | retrieved | gold | sc_contra | 0.003 | -0.000 | -0.045 | -0.029 |
| resampling | none | none | hedge_rate | -0.003 | -0.000 | 0.003 | 0.010 |
| resampling | none | none | booster_rate | 0.001 | 0.000 | -0.004 | -0.004 |
| resampling | none | none | log_length | -0.026 | 0.003 | -0.003 | -0.001 |
| resampling | none | none | specificity | 0.002 | -0.000 | 0.032 | 0.033 |
| resampling | none | none | self_contra | -0.005 | 0.000 | 0.003 | -0.002 |
| resampling | none | none | sc_embed | 0.002 | -0.000 | -0.034 | -0.030 |
| resampling | none | none | sc_entail | 0.005 | -0.000 | 0.011 | 0.010 |
| resampling | none | none | sc_contra | 0.002 | -0.000 | -0.043 | -0.042 |

### Leave one group out, and each group alone (change from all features)

Where a tier and condition has only two feature groups (the single-shot tier, and the resampling tier
without evidence), dropping one group leaves exactly the other, so "-A" and "only:B" fit the same
features and their rows are identical by construction. `same_features_as` names the twin row.

| tier | train_condition | eval_condition | subset | delta_cv_ap_mean_mean | delta_f1_mean | delta_auroc_mean | delta_auprc_mean | same_features_as |
|---|---|---|---|---|---|---|---|---|
| single_shot | retrieved | retrieved | -evidence_dependent | 0.028 | -0.001 | 0.149 | 0.127 | only:evidence_free |
| single_shot | retrieved | retrieved | -evidence_free | -0.087 | -0.000 | -0.011 | 0.000 | only:evidence_dependent |
| single_shot | retrieved | retrieved | only:evidence_dependent | -0.087 | -0.000 | -0.011 | 0.000 | -evidence_free |
| single_shot | retrieved | retrieved | only:evidence_free | 0.028 | -0.001 | 0.149 | 0.127 | -evidence_dependent |
| single_shot | retrieved | gold | -evidence_dependent | 0.028 | 0.000 | 0.196 | 0.152 | only:evidence_free |
| single_shot | retrieved | gold | -evidence_free | -0.087 | 0.000 | 0.011 | 0.013 | only:evidence_dependent |
| single_shot | retrieved | gold | only:evidence_dependent | -0.087 | 0.000 | 0.011 | 0.013 | -evidence_free |
| single_shot | retrieved | gold | only:evidence_free | 0.028 | 0.000 | 0.196 | 0.152 | -evidence_dependent |
| resampling | retrieved | retrieved | -evidence_dependent | 0.003 | -0.000 | 0.116 | 0.106 |  |
| resampling | retrieved | retrieved | -evidence_free | -0.048 | 0.000 | -0.080 | -0.042 |  |
| resampling | retrieved | retrieved | -self_consistency | 0.008 | -0.000 | -0.096 | -0.063 |  |
| resampling | retrieved | retrieved | only:evidence_dependent | -0.056 | 0.000 | -0.172 | -0.084 |  |
| resampling | retrieved | retrieved | only:evidence_free | 0.016 | -0.000 | 0.016 | 0.017 |  |
| resampling | retrieved | retrieved | only:self_consistency | -0.080 | 0.000 | 0.193 | 0.187 |  |
| resampling | retrieved | gold | -evidence_dependent | 0.003 | -0.000 | 0.172 | 0.150 |  |
| resampling | retrieved | gold | -evidence_free | -0.048 | 0.000 | -0.112 | -0.053 |  |
| resampling | retrieved | gold | -self_consistency | 0.008 | -0.000 | -0.090 | -0.049 |  |
| resampling | retrieved | gold | only:evidence_dependent | -0.056 | 0.000 | -0.202 | -0.087 |  |
| resampling | retrieved | gold | only:evidence_free | 0.016 | -0.000 | 0.072 | 0.060 |  |
| resampling | retrieved | gold | only:self_consistency | -0.080 | 0.000 | 0.249 | 0.230 |  |
| resampling | none | none | -evidence_free | -0.083 | 0.001 | 0.076 | 0.081 | only:self_consistency |
| resampling | none | none | -self_consistency | 0.013 | -0.000 | -0.100 | -0.089 | only:evidence_free |
| resampling | none | none | only:evidence_free | 0.013 | -0.000 | -0.100 | -0.089 | -self_consistency |
| resampling | none | none | only:self_consistency | -0.083 | 0.001 | 0.076 | 0.081 | -evidence_free |

## Permutation importance and stability across seeds

Drop in `val_id` average precision when a feature is permuted (30
repeats), for each seed's selected classifier.

> **Exclusion flag.** All importance results rest on `train`, which has more than 5% of its answers excluded as undecidable (`train` 9.2%); rows are marked in `flagged_label_splits`.

| tier | eval_condition | feature | importance_mean_over_seeds | importance_sd_over_seeds | rank_min | rank_max |
|---|---|---|---|---|---|---|
| single_shot | retrieved | booster_rate | -0.000 | 0.000 | 4.000 | 5.000 |
| single_shot | retrieved | hedge_rate | 0.004 | 0.001 | 1.000 | 2.000 |
| single_shot | retrieved | lex_overlap_max | -0.092 | 0.002 | 9.000 | 9.000 |
| single_shot | retrieved | log_length | -0.000 | 0.001 | 4.000 | 6.000 |
| single_shot | retrieved | nli_contra_max | -0.014 | 0.003 | 8.000 | 8.000 |
| single_shot | retrieved | nli_entail_max | 0.003 | 0.003 | 1.000 | 2.000 |
| single_shot | retrieved | self_contra | -0.000 | 0.001 | 3.000 | 6.000 |
| single_shot | retrieved | sim_max | -0.002 | 0.000 | 7.000 | 7.000 |
| single_shot | retrieved | specificity | 0.001 | 0.001 | 3.000 | 4.000 |
| single_shot | gold | booster_rate | -0.000 | 0.000 | 4.000 | 5.000 |
| single_shot | gold | hedge_rate | 0.003 | 0.002 | 2.000 | 2.000 |
| single_shot | gold | lex_overlap_max | -0.127 | 0.008 | 9.000 | 9.000 |
| single_shot | gold | log_length | -0.000 | 0.000 | 5.000 | 6.000 |
| single_shot | gold | nli_contra_max | -0.008 | 0.000 | 8.000 | 8.000 |
| single_shot | gold | nli_entail_max | 0.006 | 0.001 | 1.000 | 1.000 |
| single_shot | gold | self_contra | -0.000 | 0.003 | 3.000 | 6.000 |
| single_shot | gold | sim_max | -0.004 | 0.000 | 7.000 | 7.000 |
| single_shot | gold | specificity | 0.001 | 0.001 | 3.000 | 4.000 |
| single_shot | none | booster_rate | -0.000 | 0.000 | 2.000 | 2.000 |
| single_shot | none | hedge_rate | 0.004 | 0.000 | 1.000 | 1.000 |
| single_shot | none | log_length | -0.016 | 0.003 | 5.000 | 5.000 |
| single_shot | none | self_contra | -0.006 | 0.001 | 4.000 | 4.000 |
| single_shot | none | specificity | -0.003 | 0.001 | 3.000 | 3.000 |
| resampling | retrieved | booster_rate | -0.001 | 0.000 | 6.000 | 6.000 |
| resampling | retrieved | hedge_rate | 0.001 | 0.000 | 4.000 | 4.000 |
| resampling | retrieved | lex_overlap_max | -0.106 | 0.003 | 12.000 | 12.000 |
| resampling | retrieved | log_length | -0.013 | 0.002 | 10.000 | 10.000 |
| resampling | retrieved | nli_contra_max | -0.005 | 0.000 | 7.000 | 7.000 |
| resampling | retrieved | nli_entail_max | 0.013 | 0.002 | 3.000 | 3.000 |
| resampling | retrieved | sc_contra | 0.038 | 0.003 | 2.000 | 2.000 |
| resampling | retrieved | sc_embed | 0.041 | 0.002 | 1.000 | 1.000 |
| resampling | retrieved | sc_entail | -0.016 | 0.001 | 11.000 | 11.000 |
| resampling | retrieved | self_contra | -0.009 | 0.001 | 9.000 | 9.000 |
| resampling | retrieved | sim_max | -0.006 | 0.001 | 8.000 | 8.000 |
| resampling | retrieved | specificity | 0.000 | 0.001 | 5.000 | 5.000 |
| resampling | gold | booster_rate | -0.000 | 0.000 | 6.000 | 6.000 |
| resampling | gold | hedge_rate | 0.002 | 0.000 | 4.000 | 4.000 |
| resampling | gold | lex_overlap_max | -0.155 | 0.003 | 12.000 | 12.000 |
| resampling | gold | log_length | -0.012 | 0.002 | 9.000 | 9.000 |
| resampling | gold | nli_contra_max | -0.015 | 0.001 | 10.000 | 10.000 |
| resampling | gold | nli_entail_max | 0.023 | 0.004 | 3.000 | 3.000 |
| resampling | gold | sc_contra | 0.030 | 0.004 | 2.000 | 2.000 |
| resampling | gold | sc_embed | 0.031 | 0.003 | 1.000 | 1.000 |
| resampling | gold | sc_entail | -0.015 | 0.002 | 11.000 | 11.000 |
| resampling | gold | self_contra | -0.006 | 0.001 | 7.000 | 7.000 |
| resampling | gold | sim_max | -0.008 | 0.001 | 8.000 | 8.000 |
| resampling | gold | specificity | 0.001 | 0.000 | 5.000 | 5.000 |
| resampling | none | booster_rate | -0.001 | 0.000 | 4.000 | 4.000 |
| resampling | none | hedge_rate | -0.000 | 0.000 | 3.000 | 3.000 |
| resampling | none | log_length | -0.023 | 0.003 | 8.000 | 8.000 |
| resampling | none | sc_contra | 0.049 | 0.001 | 2.000 | 2.000 |
| resampling | none | sc_embed | 0.058 | 0.004 | 1.000 | 1.000 |
| resampling | none | sc_entail | -0.017 | 0.000 | 7.000 | 7.000 |
| resampling | none | self_contra | -0.008 | 0.001 | 5.000 | 5.000 |
| resampling | none | specificity | -0.013 | 0.001 | 6.000 | 6.000 |

| tier | eval_condition | spearman_rho_min | spearman_rho_mean | top3_jaccard_min | top3_jaccard_mean |
|---|---|---|---|---|---|
| single_shot | retrieved | 0.867 | 0.917 | 0.500 | 0.667 |
| single_shot | gold | 0.900 | 0.933 | 0.500 | 0.667 |
| single_shot | none | 1.000 | 1.000 | 1.000 | 1.000 |
| resampling | retrieved | 1.000 | 1.000 | 1.000 | 1.000 |
| resampling | gold | 1.000 | 1.000 | 1.000 | 1.000 |
| resampling | none | 1.000 | 1.000 | 1.000 | 1.000 |

## Redundancy

> **Exclusion flag.** Rows for `train` here and in the missingness and `sc_entail` tables below rest on `train`, which has more than 5% of its answers excluded as undecidable (`train` 9.2%); rows are marked in `flagged_label_splits`.

Feature pairs with |Spearman r| >= 0.7 (full matrices in `correlations.csv`):

| split | condition | feature_a | feature_b | r |
|---|---|---|---|---|
| val_id | retrieved | sc_embed | sc_entail | 0.745 |
| val_id | retrieved | sc_entail | sc_contra | -0.778 |
| val_id | none | sc_embed | sc_entail | 0.745 |
| val_id | none | sc_entail | sc_contra | -0.778 |
| val_id | gold | nli_entail_max | nli_contra_max | -0.729 |
| val_id | gold | sc_embed | sc_entail | 0.745 |
| val_id | gold | sc_entail | sc_contra | -0.778 |

VIF >= 5 (all values in `vif.csv`):

None.

## Missingness and data checks

Missing or non-finite feature values: 0 and 0 (features-v1 never imputes).

| split | condition | rows | excluded_undecidable | no_usable_resamples | truncated_primary | abstained |
|---|---|---|---|---|---|---|
| train | retrieved | 522 | 53.000 | 0.000 | 0.000 | 4.000 |
| train | none | 522 | 53.000 | 0.000 | 0.000 | 4.000 |
| val_id | retrieved | 995 | 5.000 | 0.000 | 3.000 | 0.000 |
| val_id | none | 995 | 5.000 | 0.000 | 3.000 | 0.000 |
| val_id | gold | 995 | 5.000 | 0.000 | 3.000 | 0.000 |

| check | ok |
|---|---|
| generation files match manifest | yes |
| no cluster spans two splits | yes |
| feature rows align with manifest questions, labels, and raw-answer hashes | yes |
| train and val_id share no cluster | yes |

## Note on `sc_entail`

`sc_entail` is kept in the resampling tier exactly as frozen. The pinned NLI model labels near-paraphrased
resamples as neutral, so the feature sits close to zero:

| split | label | n | median | p95 | max | share_below_0.01 |
|---|---|---|---|---|---|---|
| train | all | 522 | 0.417 | 0.993 | 0.998 | 0.140 |
| train | hallucinated | 196 | 0.447 | 0.993 | 0.996 | 0.148 |
| train | not_hallucinated | 326 | 0.398 | 0.993 | 0.998 | 0.135 |
| val_id | all | 995 | 0.590 | 0.994 | 0.998 | 0.147 |
| val_id | hallucinated | 646 | 0.395 | 0.993 | 0.998 | 0.206 |
| val_id | not_hallucinated | 349 | 0.945 | 0.994 | 0.998 | 0.037 |

Whether it carries signal is left to the pre-registered analyses: removing it (leave one feature out),
its permutation importance, and its value alone:

| train_condition | eval_condition | delta_cv_ap_mean_mean | delta_f1_mean | delta_auroc_mean | delta_auprc_mean |
|---|---|---|---|---|---|
| retrieved | retrieved | 0.004 | 0.000 | 0.006 | 0.005 |
| retrieved | gold | 0.004 | 0.000 | 0.002 | 0.002 |
| none | none | 0.005 | -0.000 | 0.011 | 0.010 |

| eval_condition | importance_mean_over_seeds | importance_sd_over_seeds | rank_min | rank_max |
|---|---|---|---|---|
| retrieved | -0.016 | 0.001 | 11.000 | 11.000 |
| gold | -0.015 | 0.002 | 11.000 | 11.000 |
| none | -0.017 | 0.000 | 7.000 | 7.000 |

| eval_condition | direction | f1 | auroc | auprc |
|---|---|---|---|---|
| retrieved | low | 0.789 | 0.729 | 0.840 |
| none | low | 0.789 | 0.729 | 0.840 |
| gold | low | 0.789 | 0.729 | 0.840 |

## Retrieval recall

BM25 top-5 over Wiki-DPR. `val_id` (HaluEval) knowledge-sentence recall@5 = 0.562 (n = 1000). `train` (TruthfulQA) has no gold evidence, so its recall is not measurable. A retrieved-evidence feature can only help where retrieval found the relevant passage; low recall caps the evidence group, and this is a retrieval limitation rather than a classifier one.

## Dataset and evidence-condition effects

`train` is TruthfulQA only and `val_id` is HaluEval only, so every validation number is already a cross-dataset result (TruthfulQA to HaluEval, same generator) and per-dataset results equal per-split results; there is nothing to pool. `train` CV average precision is TruthfulQA in-distribution. Evidence-condition effects are compared within `val_id` (same answers, same labels): retrieved versus no evidence, and gold as an exploratory oracle. Gold-evidence results are not realistic retrieval performance: HaluEval's gold passage contains the reference answer, which the labeling rule also uses.

## Runtime

| stage | seconds |
|---|---|
| cv single_shot/retrieved | 1.688 |
| refit single_shot/retrieved | 0.082 |
| cv single_shot/none | 0.596 |
| refit single_shot/none | 0.053 |
| cv resampling/retrieved | 0.710 |
| refit resampling/retrieved | 0.069 |
| cv resampling/none | 0.659 |
| refit resampling/none | 0.077 |
| selection | 3.953 |
| single features | 0.046 |
| bootstrap single_shot/retrieved | 1.503 |
| bootstrap single_shot/none | 1.090 |
| bootstrap single_shot/gold | 1.491 |
| bootstrap resampling/retrieved | 1.796 |
| bootstrap resampling/none | 1.427 |
| bootstrap resampling/gold | 1.793 |
| validation summary | 9.182 |
| ablations single_shot/retrieved | 27.283 |
| ablations single_shot/none | 2.607 |
| ablations resampling/retrieved | 8.740 |
| ablations resampling/none | 4.872 |
| ablations | 43.507 |
| permutation importance | 3.344 |
| redundancy | 0.008 |
| distributions | 0.033 |

| stage | median_ms | p95_ms | n |
|---|---|---|---|
| single_shot/retrieved/hist_gradient_boosting/seed20261008 | 0.208 | 0.221 | 200.000 |
| single_shot/none/logistic_regression/seed20261008 | 0.068 | 0.073 | 200.000 |
| resampling/retrieved/logistic_regression/seed20261008 | 0.069 | 0.078 | 200.000 |
| resampling/none/logistic_regression/seed20261008 | 0.068 | 0.076 | 200.000 |

Hardware: macOS-26.6.2-arm64-arm-64bit arm64 cpus=10 memory_gb=17.2 sklearn=1.9.1 n_jobs=8

Mean feature computation time per answer (Step 5 reports) — train: hedge 0.1 ms, length 0.0 ms, lex 0.2 ms, nli 180.3 ms, sc_embed 23.1 ms, sc_nli 125.7 ms, self_contra 32.4 ms, sim 43.7 ms; val_id: hedge 0.1 ms, length 0.0 ms, lex 0.2 ms, nli 174.5 ms, sc_embed 21.0 ms, sc_nli 110.6 ms, self_contra 48.4 ms, sim 43.8 ms

## Data access

13 files read; test-split files read: 0. Full list in `data_access.csv`.

