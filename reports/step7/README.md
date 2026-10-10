# Step 7: calibration and decision policy (val_id only)

Every number here comes from `train` (refitting the Step 6 classifiers) or `val_id` (calibration and the
policy). No test split was read (see `data_access.csv`). These are not test results.

**Label limitation.** The HaluEval labels failed the hand-audit agreement gate (Cohen's kappa 0.7117 in the first round and 0.578 in the fresh round under rule version 2, against a
minimum of 0.6). Calibration, the policy thresholds, and every rate below rest on those `val_id` labels
(amendment 2026-10-09).

**Exclusion flag.** Every classifier here was fitted on `train` (9.2% of answers excluded), above the 5% limit; every
table carries `flagged_label_splits`.

## Summary (validation only; no research-question criterion is judged here)

- **Baselines.** 0 of 4 selected classifiers beat predicting every answer hallucinated on `val_id` (lower bound of the paired 95% F1 difference interval
  above 0), and 0 of 4 beat the best single feature.
- **Policies.** 7 of 12 seed-level policies are infeasible (no accept band meets the 5%
  goal); their escalate and flag split is shown for diagnosis only, and their constraints are not met. The
  5 feasible ones are nearly degenerate: at most 3 of 995 answers accepted. Flag
  precision is 0.651–0.662, about the `val_id` prevalence of 0.649.
- **Seeds.** No deployment seed is selected; all three seeds keep their own calibrator and policy, and
  seed-level variation is reported throughout.
- **Gold evidence.** No calibration or operating-point policy is defined for the gold-evidence condition.
- **Denominators.** Rates use the 995 labeled `val_id` answers; 5 answers the hand labeler
  marked undecidable are excluded.
- Implementation choices are recorded in docs/PREREGISTRATION.md (amendment 2026-10-09, Step 7
  implementation details).

## Refit check

Each selected classifier was refitted from its Step 6 family, hyperparameters, and seed; all reproduce
their Step 6 F1 thresholds:

| tier | condition | family | seed | config | step6_threshold | refit_threshold | reproduced |
|---|---|---|---|---|---|---|---|
| single_shot | retrieved | hist_gradient_boosting | 20261008 | max_depth=2,learning_rate=0.05,max_iter=100 | 0.201 | 0.201 | yes |
| single_shot | retrieved | hist_gradient_boosting | 20261009 | max_depth=2,learning_rate=0.05,max_iter=100 | 0.201 | 0.201 | yes |
| single_shot | retrieved | hist_gradient_boosting | 20261010 | max_depth=3,learning_rate=0.05,max_iter=100 | 0.104 | 0.104 | yes |
| single_shot | none | logistic_regression | 20261008 | C=0.01,class_weight=balanced | 0.375 | 0.375 | yes |
| single_shot | none | logistic_regression | 20261009 | C=0.01,class_weight=None | 0.269 | 0.269 | yes |
| single_shot | none | logistic_regression | 20261010 | C=0.01,class_weight=balanced | 0.375 | 0.375 | yes |
| resampling | retrieved | logistic_regression | 20261008 | C=0.01,class_weight=None | 0.273 | 0.273 | yes |
| resampling | retrieved | logistic_regression | 20261009 | C=0.01,class_weight=balanced | 0.378 | 0.378 | yes |
| resampling | retrieved | logistic_regression | 20261010 | C=0.01,class_weight=None | 0.273 | 0.273 | yes |
| resampling | none | logistic_regression | 20261008 | C=0.01,class_weight=balanced | 0.406 | 0.406 | yes |
| resampling | none | logistic_regression | 20261009 | C=0.01,class_weight=None | 0.295 | 0.295 | yes |
| resampling | none | logistic_regression | 20261010 | C=0.01,class_weight=balanced | 0.406 | 0.406 | yes |

## Diagnostics: why every selected classifier flags nearly everything

| tier | condition | family | seed | classes | train_prevalence | val_prevalence | train_out_of_fold_auroc | val_auroc | val_predicted_hallucinated_share | threshold_is_lowest_val_score |
|---|---|---|---|---|---|---|---|---|---|---|
| single_shot | retrieved | hist_gradient_boosting | 20261008 | 0,1 | 0.375 | 0.649 | 0.661 | 0.358 | 0.997 |  |
| single_shot | retrieved | hist_gradient_boosting | 20261009 | 0,1 | 0.375 | 0.649 | 0.658 | 0.358 | 0.997 |  |
| single_shot | retrieved | hist_gradient_boosting | 20261010 | 0,1 | 0.375 | 0.649 | 0.660 | 0.368 | 1.000 | yes |
| single_shot | none | logistic_regression | 20261008 | 0,1 | 0.375 | 0.649 | 0.650 | 0.477 | 1.000 | yes |
| single_shot | none | logistic_regression | 20261009 | 0,1 | 0.375 | 0.649 | 0.644 | 0.477 | 1.000 | yes |
| single_shot | none | logistic_regression | 20261010 | 0,1 | 0.375 | 0.649 | 0.647 | 0.477 | 1.000 | yes |
| resampling | retrieved | logistic_regression | 20261008 | 0,1 | 0.375 | 0.649 | 0.623 | 0.458 | 0.999 |  |
| resampling | retrieved | logistic_regression | 20261009 | 0,1 | 0.375 | 0.649 | 0.614 | 0.468 | 0.999 |  |
| resampling | retrieved | logistic_regression | 20261010 | 0,1 | 0.375 | 0.649 | 0.606 | 0.458 | 0.999 |  |
| resampling | none | logistic_regression | 20261008 | 0,1 | 0.375 | 0.649 | 0.640 | 0.576 | 0.992 |  |
| resampling | none | logistic_regression | 20261009 | 0,1 | 0.375 | 0.649 | 0.633 | 0.579 | 0.992 |  |
| resampling | none | logistic_regression | 20261010 | 0,1 | 0.375 | 0.649 | 0.627 | 0.576 | 0.992 |  |

Single-feature AUROC with high values read as hallucinated (retrieved condition, resampling tier features):

| feature | auroc_train | auroc_val | mean_train | mean_val | standardized_mean_difference |
|---|---|---|---|---|---|
| sim_max | 0.507 | 0.345 | 0.504 | 0.612 | 0.667 |
| nli_entail_max | 0.498 | 0.370 | 0.059 | 0.108 | 0.212 |
| nli_contra_max | 0.536 | 0.603 | 0.387 | 0.736 | 0.842 |
| lex_overlap_max | 0.519 | 0.275 | 0.183 | 0.386 | 0.754 |
| hedge_rate | 0.496 | 0.529 | 0.011 | 0.003 | -0.487 |
| booster_rate | 0.485 | 0.502 | 0.001 | 0.000 | -0.227 |
| log_length | 0.378 | 0.537 | 3.453 | 3.074 | -0.768 |
| specificity | 0.621 | 0.492 | 0.105 | 0.342 | 1.959 |
| self_contra | 0.362 | 0.510 | 0.150 | 0.171 | 0.062 |
| sc_embed | 0.475 | 0.266 | 0.944 | 0.928 | -0.199 |
| sc_entail | 0.540 | 0.271 | 0.470 | 0.557 | 0.240 |
| sc_contra | 0.563 | 0.730 | 0.041 | 0.299 | 0.998 |

## Calibration method

Mean ECE over 5 cross-fitted `val_id` folds (grouped by `cluster_id`) and the three seeds; isotonic must
beat Platt by at least 0.005. Per-fold values in `calibration_ece_folds.csv`;
reliability diagrams in `reliability_*.png` and `reliability.csv`.

| tier | condition | family | ece_platt | ece_isotonic | platt_minus_isotonic | chosen_method |
|---|---|---|---|---|---|---|
| single_shot | retrieved | hist_gradient_boosting | 0.064 | 0.011 | 0.053 | isotonic |
| single_shot | none | logistic_regression | 0.009 | 0.012 | -0.003 | platt |
| resampling | retrieved | logistic_regression | 0.021 | 0.018 | 0.003 | platt |
| resampling | none | logistic_regression | 0.049 | 0.042 | 0.007 | isotonic |

## Decision policy

Accept if P < t_low (at most 5% of accepted answers hallucinated);
escalate if t_low <= P < t_high (at most 20% of traffic); flag otherwise.
P is the chosen calibrator refitted on all of `val_id`. Infeasible means no t_low meets the accept goal, so
the accept band is empty; the escalate and flag columns of infeasible rows are diagnostic only. Plateau
shares are the `val_id` answers whose P equals the threshold.

| tier | condition | seed | method | n_distinct | feasible | t_low | t_high | accept_rate | accept_hallucinated_share | escalate_rate | flag_rate | flag_hallucinated_share | plateau_share_t_low | plateau_share_t_high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single_shot | retrieved | 20261008 | isotonic | 4 | yes | 0.646 | 0.651 | 0.003 | 0.000 | 0.048 | 0.949 | 0.651 | 0.048 | 0.948 |
| single_shot | retrieved | 20261009 | isotonic | 4 | yes | 0.646 | 0.651 | 0.003 | 0.000 | 0.048 | 0.949 | 0.651 | 0.048 | 0.948 |
| single_shot | retrieved | 20261010 | isotonic | 5 |  | 0.538 | 0.654 | 0.000 |  | 0.102 | 0.898 | 0.654 | 0.013 | 0.897 |
| single_shot | none | 20261008 | platt | 780 |  | 0.588 | 0.635 | 0.000 |  | 0.199 | 0.801 | 0.656 | 0.001 | 0.002 |
| single_shot | none | 20261009 | platt | 780 |  | 0.587 | 0.635 | 0.000 |  | 0.200 | 0.800 | 0.656 | 0.001 | 0.001 |
| single_shot | none | 20261010 | platt | 780 |  | 0.588 | 0.635 | 0.000 |  | 0.199 | 0.801 | 0.656 | 0.001 | 0.002 |
| resampling | retrieved | 20261008 | platt | 995 | yes | 0.566 | 0.625 | 0.001 | 0.000 | 0.200 | 0.799 | 0.662 | 0.001 | 0.001 |
| resampling | retrieved | 20261009 | platt | 995 | yes | 0.591 | 0.632 | 0.001 | 0.000 | 0.200 | 0.799 | 0.662 | 0.001 | 0.001 |
| resampling | retrieved | 20261010 | platt | 995 | yes | 0.566 | 0.625 | 0.001 | 0.000 | 0.200 | 0.799 | 0.662 | 0.001 | 0.001 |
| resampling | none | 20261008 | isotonic | 9 |  | 0.375 | 0.603 | 0.000 |  | 0.051 | 0.949 | 0.661 | 0.008 | 0.527 |
| resampling | none | 20261009 | isotonic | 12 |  | 0.375 | 0.604 | 0.000 |  | 0.059 | 0.941 | 0.661 | 0.008 | 0.515 |
| resampling | none | 20261010 | isotonic | 9 |  | 0.375 | 0.603 | 0.000 |  | 0.051 | 0.949 | 0.661 | 0.008 | 0.527 |

## Baseline checks on val_id (diagnostic, not a research-question verdict)

The classifier versus predicting every answer hallucinated (paired cluster bootstrap, 10,000 replicates,
thresholds fixed, seed-averaged), and versus the best single feature (from Step 6):

| tier | condition | family | f1_classifier | f1_all_hallucinated | f1_diff_vs_all_hallucinated | f1_diff_ci_low | f1_diff_ci_high | beats_all_hallucinated_on_val | best_single_feature | f1_diff_vs_best_single | beats_best_single_on_val |
|---|---|---|---|---|---|---|---|---|---|---|---|
| single_shot | retrieved | hist_gradient_boosting | 0.788 | 0.787 | 0.001 | 0.000 | 0.002 |  | feature:lex_overlap_max | -0.018 |  |
| single_shot | none | logistic_regression | 0.787 | 0.787 | -0.000 | -0.000 | 0.000 |  | feature:log_length | -0.003 |  |
| resampling | retrieved | logistic_regression | 0.788 | 0.787 | 0.000 | -0.000 | 0.001 |  | feature:lex_overlap_max | -0.018 |  |
| resampling | none | logistic_regression | 0.788 | 0.787 | 0.000 | -0.003 | 0.003 |  | feature:log_length | -0.003 |  |

Classifiers beating all-hallucinated on `val_id`: 0 of 4; beating the best single feature: 0 of 4.

## Not implemented

The explanation generator (docs/SPEC.md, build step 7) is deferred until a separate specification is
written and recorded in the pre-registration.

## Artifacts

12 files under `data/models/step7/` (git-ignored), listed with SHA-256 in
`data/models/step7/manifest.json`, together with library versions, the git commit, config and input hashes.
Each holds the fitted classifier (with its scaler for logistic regression), feature order, seed, F1
threshold on the uncalibrated score, calibrator, and policy thresholds. Each was reloaded and reproduced
its `val_id` scores exactly.

## Data access

11 input files read, plus the artifacts written here (reopened to check them); test-split files read: 0.
