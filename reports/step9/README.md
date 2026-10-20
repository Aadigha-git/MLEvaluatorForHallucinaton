# Step 9: test evaluation

## Primary research-question outcomes (retrieved evidence, F1)

| question | tier | splits_passing | outcome |
|---|---|---|---|
| RQ1 | single_shot | 0 | no |
| RQ2 | single_shot | 0 | no |
| RQ1 | resampling | 1 | mixed |
| RQ2 | resampling | 0 | no |

## RQ1 per split

| tier | condition | split | hypothesis_tier | best_single | classifier_value | best_single_value | diff | diff_ci_low | diff_ci_high | passes |
|---|---|---|---|---|---|---|---|---|---|---|
| single_shot | retrieved | test_id | secondary | feature:lex_overlap_max | 0.728 | 0.743 | -0.015 | -0.040 | 0.010 |  |
| single_shot | retrieved | test_ood | primary | feature:lex_overlap_max | 0.482 | 0.487 | -0.006 | -0.034 | 0.020 |  |
| single_shot | retrieved | test_heldout | primary | feature:lex_overlap_max | 0.340 | 0.342 | -0.001 | -0.003 | -0.000 |  |
| resampling | retrieved | test_id | secondary | feature:lex_overlap_max | 0.724 | 0.743 | -0.020 | -0.045 | 0.006 |  |
| resampling | retrieved | test_ood | primary | feature:lex_overlap_max | 0.513 | 0.487 | 0.025 | 0.008 | 0.041 | yes |
| resampling | retrieved | test_heldout | primary | feature:lex_overlap_max | 0.340 | 0.342 | -0.002 | -0.003 | 0.000 |  |
| single_shot | none | test_id | secondary | feature:log_length | 0.724 | 0.720 | 0.003 | -0.009 | 0.016 |  |
| single_shot | none | test_ood | secondary | feature:log_length | 0.495 | 0.487 | 0.008 | -0.014 | 0.026 |  |
| single_shot | none | test_heldout | secondary | feature:log_length | 0.341 | 0.340 | 0.000 | 0.000 | 0.001 |  |
| resampling | none | test_id | secondary | feature:log_length | 0.718 | 0.720 | -0.002 | -0.018 | 0.013 |  |
| resampling | none | test_ood | secondary | feature:log_length | 0.522 | 0.487 | 0.034 | 0.003 | 0.064 | yes |
| resampling | none | test_heldout | secondary | feature:log_length | 0.340 | 0.340 | -0.000 | -0.003 | 0.002 |  |

## RQ2 per split

| tier | condition | split | hypothesis_tier | reference_value | split_value | drop | drop_within_margin | vs_trivial | vs_trivial_ci_low | passes |
|---|---|---|---|---|---|---|---|---|---|---|
| single_shot | retrieved | test_ood | primary | 0.728 | 0.482 | 0.246 |  | -0.006 | -0.034 |  |
| single_shot | retrieved | test_heldout | primary | 0.728 | 0.340 | 0.388 |  | 0.000 | 0.000 |  |
| resampling | retrieved | test_ood | primary | 0.724 | 0.513 | 0.211 |  | 0.025 | 0.008 |  |
| resampling | retrieved | test_heldout | primary | 0.724 | 0.340 | 0.384 |  | -0.000 | -0.001 |  |
| single_shot | none | test_ood | secondary | 0.724 | 0.495 | 0.229 |  | 0.008 | -0.014 |  |
| single_shot | none | test_heldout | secondary | 0.724 | 0.341 | 0.383 |  | 0.000 | 0.000 |  |
| resampling | none | test_ood | secondary | 0.718 | 0.522 | 0.197 |  | 0.034 | 0.003 |  |
| resampling | none | test_heldout | secondary | 0.718 | 0.340 | 0.378 |  | -0.000 | -0.003 |  |

## Exclusions

| split | planned | primary_api_failure | empty | labeled | excluded_undecidable | exclusion_rate | flagged_over_5pct |
|---|---|---|---|---|---|---|---|
| test_id | 500 | 0 | 0 | 500 | 0 | 0.000 |  |
| test_ood | 242 | 0 | 0 | 242 | 0 | 0.000 |  |
| test_heldout | 1000 | 0 | 0 | 1000 | 0 | 0.000 |  |

All tables are CSV files in this directory; `data_access.csv` lists every data file read.
