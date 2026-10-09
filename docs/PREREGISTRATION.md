# Pre-registration

Drafted 2026-10-08, revised 2026-10-08. **Status: draft, not frozen.** It cannot be frozen while
any item under [Open items](#open-items) is unresolved. Freeze by adding a dated "Frozen on" line
before the first test-split run. After freezing, changes go in the amendment log at the bottom
with a date and a reason; nothing above the log is edited.

## Label taxonomy

One binary label per generated answer: **hallucinated** or **not hallucinated**, as defined
per dataset in [LABELING.md](LABELING.md). The three sub-types in the spec (factually false,
unsupported, self-contradictory) are a possible later multi-label extension and are not part of
either research question.

## Tiers

- **Single-shot tier.** The verdict depends only on the question, the one answer being judged,
  and optionally an evidence passage, with no further access to the generating model.
- **Resampling tier.** Adds self-consistency features, which require re-querying the generating
  model N = 5 times at nonzero temperature. Needs query access to the generator but never its
  logits or weights.

Every result is reported for both tiers separately. The model-agnostic claim applies in full only
to the single-shot tier.

## Splits

Frozen question-level assignments in `data/splits/split_manifest.jsonl`, created in commit
`bfc3378`, SHA-256 `72398783e968f736556e6d05f489f4b9f44cb2e886506fa3b59370810fad0ef2`, config hash
`30cf64f436c0`. Every question and every near-duplicate cluster belongs to exactly one split.
Assignments stay fixed across training seeds; no split is resampled and no answers are regenerated
to vary them.

| Split | Dataset | Generator | Questions | Role |
| --- | --- | --- | --- | --- |
| `train` | TruthfulQA | Model A | 575 | fitting, preprocessing, hyperparameter cross-validation |
| `val_id` | HaluEval QA | Model A | 1,000 | model choice, decision thresholds, calibration, decision-policy thresholds, judge prompt tuning |
| `test_id` | HaluEval QA | Model A | 500 | untouched in-distribution reference for RQ2; secondary RQ1 diagnostic |
| `test_ood` | TruthfulQA | Model B | 242 | unseen model, same dataset family |
| `test_heldout` | FEVER (verifiable claims) | Model B | 1,000 | unseen model and unseen dataset |

## Evidence conditions

- **Primary: retrieved evidence** (method under [Open items](#open-items)).
- **Secondary: no evidence.** Only evidence-free features are available.
- **Exploratory: gold evidence**, on HaluEval and FEVER only; TruthfulQA has none.

A classifier is trained on `train` separately for the retrieved and no-evidence conditions. The
gold-evidence analysis reuses the retrieved-evidence classifier, because the training split has no
gold evidence.

## Fitting and selection: no test data

- Preprocessing (scaling, imputation) is fitted on `train` only.
- Hyperparameters are chosen by 5-fold cross-validation on `train`, with folds grouped by
  `cluster_id`, over fixed grids (see [Open items](#open-items)).
- **The classifier** is whichever of logistic regression and gradient-boosted trees has the higher
  mean `val_id` F1 across the three training seeds, chosen per tier and evidence condition.
- **Decision threshold.** For every evaluator (each classifier per tier, evidence condition, and
  seed, and each single feature), the threshold that maximizes hallucinated-class F1 on `val_id`.
  Candidates are the unique scores observed on `val_id`; an answer is predicted hallucinated when
  its score is at or above the threshold; ties go to the lowest threshold. For single features,
  the direction (whether high or low scores mean hallucinated) is chosen together with the
  threshold. For classifiers, the threshold is chosen on the uncalibrated score, so calibration
  cannot change any F1 result.
- **The best single feature** is the single feature with the highest `val_id` F1 among the features
  available in that tier and evidence condition. In the resampling tier this includes
  self-consistency.
- Calibration is fitted on `val_id` only (method under [Open items](#open-items)).
- Test splits are scored only after freezing, once per reported result.

## Metrics and uncertainty

- **Primary metric:** F1 for the positive class, *hallucinated*. **Secondary metric:** AUROC.
- **Bootstrap.** Resampling is by `cluster_id`, keeping every row of a selected cluster together.
  10,000 replicates and 95% percentile intervals. Comparisons within a split are paired: every
  evaluator, seed, and baseline is scored on the same resampled clusters. A comparison across two
  splits, such as the RQ2 drop, resamples each split independently in each replicate.
- **Degenerate replicates.** F1 is 0 when a replicate has no true positives, false positives, or
  false negatives. AUROC is skipped for a replicate that contains only one class. The number of
  degenerate replicates is reported for every interval.
- **Training seeds:** 20261008, 20261009, 20261010. These seeds control cross-validation folds and
  gradient-boosted tree randomness. Every verdict uses the seed-averaged statistic: within each
  bootstrap replicate, the statistic is computed for each seed and then averaged. The mean, minimum
  to maximum, and standard deviation across seeds are reported for every point estimate.

## Hypothesis tiers

- **Primary:** RQ1 and RQ2 below, for each tier, under retrieved evidence.
- **Secondary:**
  - the same tests under no evidence;
  - AUROC versions of all primary comparisons;
  - RQ1 on `test_id`;
  - all primary tests with abstentions excluded.
- **Exploratory:**
  - the gold-evidence condition;
  - ablations, single-feature baselines beyond the best one, and the feature correlation matrix;
  - calibration and decision-policy drift;
  - failure analysis and adversarial perturbations.

No correction for multiple comparisons is applied. Every primary test is reported, whatever its
outcome.

## Research question 1: does combining features beat the best single heuristic?

For each tier, D = F1(classifier) − F1(best single feature), with a paired bootstrap on
`test_ood` and on `test_heldout`.

- **Yes:** the lower bound of the 95% interval of D is above 0 on both splits.
- **Mixed:** that holds on exactly one split.
- **No:** it holds on neither.

`test_id` is reported as a secondary diagnostic. Because the primary splits are out of
distribution, a "yes" claims that combining features wins under distribution shift. Thresholds
are tuned on HaluEval, so these F1 values also include the cost of carrying a threshold over to
another dataset; the secondary AUROC comparison does not depend on thresholds.

## Research question 2: does the classifier generalize to unseen models and datasets?

For each tier and each out-of-distribution split S in {`test_ood`, `test_heldout`}:

- **Drop:** Δ(S) = F1(`test_id`) − F1(S), an absolute difference on the 0–1 F1 scale.
- **Trivial baseline:** the predictor that labels every answer hallucinated. On a sample where a
  fraction p of answers is hallucinated, its F1 is 2p / (1 + p), with p computed within each
  replicate. Nothing is fitted.

S passes if both of the following hold:

1. the point estimate of Δ(S) is at most **0.05**. A negative Δ passes. The 95% interval of Δ(S)
   is reported separately and is not part of the criterion;
2. the lower bound of the 95% paired interval of F1(S) − F1(all-hallucinated) is above 0.

Outcomes:

- **Yes:** both splits pass.
- **Mixed:** exactly one passes.
- **No:** neither passes.

`test_id` and `val_id` are both HaluEval / Model A, so `test_id` measures in-distribution F1
without the bias of tuning. Going from `test_id` to `test_ood` changes the dataset (HaluEval to
TruthfulQA) and the generator at the same time, so Δ does not isolate the effect of an unseen
model (see Known limitations).

## Decision policy

The accept/flag/escalate policy is separate from the binary classifier and is not part of either
research question. Its two thresholds are solved on calibrated `val_id` probabilities to meet the
operating-point goals under [Open items](#open-items). How the operating point drifts out of
distribution is reported as exploratory.

## Labeling commitments

- Labeling rule version 1 ([LABELING.md](LABELING.md)) is frozen before the hand audit.
- **Abstentions** are labeled not hallucinated. A secondary analysis excludes them, and abstention
  rates are reported per split and generator.
- **Unresolved answers** are hand-labeled. Answers the hand labeler cannot decide are excluded and
  counted. A split with more than 5% of its answers excluded is flagged in every result that uses
  it.
- **Audit.** 200 automatically resolved answers, stratified by dataset × generator × automatic
  label and drawn from all splits, including test splits. The minimum acceptable Cohen's kappa is
  **0.6**, overall and for each dataset. If it falls short:
  - the rules are revised using audit items from `train` and `val_id` only;
  - the rule version is bumped;
  - a fresh sample of 200 is audited;
  - the change is logged as an amendment.
- **Hand-label consistency.** 50 hand-labeled unresolved answers are labeled again at least two
  weeks later, and agreement (Cohen's kappa) is reported.

## Open items

These must be resolved, and recorded here, before freezing.

1. **Generator IDs:** exact Model A and Model B identifiers, revision, quantization, and serving
   endpoint.
2. **Generation settings:** temperature and maximum tokens for the primary answer; temperature and
   top-p for the 5 resamples; generation seeds where the API supports them.
3. **Prompts** for TruthfulQA and HaluEval (FEVER: `hdetect.data.fever.PROMPT_TEMPLATE`).
4. **Feature-model IDs:** the sentence-embedding model and the NLI model.
5. **Retrieval:** corpus and index, retriever (for example dense DPR or BM25), and top-k.
6. **Calibration method:** Platt scaling or isotonic regression.
7. **Hyperparameter grids** for logistic regression and gradient-boosted trees.
8. **Operating-point goals** for the decision policy (the spec's examples: at most 5% of accepted
   answers hallucinated; at most 20% of traffic escalated).

Not needed to freeze the research-question criteria, but needed before step 8: judge model IDs (a
small open-weight judge and a frontier judge, each from a different family than the generators).

## Negative results

A "no" or "mixed" against these criteria is a legitimate headline and will be reported as such.

## Known limitations recorded in advance

- TruthfulQA, HaluEval, and FEVER are probably in the pretraining data of both generators and
  judges.
- TruthfulQA ships no gold evidence passages, only source URLs, so the gold-evidence condition is
  unavailable for `train` and `test_ood`. Reference answers cannot stand in for evidence because
  the labels are derived from them.
- The RQ2 drop from `test_id` to `test_ood` changes dataset and generator at once.
- Decision thresholds are tuned on HaluEval and carried over to TruthfulQA and FEVER, so
  out-of-distribution F1 includes the cost of threshold transfer.
- HaluEval's gold evidence contains the reference answer, so in the gold-evidence condition the
  lexical-overlap feature nearly reproduces the HaluEval labeling rule. This is why gold evidence
  is exploratory.
- With 242 questions in `test_ood`, the interval of Δ is likely wider than 0.05. The point-estimate
  criterion can therefore pass or fail on sampling noise; the interval is reported so readers can
  judge.
- **Disclosure.** Before freezing, the labeler checks (`scripts/check_labelers.py`) consulted
  benchmark material from test partitions: TruthfulQA reference answers for all 817 questions,
  including `test_ood`, and the gold labels of all FEVER claims, all of which are in
  `test_heldout`. The HaluEval check used `val_id` only. No generated answers existed or were
  consulted.

## Pre-freeze revision history

- 2026-10-08: initial draft.
- 2026-10-08: added `test_id`; fixed the evidence conditions, threshold rule, bootstrap unit, seed
  protocol, hypothesis tiers, RQ1 and RQ2 outcome rules, and labeling commitments after review.

## Amendment log

(empty)
