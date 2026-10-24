# Results: a classical-ML hallucination detector against its pre-registered criteria

2026-10-20. Every number comes from the committed reports: `reports/step6/` through
`reports/step10/`. The criteria, hypothesis tiers, and every implementation choice are in
[PREREGISTRATION.md](PREREGISTRATION.md), frozen before any test split was scored, with dated
amendments. The test splits were scored once (Step 9, commit `8d77255`).

## Answers

| Question (primary: retrieved evidence, F1) | Single-shot tier | Resampling tier |
|---|---|---|
| **RQ1.** Does combining features beat the best single heuristic? | **No** | **Mixed** |
| **RQ2.** Does the classifier generalize to unseen models and datasets? | **No** | **No** |

These are negative results, and they are the headline. On every test split, under the primary
analysis, each classifier's F1 is within 0.025 of the F1 of predicting every answer hallucinated.

> **Baseline caveat: read this before the RQ1 result.** On both primary splits the "best single
> feature" baseline is, in effect, the flag-everything rule. With its `val_id` threshold, lexical
> overlap flags 100% of `test_ood` answers and 99.5% of `test_heldout` answers. Its F1 (0.488 and
> 0.342) equals or nearly equals flag-everything (0.488 and 0.340).
>
> So the resampling tier's RQ1 pass on `test_ood` (+0.025, 95% CI 0.008 to 0.041) shows only that
> the classifier beats flagging everything by a small margin. It is not evidence that combining
> features beats an informative single heuristic. The same comparison appears in RQ2 as "beats the
> trivial baseline" on `test_ood`; that alone does not make RQ2 pass, because the drop criterion
> fails.

## Setup in brief

- **Splits.** Each answer gets a binary label: hallucinated or not.

  | Split | Dataset | Generator | Questions | Role |
  |---|---|---|---|---|
  | `train` | TruthfulQA | Qwen3-30B-A3B ("Model A") | 575 | training |
  | `val_id` | HaluEval QA | Model A | 1,000 | thresholds, calibration, selection |
  | `test_id` | HaluEval QA | Model A | 500 | in-distribution reference for RQ2; secondary RQ1 |
  | `test_ood` | TruthfulQA | Gemma-3-27B ("Model B") | 242 | primary |
  | `test_heldout` | FEVER | Model B | 1,000 | primary |

- **Two tiers of black-box features.**
  - **Single-shot:** retrieval similarity, NLI entailment and contradiction against the top-5
    Wiki-DPR passages, lexical overlap, hedging, length, specificity, and self-contradiction.
  - **Resampling:** adds three self-consistency features from 5 resamples.
- **Classifiers.** Logistic regression or gradient-boosted trees, chosen per tier and evidence
  condition, trained on `train` only. The F1 threshold, calibration, and accept/escalate/flag
  policy were all set on `val_id`.
- **Statistics.** 10,000-replicate cluster bootstraps with paired comparisons. Every statistic is
  averaged over three training seeds within each replicate.
- **Labels.** Every test answer got a label. 189 unresolved answers were hand-labeled blind, and
  none was undecidable. No split was excluded or flagged.

## RQ1: combining features versus the best single feature

Criterion: D = F1(classifier) − F1(best single feature). The answer is "yes" if the lower bound of
the paired 95% interval is above 0 on both `test_ood` and `test_heldout`.

| Tier | Split | Classifier F1 | Best single (lexical overlap) F1 | Flag-everything F1 | D [95% CI] | Passes |
|---|---|---|---|---|---|---|
| single-shot | `test_ood` | 0.482 | 0.488 | 0.488 | −0.006 [−0.034, 0.020] | no |
| single-shot | `test_heldout` | 0.340 | 0.342 | 0.340 | −0.001 [−0.003, −0.000] | no |
| resampling | `test_ood` | 0.513 | 0.488 | 0.488 | 0.025 [0.008, 0.041] | yes* |
| resampling | `test_heldout` | 0.340 | 0.342 | 0.340 | −0.002 [−0.003, 0.000] | no |

\* The baseline here is effectively the flag-everything rule (see the caveat under "Answers").

- **Outcome.** Single-shot: no. Resampling: mixed.
- **The one pass beats flag-everything, not an informative heuristic.** On `test_ood`, lexical
  overlap flags every answer, so its F1 of 0.4875 is exactly flag-everything's at that split's
  prevalence (0.322). The resampling classifier's margin over it is the same +0.025
  [0.008, 0.041] reported under RQ2.
- **Secondary tests.**
  - `test_id`: no difference (−0.015 and −0.020; both intervals include 0).
  - No evidence: single-shot no, resampling mixed (passes `test_ood` again).
  - AUROC versions: no or mixed.
  - Excluding abstentions or truncated answers changes no outcome.
- **AUROC.** On `test_id` and `test_heldout`, the classifiers rank answers worse than the
  lexical-overlap feature alone:

  | Split | Single-shot classifier | Resampling classifier | Lexical overlap |
  |---|---|---|---|
  | `test_id` | 0.405 | 0.516 | 0.686 |
  | `test_heldout` | 0.436 | 0.472 | 0.561 |

  On both splits, the 95% intervals of both differences from the feature lie below 0. On `test_ood`,
  the classifiers' AUROC (0.63) is above the feature's (0.59), but the intervals include 0.

## RQ2: generalization to unseen models and datasets

Criterion: a split S passes if the drop Δ(S) = F1(`test_id`) − F1(S) is at most 0.05 (point
estimate), *and* F1(S) beats predicting every answer hallucinated (lower bound of the paired
interval above 0).

| Tier | Split | F1 `test_id` → S | Drop Δ [95% CI] | vs. flag-everything [95% CI] | Passes |
|---|---|---|---|---|---|
| single-shot | `test_ood` | 0.728 → 0.482 | 0.246 [0.171, 0.325] | −0.006 [−0.034, 0.020] | no |
| single-shot | `test_heldout` | 0.728 → 0.340 | 0.388 [0.336, 0.440] | 0.000 [0.000, 0.000] | no |
| resampling | `test_ood` | 0.724 → 0.513 | 0.211 [0.135, 0.290] | 0.025 [0.008, 0.041] | no |
| resampling | `test_heldout` | 0.724 → 0.340 | 0.384 [0.331, 0.436] | −0.000 [−0.001, 0.001] | no |

- **Outcome.** No for both tiers. The primary drops (0.21–0.39) are about 4 to 8 times the 0.05
  margin.
- **The drop tracks prevalence.** Hallucination prevalence is 57.2% on `test_id`,
  32.2% on `test_ood`, and 20.5% on `test_heldout`, and the F1 of flagging everything follows it
  down: 0.728, 0.488, 0.340. On `test_id` and `test_heldout`, the classifiers' F1 matches that
  trivial F1 or falls just below it.
- **The AUROC version (secondary) is mixed for every tier and condition.** It passes on
  `test_ood`, where AUROC is 0.63–0.65 against 0.41–0.61 on `test_id`, and fails on
  `test_heldout`. A higher AUROC out of distribution than on the reference is a "negative drop",
  not evidence of generalization in the sense RQ2 intends.

## Diagnostic observations (not tests)

These observations come from validation diagnostics and the test results. None was a
preregistered test, and none is a causal finding.

- **Flag-everything thresholds.** Already on `val_id` (Step 7), none of the four classifiers beat
  predicting every answer hallucinated. Their F1-maximizing thresholds sit at or near the lowest
  `val_id` score. On the test splits, the primary classifier flagged 1,152 correct answers and
  missed only 4 hallucinations.
- **The training split differs from the "in-distribution" splits.**
  - `train` is TruthfulQA, while `val_id` and `test_id` are HaluEval, with the same generator.
  - Hallucination prevalence is 37.5% in `train` and 64.9% in `val_id`.
  - Retrieved-evidence classifiers have out-of-fold AUROC of 0.61–0.66 on `train`, but
    0.36–0.47 on `val_id` and 0.41–0.52 on `test_id`. Within HaluEval, they rank answers at or
    below chance.
  - On TruthfulQA (`test_ood`, a different generator) their AUROC is 0.63.
  - So in this design, "in-distribution" means same generator, not same dataset. The pattern is
    consistent with the dataset change mattering more than the generator change for these
    features, but the design does not isolate either.
- **Lexical overlap is the strongest single ranker on HaluEval and FEVER.** It is the best single
  feature in both tiers under retrieved evidence. In the hand-sorted false positives, the most
  common category is a correct answer worded differently from the evidence (30 of 50).

## LLM judges, cost, and latency

The judges are pre-declared baselines, with prompts frozen on `val_id` (Step 8). They are fixed
binary predictions; failures count as wrong.

**F1 with retrieved evidence:**

| Evaluator | `test_id` | `test_ood` | `test_heldout` |
|---|---|---|---|
| Single-shot classifier | 0.728 | 0.482 | 0.340 |
| Resampling classifier | 0.724 | 0.513 | 0.340 |
| Nemotron-3-Nano-30B (small judge) | 0.791 | 0.516 | 0.423 |
| Gemini 3.1 Pro (frontier judge) | not run | 0.565 | **0.895** |

- **Nemotron.**
  - It beats both classifiers on `test_id` (classifier − judge ≈ −0.06 to −0.07, interval below
    0) and on `test_heldout` (−0.083 [−0.105, −0.061]).
  - On `test_ood` the intervals include 0.
  - Without evidence, it reaches 0.621 on `test_ood` and 0.320 on `test_heldout`.
- **Gemini.**
  - On FEVER it beats both classifiers by 0.555 [0.510, 0.600].
  - On `test_ood` its advantage of 0.05–0.08 has an interval that includes 0.
  - Gemini was run only on `test_ood` and `test_heldout` with retrieved evidence, to fit its
    daily quota (amendment 2026-10-12).
- **Cost and latency per verdict** (retrieved evidence). Local timings are on an Apple M5
  laptop; API calls ran at concurrency 8.

  | Evaluator | Incremental cost | Median latency | 95th percentile |
  |---|---|---|---|
  | Single-shot classifier | $0 | 0.22–0.29 s | 0.26–0.39 s |
  | Resampling classifier | $0.00008–$0.00010 (5 resamples) | 1.2–1.6 s (parallel-equivalent) | 2.0–4.7 s |
  | Nemotron judge | about $0.00006 | 0.57–0.82 s | 0.69–1.01 s |
  | Gemini judge | $0.0041–$0.0051 | 3.8–4.0 s | 8.0 s (`test_ood`), 94 s (`test_heldout`) |

  On `test_heldout`, 39 Gemini calls needed transient retries. Without them, the 95th percentile
  is 54 s.
- **Pareto chart** (`reports/step9/pareto_f1_cost.png`). The classifiers and the single feature
  sit at or near the flag-everything line; the resampling classifier is 0.025 above it on
  `test_ood`. The judges rise above it furthest, Gemini most, at about 70–90 times Nemotron's cost
  per verdict.

## Calibration and the decision policy (exploratory)

- **Calibration** was fitted on `val_id`. It holds on `test_id` (ECE 0.08) but not out of
  distribution: ECE is 0.19–0.37 on `test_ood` and 0.42–0.45 on `test_heldout`. Mean calibrated
  P(hallucination) is 0.51–0.70 on every split, while prevalence is 0.32 on `test_ood` and 0.21
  on `test_heldout`.
- **The policy** was already infeasible or degenerate on `val_id`: at most 3 of 995 answers
  accepted. On the test splits:
  - the accept band is empty or nearly empty, with one exception: 11% of `test_ood` answers are
    accepted by the single-shot, retrieved-evidence policy (mean over seeds);
  - flag precision is close to each split's prevalence;
  - the band shares drift with the score distribution, for example 0–52% escalated on `test_ood`
    depending on the model.

## Exploratory analyses

- **Gold evidence** (`test_id`, `test_heldout`):
  - With gold passages, lexical overlap reaches AUROC 0.808 on `test_id`. HaluEval's gold
    evidence contains the reference answer, a limitation recorded in advance.
  - The classifiers' AUROC with gold evidence is 0.38–0.46, below the feature's on both splits
    (intervals below 0). In this exploratory comparison, gold evidence did not lift them.
- **Pooled test splits** (1,742 answers): classifier F1 is 0.492–0.494 and flag-everything is
  0.492. Nemotron reaches 0.577–0.580.
- **Failure analysis** (Step 10). The primary classifier's errors were 1,152 false positives and
  4 false negatives. All 4 false negatives and 50 sampled false positives were hand-sorted into
  categories fixed in advance:

  | Category | False positives (50) | False negatives (4) |
  |---|---|---|
  | paraphrase (correct, worded differently from the evidence) | 30 | 0 |
  | numeric or date | 9 | 0 |
  | partial | 6 | 1 |
  | wrong entity | 1 | 3 |
  | label doubt | 2 | 0 |
  | hedged or abstained | 1 | 0 |
  | other | 1 | 0 |
  | retrieval miss | 0 | 0 |

- **Adversarial perturbations** (Step 10). These are rule-based edits of every test answer, with
  features recomputed; recomputing unedited answers reproduced the frozen features exactly.
  - **Edits that keep the facts.** On `test_ood`, padding with a content-free sentence, removing
    hedges, adding "I think", or truncating flips 4.8–13.2% of classifier verdicts, and changes the
    policy band of up to 40% of answers. On `test_id` and `test_heldout`, where the classifiers
    flag nearly everything, at most 4.5% flip.
  - **Entity swaps**, which mostly make answers wrong, move at most 3.6% of verdicts toward
    "hallucinated". Because the classifiers already flag most answers, this rate is bounded and
    says little about detection. Separately, 3 of the 4 real false negatives were wrong-entity
    errors.

## Limitations and deviations

- **Hand labels are inconsistent.** The labeler disagreed with their own earlier hand labels on 22
  of 50 answers: Cohen's kappa 0.198 [−0.026, 0.422]. Hand labels make up 53% of `train` and
  `test_ood` labels. See the 2026-10-23 addendum at the end.
- **HaluEval labels failed the hand-audit agreement gate** (Cohen's kappa 0.7117 in the first round
  and 0.578 under rule version 2, against a minimum of 0.6). This affects `val_id`, where every
  threshold and calibrator was set, and `test_id`, the RQ2 reference.
- **Training exclusions.** 9.2% of `train` answers were excluded as undecidable, above the 5% flag.
  Every classifier is marked.
- **Threshold transfer.** Thresholds were set on HaluEval and carried to TruthfulQA and FEVER, so
  out-of-distribution F1 includes that cost, as recorded in advance. The AUROC versions, which
  need no threshold, are also "no" or "mixed"; their only passes are on `test_ood`.
- **RQ2 confound.** The drop from `test_id` to `test_ood` changes dataset and generator at once.
  With 242 `test_ood` questions, its interval is wide.
- **Gemini's prompt** was chosen on only 45 `val_id` answers, under quota limits (amendments
  2026-10-10 and 2026-10-12). Self-preference by the judges toward Model B's answers cannot be
  ruled out.
- **Pretraining contamination.** The benchmarks are probably in the pretraining data of all models
  involved.
- **Disclosed test-partition consultation.** Before the freeze, the labeler checks consulted
  TruthfulQA references and FEVER gold labels from test partitions; no generated answers were
  consulted.
- **Implementation choices made during Step 9,** all recorded before scoring:
  - feature matrices in a fixed memory layout, so stored thresholds reproduce bit-for-bit;
  - seed-averaged statistics rounded to 12 decimals before comparison, so rounding noise cannot
    produce a spurious pass.

## Possible follow-ups (not tested here)

Either of these would need a new preregistration, not an amendment to this one.

- **A training set that matches where the classifier is tuned and used.** The classifiers were
  trained on TruthfulQA, then thresholded and calibrated on HaluEval. On HaluEval their ranking is
  at or below chance.
- **A paraphrase-robust evidence feature.** Paraphrase was the most common false-positive category
  in the hand-sorted sample (30 of 50), and lexical overlap, the strongest single ranker, is
  sensitive to wording by construction.

## Remaining work (separate from these results)

This item does not change any result above.

1. **Demo (build steps 12–13; an illustration, not evidence).**
   - Built, per the amendment of 2026-10-23:
     - the scoring service (`scripts/demo_scorer.py`, `src/hdetect/service/`) with the
       `demo-event-v1` schema and template explanations;
     - Kafka in Docker (`demo/docker-compose.yml`);
     - one Spring Boot app (`demo/app/`) with the question producer, the verdict relay, and
       dashboard v1.
   - Still to do:
     - the project owner reviews the trap questions (`demo/trap_questions.jsonl`);
     - record the demo.
   - docs/SPEC.md treats the demo as an illustration with no ground truth. As expected from the
     test results, the frozen policy flags nearly every demo answer.

## Addendum, 2026-10-23: hand-label consistency (preregistered)

**The hand labels are not reliable.** Re-labeling 50 of them blind, two weeks later, gave Cohen's
kappa 0.198 (95% CI −0.026 to 0.422) over three categories. The two passes agreed on 28 of 50
answers.

- **Procedure** (amendment 2026-10-23, `reports/consistency/`).
  - 50 answers were drawn from the 425-answer development hand-labeling queue: 39 TruthfulQA from
    `train` and 11 HaluEval from `val_id`.
  - They were stratified by dataset, seeded, and labeled again blind 14 days after the first pass.
  - `undecidable` is a third category.
  - The interval comes from a 10,000-replicate bootstrap over questions.
- **Agreement.**

  | Dataset | n | Raw agreement | Kappa [95% CI] |
  |---|---|---|---|
  | all | 50 | 0.560 | 0.198 [−0.026, 0.422] |
  | TruthfulQA | 39 | 0.564 | 0.237 [−0.008, 0.477] |
  | HaluEval | 11 | 0.545 | 0.035 [−0.375, 0.560] |

  For HaluEval, with 11 answers, the interval is very wide, and 1 of the 10,000 replicates had an
  undefined kappa.
- **Where the passes disagreed.**

  | Change between passes | Answers |
  |---|---|
  | hallucinated → not_hallucinated | 10 |
  | undecidable → not_hallucinated | 5 |
  | not_hallucinated → hallucinated | 4 |
  | hallucinated → undecidable | 1 |
  | not_hallucinated → undecidable | 1 |
  | undecidable → hallucinated | 1 |

  The second pass labeled more answers not hallucinated than the first.

  | Label | First pass | Second pass |
  |---|---|---|
  | hallucinated | 16 | 10 |
  | not_hallucinated | 26 | 36 |
  | undecidable | 8 | 4 |
- **What this affects.** Hand labels cover a different share of each split's labeled answers:

  | Split | Hand-labeled share |
  |---|---|
  | `train` | 53% (277 of 522) |
  | `test_ood` | 53% (129 of 242) |
  | `test_id` | 12% (60 of 500) |
  | `val_id` | 9% (90 of 995) |
  | `test_heldout` | 0% (FEVER labels are all automatic) |

  The re-labeled sample comes from the development queue only, so it does not measure the test
  queue directly. The same person labeled both queues under the same rules, so `train` and
  `test_ood` should be treated as carrying substantial label noise.
- **What it does not change.**
  - No label was edited, and no result was recomputed. The preregistration requires only that
    kappa be reported, and it sets no gate for this check.
  - The research-question outcomes stand as scored, but this caveat applies to them.
  - Noisy training labels can weaken any classifier. Noisy `test_ood` labels make every `test_ood`
    comparison less precise, including the one RQ1 pass, which was already caveated above.
  - These results cannot say how much of the negative result is due to label noise.
