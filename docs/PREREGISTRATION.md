# Pre-registration

Drafted 2026-10-08. **Status: draft, not frozen.** Freeze by adding a dated "Frozen on" line
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

Fixed by `configs/default.yaml` and the committed `data/splits/split_manifest.jsonl`.

| Split | Dataset | Generator | Role |
| --- | --- | --- | --- |
| `train` | TruthfulQA (70% of questions) | Model A | feature engineering, classifier fitting |
| `val_id` | HaluEval QA (1,000 questions) | Model A | model selection, calibration, thresholds, judge prompt tuning |
| `test_id` | HaluEval QA (500 other questions) | Model A | untouched in-distribution reference for RQ2 |
| `test_ood` | TruthfulQA (other 30%) | Model B | unseen model, same dataset family |
| `test_heldout` | FEVER (1,000 verifiable claims) | Model B | unseen model and unseen dataset |

## Common definitions

- **F1** is computed for the positive class, *hallucinated*.
- **"The classifier"** is whichever of logistic regression and gradient-boosted trees has the
  higher F1 on `val_id`, chosen once per tier before any test split is touched.
- **"The best single feature"** is the single-feature baseline with the highest `val_id` F1, its
  threshold also tuned on `val_id`.
- **Confidence intervals** come from a bootstrap over questions (not rows). Comparisons use a
  paired bootstrap on the same resampled questions. 10,000 resamples.
- Each test split is scored once per reported result.

## Research question 1: does combining features beat the best single heuristic?

**Yes**, for a given tier, if on **both** `test_ood` and `test_heldout` the classifier's F1 minus
the best single feature's F1 has a paired-bootstrap 95% interval that excludes zero and is
positive. If it holds on one test split only, the answer is reported as "mixed", not yes.

## Research question 2: does the classifier generalize to unseen models and datasets?

**Yes**, for a given tier, if both of the following hold for each of `test_ood` and
`test_heldout`:

1. The out-of-distribution F1 is no more than **5 F1 points** below the in-distribution F1
   measured on `test_id` (point estimates, with the bootstrap interval of the drop reported
   alongside).
2. The out-of-distribution F1 beats the trivial baseline that labels every answer hallucinated,
   with a paired-bootstrap 95% interval on the difference that excludes zero.

In-distribution F1 is measured on `test_id`, not `val_id`. `val_id` is used for selection,
calibration, and thresholds, so F1 there would be optimistically biased and would inflate the
drop. `test_id` is drawn from the same HaluEval / Model A pool after `val_id`. "In-distribution"
here follows the spec: same generator as training, different dataset from training.

## Negative results

A "no" against these criteria is a legitimate headline and will be reported as such.

## Known limitations recorded in advance

- TruthfulQA, HaluEval, and FEVER are probably in the pretraining data of both generators and
  judges.
- TruthfulQA ships no gold evidence passages, only source URLs, so the gold-evidence condition is
  unavailable for `train` and `test_ood`. Reference answers cannot stand in for evidence because
  the labels are derived from them.

## Amendment log

(empty)
