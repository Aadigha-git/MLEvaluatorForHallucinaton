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

- **Primary: retrieved evidence** (see [Retrieval and feature models](#retrieval-and-feature-models)).
- **Secondary: no evidence.** Only evidence-free features are available.
- **Exploratory: gold evidence**, on HaluEval and FEVER only; TruthfulQA has none.

A classifier is trained on `train` separately for the retrieved and no-evidence conditions. The
gold-evidence analysis reuses the retrieved-evidence classifier, because the training split has no
gold evidence.

## Generation

Only the primary answer (sample 0) is labeled; the five resamples are inputs to the
self-consistency features and are never labeled. Raw outputs are stored once and never
regenerated: FP8 batched serving is not bit-reproducible, even with greedy decoding.

| Setting | Primary answer (sample 0) | Resamples (samples 1–5) |
| --- | --- | --- |
| Model A (`train`, `val_id`, `test_id`) | `Qwen/Qwen3-30B-A3B-Instruct-2507` | same |
| Model B (`test_ood`, `test_heldout`) | `google/gemma-3-27b-it` | same |
| Serving | Nebius Token Factory, `https://api.tokenfactory.nebius.com/v1/`, region eu-north1, FP8, as listed in the public catalog on 2026-10-08 | same |
| Nominal weights (Hugging Face revision) | Qwen `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`; Gemma `005ad3404e59d6023443cb575daa05336842228a`. The API does not expose the served revision. | same |
| Temperature / top-p | 0 (greedy) / 1.0 | 0.7 / 1.0 |
| top-k, penalties | not sent (API defaults) | not sent |
| Max new tokens | 128 for `qa-v1`, 96 for `fever-v1` | same |
| Messages | one user message, no system message | same |
| `seed` | first 8 hex digits of SHA-256(`{question_id}/{sample_index}`) as an integer; sent, but determinism is not assumed | same |
| Recorded per call | request parameters, model ID, prompt version, response ID, system fingerprint, finish reason, token usage, timestamps, latency, raw text | same |

Prompts are versioned; any change bumps the version. HaluEval's knowledge passage is withheld,
so all generation is closed-book.

- **`qa-v1`** (TruthfulQA and HaluEval, shared so the in-distribution and out-of-distribution
  comparison is not confounded by a prompt difference):

  ```text
  Answer the following question in one or two sentences.

  Question: {question}
  ```

- **`fever-v1`** (FEVER): `hdetect.data.fever.PROMPT_TEMPLATE`.

### Pilot

Before freezing, a 20-question pilot checks formats and model availability only. Pilot questions
come from `train` and `val_id`, plus FEVER claims that are in no split and share no
near-duplicate cluster with any split. Both models answer every pilot question, including
Model B, which otherwise never sees `train` or `val_id`. Pilot outputs are not labeled, are not
used for any tuning, and are discarded after the format checks. If the pilot changes a prompt or a
setting, its version is bumped here before freezing.

**Outcome (2026-10-08, `scripts/run_pilot.py`, generation config hash `dd62601d8ab3`).** The pilot
used 7 `train` TruthfulQA questions, 7 `val_id` HaluEval questions, and 6 unassigned FEVER claims.
Each model made 140 calls: a primary answer, a second greedy call, and 5 resamples per question.

- All 280 calls succeeded. Each response reported the requested model ID and a single system
  fingerprint per model.
- Every call ended with finish reason `stop`: none was truncated, empty, or contained a reasoning
  block. The longest answers were 96 of 128 tokens (`qa-v1`) and 59 of 96 (`fever-v1`).
- All 84 FEVER answers open with a parseable verdict.
- Some answers use Markdown emphasis (19 of 280), and Gemma sometimes ends with blank lines.
- **Greedy decoding is not repeatable.** A second greedy call returned identical text for 8 of 14
  `qa-v1` questions and 4 of 6 FEVER claims with Qwen, and for 2 of 14 and 3 of 6 with Gemma.
  Token F1 between differing pairs ranged from 0.47 to 0.98. In every differing FEVER pair, both
  calls gave the same verdict.
- Resamples at temperature 0.7 differed within 39 of 40 question–model pairs (mean pairwise token
  F1 0.76).
- Median latency per call at concurrency 8 was 3.0–3.4 s for Qwen and 0.7–0.9 s for Gemma. The
  pilot cost $0.0051.

No prompt or setting changed, so `qa-v1` and `fever-v1` stand. Because greedy output is not
repeatable, the primary answer is a single stored draw and is never regenerated, as already stated
above. The pilot outputs were deleted after these checks.

## Retrieval and feature models

- **Corpus:** Wiki-DPR, 21,015,300 passages of 100 words from the 2018-12-20 English Wikipedia
  dump (`facebook/wiki_dpr`, revision `0ae2454140a2d6864475c83f26e6dc9cd4ab9ce4`, CC-BY-NC-4.0).
- **Retriever:** BM25 using Pyserini's prebuilt Lucene index `wikipedia-dpr-100w`, k1 = 0.9,
  b = 0.4, top 5 passages.
- **Query:** the question text, or the claim for FEVER. It does not depend on the generated answer,
  so both generators get the same passages.
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (maximum 256 word pieces).
- **NLI model:** `cross-encoder/nli-deberta-v3-small`, revision
  `fa2804872c3b4bd748f38c0185cc85775361e735` (labels: 0 contradiction, 1 entailment, 2 neutral).
  Premise = one passage, hypothesis = the answer; passages are scored one at a time.
- Neither model is used anywhere in labeling.

## Fitting and selection: no test data

- Preprocessing (scaling, imputation) is fitted on `train` only.
- **Hyperparameters** are chosen by `StratifiedGroupKFold(n_splits=5, shuffle=True,
  random_state=seed)` on `train`, grouped by `cluster_id`, scored by average precision. Ties go to
  the simpler setting (smaller `C`; then smaller `max_depth`, then fewer `max_iter`). Grids, all
  scikit-learn:
  - logistic regression on standardized features: L2 penalty, `C` ∈ {0.01, 0.1, 1, 10},
    `class_weight` ∈ {None, balanced}, `lbfgs` solver, `max_iter=1000`;
  - `HistGradientBoostingClassifier`: `max_depth` ∈ {2, 3}, `learning_rate` ∈ {0.05, 0.1},
    `max_iter` ∈ {100, 300}, `min_samples_leaf=20`, no early stopping, `random_state=seed`.
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
- **Calibration** is fitted on `val_id` only. Platt scaling and isotonic regression are compared by
  expected calibration error (10 equal-width bins on P(hallucinated)), cross-fitted within
  `val_id`: 5 folds grouped by `cluster_id`, each method fitted on 4 folds and measured on the
  fifth, averaged over folds and the three training seeds. The lower mean ECE wins; a difference
  under 0.005 goes to Platt. The chosen method is then refitted on all of `val_id`. Measuring ECE
  on the data a calibrator was fitted to would favor isotonic regression by construction.
- Test splits are scored only after freezing, once per reported result.

## Metrics and uncertainty

- **Primary metric:** F1 for the positive class, *hallucinated*. **Secondary metric:** AUROC,
  always computed on uncalibrated scores, since isotonic plateaus would create ties.
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
research question. With P the calibrated probability of hallucination on `val_id`:

- **Accept** if P < t_low, where t_low is the largest value at which at most **5%** of accepted
  answers are hallucinated.
- **Escalate** if t_low ≤ P < t_high, where t_high is the largest value at which at most **20%** of
  traffic is escalated. This spends the escalation budget on making flags as precise as possible.
- **Flag** if P ≥ t_high.

Candidate thresholds are the distinct calibrated values on `val_id`. If no t_low meets the 5% goal,
the accept band is empty and the policy is reported as infeasible for that classifier.
Isotonic calibration produces plateaus, so achieved rates move in steps. For every policy the
report gives:

- the number of distinct calibrated values;
- the share of answers on each threshold's plateau, for `val_id` and each test split;
- the achieved accept, escalate, and flag rates, next to the goals;
- a reliability diagram for each calibration method.

How the operating point drifts out of distribution is reported as exploratory.

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

1. ~~**Pilot outcome**~~: resolved; see "Generation", "Pilot".
2. **Feature formulas:**
   - how scores combine over the 5 passages;
   - the lexical-overlap metric;
   - a versioned hedging lexicon;
   - the length and specificity measures;
   - how self-consistency compares the 5 resamples;
   - the internal-contradiction check.
3. **Retrieval recall** per dataset: FEVER by gold page title, HaluEval by answer-string match;
   not measurable for TruthfulQA.
4. **Generation failures:** what happens to truncated, empty, refused, or failed answers, and
   whether those questions are excluded and counted.
5. **Cost and latency measurement:** a dated price snapshot, the concurrency used for wall-clock
   latency, and the spec's "incremental cost" definition.

Not needed to freeze the research-question criteria, but needed before step 8: judge model IDs (a
small open-weight judge and a frontier judge, each from a family other than Qwen and Gemma).

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
- The spec suggests 7–9B generators; Nebius serves none, so the generators are a 30B
  mixture-of-experts model (3B active parameters) and a 27B dense model. Served weights are FP8,
  their exact revision is not exposed, and outputs are not bit-reproducible.
- The retrieval corpus (Wiki-DPR) is licensed CC-BY-NC-4.0, so results and the demo are
  non-commercial.
- **Disclosure.** Before freezing, the labeler checks (`scripts/check_labelers.py`) consulted
  benchmark material from test partitions: TruthfulQA reference answers for all 817 questions,
  including `test_ood`, and the gold labels of all FEVER claims, all of which are in
  `test_heldout`. The HaluEval check used `val_id` only. No generated answers existed or were
  consulted.

## Pre-freeze revision history

- 2026-10-08: initial draft.
- 2026-10-08: added `test_id`; fixed the evidence conditions, threshold rule, bootstrap unit, seed
  protocol, hypothesis tiers, RQ1 and RQ2 outcome rules, and labeling commitments after review.
- 2026-10-08: recorded generators, generation settings, prompts, pilot design, retrieval, feature
  models, hyperparameter grids, calibration selection, AUROC on uncalibrated scores, and the
  decision-policy rule.
- 2026-10-08: recorded the pilot outcome; no prompt or setting changed.

## Amendment log

(empty)
