# Pre-registration

Drafted 2026-10-08, revised 2026-10-08. **Status: frozen.**

**Frozen on 2026-10-08**, before full generation and before any test-split run. All open items
required for freezing are resolved. From here on, changes go in the amendment log at the bottom
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

### Storage, retries, and failures

- **Storage.** Raw responses and call metadata go to one append-only file per split,
  `data/generations/{split}.jsonl`, keyed by `answer_id` (`{question_id}/{generator}/{sample_index}`).
  These files are git-ignored and are not committed. A committed manifest records each file's
  SHA-256, row count, and the generation config hash. Completed files are made read-only and
  copied to a separate durable backup outside the repository; the backup's hashes are verified
  against the manifest, and a restore into a scratch directory is verified to reproduce them.
- **Cache.** A successful row is never requested again or overwritten. If a stored row's request
  parameters (model, prompt, temperature, top-p, max tokens, seed) differ from the plan, the run
  aborts before making any call.
- **Retries.** Transient errors (rate limits, timeouts, server errors, connection failures) are
  retried 5 times with exponential backoff inside each attempt. A call that still fails gets an
  error row and is attempted again on later passes, up to **5 attempts** in total. Errors caused by
  the setup rather than the question (HTTP 400, 401, 403, 404) stop the run at once and are not
  recorded against any question.
- **Permanent failure:** no successful row after 5 attempts. A failed primary answer removes the
  question from every result, counted as `api_failure`. Failed resamples are handled under
  "Features", F9.
- **Empty answers.** A successful call whose cleaned text (see "Text normalization") is empty,
  including a content-filter stop, is not retried, since retrying would regenerate it. An empty
  primary answer is excluded from every result and counted as `empty`. It is never labeled
  hallucinated.
- **Refusals** are model output, not failures, and are labeled by the existing rules: an
  abstention is not hallucinated; anything else unclear goes to hand-labeling. No separate refusal
  detector is used.
- **Truncation** (finish reason `length`): the answer is kept and labeled as is, and the count is
  reported. A secondary analysis repeats the primary tests without truncated primary answers.
- **Reporting.** For every split and generator: planned, successful, `api_failure`, `empty`,
  truncated, abstention, sent to hand-labeling, and excluded after hand-labeling. Total exclusions
  (`api_failure` + `empty` + hand-excluded) count toward the 5% flag under "Labeling commitments".
  Every metric states its n.
- **Budget guard.** Before each call, the spend so far plus an upper bound on the call's cost (its
  prompt length in characters as input tokens, plus the maximum new tokens as output tokens,
  priced at the snapshot) is checked against a cap of **$1.00** for full generation. If the cap
  would be exceeded, the run stops cleanly and resumes only after the cap is raised by amendment.

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

### Retrieval recall

Recall is descriptive. It is computed for every split after freezing, and no retrieval setting
changes in response to it.

- **FEVER, page recall@5:** the share of claims where at least one of the top 5 passage titles
  equals a gold page title. Titles are compared after `hdetect.data.fever.page_title`, NFKC, and
  case-folding.
- **FEVER, sentence recall@5:** the share of claims where at least one gold sentence's normalized
  tokens appear as a contiguous span in one of the top 5 passages.
- **HaluEval, knowledge-sentence recall@5:** the share of questions where at least one sentence of
  the gold knowledge passage appears, as a contiguous span of normalized tokens, in one of the top
  5 passages. The knowledge passage has no page title, so there is no page-level measure. Matching
  the reference answer string is not used, because it is not gold evidence.
- **TruthfulQA: not measurable.** It has no gold passages, only source URLs, and is reported as
  "n/a: no gold evidence". No proxy is used.

FEVER was built from a 2017 Wikipedia dump and Wiki-DPR from a 2018 dump, so both measures
undercount.

## Text normalization

`hdetect` applies `clean_answer` v1 to every answer before labeling and before feature extraction.
The raw text is stored unchanged and never overwritten. In order:

1. Unicode NFKC.
2. Remove Markdown emphasis that wraps text: `**x**`, `__x__`, `*x*`, `_x_`, and backticks.
3. At the start of a line, remove heading markers (`#`), quote markers (`>`), and bullet markers
   (`-`, `*`, `+` followed by a space). Numbered-list markers are kept.
4. Collapse every run of whitespace, including newlines, to one space, and trim both ends.

Labeling rule version 1 takes the cleaned answer as input. No generated answer had been labeled
when this was decided, and the hand audit had not started. Labeling already turns every character
that steps 2–3 remove into a space, so cleaning changes a label in one case only: FEVER verdict
parsing reads the text before normalization, so emphasis that splits a multi-word verdict (for
example `*NOT* ENOUGH INFO`) leaves the raw answer unparseable (sent to hand-labeling), while the
cleaned answer parses. `tests/test_cleaning.py` checks 1,449 Markdown variants of reference
answers and verdicts; labels match except for that case.

## Features (`features-v1`)

Notation: *q* is the question (the claim for FEVER); *a* is the cleaned primary answer; *r*₁…*r*₅
are the cleaned resamples; *P*₁…*P*ₖ are the evidence passages. *T*(*x*) is the set of
`hdetect.labeling.text.content_tokens` of *x* (normalized tokens without stopwords, negations
kept). *n* is the number of `hdetect.labeling.text.tokens` of *a*. NLI(premise, hypothesis) gives
softmax probabilities of contradiction, entailment, and neutral.

- **F0, input.** *a* is used as is for every dataset. FEVER answers keep their leading verdict
  word; nothing is rewritten per dataset.
- **F1, passages.** Retrieved: the top 5 BM25 passages, each as `"{title}: {text}"` (*k* = 5).
  Gold: the one HaluEval knowledge passage (*k* = 1), or each FEVER gold sentence as
  `"Title: sentence"` (*k* varies). No evidence: F2–F5 are absent, not imputed, and that
  classifier uses F6–F9 only.
- **F2, semantic similarity.** `sim_max` = max over *i* of the cosine similarity between the
  normalized embeddings of *a* and *P*ᵢ (library default pooling, inputs truncated at 256 word
  pieces).
- **F3, NLI.** `nli_entail_max` = max over *i* of P(entailment | premise *P*ᵢ, hypothesis *a*);
  `nli_contra_max` = max over *i* of P(contradiction | *P*ᵢ, *a*). Maximum input length 512
  tokens; only the premise is truncated.
- **F4, aggregation over passages.** Every evidence feature takes the maximum over passages.
- **F5, lexical overlap.** With *A* = *T*(*a*) \ *T*(*q*), `lex_overlap_max` =
  max over *i* of |*A* ∩ *T*(*P*ᵢ)| / |*A*|, and 0 when *A* is empty.
- **F6, hedging.** `hedge_rate` = matches of the hedge lexicon / *n*; `booster_rate` = matches of
  the booster lexicon / *n*; both 0 when *n* = 0. Matching runs on normalized tokens, longest
  phrase first, without overlaps. Lexicon `hedges-v1`:
  - hedges: might, may, could, possibly, possible, perhaps, probably, likely, unlikely,
    apparently, seemingly, seem, seems, appear, appears, suggest, suggests, reportedly, allegedly,
    supposedly, roughly, approximately, estimated, generally, typically, usually, often,
    sometimes, somewhat, unclear, uncertain, unknown, debated, disputed, "i think", "i believe",
    "not sure", "it is possible", "it depends", "some sources", "is thought", "is believed",
    "is said";
  - boosters: definitely, certainly, clearly, undoubtedly, always, never, absolutely, indeed,
    "in fact", obviously, surely, proven, "without doubt", "of course".
- **F7, length and specificity.** `log_length` = ln(1 + *n*). `specificity` = (tokens containing
  a digit + capitalized tokens that do not start a sentence) / the number of whitespace-separated
  tokens of *a*. A sentence starts at the beginning of the text or after a token ending in `.`,
  `!`, or `?`; the token "I" is not counted. 0 when *a* is empty.
- **F8, internal contradiction.** Split *a* into sentences at the regex
  `(?<=[.!?])\s+(?=[A-Z0-9"'(])`. `self_contra` = max over ordered pairs of distinct sentences
  (*s*ᵢ, *s*ⱼ) of P(contradiction | *s*ᵢ, *s*ⱼ); 0 with fewer than 2 sentences.
- **F9, self-consistency (resampling tier only).** *R* = the resamples that succeeded and are not
  empty after cleaning. `sc_embed` = mean over *r* ∈ *R* of cos(*a*, *r*); `sc_entail` = mean of
  P(entailment | premise *r*, hypothesis *a*); `sc_contra` = mean of P(contradiction | *r*, *a*).
  The number |*R*| is recorded. With |*R*| = 0, the question is left out of resampling-tier
  results and counted.
- **F10, determinism.** Features are computed once on CPU in float32 at batch size 1, with library
  and model versions recorded, and cached. A missing value outside the cases above stops the run;
  nothing is imputed.
- **F11, exclusions.** No labeling-rule output (abstention detection, verdict parsing) is a
  feature.

The single-shot tier has 9 features (4 evidence-dependent, 5 evidence-free); the resampling tier
adds 3. Each is a candidate for "best single feature" in its tier and evidence condition.

## Cost and latency

- **Price snapshot:** the Nebius Token Factory public catalog on 2026-10-08, recorded in
  `configs/generation.yaml`: $0.10 per million input tokens and $0.30 per million output tokens
  for both generators. Every dollar figure is API-reported tokens × snapshot price. The Nebius
  bill is reported only as a cross-check. Judge prices get their own dated snapshot at step 8.
- **Token accounting:** the usage the API reports per call. Failed calls are counted and their
  cost is reported as unknown, not estimated. A successful row without usage is counted and the
  cost total is marked incomplete.
- **Incremental cost per verdict** (the spec's definition): the mean over evaluated questions,
  excluding the primary answer. Resampling tier and self-consistency alone: the 5 resample calls.
  LLM judge: the judge call. Single-shot classifier and the single features: $0 in API spend.
  Local compute (retrieval, embedding, NLI, classifier) is reported in seconds on the stated
  hardware and is not converted to dollars. The cost of the primary answer is reported for
  context.
- **Concurrency:** bulk generation uses 8 concurrent requests. It is stated next to every
  generation latency.
- **Latency boundaries.** From the moment the primary answer text is available to the decision
  (verdict, probability, and band). This includes retrieval, embedding, NLI, feature computation,
  the classifier, calibration, and the policy. The resampling tier adds resampling wall-clock
  time. For the judge, it includes retrieval, because the judge gets the same evidence, and ends
  when the judge's verdict is parsed. Primary generation and model loading are excluded; models
  are warm.
- **Local latency** is timed per question at batch size 1 on CPU, warm, for every evaluated
  question, and reported as the median and 95th percentile with the hardware (Apple M5, 16 GB).
- **Resampling cost and wall-clock are reported separately.** Cost = the sum over the 5 calls.
  Parallel-equivalent latency = the maximum of the 5 per-call latencies; sequential latency =
  their sum. Both are derived from per-call timings in the bulk run at concurrency 8, not from a
  separate parallel benchmark.

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
  - all primary tests with abstentions excluded;
  - all primary tests with truncated primary answers excluded.
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
2. ~~**Feature formulas**~~: resolved; see "Text normalization" and "Features".
3. ~~**Retrieval recall**~~: resolved; see "Retrieval and feature models", "Retrieval recall".
4. ~~**Generation failures**~~: resolved; see "Generation", "Storage, retries, and failures".
5. ~~**Cost and latency measurement**~~: resolved; see "Cost and latency".

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
- 2026-10-08: recorded text normalization, feature formulas, retrieval recall, generation storage
  and failure rules, the budget guard, and cost and latency measurement; added the
  truncation-excluded secondary analysis.

## Amendment log

(empty)
