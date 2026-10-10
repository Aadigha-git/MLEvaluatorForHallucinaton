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

- **2026-10-09, corpus description and index version (no protocol change).** "Retrieval and
  feature models" describes the corpus as 21,015,300 passages, a figure from the Hugging Face
  `facebook/wiki_dpr` dataset card. The pre-registered retriever, Pyserini's prebuilt
  `wikipedia-dpr-100w` index, holds the full DPR 100-word Wikipedia split: **21,015,324**
  passages, confirmed by Pyserini's document count after download. The index used is
  `lucene-inverted.wikipedia-dpr-100w.20260508.deb4c7b.tar` (Lucene 10.4.0, 11,078,635,520
  bytes, MD5 `1ef94a97f2ac418577d1e6a9ecf44806`), the catalog entry pinned by Pyserini 2.4.0,
  searched with Anserini 2.3.0 on Temurin JDK 21.0.12.1. Retriever, index name, BM25 parameters,
  k, and query are unchanged.
- **2026-10-09, implementation notes for `features-v1` (no definition change).** Details the text
  above leaves open, fixed in code before any feature was computed:
  - F7 `specificity`: a token can count in both terms (digit and capitalized); leading
    punctuation such as quotes or brackets is ignored when testing capitalization; only the
    exact token "I" is excluded.
  - HaluEval knowledge-sentence recall splits the knowledge passage with the F8 sentence regex.
  - Retrieved passage titles are taken from the first line of each Wiki-DPR document; passage
    text is used as stored, including the corpus's doubled quotation marks.
  - Self-consistency, NLI, and embeddings run with `torch.set_num_threads(8)` and deterministic
    algorithms; the settings are in `configs/features.yaml`.
- **2026-10-09, labeling workflow and Step 6 analysis details (no protocol change).** Recorded
  before any classifier was fitted on real labels, and before any label of an unresolved answer
  was known.
  - Labels. The automatic rules (version 1) label every primary answer once and store the result
    under `data/labels/auto/`. The test splits are labeled only through `ml/evaluation/`, and their
    labels are not printed or inspected. The 200-answer audit sample is drawn from all five splits
    by `ml/evaluation/export_audit_sample.py`. Each split's unresolved answers go to a blind queue,
    starting with `train` and `val_id`. In that queue, `undecidable` marks an answer the hand
    labeler cannot decide. Such answers are excluded and counted. Training refuses to start while
    any unresolved `train` or `val_id` answer lacks a hand decision.
  - Ties beyond those already fixed. Mean cross-validated AP is compared after rounding to 12
    decimals. For logistic regression, a tie in `C` goes to `class_weight=None`. For boosted trees,
    remaining ties go to the smaller `learning_rate`. A tie in mean `val_id` F1 between the two
    families goes to logistic regression.
  - Single features. If both directions reach the same F1, "higher means hallucinated" wins. AUROC
    and AUPRC use the score in the chosen direction.
  - Validation reporting. F1, AUROC, and AUPRC are reported on `val_id` for each seed, with the mean,
    minimum, maximum, and standard deviation over seeds. F1 uses the threshold chosen on `val_id`
    under the frozen rule, so it is optimistic in the same way for every evaluator. Uncertainty
    comes from the frozen cluster bootstrap over `val_id` (10,000 replicates, paired, seed-averaged
    within each replicate) with thresholds held fixed.
  - No imputer is fitted, because `features-v1` has no missing values and F10 forbids imputation.
    Standardization is fitted inside each cross-validation training fold, then refitted on all of
    `train`.
  - Ablations (exploratory) use the selected family for each tier and training condition.
    - Every feature subset gets its own cross-validated hyperparameters per seed on `train`.
      Results are `val_id` metrics averaged over the three seeds.
    - The bare baseline is intercept-only, so `val_id` AUROC 0.5, AUPRC equal to prevalence, and
      F1 equal to the all-hallucinated F1.
    - Add-one-at-a-time is greedy forward selection. At each step it adds the feature with the
      highest mean `train` cross-validated AP over seeds, with ties going to feature order;
      `val_id` never chooses a feature.
    - Leave-one-group-out uses three groups: evidence-dependent (F2–F5), evidence-free single-shot
      (F6–F8), and self-consistency (F9). Each group is also reported alone.
    - Gold-evidence ablations apply the retrieved-evidence subset models to gold features.
  - Permutation importance is computed on `val_id` for each seed's selected classifier: 30 repeats,
    `random_state` set to the seed, scored by AP. Stability across seeds is reported as pairwise
    Spearman correlation of importances and the Jaccard overlap of the top three features.
  - Redundancy. Pearson and Spearman correlation matrices are computed on `train` and on `val_id`,
    per evidence condition. VIF is the diagonal of the inverse Pearson correlation matrix of each
    tier's features.
  - Datasets. `train` is TruthfulQA only and `val_id` is HaluEval only, so per-dataset results equal
    per-split results. Evidence-condition effects are compared within `val_id`.
- **2026-10-09, labeling audit failed the TruthfulQA gate; rule version 2 (procedure under
  "Labeling commitments").** Frozen and committed before any version 2 label, sample, or hand
  label existed.
  - **Version 1 audit result.** 200 answers, every one hand-labeled. Cohen's kappa: overall 0.7447,
    FEVER 0.7690, HaluEval 0.7117, TruthfulQA 0.5768. The gate requires at least 0.6 overall and
    per dataset, so it failed on TruthfulQA. The TruthfulQA estimate rests on 34 answers, 25 from
    `train` and 9 from `test_ood`. The version 1 sample, its key, the hand labels, and the
    version 1 automatic labels are kept unchanged for the record. No hand label was edited after
    scoring.
  - **Evidence used for the revision.** Only the 25 TruthfulQA `train` audit items. Their
    automatic labels were read from the `train` label file, not from the audit key, so no
    test-split label was loaded. The 63 HaluEval `val_id` items were not needed, because only the
    TruthfulQA rule changes. Not used: the 9 `test_ood`, 32 `test_id`, and 71 FEVER audit items.
  - **What went wrong.** Version 1 agreed with the hand label on 19 of the 25 `train` items
    (kappa 0.41). The six disagreements had four causes:
    - negation checked over the whole answer (2);
    - references left with a single answer-specific word after the question's words are removed (2);
    - an answer that mentions an incorrect claim while correcting it (1);
    - a hand label that departs from TruthfulQA's listed incorrect answer (1). This one is outside
      any rule's reach and was left as recorded.
  - **Change: TruthfulQA matching only** (LABELING.md, "Version 2 matching"). Scoring and the
    negation check now work per answer sentence. A reference with fewer than 2 answer-specific
    words cannot decide a label, but a full match on one blocks the opposite label. Unchanged:
    `min_match` and `margin`; the HaluEval and FEVER rules; the label definitions; the shared text
    utilities that also define `features-v1` tokens; features; answers; splits.
  - **Rejected draft.** Ignoring short references outright was rejected. An answer asserting both a
    correct reference and a one-word incorrect one (for example, "a French folk song composed by
    Mozart") would then be labeled not hallucinated, against the definition.
  - **In-sample agreement on the 25 `train` items.** Version 1 resolved 25 and agreed on 19 (kappa
    0.41). Version 2 resolves 15, agrees on 13, and reaches kappa **0.595**, below the gate even on
    its own development items. It was frozen on the project owner's decision, and the fresh audit
    decides.
  - **Effect on development labels.**
    - `train` goes from 109 hallucinated, 237 not hallucinated, and 229 unresolved to 79, 166, and
      330.
    - 156 labels change: 127 resolved answers become unresolved, 26 unresolved answers become
      resolved, and 3 change from not hallucinated to hallucinated.
    - `val_id` (HaluEval) is unchanged.
    - Changes on the test splits are not counted or reported before Step 9.
  - **Fresh audit.**
    - 200 automatically resolved answers drawn from the version 2 labels of all five splits, with
      the same stratification, seed 20261011, and every answer from the version 1 sample excluded.
      It is drawn once, by `ml/evaluation/export_audit_sample.py` (`AUDIT_ROUNDS`).
    - It is scored by `ml/evaluation/score_audit.py` against the same gate: kappa of at least 0.6
      overall and per dataset, on the point estimates.
    - The auditor instructions now restate the existing TruthfulQA definition: listed incorrect
      answers count as hallucinated. This restates the definition; it does not change it.
    - **If the fresh audit fails for any dataset, work stops for a decision.** No further revision
      is pre-specified.
  - **Disclosures.**
    - The auditor has seen every answer in the version 1 sample, including the test-split items.
      None informed the revision, and none can enter the fresh sample.
    - Changing `configs/default.yaml` changes the config hash recorded on labeled rows. The split
      assignments and manifest are unchanged; the split config hash `30cf64f436c0` above refers to
      the configuration at split creation.
    - Labels are stored per rule version under `data/labels/v{version}/`. Training reads version 2
      labels only and checks the version. The version 1 dev unresolved queue, which had no hand
      labels, is superseded by the version 2 queue.
- **2026-10-09, fresh audit under rule version 2 failed the HaluEval gate; recorded as a
  limitation.**
  - **Result.** 200 answers, every one hand-labeled. Cohen's kappa, with 95% bootstrap intervals:
    - overall 0.667 (0.560–0.767), raw agreement 0.84;
    - FEVER 0.666 (0.401–0.872), 74 answers;
    - HaluEval **0.578** (0.415–0.734), 100 answers;
    - TruthfulQA 0.641 (0.273–0.913), 26 answers.

    The gate fails on HaluEval. TruthfulQA, the dataset whose rule changed, passes. The HaluEval
    rule is identical under versions 1 and 2. It scored 0.7117 on the first sample and 0.578 on
    this one, and the two intervals overlap.
  - **Diagnosis.** Only `val_id` audit items were used, matched by answer ID against the `val_id`
    label file: 63 from the first round and 78 from the second. No `test_id` audit row was read.
    - **First round, 8 disagreements.** In 6, the answer gives the reference answer in a form the
      rule does not match: plurals, a shorter form of a full name, or different punctuation or
      wording. These were labeled hallucinated. In the other 2, the answer contains the reference
      answer but adds a false detail. These were labeled not hallucinated; this is the limitation
      already documented under "HaluEval" in LABELING.md.
    - **Second round, 13 disagreements**, all labeled hallucinated by the rule and not
      hallucinated by hand:
      - 3 are rule misses of the same kind;
      - 4 use a near-equivalent or approximate term in place of the reference answer;
      - 6 give a different entity, value, or yes/no verdict than the reference answer.
    - Under the written definition ("the answer to the question asked is not the reference
      answer"), those last six are hallucinated. A rule revision would address at most 3 of the 13.
  - **Decision.** The project owner chose to record this failure as a limitation rather than
    revise further:
    - no version 3 rule;
    - no change to the auditor instructions;
    - no third audit round;
    - no hand label edited after scoring.

    Both audit rounds, their keys, and their hand labels are kept as recorded. No kappa was
    recomputed under alternative hand labels.
  - **Use of the labels.** HaluEval labels from rule version 2 are used as they are, for `val_id`
    and `test_id`. Every report that shows HaluEval results states that the HaluEval agreement gate
    failed and gives both rounds' kappas (0.7117 and 0.578). Nothing else changes.
  - **Considered and dropped: a sensitivity analysis.** It would have re-reported HaluEval results
    without the answers labeled hallucinated by the "asserts a different answer" branch. That
    branch produces 592 of the 597 automatic hallucinated labels in `val_id`, so removing those
    answers would leave almost no hallucinated cases, and the analysis would mostly measure the
    change in class balance. Counts come from the `val_id` label file only; `test_id` was not
    inspected.
- **2026-10-09, Step 7 implementation details (no change to selection, thresholds, or the research
  questions).** These are choices that "Calibration" and "Decision policy" above leave open. The
  project owner recorded them after Step 7 was implemented and run on `train` and `val_id` only,
  and before any test split was scored. Selected families, hyperparameters, and F1 thresholds are
  the Step 6 ones: each classifier is refitted deterministically from its Step 6 family,
  hyperparameters, and seed, and the run stops unless the refit reproduces its Step 6 threshold.
  - **Calibrators.** scikit-learn's `CalibratedClassifierCV` wrapped around the frozen classifier
    (`FrozenEstimator`), so the classifier is never refitted:
    - `method="sigmoid"` for Platt scaling, fitted on the classifier's decision function (the
      logit), not on its predicted probability;
    - `method="isotonic"` for isotonic regression, which is unaffected by that choice because it
      is invariant to monotone transforms.
  - **Calibration folds.** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)` on
    `val_id`, grouped by `cluster_id`, with the seed of the classifier being calibrated. ECE uses 10
    equal-width bins, with P = 1 in the top bin.
  - **Denominators.**
    - Calibration, ECE, policy thresholds, and every achieved rate use the 995 labeled `val_id`
      answers.
    - The 5 `val_id` answers the hand labeler marked undecidable (0.5%, below the 5% flag) have
      features but no label, so they are excluded from every rate and every denominator.
    - The accept and escalate goals are compared as exact fractions: at most 5% of accepted
      answers hallucinated, and at most 20% of the 995 answers escalated.
  - **Infeasible accept band.** When no t_low meets the 5% goal, t_low is set to the lowest
    calibrated value, so the accept band is empty. The escalate and flag split is still computed for
    diagnostic purposes, and the policy is marked infeasible. Its constraints are not claimed to be
    met.
  - **Seeds.** No single deployment seed is selected. All three seeds keep their own calibrator and
    policy, and both are evaluated with seed-level variation reported.
  - **Gold evidence.** No calibration or operating-point policy is defined for the gold-evidence
    condition. It stays exploratory and uses the uncalibrated scores and F1 thresholds only.
  - **Explanation generator** (docs/SPEC.md, build step 7). Deferred until a separate
    specification is written and recorded here; nothing in the research questions depends on it.
  - **Artifacts.**
    - One file per tier, evidence condition, and seed under `data/models/step7/` (git-ignored), each
      holding the classifier and its scaler, the feature order, seed, F1 threshold, calibrator, and
      policy thresholds.
    - `manifest.json` records their SHA-256 hashes, library versions, input and config hashes, and
      the git commit. Artifacts are stamped from a clean commit by rerunning
      `scripts/calibrate.py`, which is deterministic.
    - The test splits will be scored with these frozen artifacts.
  - **Validation-only diagnostics** (reported in `reports/step7/`, and not a research-question
    verdict):
    - none of the four selected classifiers beats predicting every answer hallucinated on `val_id`
      (the lower bound of the paired 95% F1 difference interval is not above 0), and none beats the
      best single feature;
    - every policy is infeasible or nearly degenerate: at most 3 of 995 answers accepted, and flag
      precision equal to the `val_id` prevalence.

    These findings change nothing in the protocol. RQ1 and RQ2 are judged only on the test splits,
    as pre-registered.
  - **Reporting fixes in `reports/step6/`** (no number changed):
    - every result that uses `train` (9.2% of answers excluded) is marked;
    - degenerate and skipped bootstrap replicate counts are given for every interval, including
      differences;
    - identical rows in the two-group leave-one-group-out ablations are labeled as such.
- **2026-10-09, Step 8 LLM-judge configuration (provisional; recorded before any judge call).**
  This resolves the open item "judge model IDs" and fixes the judge protocol that docs/SPEC.md
  ("LLM-judge fairness") leaves open. The settings are in `configs/judges.yaml`, which has
  `status: provisional`. In that state, the only call allowed is one synthetic, non-benchmark smoke
  test per judge (`scripts/judge_smoke.py --confirm`). Benchmark calls on `val_id` refuse to start
  until a later amendment sets `status: frozen`. That amendment needs both smoke tests to pass and
  a final budget cap. No test split is read or judged in Step 8.
  - **Judges.**
    - Small open-weight judge: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` on Nebius Token Factory
      (eu-north1, FP8, key `NEBIUS_API_KEY`). It is NVIDIA's own model, trained from scratch, and
      the same size class as Model A (30B total, 3B active). Settings: temperature 0, top-p 1.0,
      at most 64 output tokens, and reasoning off via `chat_template_kwargs: {enable_thinking:
      false}`.
    - Frontier judge: `gemini-3.1-pro-preview` on the Gemini API (key `GEMINI_API_KEY`), called
      through the native `generateContent` endpoint so that thinking tokens are reported. Settings:
      temperature 1.0, thinking level `low`, and at most 2,048 output tokens, thinking included.
    - Prices are from the dated snapshot `reports/step8/price_snapshot.json` (retrieved
      2026-10-10 UTC, with archived copies and SHA-256 hashes), in USD per million API-reported
      tokens:
      - Nemotron: $0.06 input, $0.24 output;
      - Gemini: $2.00 input and $12.00 output, paid tier, prompts up to 200,000 tokens. Output
        includes thinking tokens, so billed output = `candidatesTokenCount` +
        `thoughtsTokenCount`.
  - **Verified provider constraints** (provider documentation, 2026-10-09):
    - Gemini 3.1 Pro cannot turn thinking off; its lowest level is `low`. Google "strongly
      recommend[s]" temperature 1.0 for Gemini 3 models and warns that lower values may cause
      looping or degraded performance.
    - The Nemotron reasoning switch is documented by NVIDIA but not by Nebius. The smoke test
      checks that no reasoning text is returned. If reasoning cannot be turned off, work stops for
      a decision; there is no silent fallback.
    - Also considered, not chosen: GPT-6 Sol (`gpt-6-sol`) accepts sampling parameters only at
      reasoning effort `none`; GPT-6.1 Sol, GPT-6 Astra, and the Claude 5.5 models do not accept
      temperature 0.
  - **Deviations, approved by the project owner and disclosed in every judge result:**
    - The frontier judge runs at temperature 1.0, not the spec's temperature zero.
    - The frontier judge thinks before answering (level `low`).
    - Gemini and generator Model B (Gemma) both come from Google, and Gemma is built from Gemini
      research. The pre-registration excludes the Qwen and Gemma families by name. Gemini is a
      different family, but self-preference toward Model B's answers (`test_ood`,
      `test_heldout`) cannot be ruled out. This is recorded as a limitation.
    - `gemini-3.1-pro-preview` is a preview model and can be withdrawn with two weeks' notice. If it
      becomes unavailable before Step 9, work stops for a decision; no other model is substituted.
  - **Inputs.** The judge gets the classifier's inputs:
    - the question (the claim for FEVER);
    - the cleaned primary answer;
    - under retrieved evidence, the same top 5 BM25 passages as `"{title}: {text}"`.

    Under no evidence, the evidence section is left out. Each prompt is sent as one user message
    with no system message. No seed is sent, and log probabilities are not requested.
  - **Conditions.** Retrieved and no evidence. Gold evidence is exploratory in this protocol and is
    not run for the judges.
  - **Prompt variants.** Four are pre-specified in `src/hdetect/judging/prompts.py`, with template
    hashes pinned in `configs/judge_prompt_hashes.json`:
    - `judge-v1a`: a one-sentence definition;
    - `judge-v1b`: a full rubric following LABELING.md (only the asked-for answer or verdict is
      judged, abstentions are not hallucinated, a hedged guess is judged on the guess);
    - `judge-v1c`: the rubric plus guidance on using evidence;
    - `judge-v1d`: the rubric plus three invented examples.

    Every variant asks for strict JSON: `{"verdict": "hallucinated"}` or
    `{"verdict": "not_hallucinated"}`.
  - **Parsing.** A verdict is valid only as a JSON object with exactly the key `verdict` and one of
    the two values, optionally inside one Markdown code fence. Every other output is a parse
    failure with a named outcome: `empty`, `not_json`, `wrong_schema`, `invalid_verdict`,
    `blocked` (provider safety or recitation stop), or `reasoning_leak`. Nothing is repaired.
  - **Retries and failures.** These follow "Storage, retries, and failures":
    - Transient API errors (408, 429, 5xx, and transport errors; 409 too for Nebius, as the
      `openai` package retries it) are retried 5 times with backoff inside an attempt, for up to 5
      attempts.
    - Other 4xx errors stop the run.
    - A call whose HTTP request succeeded is final, so malformed output is never re-requested.
  - **Scoring.** Only the 995 labeled `val_id` answers are judged. The 5 undecidable answers are
    never sent to a judge.
    - In the primary F1, an answer with no valid verdict (any parse failure, or an API failure
      after 5 attempts) counts as wrong: a false negative if hallucinated, a false positive if not.
    - Parse-failure and API-failure counts by outcome, and F1 on parsed verdicts only (with its n),
      are reported beside it.
  - **Prompt selection.** For each judge and evidence condition, the winner is the variant with the
    highest hallucinated-class F1 on the 995 answers. Ties, after rounding F1 to 12 decimals, go to
    fewer failures and then to the earlier variant. The choice is written to
    `reports/step8/selected_prompts.json` and committed before any test-split judge call.
  - **Repeatability.** The subset is `configs/judge_repeat_subset.txt`: 200 answer IDs drawn by
    seeded simple random sampling (seed 20261012) from the 995 labeled `val_id` answers. It was
    written before any judge call, with SHA-256
    `e4694d3de242b621bd09914b28e31d4309df968cd5a52291d3a9aa199372484c`.
    - The frozen prompt runs twice more per answer and evidence condition. The tuning run counts
      as run 0.
    - Reported: the share of answers with identical outcomes in all three runs, pairwise
      agreement, and verdict flips. A failure counts as its own outcome.
  - **Planned calls** per judge: 4 variants × 2 conditions × 995 answers = 7,960 tuning calls, plus
    2 runs × 2 conditions × 200 answers = 800 repeat calls, for 8,760 in all. Estimated cost,
    measured on `val_id` prompt sizes (about 1,116 input tokens with retrieved evidence and 333
    without):
    - Nemotron: about $0.40.
    - Gemini input: about $12.70.
    - Gemini output: about $0.12 per 1,000 calls for each 10 output tokens per call. At 100
      thinking tokens and 10 answer tokens per call, Gemini's total is about $24.26, above the
      provisional cap on its own.
  - **Budget guard.** One cap covers every Step 8 call, smoke tests included. It is provisionally
    **$20.00**, and the final cap is set by amendment after the smoke tests measure Gemini's
    thinking tokens.
    - Before each call, the spend so far, plus the upper bounds reserved for calls in flight, plus
      this call's upper bound, must stay within the cap.
    - A call's upper bound is its prompt length in characters as input tokens, plus its maximum
      output tokens, priced at the snapshot.
    - Spend is rebuilt from the stored rows, so the cap holds across runs.
    - A call with unreported usage is charged its upper bound. A failed call is charged nothing,
      and its cost is reported as unknown.
  - **Cost reporting.** API-reported input and output tokens are multiplied by the snapshot prices.
    Cost per verdict is the mean over the 995 judged answers.
  - **Latency.** Per "Cost and latency": retrieval time (retrieved condition only), plus API
    wall-clock including in-call retries, plus verdict parsing.
    - Calls run at concurrency 8, which is stated with every latency.
    - Medians and 95th percentiles are reported for all calls and, separately, for calls whose
      output failed to parse.
  - **Recorded per call** (`data/judges/`, git-ignored):
    - request settings, the model ID returned, the response ID, and the finish reason;
    - input, output, and thinking tokens;
    - the raw output, parse outcome, verdict, cost, latency, timestamps, and attempt number;
    - the prompt's SHA-256, never the prompt itself.
  - **Comparison scope.** In Step 9, the judges enter the pre-declared baseline table and the
    F1-versus-cost Pareto chart. Their costs are not added to the classifier's escalation policy,
    because the protocol defines no combined cascade.
  - **Isolation.** The judge code reads `val_id` only and has no switch for test splits. Tests check
    that it names no test split, and every run records the files it opens under `data/` and fails
    if one belongs to a test split. Judging the test splits needs its own entry point under
    `ml/evaluation/`, added in Step 9.
  - **Audit plan.**
    - Before the first benchmark call:
      - both smoke tests pass;
      - the final cap and `status: frozen` are recorded by amendment and committed from a clean
        tree;
      - the prompt-template, repeat-subset, and judge-config hashes match this entry.
    - After each stage:
      - every planned call is accounted for (successful plus failed equals planned);
      - parse outcomes are tabulated per judge, variant, and condition;
      - cost is recomputed from stored tokens and snapshot prices, and must equal the guard's
        ledger;
      - the data-access table shows no test-split file.
    - `reports/step8/manifest.json` records the output-file hashes, config hashes, spend, and git
      commit.
  - **Implementation choices made here, which the protocol did not specify:**
    - failures are scored as wrong;
    - the prompt is selected per judge and evidence condition, with the tie-break rule above;
    - one surrounding code fence is accepted;
    - no seed is sent;
    - output caps are 64 tokens (Nemotron) and 2,048 tokens including thinking (Gemini);
    - the repeat subset is a simple random sample.
- **2026-10-09, Step 8 smoke tests and final budget cap.** Each judge got one synthetic smoke
  call (invented item, `judge-v1b`, retrieved condition). No benchmark data was read or sent.
  - **Nemotron.** It passed every check:
    - returned model `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`, finish reason `stop`;
    - parsed verdict `not_hallucinated`, which is correct for the invented item;
    - 361 input and 12 output tokens, cost $0.0000245, API latency 0.85 s;
    - no reasoning field and no `<think>` text, so Nebius honoured `enable_thinking: false`.
  - **Gemini.** It passed every check:
    - returned model `gemini-3.1-pro-preview`, finish reason `STOP`;
    - parsed verdict `not_hallucinated`, which is correct;
    - 348 input and 88 output tokens, of which 72 were thinking tokens;
    - cost $0.001752, API latency 3.45 s.
  - Spend after both smoke tests, rebuilt from the stored rows: $0.0018.
  - **Projection.** For the 8,760 planned calls per judge, at the measured 88 Gemini output tokens
    per call, the total is about $22.35 (Gemini $21.94, Nemotron $0.41). With a 25% margin, that
    is about $27.94. The smoke item is short and easy, so real `val_id` items may need more
    thinking: at 150 output tokens per call, the total is about $28.87.
  - **Final cap.** The project owner set the Step 8 cap to **$35.00**. It replaces the provisional
    $20.00 and covers every Step 8 call, smoke tests included. The guard is unchanged. If the cap
    would be exceeded, the run stops cleanly, and any further raise needs its own amendment.
  - `configs/judges.yaml` stays `status: provisional`. Freezing it is a separate amendment,
    committed from a clean tree as the audit plan requires.
- **2026-10-09, Step 8 configuration frozen (recorded before any benchmark judge call).**
  - The project owner approved the six implementation choices listed in the Step 8 configuration
    entry, and the configuration is frozen as specified there. The final cap is **$35.00**.
  - `configs/judges.yaml` now has `status: frozen`, with judge-config hash `485ca71cb4b1` (as
    computed by `hdetect.judging.config`) and file SHA-256
    `46519dd88985e9f7c49e240e04006fc5a3007ef63c288a4639efbf0963d86b79`.
  - Checked before the first benchmark call:
    - all four prompt-template hashes match `configs/judge_prompt_hashes.json`;
    - the repeat subset's SHA-256 matches the value recorded above;
    - both smoke tests passed.
  - Next: prompt tuning on the 995 labeled `val_id` answers, for both judges and both evidence
    conditions. No test split is read or judged.
