# Classical ML Evaluator for Hallucination Detection

Oct 6, 2026 · @BAG

## Research questions

This build is driven by two questions, not a feature wishlist:

1. Can a classifier trained on heterogeneous, black-box textual signals (semantic similarity, NLI entailment, self-consistency, lexical overlap, hedging language) outperform any single hallucination-detection heuristic used alone?
2. Does that classifier generalize to generating models and datasets it was never trained on — or does it just memorize quirks of one model/dataset pairing?

Everything below exists to answer these two questions as cleanly as possible. Framing: this is not "build a better hallucination detector" — it's testing whether a cheap, transparent, model-agnostic evaluator can capture most of the useful signal of an expensive LLM judge, at a fraction of the cost and latency, with the honesty to report where it falls short.

## What we're building

A single-shot, model-agnostic classifier — in the spirit of GEV's single-shot classifier — that takes a question, a generated answer, and optionally retrieved evidence, and outputs one of three decisions: accept, flag for review, or escalate to a full LLM judge. Think of it as a spam filter for hallucinated LLM output: cheap and fast enough to run on everything, with escalation reserved for the genuinely ambiguous cases.

It is model-agnostic by construction — it never touches the generating model's logits, attention, or weights, only the text it produced — so the same classifier works whether the answer came from GPT-4, Claude, Llama, or anything else. It is "single-shot" in that one forward pass through the classifier produces a verdict, with no multi-turn reasoning chain like an LLM judge uses.

Two tiers, defined precisely. The single-shot tier: the verdict depends only on the question, the one answer being judged, and optionally an evidence passage, with no further access to the generating model. The resampling tier: adds self-consistency features, which require re-querying the generating model N times (fix N at 5), so it needs query access to the generator but still never touches logits or weights. Every result in this project is reported for both tiers separately, and the model-agnostic claim applies in full only to the single-shot tier.

Critically, it does not output a bare probability. Each verdict comes with the feature values that drove it, so a flagged answer can be explained in one sentence — e.g. "low NLI entailment with retrieved evidence, high self-contradiction across resamples."

## Hallucination taxonomy (decide this before labeling anything)

LLM judges can catch several distinct failure modes under one "hallucination" label. Pick one of these before writing any labeling or feature code, since it determines which datasets apply and how the classifier's output is structured:

- Factually false — the answer states something untrue, checkable against evidence
- Unsupported — the answer isn't contradicted by evidence, but evidence doesn't back it either
- Self-contradictory — the answer conflicts with itself, or conflicts across resampled generations
- Combined — a multi-label setup where an answer can be more than one of the above at once

Recommendation: start with a single combined "hallucinated / not hallucinated" binary label for research question one, then if time allows, break it into the three sub-types as a multi-label extension — this keeps the first pass simple while leaving room to show the taxonomy was considered deliberately, not skipped.

## Feature set (black-box, model-agnostic)

Every feature below is computed from text in/text out plus optional retrieved evidence — never from the generating model's internals, which is what keeps the system usable on any model, open or closed.

| Feature | What it captures |
| --- | --- |
| Semantic similarity | Embedding-space closeness between the answer and retrieved evidence passages |
| NLI entailment | Whether the evidence actually entails the answer, contradicts it, or is neutral — catches fluent answers that sound right but aren't supported |
| Self-consistency | Agreement across multiple sampled generations for the same prompt (same model, resampled at nonzero temperature) — low agreement is a hallucination signal |
| Lexical overlap | N-gram / token overlap between answer and evidence |
| Hedging and confidence language | Frequency of hedge words ("might", "possibly") vs. confident assertions |
| Answer length and specificity | Overly short or overly vague answers correlate with ungrounded claims |
| Contradiction detection | Whether the answer contradicts itself internally or across resamples |

Each feature alone is a weak signal with known failure modes; the classifier's value is in learning how they combine.

## Cost/latency experiment: don't assume the comparison, measure it

The naive framing — one cheap classifier call versus one LLM judge call — is wrong once self-consistency is in the feature set. Self-consistency needs N resamples of the model being evaluated, so the real comparison is N generations plus embedding plus NLI plus classifier, versus one judge call. That could still win on cost and latency, but it has to be measured, not assumed.

Run this as an explicit tiered experiment, each row evaluated on the same held-out examples:

| Evaluator | Generations needed | Cost | Latency | F1 |
| --- | --- | --- | --- | --- |
| Semantic similarity alone | 1 | $ | low |  |
| NLI alone | 1 | $ | low |  |
| Self-consistency alone | 5 | $$ | low (parallel) |  |
| Classical ML, no self-consistency | 1 | $ | low |  |
| Classical ML + self-consistency | 5 | $$ | low (parallel) |  |
| LLM judge (small open-weight) | 1 | $$ | medium |  |
| LLM judge (frontier-class) | 1 | $$$ | medium–high |  |

A few things that make this rigorous rather than just a nice table:

Measure *incremental* cost over the original answer, not total cost. The first generation happens regardless of evaluation, so what matters is the extra resamples plus embedding/NLI compute versus the judge's one extra call.

Report cost and latency separately for the resampling rows. Resamples can run in parallel, so latency barely grows while cost multiplies — that gap is itself a finding worth stating explicitly.

Use two judge baselines if possible: a small open-weight judge on Nebius and a frontier-class judge. Which one wins the cost/accuracy tradeoff can flip depending on which judge it's compared against, and that's exactly the kind of nuance that makes the writeup credible.

Plot F1 against cost-per-verdict as a Pareto chart once the table is filled in — it's the single figure that makes the whole cost argument visually, and it doubles as a strong demo/resume visual.

This table also settles the earlier open question about whether self-consistency breaks the black-box, model-agnostic claim. It does, partially: a classifier using self-consistency needs query access to resample the generating model, not just a single text pair, while similarity, NLI, lexical overlap, and hedging stay strictly single-shot and text-only. Treat it as two tiers — a pure single-shot tier and a resampling tier — and let the table show what the extra tier buys in F1 against what it costs in generations.

## Data sources, split for generalization testing

Research question two requires training and test data to come from different models and datasets, not just a random split of one pile. Structure it like this:

| Source | Role |
| --- | --- |
| TruthfulQA, generated with Model A | Training split — feature engineering and classifier fitting |
| HaluEval, generated with Model A | In-distribution validation — same model, different dataset |
| TruthfulQA or FEVER, generated with Model B (a different open-weight model) | Out-of-distribution test — same dataset family, unseen model |
| FEVER, generated with Model B | Fully held-out test — unseen model AND unseen dataset |
| Wikipedia / Wiki-DPR | Free evidence/grounding passages for the retrieval-based features, used across all splits |

The gap between in-distribution and out-of-distribution performance is the actual answer to research question two, so report it explicitly and don't bury it. One correction to an earlier claim: pre-labeled benchmarks do not remove the labeling burden. Their labels describe the benchmark's own questions, claims, or pre-written answers, not whatever Model A or Model B generates, so labeling the generated answers is real work, covered under Experimental rigor below. Also split by question, never by row.

## Compute plan (student budget)

The heavy part is generating model outputs and embeddings — training the classifier itself is cheap and runs on a laptop CPU in minutes, since it's logistic regression or gradient-boosted trees over a small feature vector, not a neural network.

- Generating model: an open-weight model such as Llama 3 8B or Qwen2.5 7B, run via Nebius AI Cloud using existing credits. Open-weight models also keep the door open later to compare against logit-based features, even though the core system doesn't depend on them.
- Embeddings / semantic similarity: a small open-source sentence embedding model (e.g. something in the sentence-transformers family), cheap enough to run on Nebius or even CPU-only.
- NLI entailment: a small open-source NLI model, same story — lightweight enough not to need serious GPU time.
- Self-consistency: just resampling the same generating model multiple times at nonzero temperature — no extra infrastructure, just extra API calls against the Nebius-hosted model.
- Classifier training: scikit-learn, CPU-only, free.

Net effect: the only real compute cost is inference calls against the Nebius-hosted model for generation and resampling — everything downstream is lightweight.

## End-to-end pipeline

&#91;embedded content: pipeline · 5 stages, data to classifier to comparison\]

Each stage hands off a concrete artifact to the next: labeled questions become generated answers, answers become feature vectors, feature vectors become a trained classifier, and the classifier's verdicts get compared against an LLM-as-judge baseline on the same held-out examples to see where each approach wins.

## What we're evaluating, and how

Five axes, measured on the same held-out examples for both the classifier and an LLM-as-judge baseline. This table is the project's actual result — fill it in once experiments run:

| Axis | What it measures | How to measure it |
| --- | --- | --- |
| Accuracy | Does the verdict match the ground-truth label | Precision/recall/F1 on in-distribution and out-of-distribution test splits separately (this separation is non-negotiable — it's the answer to research question two) |
| Calibration | Does a 0.9 "accept" confidence mean accept is right 90% of the time | Reliability diagram / expected calibration error |
| Latency | Time from input to verdict | Wall-clock per example, classifier vs. LLM judge |
| Cost | API/compute spend per verdict | Token or inference cost per example, classifier vs. LLM judge |
| Interpretability | Can the verdict be explained in one sentence from feature values | Qualitative — feature importances plus a handful of worked examples showing the explanation alongside the verdict |

The headline result is research question one: does the trained classifier (combining all features) beat the single best individual heuristic used alone, on accuracy, at a fraction of the cost and latency of an LLM judge? The second headline is research question two: how much does accuracy drop from in-distribution to out-of-distribution, and is that drop small enough for the classifier to be trustworthy on models/datasets it never saw.

## Experimental rigor: make the claims hard to falsify

The goal is not a bigger project. It is claims that survive attack. Almost everything below runs on CPU over features that already exist, so it costs time rather than compute credits. Nothing in this section adds a dataset, framework, or feature.

### Write down what counts as yes or no, before running anything

Research question one is a yes only if the classifier beats the best single feature on F1, with a paired-bootstrap 95% interval on the difference that excludes zero. Research question two is a yes only if the drop from in-distribution to out-of-distribution F1 stays within a margin chosen now (for example 5 points) and out-of-distribution performance clears a trivial baseline. Record these criteria in the repo with a date before the first test-set run. A negative result against pre-set criteria is still a legitimate headline for this project.

### Label construction (construct validity)

Benchmark labels describe the benchmark's own questions, claims, or pre-written answers, not what Model A or Model B actually generates, so labeling generated answers is real work. Label each generated answer by comparing it to the dataset's reference answers or gold evidence, using a written definition of 'hallucinated' for each dataset. Keep the LLM judge out of the labeling path entirely, and do not label with the same NLI model that is also a feature, or the NLI feature will look better than it is and the judge comparison becomes circular. Hand-audit about 200 labeled examples and report chance-corrected agreement (Cohen's kappa) between automatic and hand labels. TruthfulQA (imitative falsehoods), HaluEval (pre-built hallucination examples), and FEVER (claim verification rather than free-form generation) measure related but different things, so report results per dataset as well as pooled.

### Leakage controls

- Split by question, never by row: every resample and variant of a question stays in one split, and near-duplicate questions are deduplicated across splits.
- Fit all preprocessing, calibration, and thresholds on training or validation data only. Each test split is touched once per reported result.
- Keep gold evidence out of any feature that effectively reveals the label.
- These benchmarks are probably in the pretraining data of both the generators and the judges. State this as a limitation in the writeup, not a footnote.

### Evidence conditions

Report every result under three conditions: gold evidence (an oracle upper bound), retrieved evidence from Wiki-DPR (the realistic case), and no evidence. Measure retrieval recall separately so retrieval failures are not blamed on the classifier. Group the features into evidence-dependent (similarity, NLI, lexical overlap) and evidence-free (hedging, length, self-consistency). The no-evidence condition shows how much the evidence-free group carries alone, which is what a deployed support bot without a knowledge base would actually get.

### Ablations and feature redundancy

Run these as one analysis: leave-one-feature-out, leave-one-group-out, add-one-at-a-time from a bare baseline, and each single feature alone as a baseline. Add a feature correlation matrix. Redundant features make coefficient-based importances unstable, so use permutation importance for explanations, and check that explanations for the same example stay stable across seeds before showing them on the dashboard.

### LLM-judge fairness

Give the judge the same inputs the classifier gets, including evidence. Tune its prompt on the validation split with effort comparable to what goes into tuning the classifier, then freeze it. Use temperature zero, a fixed rubric, and a judge from a different model family than the generator to avoid self-preference. Report its parse-failure rate and, on a subset, its run-to-run variation.

### Uncertainty

Report bootstrap 95% confidence intervals over test questions (not rows) for every headline metric, and a paired bootstrap for every classifier-versus-baseline comparison. Repeat classifier training and splits across several seeds and report the spread, not the best run. Repeat generation runs only on a subset, since that is where credits go.

### Failure analysis

Hand-sort about 50 false positives and 50 false negatives into named categories (for example missing evidence, correct paraphrase, numeric error, hedged but correct) and report counts per category and per model. Include a few worked examples next to their explanations. This is also the section interviewers will ask about.

### Adversarial tests (minimal)

Perturb existing test examples programmatically instead of collecting new data: strip or insert hedging, paraphrase the answer, pad or truncate length, swap an entity. Report how often the verdict flips. Hedging and length are the easiest features to game, so this shows how brittle the signal is.

### Scope guardrails

Do not add frameworks, features, or datasets unless one is required to answer a research question. The dashboard is built last and kept to v1.

## Decision policy: keep the model binary, tune the policy separately

The classifier itself should stay strictly binary — hallucinated or not — rather than trying to train it directly against three output classes. Accept/flag/escalate is a decision policy built on top of that binary probability, not a property of the model. Keeping them separate makes both easier to get right.

First, calibrate the raw probability on a held-out split (Platt scaling or isotonic regression) before any thresholds are chosen — an uncalibrated 0.6 doesn't reliably mean a 60% hallucination rate, so thresholds set against raw scores are arbitrary by construction.

Second, set thresholds from an operating-point goal, not a guess. Pick something like "no more than 5% of accepted answers may be hallucinations" and "escalate at most 20% of traffic," then solve for the thresholds that hit those targets on the validation split — never the test split, which stays untouched for the final report.

Third, the two thresholds split calibrated P(hallucination) into three bands, and the direction matters for the story: low probability accepts, the middle band escalates to the LLM judge or a human as the genuinely ambiguous case, and high probability flags as a confident catch — the classifier is sure enough to act on its own, with the explanation attached. That keeps the three-way split coherent with the spam-filter framing: the net resolves both obvious ends on its own and only forwards the ambiguous middle.

Fourth, report what happens to the operating point out of distribution. Thresholds tuned on Model A's validation split may drift when applied to Model B's held-out data — showing that drift (or the lack of it) feeds directly into research question two.

On the dashboard, expose the two thresholds as live sliders rather than fixed constants. Moving them visibly reshapes the accept/flag/escalate split in real time, and a running cost ticker — "judge-on-everything would cost $X, the net cost $Y and escalated only 18%" — turns the tradeoff into one number that lands fast with an interviewer, stronger than reciting an F1 score.

## Architecture: Python core, Java Spring Boot + Kafka around it

The ML stays in Python — fighting scikit-learn, the NLI model, and embeddings in Java costs far more than it's worth. The engineering around it is where Spring Boot and Kafka earn their place, and it's a legitimate architectural need here, not a bolt-on: simulating a live stream of incoming questions is exactly the kind of problem Kafka exists for.

&#91;embedded content: architecture · Python ML core, Java Spring Boot + Kafka around it\]

Four pieces: a Kafka producer (Spring Boot) reads from a public dataset and publishes one question at a time onto a questions topic, with a small randomized delay so it reads as live traffic rather than a batch dump. A Python scoring service consumes that topic, calls the chatbot on Nebius for an answer, runs the classifier, and publishes the verdict — accept, flag, or escalate, plus its one-line explanation — onto a verdicts topic. A Spring Boot dashboard consumes the verdicts topic and pushes updates to the browser as they arrive, so questions and verdicts appear to stream in live.

Why this split: the producer and dashboard are a real Spring Boot story — REST, WebSocket or SSE push to the browser, Kafka consumer/producer wiring — and the Python service is a thin wrapper around the actual ML, called over a simple internal API. Nothing here is faked to justify the tech stack; each piece is doing a job it would do in a real deployment.

Versioned events and observability: every Kafka message carries a schema version, a question id, the generating model version, the tier (single-shot or resampling), the evidence condition, the classifier, calibration, and threshold versions, and a timestamp for each stage (produced, answered, scored, displayed). The same events feed the latency and cost numbers in the evaluation, so the demo and the experiment measure things the same way, and any run can be replayed. Skip a full observability stack; structured JSON logs and these fields are enough.

## Live demo plan

Data source: a public customer-support/FAQ dataset (e.g. the Bitext customer-support intent dataset) or a handful of scraped real product FAQ pages — cite whichever is used, since it's a real, attributable source rather than synthetic data.

Demo flow: the producer streams questions from that dataset through a chatbot on Nebius, the scoring service classifies each answer, and the dashboard shows live rows of question, answer, and verdict as they land. Deliberately seed a handful of questions known to be tricky — ones with subtle factual traps or ones the chatbot is likely to answer confidently but wrong — so the audience watches the classifier actually catch a hallucination in real time, not just a reported accuracy number.

This is the demo of the "cheap net" in action: most questions flow through as quick accepts, a few get flagged, and the seeded trap questions are the moment that sells the whole project.

Be honest about what the demo is. The public support dataset has no ground truth for hallucination, so the demo illustrates the system and is not evidence for either research question. The seeded trap questions do have known answers, so you can report a catch rate on them, but present it as an illustration. The research results come from the labeled benchmark splits.

## Dashboard UI: layout and visual spec

The dashboard is the centerpiece of the live demo, so it should read clearly from across a room, not just up close on a laptop. Keep it to one screen, no scrolling during the demo.

Scope: build v1 first, which is the live feed with verdict pills plus the cost ticker. The threshold sliders, detail panel, and summary strip below are only worth building after the experimental rigor work is done. A modest dashboard on top of defensible results beats an elaborate one on top of weak claims.

Layout, top to bottom:

Header bar: project name, a live connection indicator (Kafka consumer connected/disconnected), and the two threshold sliders (accept/flag boundary, flag/escalate boundary) with their current calibrated P(hallucination) cutoffs shown as numbers next to each slider.

Cost ticker, directly under the header: three running numbers side by side — total verdicts processed, what judge-on-everything would have cost, what the net actually cost — updating live as events stream in. This is the single number that should land with a viewer in the first five seconds.

Live feed, the largest area, center: a reverse-chronological table or card list, one row per question as it arrives — question text, a truncated answer, the verdict as a colored pill (green accept, amber flag, red escalate), and a one-line feature-based explanation. New rows animate in at the top so the "live" feeling is obvious.

Detail panel, right side or a click-to-expand row: when a row is selected, show the full answer, the evidence passage it was checked against, and a small bar chart of the feature values that drove the verdict — this is where the interpretability story becomes visible rather than asserted.

Summary strip, bottom: running counts of accept/flag/escalate as both numbers and a thin horizontal stacked bar, so the proportions are visible at a glance over the session.

Visual style: a dark background reads best for a projector or screen share and makes the colored verdict pills pop. Keep the palette to three accent colors tied consistently to the three verdicts, neutral grays everywhere else, and one legible monospace or clean sans-serif font throughout — this is a technical demo, not a consumer product, so restraint reads as more credible than polish for its own sake. Avoid gratuitous animation beyond the row entrance and the live counters ticking up.

## Pointers for building this in Cursor

Give Cursor the shape before the code. Paste the layout description above as a single prompt and ask it to scaffold the component tree first — header, cost ticker, live feed, detail panel, summary strip — as empty components with props defined, before filling in any logic. Reviewing the skeleton first catches structural misunderstandings while they're still cheap to fix.

Be explicit about the stack in the first prompt: Spring Boot backend exposing a WebSocket or SSE endpoint that the frontend subscribes to, and name the frontend stack directly — plain Spring Boot server-rendered templates with a small JS layer is the fastest path if frontend time is limited; a separate React app is worth it only if comfortable maintaining two codebases during a live demo.

Feed Cursor one real example event (a full JSON payload: question, answer, verdict, feature values) rather than describing the shape in prose — concrete examples produce much more accurate component code than abstract field descriptions.

Build and verify in this order: static layout with hardcoded fake data first, confirm it looks right, then wire the WebSocket/SSE connection against a mock event stream, then finally connect it to the real Kafka consumer. Each stage is independently demoable, which matters since live demos fail in layers, not all at once.

Ask Cursor to keep the verdict-to-color mapping and threshold values in one small config object, not scattered through components — that's what lets the sliders actually drive the UI live during the demo instead of requiring a rebuild.

## Build order

1. Write down the success criteria for both research questions, freeze the taxonomy (binary label), and freeze the single-shot versus resampling tier definitions.
2. Pull TruthfulQA, HaluEval, and FEVER; deduplicate, then split by question into train, in-distribution, out-of-distribution, and fully held-out sets.
3. Define the labeling procedure from reference answers or gold evidence (no LLM judge, no feature NLI model in the labeling path) and hand-audit about 200 examples.
4. Stand up Model A and Model B on Nebius; generate answers for every split, and resample N=5 for the resampling tier.
5. Build the Python feature extraction pipeline under three evidence conditions (gold, retrieved, none), logging cost and latency per stage.
6. Train the binary classifier (logistic regression and gradient-boosted trees) on the training split only; run ablations, single-feature baselines, and the redundancy analysis.
7. Calibrate on validation, set thresholds from operating-point goals, and build the explanation generator.
8. Build the LLM-judge baselines (small open-weight and frontier-class) under the fairness protocol.
9. Run the full evaluation: the axes table and the cost/latency experiment, with bootstrap confidence intervals, in-distribution and out-of-distribution separately.
10. Do the failure analysis (about 50 false positives and 50 false negatives) and the adversarial perturbation tests.
11. Write up both research-question answers against the pre-set criteria, including any negative results.
12. Wrap the scoring step as a Python service with the versioned event schema; build the Spring Boot producer and the Kafka wiring.
13. Build dashboard v1 (live feed and cost ticker), add sliders and the detail panel only if time remains, seed the trap questions, record the demo, and add it to the writeup.
