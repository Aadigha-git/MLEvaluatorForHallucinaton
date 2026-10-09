# Labeling generated answers

Labeling rule version **2** (`labeling.rule_version` in `configs/default.yaml`). Any change to a
rule or threshold below bumps the version, and every labeled row records the version and the
config hash. Version 2 changes only the TruthfulQA matching rule, after the version 1 audit missed
the TruthfulQA agreement gate (PREREGISTRATION.md, amendment log, 2026-10-09). Version 1 stays
reproducible with `rule_version: "1"`.

## Principles

- Labels come only from the dataset's reference answers, gold label, or gold evidence. No LLM
  judge, no NLI model, and no embedding model is used anywhere in labeling, so neither the judge
  comparison nor the NLI and similarity features are circular. `tests/test_labeling_isolation.py`
  enforces this by checking what `src/hdetect/labeling/` imports.
- Each answer gets one of three outcomes: **hallucinated**, **not hallucinated**, or
  **unresolved**. Unresolved answers are labeled by hand (see below) and the unresolved rate is
  reported per dataset and generator.
- **Abstentions are not hallucinated** in every dataset ("I have no comment", "I don't know",
  "NOT ENOUGH INFO", and similar). A hedged guess ("I'm not sure, but I think it's X") is not an
  abstention and is judged on X. The abstention rate is reported separately, per dataset and
  generator, so a model cannot look good just by declining to answer.
- The labeler's input is the cleaned answer, `hdetect.cleaning.clean_answer` (Markdown emphasis
  and line markers removed, whitespace collapsed; see PREREGISTRATION.md, "Text normalization").
  The raw answer is stored unchanged.
- Text is normalized SQuAD-style before matching: lowercased, punctuation and articles removed.

## TruthfulQA

**Hallucinated:** the answer asserts one of the benchmark's incorrect answers (an imitative
falsehood) rather than a correct one.

Rule: for each reference, compute the share of its answer-specific content words found in the
answer. Answer-specific means words that do not also appear in the question, since references
usually restate it. Stopwords are ignored, and the score is zero if the answer and the reference
disagree on negation ("you digest the seeds" vs "you will not digest the seeds"). Let *c* be the
best score over correct references and *i* the best over incorrect references. Then:

- *c* and *i* both below `min_match` (0.5): **unresolved** (no reference matched).
- *c* − *i* ≥ `margin` (0.2): **not hallucinated**.
- *i* − *c* ≥ `margin`: **hallucinated**.
- otherwise: **unresolved**.

**Version 2 matching** (thresholds unchanged):

- Scores are computed per answer sentence (split after `.`, `!`, or `?` followed by whitespace),
  taking the best sentence. Negation agreement is checked within that sentence, so a negation
  elsewhere in the answer ("nothing dramatic happens") no longer switches a match on or off.
- A reference with fewer than 2 answer-specific content words ("composed by Mozart" for "Who
  composed…?" leaves only "mozart") cannot decide a label: it scores 0 in *c* and *i*. If one
  sentence fully contains such a short reference, it blocks the opposite label instead: a short
  incorrect reference turns **not hallucinated** into **unresolved**, and a short correct
  reference turns **hallucinated** into **unresolved**.

Version 1 scored the whole answer at once and let any reference decide. Measured on the
benchmark's own references, version 1 resolved about 44% of answers and was right about 89% of the
time when it resolved; the misses were mostly comparatives ("cooler" vs "warmer", "more" vs "less")
that word overlap cannot separate. Version 2 resolves fewer answers, so **expect more than half of
TruthfulQA answers in the hand-labeling queue.**

## HaluEval (QA)

**Hallucinated:** the answer to the question asked is not the reference answer.

Rules, in order:

1. Abstention: **not hallucinated**.
2. The answer contains the reference answer (whole-word span match):
   - every mention has a negation within the 3 words before it ("not in Delhi"): **unresolved**;
   - the reference answer is one of the options named in the question (comparison questions),
     and the answer is a longer statement (token F1 < 0.6): **unresolved**, because naming an
     option proves nothing when the question names all of them;
   - otherwise: **not hallucinated**.
3. Yes/no questions: the answer's leading yes/no (or its only yes/no) is compared with the
   reference. No yes/no at all: **unresolved**.
4. Token F1 with the reference ≥ `f1_threshold` (0.6): **not hallucinated**.
5. Token F1 ≥ `unresolved_floor` (0.3): **unresolved**.
6. Otherwise: **hallucinated** (asserts a different answer).

**Known limitation.** The rule judges only the asked-for answer. An answer that gives the right
entity plus false side details ("Figment is the character and the song is One Little Spark") is
labeled not hallucinated. On HaluEval's own pre-written hallucinated answers, 12% are accepted
this way, 10% are unresolved, and 78% are caught. All of its right answers are accepted. The hand
audit checks how often this matters for real generations.

## FEVER

Generators get the claim and are told to start with SUPPORTED, REFUTED, or NOT ENOUGH INFO,
followed by a one-sentence justification (`hdetect.data.fever.PROMPT_TEMPLATE`). Only verifiable
claims (gold SUPPORTS or REFUTES) are used.

**Hallucinated:** the verdict contradicts the gold label.

- The verdict is the one the answer opens with. If it doesn't open with one, the only verdict word
  it mentions is used. If it mentions none, or more than one: **unresolved**.
- NOT ENOUGH INFO: abstention, **not hallucinated**.
- Verdict matches gold: **not hallucinated**; otherwise **hallucinated**.

**Known limitation.** The justification sentence is not checked. A correct verdict with a false
justification is labeled not hallucinated.

## Hand labeling and audit

Automatic labels are written per rule version to `data/labels/v{version}/auto/{split}.jsonl`
(git-ignored): `scripts/label_answers.py` for `train` and `val_id`,
`ml/evaluation/label_test_answers.py` for the test splits. Two kinds of blind CSV go to the
auditor. Neither shows the automatic label, so auditors are not anchored by it. Version 1 files
keep their original names; later versions add `_v{version}`.

- `audit_sample[_v2]_blind.csv` (`ml/evaluation/export_audit_sample.py`): a random sample of 200
  automatically resolved answers from all splits, stratified in proportion to dataset ×
  generator × automatic label. Used only to measure agreement. Each round's seed is fixed in the
  script before it is drawn (version 1: 20261008; version 2: 20261011), and a later round
  excludes every answer audited in earlier rounds.
- `unresolved_dev[_v2]_blind.csv` (`scripts/label_answers.py`): every unresolved answer in
  `train` and `val_id`. Hand labels from this file become the labels for those answers. They are
  kept separate from the agreement sample so they do not bias the kappa estimate. The test
  splits' queues are exported separately before the test evaluation.

Auditors apply the same written definitions above, using the references, gold label, and gold
evidence shown in the CSV, and write `hallucinated` or `not_hallucinated` in `hand_label`. For
TruthfulQA, the definition is the benchmark's: an answer that asserts one of the listed incorrect
answers is hallucinated, even where the auditor's own view of the question differs. In the
unresolved queues only, `undecidable` marks an answer the auditor cannot decide; it is excluded
and counted. `ml/evaluation/score_audit.py` reports raw agreement and Cohen's kappa, with a 95%
bootstrap interval that resamples questions, overall and per dataset; the gate (kappa ≥ 0.6
overall and per dataset) applies to the point estimates.
