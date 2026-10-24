# hdetect: a classical-ML hallucination evaluator, tested against pre-registered criteria

This project asks whether a cheap classifier over black-box text signals can detect hallucinated
answers from LLMs. The signals are retrieval similarity, NLI entailment and contradiction, lexical
overlap, hedging, length, and self-consistency. The question has two parts: can the classifier beat
the best single heuristic, and does it hold up on an unseen model and dataset?

Everything was pre-registered before any test split was scored. Every later change is a dated
amendment.

- Specification: [docs/SPEC.md](docs/SPEC.md)
- Pre-registration and amendment log: [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md)
- Labeling rules: [docs/LABELING.md](docs/LABELING.md)
- **Results write-up: [docs/RESULTS.md](docs/RESULTS.md)**

## Results at a glance

Primary analysis: retrieved evidence, hallucinated-class F1, splits `test_ood` and `test_heldout`.

| Research question | Single-shot tier | Resampling tier |
|---|---|---|
| RQ1: does combining features beat the best single heuristic? | **No** | **Mixed** |
| RQ2: does the classifier generalize to unseen models and datasets? | **No** | **No** |

These are negative results. On every test split, each classifier's F1 is within 0.025 of the F1 of
predicting every answer hallucinated.

**Read with these caveats** (details in [docs/RESULTS.md](docs/RESULTS.md)):

- **The one RQ1 pass beats flag-everything, not an informative heuristic.** On both primary splits,
  the best single feature (lexical overlap, with its `val_id` threshold) flags 99.5–100% of answers.
  So the resampling tier's RQ1 pass on `test_ood` (+0.025, 95% CI 0.008 to 0.041) shows only a small
  margin over flagging everything.
- **The RQ2 drop tracks hallucination prevalence**, which is 57% on `test_id`, 32% on `test_ood`, and
  21% on `test_heldout`. Each drop is 0.21–0.39 F1, against a 0.05 margin.
- **Labels are noisy.**
  - HaluEval labels failed the hand-audit agreement gate: Cohen's kappa 0.7117, then 0.578 under
    rule version 2, against a minimum of 0.6.
  - A blind re-labeling of 50 hand labels two weeks later gave kappa 0.198 [−0.026, 0.422].
  - Hand labels make up 53% of `train` and `test_ood` labels.
- **LLM judges did better, at a cost.** A small open-weight judge (Nemotron, about $0.00006 per
  verdict) beat the classifiers on `test_id` and `test_heldout`. A frontier judge (Gemini 3.1 Pro,
  about $0.004–$0.005 per verdict) reached F1 0.895 on FEVER, against 0.34 for the classifiers.

## Status

Every build step in docs/SPEC.md is complete and committed:

- Data, labels, and generation: steps 1–4.
- Features, training, calibration, and judges: steps 5–8.
- One-shot test evaluation: step 9.
- Failure analysis and adversarial tests: step 10.
- Write-up: step 11.
- Hand-label consistency check: complete.
- Live demo: steps 12–13. The demo illustrates the system and is not evidence for either research
  question.

## Repository layout

| Path | Contents |
|---|---|
| `docs/` | Specification, pre-registration with amendments, labeling rules, results |
| `configs/` | Every setting, pinned: splits, generation, features, training, calibration, judges, evaluation, Step 10, demo |
| `src/hdetect/` | Library: `data`, `generation`, `labeling`, `features`, `training`, `judging`, `evaluation`, `service` |
| `scripts/` | Development-split entry points (steps 1–8, the consistency check, demo tooling). They cannot read test splits. |
| `ml/evaluation/` | The only code allowed to read test splits: test labeling, features, judges, Step 9 evaluation, Step 10 |
| `reports/step6/` … `reports/step10/`, `reports/consistency/` | Committed results: aggregate tables, figures, manifests, data-access logs. No answer text or labels. |
| `demo/` | Kafka (`docker-compose.yml`), Spring Boot app (`app/`), trap questions |
| `tests/` | Unit, isolation, and reproducibility tests |
| `data/`, `logs/`, `.tools/`, `.env` | Local only and git-ignored: raw data, generations, labels, features, models, judge outputs, audit files, the JDK, API keys |

## Data handling rules

These are enforced in code and tests, not just by convention.

- **Test-split isolation.**
  - Only `ml/evaluation/` may pass `allow_test_splits=True`, and a test checks this.
  - Training and judging code cannot name a test split.
  - The Step 6–10 runs record every file they open under `data/` and write the list to
    `data_access*.csv` in their report directory.
- **One-shot test scoring.**
  - `ml/evaluation/evaluate.py` and `ml/evaluation/step10.py` refuse to run unless their config
    says `status: frozen`.
  - They never overwrite a report.
- **Nothing private is committed.** Answer text, labels, hand-labeling files, generated corpora,
  feature tables, model artifacts, and judge outputs stay under the git-ignored `data/` tree.
- **API keys** are read from `.env` and are never printed or committed.
- **Budget guards.** Every paid API call goes through a budget guard that rebuilds spend from
  stored rows:
  - generation: $1 cap;
  - Step 8 judges: $45 cap;
  - Step 9 judges: $10 cap;
  - demo chatbot: $0.50 cap.

## Setup

Requirements:

- macOS or Linux;
- [uv](https://docs.astral.sh/uv/) with Python 3.12;
- about 15 GB of disk for the retrieval index and models;
- for the demo only: Docker and Maven.

```bash
uv sync --group features        # core + dev dependencies, plus torch, transformers, pyserini
```

- **JDK 21 for retrieval.** Pyserini needs a JDK 21 placed under `.tools/`, matching
  `jdk_glob` in `configs/features.yaml`. It is checked at runtime.
- **Retrieval index.** The pre-built Pyserini Lucene index `wikipedia-dpr-100w` is unpacked under
  `data/indexes/`. `configs/features.yaml` pins its file name, MD5, and document count
  (21,015,324), and the run stops if they differ. Wiki-DPR is licensed CC-BY-NC-4.0, so the results
  and the demo are non-commercial.
- **Feature models.** `all-MiniLM-L6-v2` and `nli-deberta-v3-small` are pinned to exact Hugging
  Face revisions.
- **API keys** go in `.env` (key names only shown here):

  ```bash
  NEBIUS_API_KEY=...   # generation (Qwen, Gemma), the Nemotron judge, the demo chatbot
  GEMINI_API_KEY=...   # the Gemini judge
  ```

## Reproducing the pipeline

Run the steps in order from the repository root, each with `uv run python <script>`. The steps that
call paid APIs are marked. Steps that read test splits are one-shot and refuse to overwrite.

| Step | Command | Output |
|---|---|---|
| 1–2 Data and splits | `scripts/prepare_data.py` | `data/processed/`, `data/splits/split_manifest.jsonl` |
| Pilot | `scripts/run_pilot.py` (API) | `data/pilot/` |
| 4 Generation | `scripts/generate.py` (API), then `scripts/backup_generations.py --dest <dir>` | `data/generations/`, `data/manifests/generations.json` |
| 3 Labels (dev) | `scripts/check_labelers.py`, `scripts/label_answers.py` | `data/labels/v2/`, `data/audit/unresolved_dev_v2_blind.csv` |
| 3 Labels (test) | `ml/evaluation/label_test_answers.py`, `ml/evaluation/export_test_queue.py` | test auto labels, blind test queue |
| 3 Label audit | `ml/evaluation/export_audit_sample.py`, then `ml/evaluation/score_audit.py <filled csv>` | Cohen's kappa |
| 5 Features | `scripts/extract_features.py`, `ml/evaluation/extract_test_features.py` | `data/features/tables/{retrieved,none,gold}/` |
| 6 Training | `scripts/train_classifiers.py` | `reports/step6/` |
| 7 Calibration and policy | `scripts/calibrate.py` | `data/models/step7/`, `reports/step7/` |
| 8 Judges (val_id) | `scripts/judge_smoke.py --judge small --confirm` (API); `scripts/run_judges.py {freeze-subset,tune,select,repeat,report}` (API) | `reports/step8/` |
| 9 Test judges | `ml/evaluation/run_test_judges.py --wait-for-daily-quota` (API) | `data/judges/test/` |
| 9 Test evaluation | `ml/evaluation/evaluate.py --stand-in` (dry run on val_id), then `ml/evaluation/evaluate.py` (once) | `reports/step9/` |
| 10 Failure analysis | `ml/evaluation/step10.py export-failures`, hand-sort, then `ml/evaluation/step10.py summarize-failures` | `reports/step10/` |
| 10 Adversarial tests | `ml/evaluation/step10.py perturb` | `reports/step10/perturbation_*.csv` |
| Consistency check | `scripts/consistency_check.py export`, relabel, then `scripts/consistency_check.py score` | `reports/consistency/` |

Notes:

- **Determinism.**
  - Features are computed on CPU at batch size 1 and cached.
  - Classifier artifacts are deterministic refits; `scripts/calibrate.py` checks that each one
    reproduces its Step 6 threshold.
  - Step 9 checks that every frozen threshold is reproduced bit for bit on `val_id` before it
    scores anything.
- **Generation is not bit-reproducible.** The Nebius models are served as FP8 without exposed
  revisions, so generated answers are stored once and never regenerated.
- **Gemini's daily quota** (250 requests) is why the Gemini runs take several days.
  `--wait-for-daily-quota` sleeps until the reset, and reruns resume from the stored rows.

## Running the demo

The demo streams questions through Model A, scores each answer with the frozen single-shot
classifier, and shows verdicts live. It is an illustration with no ground truth, as stated in
docs/SPEC.md and the 2026-10-23 amendment. The frozen policy flags nearly every answer, and the
dashboard says so.

```bash
docker compose -f demo/docker-compose.yml up -d          # Kafka (KRaft) on localhost:9092, topics questions.v1 and verdicts.v1
uv run python scripts/build_demo_questions.py            # 200 never-sampled HaluEval questions + 10 trap questions
uv run python scripts/demo_scorer.py                     # scoring service (Nebius; $0.50 cap)
(cd demo/app && JAVA_HOME=$(ls -d ../../.tools/jdk-21*/Contents/Home | head -1) mvn -q package -DskipTests)
.tools/jdk-21*/Contents/Home/bin/java -jar demo/app/target/hdetect-demo-0.1.0.jar
open http://localhost:8080                               # then press "start stream"
```

- **Smoke test without Kafka:** `uv run python scripts/demo_scorer.py --once "What is the capital
  of Australia?"`.
- **Events** are versioned JSON (`demo-event-v1`, `src/hdetect/service/schema.py`). Each carries
  model, artifact, and policy versions and a timestamp per stage.
- **Explanations** name the two features whose replacement by their `val_id` median moves the score
  most. They describe the classifier's score, not why an answer is wrong.
- **The sliders** re-band answers in the browser for illustration only. The scoring service always
  uses the frozen policy.

## Tests

```bash
uv run pytest            # 416 tests; network tests are deselected by default
```

The tests cover:

- labeling rules;
- feature formulas and isolation: no labels in features, test splits only through `ml/evaluation/`;
- budget guards and judge parsing;
- the bootstrap and research-question rules on synthetic data, including exact-zero differences
  for identical predictions;
- the Step 10 perturbations;
- the demo event schema and explanations.

## Limitations

The full list, with numbers, is in [docs/RESULTS.md](docs/RESULTS.md). In brief:

- **Label noise.** HaluEval labels failed the agreement gate, and hand labels proved inconsistent
  (kappa 0.198).
- **Dataset shift in "in-distribution".** `train` is TruthfulQA while `val_id` and `test_id` are
  HaluEval, so "in-distribution" means same generator, not same dataset.
- **Threshold transfer.** Thresholds tuned on HaluEval carry over to TruthfulQA and FEVER.
- **RQ2 confound.** The drop from `test_id` to `test_ood` changes dataset and generator at once.
- **Gemini's prompt** was selected on only 45 answers, because of quota limits.
- **Pretraining contamination.** The benchmarks are probably in every model's pretraining data.
- **Retrieval licence.** The Wiki-DPR corpus is non-commercial (CC-BY-NC-4.0).
