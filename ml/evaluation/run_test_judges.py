"""Step 9: run the frozen LLM judges on the test splits (docs/PREREGISTRATION.md, amendment log).

The only entry point that judges test answers. Uses the Step 8 judge settings (configs/judges.yaml)
and the prompts frozen in reports/step8/selected_prompts.json, under the plan and caps in
configs/evaluation.yaml. Every non-empty primary answer is judged; the judge never sees a label, and
answers later excluded from scoring (undecidable after hand labeling) are simply not scored. Rows go
to data/judges/test/{judge}/{split}.jsonl (git-ignored). Refuses to run unless
configs/evaluation.yaml has `status: frozen`. Prints call counts and spend only, never a verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from hdetect.cleaning import clean_answer
from hdetect.config import load_config
from hdetect.data.io import read_jsonl
from hdetect.data.splits import SplitManifest
from hdetect.features import TEST_SPLITS
from hdetect.features.access import load_answers, load_inputs
from hdetect.features.pipeline import load_feature_config
from hdetect.features.retrieval import passages_from_row
from hdetect.judging.clients import QuotaExhausted, load_dotenv, make_client
from hdetect.judging.config import load_judge_config
from hdetect.judging.prompts import template_hash
from hdetect.judging.runner import Budget, JudgeItem, RateLimiter, plan, read_rows, run_requests

ROOT = Path(__file__).resolve().parents[2]


def test_items(split: str, cfg, fcfg, evidence_passages: int) -> list[JudgeItem]:
    """The classifier's inputs for every primary answer with non-empty cleaned text, in question order."""
    if split not in TEST_SPLITS:
        raise SystemExit(f"{split} is not a test split")
    manifest = SplitManifest.read(Path(cfg.paths["splits"]) / "split_manifest.jsonl")
    examples = Path(cfg.paths["processed"]) / "examples.jsonl"
    questions = {q.question_id: q for q in load_inputs(split, examples, manifest, allow_test_splits=True)}
    answers = load_answers(split, fcfg.paths["generations"], fcfg.paths["generation_manifest"],
                           allow_test_splits=True)
    retrieved = {r["question_id"]: r for r in read_jsonl(Path(fcfg.paths["features"]) / "retrieval" / f"{split}.jsonl")}
    items = []
    for qid in sorted(questions):
        if qid not in answers:
            continue
        primary = answers[qid].primary
        answer = clean_answer(primary["text"])
        if not answer:
            continue
        passages = passages_from_row(retrieved[qid])
        if len(passages) != evidence_passages:
            raise SystemExit(f"{qid}: {len(passages)} retrieved passages, expected {evidence_passages}")
        items.append(JudgeItem(answer_id=primary["answer_id"], question_id=qid, question=questions[qid].query,
                               answer=answer, passages=tuple(p.formatted() for p in passages),
                               retrieval_s=float(retrieved[qid]["retrieval_s"])))
    return items


def seconds_until_quota_reset() -> float:
    tz = ZoneInfo("America/Los_Angeles")
    now = datetime.now(tz)
    reset = datetime.combine(now.date() + timedelta(days=1), dtime(0, 5), tz)
    return max(60.0, (reset - now).total_seconds())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--feature-config", default="configs/features.yaml")
    p.add_argument("--judge-config", default="configs/judges.yaml")
    p.add_argument("--evaluation-config", default="configs/evaluation.yaml")
    p.add_argument("--judge", choices=["small", "frontier"], help="limit to one judge")
    p.add_argument("--wait-for-daily-quota", action="store_true",
                   help="when the provider's daily quota is used up, sleep until it resets and continue")
    args = p.parse_args()

    ecfg = yaml.safe_load((ROOT / args.evaluation_config).read_text())
    if ecfg["status"] != "frozen":
        sys.exit("configs/evaluation.yaml is provisional: test-split judge calls need `status: frozen`")
    cfg = load_config(ROOT / args.config)
    fcfg = load_feature_config(ROOT / args.feature_config)
    jcfg = load_judge_config(ROOT / args.judge_config)
    selected = json.loads((ROOT / ecfg["paths"]["selected_prompts"]).read_text())
    for v, h in selected["template_sha256"].items():
        if template_hash(v) != h:
            sys.exit(f"prompt template {v} changed since Step 8 selection")
    jc = ecfg["judges"]
    specs = {**jcfg.judges, "frontier": replace(jcfg.judges["frontier"], cap_usd=jc["frontier_cap_usd"])}
    out_root = ROOT / ecfg["paths"]["judge_outputs"]
    load_dotenv(ROOT / ".env")

    items = {s: test_items(s, cfg, fcfg, jcfg.evidence_passages) for s in ecfg["splits"]}
    for j in ([args.judge] if args.judge else list(jc["plan"])):
        spec = specs[j]
        limiters = {j: RateLimiter(spec.max_requests_per_minute)} if spec.max_requests_per_minute else None
        for split in jc["plan"][j]["splits"]:
            reqs = [r for c in jc["plan"][j]["conditions"]
                    for r in plan(items[split], spec, [selected["selected"][j][c]], [c], run=0)]
            out = out_root / j / f"{split}.jsonl"
            while True:
                rows = [r for f in sorted(out_root.glob("**/*.jsonl")) for r in read_rows(f)]
                budget = Budget.from_rows(jc["cap_usd"], specs, rows)
                print(f"{j} on {split}: {len(reqs)} planned calls; Step 9 judges ${budget.spent:.4f} of "
                      f"${jc['cap_usd']:.2f}, {j} ${budget.judge_spent[j]:.4f}", flush=True)
                try:
                    counts = run_requests(reqs, {j: make_client(spec, jcfg)}, specs, out, budget,
                                          concurrency=jcfg.concurrency, max_attempts=jcfg.max_attempts,
                                          progress_every=100, limiters=limiters)
                except QuotaExhausted as e:
                    if e.kind != "daily" or not args.wait_for_daily_quota:
                        sys.exit(f"{j}: provider quota exhausted ({e.kind}); stopped without using an attempt")
                    wait = seconds_until_quota_reset()
                    print(f"  {j}: daily quota used up; sleeping {wait / 3600:.1f} h until it resets", flush=True)
                    time.sleep(wait)
                    continue
                print(f"  {counts}", flush=True)
                break


if __name__ == "__main__":
    main()
