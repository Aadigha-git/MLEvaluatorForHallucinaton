"""Step 8 smoke test: one synthetic, non-benchmark call per judge, counted against the Step 8 cap.

Checks that the judge accepts the configured settings and reports what the protocol needs: model ID,
token usage (thinking tokens for Gemini), a parseable verdict, and, for Nemotron, no reasoning output.
Refuses to run without --confirm. A judge with a stored successful smoke row is never called again.
The API key is read from the environment and never printed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hdetect.judging.clients import load_dotenv, make_client
from hdetect.judging.config import load_judge_config
from hdetect.judging.runner import Budget, JudgeItem, plan, read_rows, run_requests

ROOT = Path(__file__).resolve().parents[1]

# Invented entities, so the item cannot come from any benchmark split.
SYNTHETIC = JudgeItem(
    answer_id="smoke/synthetic/0",
    question_id="smoke-synthetic",
    question="Which river flows through the town of Varnholt?",
    answer="The Lissel River flows through Varnholt.",
    passages=(
        "Varnholt: Varnholt is a fictional market town used in this test. The Lissel River runs through its centre.",
        "Lissel River: The Lissel is a fictional river that rises in the Brenn Hills and passes Varnholt.",
        "Brenn Hills: The Brenn Hills are a fictional range north of Varnholt.",
        "Varnholt Bridge: A stone bridge over the Lissel in the fictional town of Varnholt.",
        "Market towns: A market town holds a regular market; Varnholt is an invented example.",
    ),
    retrieval_s=0.0,
)
SMOKE_VARIANT = "judge-v1b"
SMOKE_CONDITION = "retrieved"


def checks(spec, row: dict) -> list[tuple[str, bool]]:
    out = [
        ("call succeeded", row["status"] == "ok"),
        ("verdict parsed", row.get("parse_outcome") == "ok"),
        ("model ID reported", bool(row.get("model_returned"))),
        ("input and output tokens reported", row.get("input_tokens") is not None and row.get("output_tokens") is not None),
    ]
    if spec.provider == "nebius":
        out += [("no reasoning returned (enable_thinking=false honoured)", not row.get("reasoning_returned")
                 and row.get("parse_outcome") != "reasoning_leak"),
                ("finish reason stop", row.get("finish_reason") == "stop")]
    if spec.provider == "gemini":
        out += [("thinking tokens reported", row.get("thinking_tokens") is not None),
                ("finish reason STOP", row.get("finish_reason") == "STOP")]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--judge-config", default="configs/judges.yaml")
    p.add_argument("--judge", choices=["small", "frontier"], required=True)
    p.add_argument("--confirm", action="store_true", help="required: this makes one paid API call")
    args = p.parse_args()
    if not args.confirm:
        sys.exit("refusing to call the API without --confirm")

    load_dotenv(ROOT / ".env")
    cfg = load_judge_config(ROOT / args.judge_config)
    spec = cfg.judges[args.judge]
    out_dir = ROOT / cfg.paths["outputs"]
    out = out_dir / "smoke.jsonl"
    budget = Budget.from_rows(cfg.cap_usd, cfg.judges, (r for f in sorted(out_dir.glob("**/*.jsonl"))
                                                       for r in read_rows(f)))
    reqs = plan([SYNTHETIC], spec, [SMOKE_VARIANT], [SMOKE_CONDITION], run=0)
    counts = run_requests(reqs, {spec.name: make_client(spec, cfg)}, cfg.judges, out, budget,
                          max_attempts=1)
    row = [r for r in read_rows(out) if r["request_id"] == reqs[0].request_id][-1]
    print(f"{spec.name} ({spec.model_id}): {counts}")
    for k in ("status", "model_returned", "finish_reason", "parse_outcome", "verdict", "input_tokens",
              "output_tokens", "thinking_tokens", "reasoning_returned", "cost_usd", "api_latency_s", "error"):
        print(f"  {k}: {row.get(k)}")
    print(f"  Step 8 spend so far: ${budget.spent:.4f} of ${cfg.cap_usd:.2f}")
    results = checks(spec, row)
    for name, ok in results:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    if not all(ok for _, ok in results):
        sys.exit(f"{spec.name}: smoke test failed; stop and decide before any benchmark call")


if __name__ == "__main__":
    main()
