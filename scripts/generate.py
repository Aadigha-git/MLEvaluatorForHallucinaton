"""Full step-4 generation (docs/PREREGISTRATION.md, "Generation").

Each split is answered only by its assigned generator: a greedy primary answer plus 5 resamples per
question. Outputs go to git-ignored data/generations/{split}.jsonl; the committed manifest
data/manifests/generations.json records hashes and the run report. Prints counts only, never
answer text.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from hdetect.config import SPLIT_ROLES, load_config
from hdetect.data.io import read_jsonl
from hdetect.data.splits import SplitManifest
from hdetect.generation.client import OpenAICompatibleClient
from hdetect.generation.config import load_generation_config
from hdetect.generation.report import complete, sha256_file, split_report, totals
from hdetect.generation.runner import Budget, BudgetExceeded, FatalApiError, plan_requests, read_rows, run_requests
from hdetect.schema import Example

MANIFEST_SHA256 = "72398783e968f736556e6d05f489f4b9f44cb2e886506fa3b59370810fad0ef2"
SPLIT_ORDER = ("train", "val_id", "test_id", "test_ood", "test_heldout")
PASS_PAUSE_S = 30


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def plan_all(cfg, gen, manifest: SplitManifest, splits) -> dict:
    by_id = {
        row["question_id"]: Example.from_dict(row)
        for row in read_jsonl(Path(cfg.paths["processed"]) / "examples.jsonl")
    }
    plans = {}
    for split in splits:
        dataset, generator = SPLIT_ROLES[split]
        qids = manifest.question_ids(split)
        examples = [by_id[q] for q in qids]
        if any(ex.dataset != dataset for ex in examples):
            raise SystemExit(f"{split}: manifest lists a question from the wrong dataset")
        targets = {q: (split, [generator]) for q in qids}
        plans[split] = plan_requests(examples, targets, gen, resamples=True)
    return plans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--generation-config", default="configs/generation.yaml")
    parser.add_argument("--out-dir", default="data/generations")
    parser.add_argument("--manifest-out", default="data/manifests/generations.json")
    parser.add_argument("--dry-run", action="store_true", help="plan and report only; make no API calls")
    args = parser.parse_args()

    cfg = load_config(args.config)
    gen = load_generation_config(args.generation_config)
    manifest_path = Path(cfg.paths["splits"]) / "split_manifest.jsonl"
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise SystemExit("split manifest does not match the pre-registered SHA-256")
    manifest = SplitManifest.read(manifest_path)
    out_dir = Path(args.out_dir)
    plans = plan_all(cfg, gen, manifest, SPLIT_ORDER)
    paths = {s: out_dir / f"{s}.jsonl" for s in SPLIT_ORDER}
    prices = gen.prices()

    budget = Budget(gen.run.budget_usd, prices)
    for split in SPLIT_ORDER:
        for row in read_rows(paths[split]):
            if row["status"] == "ok":
                budget.settle(0.0, row["generator"], row["prompt_tokens"], row["completion_tokens"])
    n_calls = sum(len(p) for p in plans.values())
    print(f"{n_calls} planned calls; ${budget.spent:.4f} already spent of the ${budget.cap:.2f} cap")
    for split in SPLIT_ORDER:
        print(f"  {split}: {len(plans[split])} calls, {SPLIT_ROLES[split][1]}")

    halted = None
    if not args.dry_run:
        load_dotenv(Path(".env"))
        client = OpenAICompatibleClient(gen.api)
        try:
            for attempt_pass in range(1, gen.run.max_attempts + 1):
                print(f"pass {attempt_pass}", flush=True)
                for split in SPLIT_ORDER:
                    counts = run_requests(
                        plans[split], client, paths[split],
                        concurrency=gen.api.concurrency, progress_every=1000,
                        max_attempts=gen.run.max_attempts, budget=budget,
                    )
                    print(f"  {split}: {json.dumps(counts)}; ${budget.spent:.4f} spent", flush=True)
                reports = {
                    s: split_report(plans[s], read_rows(paths[s]), prices, gen.run.max_attempts) for s in SPLIT_ORDER
                }
                if all(complete(r) for r in reports.values()):
                    break
                time.sleep(PASS_PAUSE_S)
        except (FatalApiError, BudgetExceeded) as e:
            halted = f"{type(e).__name__}: {e}"
            print(f"HALTED: {halted}", file=sys.stderr)

    reports = {s: split_report(plans[s], read_rows(paths[s]), prices, gen.run.max_attempts) for s in SPLIT_ORDER}
    done = all(complete(r) for r in reports.values())
    files = {}
    for split in SPLIT_ORDER:
        if paths[split].exists():
            files[split] = {
                "path": str(paths[split]),
                "sha256": sha256_file(paths[split]),
                "rows": len(read_rows(paths[split])),
                "generator": SPLIT_ROLES[split][1],
            }
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "complete": done,
        "halted": halted,
        "generation_config_hash": gen.hash,
        "split_config_hash": cfg.hash,
        "split_manifest_sha256": MANIFEST_SHA256,
        "price_snapshot": gen.catalog_snapshot,
        "concurrency": gen.api.concurrency,
        "budget_usd": gen.run.budget_usd,
        "files": files,
        "totals": totals(reports.values()),
        "splits": reports,
    }
    if args.dry_run:
        print(json.dumps({"totals": record["totals"]}, indent=2))
        return
    if done:
        for p in paths.values():
            p.chmod(0o444)
    manifest_out = Path(args.manifest_out)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if halted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
