"""Run the pre-freeze generation pilot (docs/PREREGISTRATION.md, "Pilot") and print format checks.

Outputs go to data/pilot/ (git-ignored), are never labeled, and are deleted after the checks.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from hdetect.config import load_config
from hdetect.data import fetch_all
from hdetect.data.dedup import deduplicate
from hdetect.data.splits import SplitManifest
from hdetect.generation.client import OpenAICompatibleClient
from hdetect.generation.config import load_generation_config
from hdetect.generation.pilot import format_report, select_pilot
from hdetect.generation.runner import load_results, plan_requests, run_requests


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--generation-config", default="configs/generation.yaml")
    parser.add_argument("--out-dir", default="data/pilot")
    parser.add_argument("--dry-run", action="store_true", help="select and plan only; make no API calls")
    args = parser.parse_args()

    cfg = load_config(args.config)
    gen = load_generation_config(args.generation_config)
    manifest = SplitManifest.read(Path(cfg.paths["splits"]) / "split_manifest.jsonl")

    examples = [ex for exs, _ in fetch_all(cfg).values() for ex in exs]
    dedup = deduplicate(examples, cfg.dedup)
    mismatched = [a.question_id for a in manifest.assignments if dedup.cluster_of.get(a.question_id) != a.cluster_id]
    if mismatched:
        raise SystemExit(f"recomputed clusters disagree with the manifest for {len(mismatched)} questions")

    selected = select_pilot(dedup.kept, dedup.cluster_of, manifest, gen.pilot.questions, gen.pilot.seed)
    generators = sorted(gen.generators)
    targets = {ex.question_id: (split, generators) for ex, split in selected}
    requests = plan_requests(
        (ex for ex, _ in selected), targets, gen, resamples=True, repeat_primary=gen.pilot.repeat_primary
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pilot_questions.json").write_text(
        json.dumps([{"question_id": ex.question_id, "dataset": ex.dataset, "split": split} for ex, split in selected], indent=2)
        + "\n"
    )
    print(f"{len(selected)} pilot questions, {len(requests)} planned calls (generation config {gen.hash})")
    if args.dry_run:
        return

    load_dotenv(Path(".env"))
    client = OpenAICompatibleClient(gen.api)
    outputs = out_dir / "pilot_outputs.jsonl"
    counts = run_requests(requests, client, outputs, concurrency=gen.api.concurrency, progress_every=40)
    print(json.dumps(counts))

    prices = {
        name: (spec.usd_per_million_input_tokens, spec.usd_per_million_output_tokens)
        for name, spec in gen.generators.items()
    }
    report = format_report(load_results(outputs).values(), prices)
    (out_dir / "pilot_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
