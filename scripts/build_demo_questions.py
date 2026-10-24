"""Build the demo question stream (configs/demo.yaml; amendment 2026-10-23). Not evidence.

A seeded sample of HaluEval QA questions never sampled into any split (from the pinned raw file),
plus the trap questions in demo/trap_questions.jsonl, inserted at seeded positions. Writes
data/demo/questions.jsonl (git-ignored); the Spring Boot producer streams it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import yaml

from hdetect.config import load_config
from hdetect.data import load_dataset
from hdetect.data.io import download, read_jsonl
from hdetect.data.splits import SplitManifest

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    d = yaml.safe_load((ROOT / "configs/demo.yaml").read_text())["questions"]
    cfg = load_config(ROOT / "configs/default.yaml")
    examples, _ = load_dataset("halueval", download(cfg.datasets["halueval"], cfg.paths["raw"]))
    used = {r["question_id"] for r in read_jsonl(Path(cfg.paths["processed"]) / "examples.jsonl")}
    used |= {a.question_id for a in SplitManifest.read(Path(cfg.paths["splits"]) / "split_manifest.jsonl").assignments}
    pool = sorted((e for e in examples if e.question_id not in used), key=lambda e: e.question_id)
    rng = random.Random(d["seed"])
    rows = [{"question_id": f"demo-{e.question_id}", "question": e.prompt_text, "source": "halueval_unsampled",
             "reference_answer": e.correct_refs[0] if e.correct_refs else None}
            for e in rng.sample(pool, d["halueval_unsampled"])]
    traps = read_jsonl(ROOT / d["traps"])
    for t in traps:
        rows.insert(rng.randrange(len(rows) + 1), {"question_id": t["question_id"], "question": t["question"],
                                                   "source": "trap", "reference_answer": t["reference_answer"]})
    out = ROOT / d["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} demo questions ({len(pool)} unsampled HaluEval available, {len(traps)} traps) -> "
          f"{out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
