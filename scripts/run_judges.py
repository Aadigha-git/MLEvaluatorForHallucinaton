"""Step 8: LLM-judge prompt tuning and repeatability on val_id (docs/PREREGISTRATION.md, amendment log).

Stages, in order:
  freeze-subset  write the 200-answer repeatability subset (answer IDs only); no API call
  tune           4 prompt variants x 2 evidence conditions x every labeled val_id answer, per judge
  select         score the tuning run and freeze the winning variant per judge and condition
  repeat         2 more runs of the frozen prompt on the subset, per judge and condition
  report         write reports/step8/ tables and the manifest

`tune` and `repeat` call the APIs and refuse to start unless configs/judges.yaml has `status: frozen`.
Every Step 8 call, smoke tests included, counts against one cap. Reads no test split: every file
opened under data/ is recorded, and the run fails if any belongs to a test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from hdetect.config import load_config
from hdetect.features.pipeline import load_feature_config
from hdetect.judging.clients import load_dotenv, make_client
from hdetect.judging.config import load_judge_config
from hdetect.judging.inputs import load_judge_inputs
from hdetect.judging.prompts import template_hash
from hdetect.judging.runner import Budget, plan, read_rows, run_requests, spent_usd
from hdetect.judging.scoring import agreement, final_rows, repeat_subset, score_cell, select_variant
from hdetect.training.access import OPENED, data_access_table, install_audit_hook, is_test_path

ROOT = Path(__file__).resolve().parents[1]
KAPPAS = "Cohen's kappa 0.7117 in the first round and 0.578 in the fresh round under rule version 2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state() -> dict[str, str | bool]:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                    capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": True}
    return {"commit": head, "dirty": dirty}


def read_subset(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def all_rows(out_dir: Path) -> list[dict]:
    return [r for f in sorted(out_dir.glob("**/*.jsonl")) for r in read_rows(f)]


def cell_rows(rows: list[dict], judge: str, variant: str, condition: str, run: int) -> dict[str, dict]:
    keep = [r for r in rows if (r["judge"], r["variant"], r["condition"], r["run"]) == (judge, variant, condition, run)]
    return {r["answer_id"]: r for r in final_rows(keep).values()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["freeze-subset", "tune", "select", "repeat", "report"])
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--feature-config", default="configs/features.yaml")
    p.add_argument("--training-config", default="configs/training.yaml")
    p.add_argument("--judge-config", default="configs/judges.yaml")
    p.add_argument("--judge", choices=["small", "frontier"], help="limit tune/repeat to one judge")
    args = p.parse_args()

    install_audit_hook()
    cfg = load_config(ROOT / args.config)
    fcfg = load_feature_config(ROOT / args.feature_config)
    tpaths = {k: ROOT / v for k, v in yaml.safe_load((ROOT / args.training_config).read_text())["paths"].items()}
    jcfg = load_judge_config(ROOT / args.judge_config)
    out_dir = ROOT / jcfg.paths["outputs"]
    reports = ROOT / jcfg.paths["reports"]
    subset_path = ROOT / jcfg.repeat_subset_file
    selected_path = reports / "selected_prompts.json"
    judges = [args.judge] if args.judge else list(jcfg.judges)

    inputs = load_judge_inputs(cfg, fcfg, tpaths, evidence_passages=jcfg.evidence_passages)
    ids = [i.answer_id for i in inputs.items]
    print(f"val_id: {len(ids)} labeled answers judged, {len(inputs.excluded)} undecidable excluded")

    if args.stage == "freeze-subset":
        subset = repeat_subset(ids, jcfg.repeat_subset_size, jcfg.repeat_subset_seed)
        text = "\n".join(subset) + "\n"
        if subset_path.exists() and subset_path.read_text() != text:
            sys.exit(f"{subset_path} exists and differs; the subset is frozen")
        subset_path.write_text(text)
        print(f"wrote {subset_path.relative_to(ROOT)}: {len(subset)} answer IDs, sha256 {sha256(subset_path)}")

    elif args.stage in ("tune", "repeat"):
        if not jcfg.frozen:
            sys.exit("configs/judges.yaml is provisional: benchmark calls need `status: frozen` (by amendment)")
        load_dotenv(ROOT / ".env")
        budget = Budget(jcfg.cap_usd, jcfg.judges, spent_usd(all_rows(out_dir)))
        if args.stage == "tune":
            work = {j: plan(inputs.items, jcfg.judges[j], jcfg.variants, jcfg.conditions, run=0) for j in judges}
        else:
            if not selected_path.exists():
                sys.exit("run `select` first: the repeat runs use the frozen prompts")
            selected = json.loads(selected_path.read_text())["selected"]
            subset = set(read_subset(subset_path))
            items = [i for i in inputs.items if i.answer_id in subset]
            if len(items) != jcfg.repeat_subset_size:
                sys.exit("the repeatability subset does not match the labeled val_id answers")
            work = {j: [r for run in range(1, jcfg.repeat_extra_runs + 1) for c in jcfg.conditions
                        for r in plan(items, jcfg.judges[j], [selected[j][c]], [c], run=run)] for j in judges}
        for j, reqs in work.items():
            spec = jcfg.judges[j]
            print(f"{j} ({spec.model_id}): {len(reqs)} planned calls; ${budget.spent:.4f} of ${jcfg.cap_usd:.2f} spent")
            counts = run_requests(reqs, {j: make_client(spec, jcfg)}, jcfg.judges, out_dir / f"{j}.jsonl", budget,
                                  concurrency=jcfg.concurrency, max_attempts=jcfg.max_attempts, progress_every=500)
            print(f"  {counts}")

    elif args.stage == "select":
        rows = all_rows(out_dir)
        records, selected = [], {}
        for j in jcfg.judges:
            selected[j] = {}
            for c in jcfg.conditions:
                scores = {v: score_cell(cell_rows(rows, j, v, c, 0), inputs.labels, ids) for v in jcfg.variants}
                for v, s in scores.items():
                    print(f"  {j}/{c}/{v}: F1 {s['f1']:.4f}, parse failures {s['parse_failures']}, "
                          f"API failures {s['api_failures']} (both scored as wrong)")
                selected[j][c] = select_variant(scores, jcfg.variants)
                records += [{"judge": j, "condition": c, "variant": v, "selected": v == selected[j][c], **s}
                            for v, s in scores.items()]
        reports.mkdir(parents=True, exist_ok=True)
        payload = {"selected": selected, "rule": "max val_id hallucinated-class F1; ties: fewer failures, then order",
                   "template_sha256": {v: template_hash(v) for v in jcfg.variants}, "judge_config_hash": jcfg.hash}
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if selected_path.exists() and selected_path.read_text() != text:
            sys.exit(f"{selected_path} exists and differs; the selected prompts are frozen")
        selected_path.write_text(text)
        pd.DataFrame(records).to_csv(reports / "tuning.csv", index=False)
        print(json.dumps(selected, indent=2))

    elif args.stage == "report":
        rows = all_rows(out_dir)
        selected = json.loads(selected_path.read_text())["selected"]
        subset = read_subset(subset_path)
        rep = []
        for j in jcfg.judges:
            for c in jcfg.conditions:
                v = selected[j][c]
                runs = [cell_rows(rows, j, v, c, run) for run in range(jcfg.repeat_extra_runs + 1)]
                rep.append({"judge": j, "condition": c, "variant": v, **agreement(runs, subset)})
        reports.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rep).to_csv(reports / "repeatability.csv", index=False)
        data_access_table(ROOT).to_csv(reports / "data_access.csv", index=False)
        manifest = {
            "git": git_state(), "python": platform.python_version(), "judge_config_hash": jcfg.hash,
            "price_snapshot": jcfg.price_snapshot, "spent_usd": spent_usd(rows), "cap_usd": jcfg.cap_usd,
            "outputs_sha256": {f.relative_to(ROOT).as_posix(): sha256(f) for f in sorted(out_dir.glob("**/*.jsonl"))},
            "repeat_subset_sha256": sha256(subset_path), "labels_note": f"HaluEval agreement gate failed: {KAPPAS}",
        }
        (reports / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    opened_test = [p for p in OPENED if is_test_path(p)]
    if opened_test:
        sys.exit(f"a test split was opened: {sorted(set(opened_test))}")


if __name__ == "__main__":
    main()
