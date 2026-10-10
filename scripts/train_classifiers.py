"""Step 6: train on train, select and validate on val_id, ablations and redundancy; write reports/step6/.

Refuses to run until every unresolved train and val_id answer has a hand decision. Reads no test
split: every file opened under data/ is recorded (Python audit hook plus the loader's own log) and
the run fails if any belongs to a test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from hdetect.config import load_config
from hdetect.features.pipeline import load_feature_config
from hdetect.training.access import data_access_table, install_audit_hook, is_test_path  # noqa: F401
from hdetect.training.inputs import load_dev_inputs
from hdetect.training.report import annotate_tables, write_readme
from hdetect.training.run import REPORT_FILES, Inputs, Step6, write_tables

ROOT = Path(__file__).resolve().parents[1]


def notes(fcfg_paths: dict, train: dict, val: dict) -> dict[str, str]:
    report = json.loads(Path(fcfg_paths["features"], "report.train-val_id.full.json").read_text())
    recall = report["splits"]["val_id"]["recall@5"]
    times = report["splits"]
    ft = "; ".join(f"{s}: " + ", ".join(f"{k[5:-3]} {v['mean']:.1f} ms" for k, v in sorted(times[s]["time_ms"].items()))
                   for s in ("train", "val_id"))
    return {
        "retrieval": (
            "BM25 top-5 over Wiki-DPR. `val_id` (HaluEval) knowledge-sentence recall@5 = "
            f"{recall['halueval']['knowledge_sentence_recall@5']:.3f} (n = {recall['halueval']['n']}). `train` "
            "(TruthfulQA) has no gold evidence, so its recall is not measurable. A retrieved-evidence feature can only "
            "help where retrieval found the relevant passage; low recall caps the evidence group, and this is a "
            "retrieval limitation rather than a classifier one."
        ),
        "datasets": (
            "`train` is TruthfulQA only and `val_id` is HaluEval only, so every validation number is already a "
            "cross-dataset result (TruthfulQA to HaluEval, same generator) and per-dataset results equal per-split "
            "results; there is nothing to pool. `train` CV average precision is TruthfulQA in-distribution. "
            "Evidence-condition effects are compared within `val_id` (same answers, same labels): retrieved versus "
            "no evidence, and gold as an exploratory oracle. Gold-evidence results are not realistic retrieval "
            "performance: HaluEval's gold passage contains the reference answer, which the labeling rule also uses."
        ),
        "feature_times": "Mean feature computation time per answer (Step 5 reports) — " + ft,
        "sc_entail": "",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--feature-config", default="configs/features.yaml")
    p.add_argument("--training-config", default="configs/training.yaml")
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--rewrite-reports", action="store_true",
                   help="rebuild annotations and README.md from the existing CSVs; fits nothing")
    args = p.parse_args()
    install_audit_hook()

    cfg = load_config(args.config)
    fcfg = load_feature_config(args.feature_config)
    tcfg = yaml.safe_load(Path(args.training_config).read_text())
    paths = tcfg["paths"]
    if args.rewrite_reports:
        rewrite_reports(Path(paths["reports"]), fcfg, tcfg)
        return

    dev = load_dev_inputs(cfg, fcfg, paths)
    train, val = dev.train, dev.val
    inputs = Inputs(train, val, [dev.labels[s][0].summary() for s in ("train", "val_id")], dev.checks,
                    notes(fcfg.paths, train, val))
    step = Step6(tcfg, inputs, args.n_jobs)
    tables = step.run()
    tables["data_access"] = data_access_table(ROOT)
    if tables["data_access"]["test_split"].any():
        raise SystemExit("a test-split file was opened; reports not written")

    out = Path(paths["reports"])
    tables = annotate_tables(tables)
    write_tables(tables, out)
    (out / "README.md").write_text(write_readme(tables, inputs.notes, tcfg) + "\n")
    print(f"reports -> {out}")


def rewrite_reports(out: Path, fcfg, tcfg) -> None:
    names = [f[:-4] for f in REPORT_FILES if f.endswith(".csv")]
    missing = [n for n in names if not (out / f"{n}.csv").exists()]
    if missing:
        raise SystemExit(f"missing Step 6 tables: {missing}")
    tables = annotate_tables({n: pd.read_csv(out / f"{n}.csv") for n in names})
    write_tables(tables, out)
    (out / "README.md").write_text(write_readme(tables, notes(fcfg.paths, {}, {}), tcfg) + "\n")
    print(f"reports rewritten -> {out}")


if __name__ == "__main__":
    main()
