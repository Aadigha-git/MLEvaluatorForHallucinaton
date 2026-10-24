"""Hand-label consistency check (docs/PREREGISTRATION.md, "Labeling commitments"; amendment 2026-10-23).

`export`: a seeded sample of 50 answers from the development hand-labeling queue, stratified by
dataset in proportion to the queue, written blind (no earlier hand label, no automatic label, rows
in seeded random order) to data/audit/consistency_dev_v2_blind.csv. Never overwrites.

`score`: Cohen's kappa between the first and second hand labels over three categories
(hallucinated, not_hallucinated, undecidable), with a 95% bootstrap interval resampling questions,
overall and per dataset, plus the confusion counts. Writes counts and statistics only.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from hdetect.labeling.audit import HAND_LABELS, UNDECIDABLE, _allocate, read_hand_labels

ROOT = Path(__file__).resolve().parents[1]
FIRST = ROOT / "data" / "audit" / "unresolved_dev_v2_blind.csv"
SECOND = ROOT / "data" / "audit" / "consistency_dev_v2_blind.csv"
REPORTS = ROOT / "reports" / "consistency"
N, SEED, N_BOOT = 50, 20261008, 10000
CATEGORIES = sorted(HAND_LABELS | {UNDECIDABLE})


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Proportional allocation by dataset, then a seeded simple random sample within each dataset."""
    by_dataset: dict[tuple, list[dict]] = {}
    for r in sorted(rows, key=lambda r: r["answer_id"]):
        by_dataset.setdefault((r["dataset"],), []).append(r)
    quota = _allocate({k: len(v) for k, v in by_dataset.items()}, n)
    picked = [r for k in sorted(by_dataset) for r in random.Random(f"{seed}:{k[0]}").sample(by_dataset[k], quota[k])]
    random.Random(seed).shuffle(picked)
    return picked


def kappa(a: list[str], b: list[str]) -> float:
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore")
        return float(cohen_kappa_score(a, b, labels=CATEGORIES))


def agreement(pairs: pd.DataFrame, n_boot: int, seed: int) -> dict:
    a, b = pairs["first"].tolist(), pairs["second"].tolist()
    groups: dict[str, list[int]] = {}
    for i, q in enumerate(pairs["question_id"]):
        groups.setdefault(q, []).append(i)
    questions = sorted(groups)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        rows = [i for q in rng.choice(len(questions), size=len(questions)) for i in groups[questions[q]]]
        boots.append(kappa([a[i] for i in rows], [b[i] for i in rows]))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5]) if not np.all(np.isnan(boots)) else (np.nan, np.nan)
    return {"n": len(a), "n_questions": len(questions), "raw_agreement": float(np.mean([x == y for x, y in zip(a, b)])),
            "kappa": kappa(a, b), "kappa_ci_low": float(lo), "kappa_ci_high": float(hi),
            "skipped_replicates": int(np.isnan(boots).sum())}


def export() -> None:
    if SECOND.exists():
        sys.exit(f"{SECOND} exists; not overwritten")
    first = read_hand_labels(FIRST, allow_undecidable=True)
    rows = [r for r in read_rows(FIRST) if r["answer_id"] in first]
    picked = sample(rows, N, SEED)
    with open(SECOND, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({**r, "hand_label": "", "notes": ""} for r in picked)
    print(f"consistency sample: {len(picked)} answers {dict(Counter(r['dataset'] for r in picked))} -> "
          f"{SECOND.relative_to(ROOT)}")


def score() -> None:
    first = read_hand_labels(FIRST, allow_undecidable=True)
    second = read_hand_labels(SECOND, allow_undecidable=True)
    rows = read_rows(SECOND)
    missing = [r["answer_id"] for r in rows if r["answer_id"] not in second]
    if missing:
        sys.exit(f"{len(missing)} of {len(rows)} answers have no second hand label")
    pairs = pd.DataFrame([{"answer_id": r["answer_id"], "question_id": r["question_id"], "dataset": r["dataset"],
                           "first": first[r["answer_id"]], "second": second[r["answer_id"]]} for r in rows])
    out = [{"dataset": "all", **agreement(pairs, N_BOOT, SEED)}]
    out += [{"dataset": d, **agreement(g.reset_index(drop=True), N_BOOT, SEED)} for d, g in pairs.groupby("dataset")]
    confusion = pairs.groupby(["first", "second"]).size().rename("n").reset_index()
    REPORTS.mkdir(parents=True, exist_ok=True)
    for name, df in (("agreement", pd.DataFrame(out)), ("confusion", confusion)):
        path = REPORTS / f"{name}.csv"
        if path.exists():
            sys.exit(f"{path} exists; not overwritten")
        df.round(6).to_csv(path, index=False)
    print(pd.DataFrame(out).round(3).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["export", "score"])
    {"export": export, "score": score}[p.parse_args().command]()
