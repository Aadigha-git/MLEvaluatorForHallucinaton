"""Run the labelers on answers whose truth is known from the benchmark itself.

TruthfulQA: each correct and incorrect reference, scored against the remaining references.
HaluEval: the right_answer and the hallucinated_answer.
FEVER: a synthetic verdict line for the gold label and for the opposite label.
Reports, per dataset and per kind of answer, how often the labeler is right, wrong, or unresolved.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter

from hdetect.config import load_config
from hdetect.data.fever import REFUTES, SUPPORTS
from hdetect.data.io import read_jsonl
from hdetect.labeling import label
from hdetect.schema import Example, Label


def probes(ex: Example):
    """(kind, answer, expected label, example to label against) tuples."""
    if ex.dataset == "truthfulqa":
        for i, ref in enumerate(ex.correct_refs):
            rest = dataclasses.replace(ex, correct_refs=ex.correct_refs[:i] + ex.correct_refs[i + 1 :])
            yield "correct_ref", ref, Label.NOT_HALLUCINATED, rest
        for i, ref in enumerate(ex.incorrect_refs):
            rest = dataclasses.replace(ex, incorrect_refs=ex.incorrect_refs[:i] + ex.incorrect_refs[i + 1 :])
            yield "incorrect_ref", ref, Label.HALLUCINATED, rest
    elif ex.dataset == "halueval":
        yield "right_answer", ex.correct_refs[0], Label.NOT_HALLUCINATED, ex
        yield "hallucinated_answer", ex.incorrect_refs[0], Label.HALLUCINATED, ex
    elif ex.dataset == "fever":
        verdict = {SUPPORTS: "SUPPORTED", REFUTES: "REFUTED"}
        opposite = {SUPPORTS: REFUTES, REFUTES: SUPPORTS}
        yield "gold_verdict", f"{verdict[ex.gold_label]}. The evidence says so.", Label.NOT_HALLUCINATED, ex
        yield "wrong_verdict", f"{verdict[opposite[ex.gold_label]]}. The evidence says so.", Label.HALLUCINATED, ex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    tally: dict[tuple[str, str], Counter] = {}
    for ex in map(Example.from_dict, read_jsonl(f"{cfg.paths['processed']}/examples.jsonl")):
        for kind, answer, expected, target in probes(ex):
            got = label(target, answer, cfg.labeling).label
            outcome = "unresolved" if got == Label.UNRESOLVED else ("right" if got == expected else "wrong")
            tally.setdefault((ex.dataset, kind), Counter())[outcome] += 1

    report = {}
    for (dataset, kind), c in sorted(tally.items()):
        n = sum(c.values())
        report[f"{dataset}/{kind}"] = {"n": n, **{k: round(c[k] / n, 3) for k in ("right", "wrong", "unresolved")}}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
