"""Judge inputs for val_id: the classifier's inputs and final labels. There is no switch for test splits.

The question text, cleaned primary answer, and top 5 retrieved passages are exactly what feature
extraction used. Generation files are verified against the committed manifest before any row is read.
Only the labeled answers are returned; the hand-labeler's undecidable answers are counted and never
judged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hdetect.cleaning import clean_answer
from hdetect.config import Config
from hdetect.data.io import read_jsonl
from hdetect.features.access import SplitAccessError, load_answers, load_inputs
from hdetect.features.pipeline import FeatureConfig
from hdetect.features.retrieval import passages_from_row
from hdetect.judging.runner import JudgeItem
from hdetect.training.data import ACCESS_LOG, load_labels, read_hand, read_manifest

JUDGE_DEV_SPLIT = "val_id"


@dataclass
class JudgeInputs:
    items: list[JudgeItem]
    labels: dict[str, int]
    excluded: tuple[str, ...]


def load_judge_inputs(cfg: Config, fcfg: FeatureConfig, training_paths: dict, split: str = JUDGE_DEV_SPLIT,
                      evidence_passages: int = 5) -> JudgeInputs:
    if split != JUDGE_DEV_SPLIT:
        raise SplitAccessError(f"Step 8 judges read only {JUDGE_DEV_SPLIT}, not {split!r}")
    manifest = read_manifest(Path(cfg.paths["splits"]) / "split_manifest.jsonl")
    examples = Path(cfg.paths["processed"]) / "examples.jsonl"
    ACCESS_LOG.append(examples.as_posix())
    questions = {q.question_id: q for q in load_inputs(split, examples, manifest)}
    answers = load_answers(split, fcfg.paths["generations"], fcfg.paths["generation_manifest"])
    retrieval_path = Path(fcfg.paths["features"]) / "retrieval" / f"{split}.jsonl"
    ACCESS_LOG.append(retrieval_path.as_posix())
    retrieved = {row["question_id"]: row for row in read_jsonl(retrieval_path)}
    hand = read_hand(training_paths["hand_labels"])
    labels, _ = load_labels(split, training_paths["labels"], hand, cfg.labeling.rule_version)

    items = []
    for qid in sorted(questions):
        primary = answers[qid].primary
        if primary["answer_id"] not in labels.y:
            continue
        passages = passages_from_row(retrieved[qid])
        if len(passages) != evidence_passages:
            raise ValueError(f"{qid}: {len(passages)} retrieved passages, expected {evidence_passages}")
        items.append(JudgeItem(
            answer_id=primary["answer_id"], question_id=qid, question=questions[qid].query,
            answer=clean_answer(primary["text"]), passages=tuple(p.formatted() for p in passages),
            retrieval_s=float(retrieved[qid]["retrieval_s"]),
        ))
    if {i.answer_id for i in items} != set(labels.y):
        raise ValueError(f"{split}: judge items do not match the labeled answers")
    return JudgeInputs(items, dict(labels.y), labels.excluded)
