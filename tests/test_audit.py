import csv
from collections import Counter

import pytest

from hdetect.labeling.audit import (
    BLIND_COLUMNS,
    agreement,
    audit_sample,
    read_hand_labels,
    unresolved_queue,
    write_blind_csv,
)

H, N, U = "hallucinated", "not_hallucinated", "unresolved"


def records(n=1000):
    out = []
    for i in range(n):
        dataset = ("truthfulqa", "halueval", "fever")[i % 3]
        out.append(
            {
                "answer_id": f"a{i:04d}",
                "question_id": f"{dataset}-q{i // 2}",
                "dataset": dataset,
                "generator": "model_a" if i % 5 else "model_b",
                "answer": f"answer {i}",
                "label": U if i % 10 == 0 else (H if i % 4 == 0 else N),
            }
        )
    return out


def test_sample_size_and_excludes_unresolved():
    sample = audit_sample(records(), 200, seed=1)
    assert len(sample) == 200
    assert all(r["label"] != U for r in sample)
    assert len({r["answer_id"] for r in sample}) == 200


def test_sample_is_proportional_by_stratum():
    recs = records()
    resolved = [r for r in recs if r["label"] != U]
    key = lambda r: (r["dataset"], r["generator"], r["label"])
    pool, got = Counter(map(key, resolved)), Counter(map(key, audit_sample(recs, 200, seed=1)))
    for stratum, size in pool.items():
        assert abs(got[stratum] - 200 * size / len(resolved)) < 1


def test_sample_deterministic_and_seeded():
    recs = records()
    assert audit_sample(recs, 200, seed=1) == audit_sample(list(reversed(recs)), 200, seed=1)
    assert audit_sample(recs, 200, seed=1) != audit_sample(recs, 200, seed=2)


def test_sample_larger_than_pool():
    recs = records(30)
    assert len(audit_sample(recs, 200, seed=1)) == sum(r["label"] != U for r in recs)


def test_unresolved_queue():
    queue = unresolved_queue(records(100))
    assert len(queue) == 10 and all(r["label"] == U for r in queue)


def test_blind_csv_hides_auto_label_and_roundtrips(fixture_examples, tmp_path):
    ex = fixture_examples["halueval"][0]
    recs = [
        {"answer_id": "a1", "question_id": ex.question_id, "dataset": "halueval", "generator": "model_a",
         "answer": "Arthur's Magazine", "label": N},
        {"answer_id": "a2", "question_id": ex.question_id, "dataset": "halueval", "generator": "model_a",
         "answer": "First for Women", "label": H},
    ]
    path = tmp_path / "blind.csv"
    write_blind_csv(recs, {ex.question_id: ex}, path)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert tuple(rows[0]) == BLIND_COLUMNS
    assert "label" not in rows[0]
    assert all(r["hand_label"] == "" for r in rows)
    assert rows[0]["correct_refs"] == ex.correct_refs[0]
    assert ex.gold_evidence[0] in rows[0]["gold_evidence"]

    rows[0]["hand_label"] = " Not_Hallucinated "
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BLIND_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    assert read_hand_labels(path) == {"a1": N}


def test_read_hand_labels_rejects_unknown_values(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("answer_id,hand_label\na1,maybe\n")
    with pytest.raises(ValueError, match="bad.csv:2"):
        read_hand_labels(path)


def test_kappa_known_value():
    auto = {"a": H, "b": H, "c": N, "d": N}
    hand = {"a": H, "b": N, "c": N, "d": N}
    result = agreement(auto, hand, {k: k for k in auto}, n_boot=200, seed=0)
    assert result.n == 4 and result.n_questions == 4
    assert result.raw_agreement == 0.75
    assert result.kappa == pytest.approx(0.5)


def test_kappa_perfect_and_ci_brackets_estimate():
    auto = {f"a{i}": (H if i % 3 == 0 else N) for i in range(120)}
    hand = dict(auto)
    for i in (1, 2, 4, 6):
        hand[f"a{i}"] = H if hand[f"a{i}"] == N else N
    result = agreement(auto, hand, {k: f"q{int(k[1:]) // 2}" for k in auto}, n_boot=500, seed=0)
    assert result.n_questions == 60
    assert result.kappa_ci_low <= result.kappa <= result.kappa_ci_high
    perfect = agreement(auto, auto, {k: k for k in auto}, n_boot=50, seed=0)
    assert perfect.kappa == 1.0


def test_kappa_only_uses_overlap():
    result = agreement({"a": H, "b": N}, {"b": N, "c": H}, {"a": "q", "b": "q", "c": "q"}, n_boot=10, seed=0)
    assert result.n == 1
    with pytest.raises(ValueError):
        agreement({"a": H}, {"b": H}, {}, n_boot=10, seed=0)
