from collections import Counter

import pytest

from hdetect.config import parse_config
from hdetect.data.splits import Assignment, LeakageError, SplitManifest, assign_splits, check_integrity
from tests.conftest import make_example

CATEGORIES = {"A": 80, "B": 60, "C": 40, "D": 20}


def synthetic_pool():
    """TruthfulQA questions in clusters of 1-3, plus HaluEval and labeled FEVER singletons."""
    examples, cluster_of = [], {}
    for cat, count in CATEGORIES.items():
        made = 0
        while made < count:
            size = min(1 + (made // 3) % 3, count - made)
            group = [make_example("truthfulqa", f"{cat} question {made + k}", meta={"category": cat}) for k in range(size)]
            cid = min(e.question_id for e in group)
            for e in group:
                cluster_of[e.question_id] = cid
            examples += group
            made += size
    for i in range(300):
        e = make_example("halueval", f"halueval question {i}", correct_refs=("x",))
        examples.append(e)
        cluster_of[e.question_id] = e.question_id
    for i in range(300):
        e = make_example("fever", f"fever claim {i}", gold_label="SUPPORTS" if i % 3 else "REFUTES")
        examples.append(e)
        cluster_of[e.question_id] = e.question_id
    return examples, cluster_of


def shrink(raw_cfg, val_id=100, test_id=50, test_heldout=90):
    raw_cfg["splits"]["val_id"]["n"] = val_id
    raw_cfg["splits"]["test_id"]["n"] = test_id
    raw_cfg["splits"]["test_heldout"]["n"] = test_heldout
    return parse_config(raw_cfg)


@pytest.fixture
def small_cfg(raw_cfg):
    return shrink(raw_cfg)


@pytest.fixture
def pool():
    return synthetic_pool()


def test_each_question_in_exactly_one_split(small_cfg, pool):
    examples, cluster_of = pool
    assignments = assign_splits(examples, cluster_of, small_cfg)
    ids = [a.question_id for a in assignments]
    assert len(ids) == len(set(ids))


def test_clusters_stay_whole(small_cfg, pool):
    examples, cluster_of = pool
    assignments = assign_splits(examples, cluster_of, small_cfg)
    split_by_cluster = {}
    for a in assignments:
        assert split_by_cluster.setdefault(a.cluster_id, a.split) == a.split
    assert any(n > 1 for n in Counter(a.cluster_id for a in assignments).values())


def test_all_truthfulqa_assigned_with_target_fractions(small_cfg, pool):
    examples, cluster_of = pool
    assignments = [a for a in assign_splits(examples, cluster_of, small_cfg) if a.dataset == "truthfulqa"]
    assert len(assignments) == sum(CATEGORIES.values())
    share = Counter(a.split for a in assignments)["train"] / len(assignments)
    assert share == pytest.approx(0.7, abs=0.03)


def test_truthfulqa_stratified_by_category(small_cfg, pool):
    examples, cluster_of = pool
    assignments = [a for a in assign_splits(examples, cluster_of, small_cfg) if a.dataset == "truthfulqa"]
    for cat, total in CATEGORIES.items():
        train = sum(1 for a in assignments if a.stratum == cat and a.split == "train")
        assert train / total == pytest.approx(0.7, abs=0.1), cat


def test_dataset_and_generator_roles(small_cfg, pool):
    examples, cluster_of = pool
    roles = {(a.split, a.dataset, a.generator) for a in assign_splits(examples, cluster_of, small_cfg)}
    assert roles == {
        ("train", "truthfulqa", "model_a"),
        ("test_ood", "truthfulqa", "model_b"),
        ("val_id", "halueval", "model_a"),
        ("test_id", "halueval", "model_a"),
        ("test_heldout", "fever", "model_b"),
    }


def test_sampled_splits_hit_n_and_stratify(small_cfg, pool):
    examples, cluster_of = pool
    counts = Counter()
    fever_labels = Counter()
    for a in assign_splits(examples, cluster_of, small_cfg):
        counts[a.split] += 1
        if a.split == "test_heldout":
            fever_labels[a.stratum] += 1
    assert counts["val_id"] == 100
    assert counts["test_id"] == 50
    assert counts["test_heldout"] == 90
    assert fever_labels == {"SUPPORTS": 60, "REFUTES": 30}


def test_sample_larger_than_pool_takes_everything(raw_cfg, pool):
    examples, cluster_of = pool
    counts = Counter(a.split for a in assign_splits(examples, cluster_of, shrink(raw_cfg, val_id=10_000)))
    assert counts["val_id"] == 300
    assert counts["test_id"] == 0


def test_later_sampled_split_does_not_disturb_earlier_one(raw_cfg, pool):
    examples, cluster_of = pool
    val = lambda cfg: [a for a in assign_splits(examples, cluster_of, cfg) if a.split == "val_id"]
    assert val(shrink(raw_cfg, test_id=50)) == val(shrink(raw_cfg, test_id=10))


def test_deterministic_given_seed(small_cfg, raw_cfg, pool):
    examples, cluster_of = pool
    first = assign_splits(examples, cluster_of, small_cfg)
    assert assign_splits(list(reversed(examples)), cluster_of, small_cfg) == first
    raw_cfg["seed"] += 1
    assert assign_splits(examples, cluster_of, shrink(raw_cfg)) != first


def test_changing_one_dataset_leaves_others_untouched(small_cfg, raw_cfg, pool):
    examples, cluster_of = pool
    before = [a for a in assign_splits(examples, cluster_of, small_cfg) if a.dataset == "truthfulqa"]
    after = [a for a in assign_splits(examples, cluster_of, shrink(raw_cfg, val_id=50)) if a.dataset == "truthfulqa"]
    assert before == after


def test_manifest_roundtrip(small_cfg, pool, tmp_path):
    examples, cluster_of = pool
    manifest = SplitManifest(assign_splits(examples, cluster_of, small_cfg))
    manifest.write(tmp_path / "m.jsonl")
    again = SplitManifest.read(tmp_path / "m.jsonl")
    assert again.assignments == manifest.assignments
    assert again.counts() == manifest.counts()


def test_resamples_inherit_question_split(small_cfg, pool):
    examples, cluster_of = pool
    manifest = SplitManifest(assign_splits(examples, cluster_of, small_cfg))
    qid = manifest.question_ids("test_ood")[0]
    rows = [{"question_id": qid, "answer_id": f"{qid}/r{k}"} for k in range(5)]
    assert {r["split"] for r in manifest.attach(rows)} == {"test_ood"}
    with pytest.raises(KeyError):
        list(manifest.attach([{"question_id": "truthfulqa-unknown"}]))


def _a(qid, split="train", cluster="c1", dataset="truthfulqa", generator="model_a"):
    return Assignment(question_id=qid, dataset=dataset, split=split, generator=generator, cluster_id=cluster)


def test_integrity_rejects_duplicate_question():
    with pytest.raises(LeakageError, match="assigned twice"):
        check_integrity([_a("q1"), _a("q1", split="test_ood", generator="model_b")])


def test_integrity_rejects_split_cluster():
    with pytest.raises(LeakageError, match="spans"):
        check_integrity([_a("q1"), _a("q2", split="test_ood", generator="model_b")])


def test_integrity_rejects_wrong_role():
    with pytest.raises(LeakageError, match="not allowed"):
        check_integrity([_a("q1", generator="model_b")])
