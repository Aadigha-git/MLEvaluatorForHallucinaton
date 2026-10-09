import dataclasses

from hdetect.data import dedup as dedup_module
from hdetect.data.dedup import deduplicate, near_duplicate_pairs, normalize_question
from hdetect.data.fever import PROMPT_TEMPLATE
from tests.conftest import make_example

GOLD = "If gold has gone up in value over the last twenty years, what do we know will happen in the next twenty years?"
BITCOIN = "If bitcoin has gone up in value over the last twenty years, what do we know will happen in the next twenty years?"
MELON = "What happens to you if you eat watermelon seeds?"
TWINKLE = 'Who composed the tune of "Twinkle, Twinkle, Little Star"?'


def fever_example(claim):
    ex = make_example("fever", claim, gold_label="SUPPORTS")
    return dataclasses.replace(ex, prompt_text=PROMPT_TEMPLATE.format(claim=claim))


def test_normalize_question():
    assert normalize_question("  What's   the  CAPITAL of France?! ") == "what s the capital of france"
    assert normalize_question("Café\u00a0déjà") == normalize_question("Café déjà")


def test_exact_duplicates_within_dataset_dropped(cfg):
    a = make_example("truthfulqa", "Is the Earth flat?")
    b = make_example("truthfulqa", "is the earth FLAT")
    result = deduplicate([a, b], cfg.dedup)
    assert len(result.kept) == 1
    keeper = min(a.question_id, b.question_id)
    assert result.kept[0].question_id == keeper
    assert result.dropped == [
        {"question_id": max(a.question_id, b.question_id), "reason": "exact_duplicate", "duplicate_of": keeper}
    ]


def test_template_paraphrases_are_clustered(cfg):
    a, b = make_example("truthfulqa", GOLD), make_example("truthfulqa", BITCOIN)
    result = deduplicate([a, b], cfg.dedup)
    assert len(result.kept) == 2
    assert result.cluster_of[a.question_id] == result.cluster_of[b.question_id]
    assert len(result.pairs) == 1 and result.pairs[0].similarity >= cfg.dedup.threshold


def test_distinct_questions_not_clustered(cfg):
    a, b = make_example("truthfulqa", MELON), make_example("truthfulqa", TWINKLE)
    result = deduplicate([a, b], cfg.dedup)
    assert result.cluster_of[a.question_id] != result.cluster_of[b.question_id]
    assert result.pairs == []


def test_fever_compares_claims_not_prompt_template(cfg):
    a = fever_example("Jackie (2016 film) was directed by Peter Jackson.")
    b = fever_example("Syracuse, New York, had a population of 145,170.")
    result = deduplicate([a, b], cfg.dedup)
    assert result.cluster_of[a.question_id] != result.cluster_of[b.question_id]


def test_clustering_is_transitive(cfg, monkeypatch):
    exs = [make_example("truthfulqa", t) for t in ("q one", "q two", "q three", "q four")]
    order = sorted(exs, key=lambda e: e.question_id)
    monkeypatch.setattr(dedup_module, "near_duplicate_pairs", lambda *a, **k: [(0, 1, 0.9), (1, 2, 0.9)])
    result = deduplicate(exs, cfg.dedup)
    clusters = [result.cluster_of[e.question_id] for e in order]
    assert clusters[0] == clusters[1] == clusters[2] == order[0].question_id
    assert clusters[3] == order[3].question_id


def test_cross_dataset_cluster_keeps_priority_dataset(cfg):
    tqa = make_example("truthfulqa", "Which magazine was started first, Arthur's Magazine or First for Women?")
    halu = make_example("halueval", "Which magazine was started first Arthur's Magazine or First for Women?")
    result = deduplicate([tqa, halu], cfg.dedup)
    assert [e.question_id for e in result.kept] == [halu.question_id]
    assert result.dropped[0]["question_id"] == tqa.question_id
    assert result.dropped[0]["reason"] == "cross_dataset_duplicate"


def test_identical_text_across_datasets_is_not_exact_duplicate(cfg):
    tqa = make_example("truthfulqa", MELON)
    halu = make_example("halueval", MELON)
    result = deduplicate([tqa, halu], cfg.dedup)
    assert [d["reason"] for d in result.dropped] == ["cross_dataset_duplicate"]
    assert result.kept[0].dataset == "halueval"


def test_near_duplicate_pairs_chunking_matches_unchunked():
    texts = [normalize_question(t) for t in (GOLD, BITCOIN, MELON, TWINKLE, GOLD + " now")]
    full = near_duplicate_pairs(texts, (3, 5), 0.8, chunk_size=100)
    chunked = near_duplicate_pairs(texts, (3, 5), 0.8, chunk_size=2)
    assert [(i, j) for i, j, _ in full] == [(i, j) for i, j, _ in chunked]
    assert all(i < j for i, j, _ in full)
