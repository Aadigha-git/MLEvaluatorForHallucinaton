from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np
import pytest

from hdetect.features import EVIDENCE_FEATURES, EVIDENCE_FREE_FEATURES, RESAMPLING_FEATURES, feature_columns
from hdetect.features.access import Answers, QuestionInput
from hdetect.features.extract import (
    BlockCache,
    FeatureExtractor,
    MissingFeatureValue,
    Versions,
    check_complete,
    sha256_text,
)
from hdetect.features.lexicon import BOOSTERS_V1, HEDGES_V1, booster_count, hedge_count
from hdetect.features.models import ModelSpec
from hdetect.features.recall import NOT_MEASURABLE, fever_gold_sentences, recall
from hdetect.features.retrieval import Passage, parse_dpr_contents
from hdetect.features.text import hedge_rates, lex_overlap_max, log_length, specificity, split_sentences
from hdetect.labeling.text import tokens


class FakeEmbedder:
    """Deterministic unit vectors from a text hash; identical texts embed identically."""

    def __init__(self):
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        rng = np.random.default_rng(int(hashlib.sha256(text.encode()).hexdigest()[:8], 16))
        v = rng.normal(size=8)
        return v / np.linalg.norm(v)


class FakeNLI:
    def __init__(self):
        self.calls = []

    def probs(self, premise, hypothesis):
        self.calls.append((premise, hypothesis))
        h = int(hashlib.sha256(f"{premise}|{hypothesis}".encode()).hexdigest()[:8], 16)
        c, e = (h % 97) / 300, (h % 89) / 300
        return {"contradiction": c, "entailment": e, "neutral": 1 - c - e}


VERSIONS = Versions.current(ModelSpec("emb", "rev-e"), ModelSpec("nli", "rev-n"))


def extractor(tmp_path, versions=VERSIONS):
    return FeatureExtractor(FakeEmbedder(), FakeNLI(), BlockCache(tmp_path / "cache.sqlite"), versions)


def q_input(dataset="halueval", split="val_id", gold=("Knowledge passage about Paris. It is in France.",)):
    return QuestionInput("q-1", dataset, split, "model_a", "Where is Paris?", tuple(gold), ())


def row(text, sample_index, finish="stop"):
    return {"answer_id": f"q-1/model_a/{sample_index}", "question_id": "q-1", "sample_index": sample_index,
            "text": text, "finish_reason": finish, "status": "ok"}


def answers(primary="**Paris** is in France. It is the capital.\n\n", resamples=None, finish="stop"):
    resamples = resamples if resamples is not None else [f"Paris is in France, take {i}." for i in range(1, 6)]
    return Answers("q-1", row(primary, 0, finish), tuple(row(t, i) for i, t in enumerate(resamples, 1)))


CONDITIONS = {"retrieved": ["Paris: Paris is the capital of France.", "France: A country."], "none": None}


# --- pure features ---


def test_lexicon_is_frozen():
    assert len(HEDGES_V1) == 43 and len(BOOSTERS_V1) == 14
    digest = hashlib.sha256(json.dumps([HEDGES_V1, BOOSTERS_V1]).encode()).hexdigest()
    assert digest == "09b5f3175b5c8643fad3767d1ae1d06b8fd21b0b6efb49fa56e3d94c5bd4d87d"


def test_hedge_matching_longest_first_without_overlap():
    assert hedge_count(tokens("It is possible that it might rain, I think.")) == 3
    assert hedge_count(tokens("Perhaps; perhaps.")) == 2
    assert booster_count(tokens("In fact it is definitely, of course, true.")) == 3
    assert hedge_rates("") == (0.0, 0.0)
    h, b = hedge_rates("Probably yes, definitely.")
    assert (h, b) == (1 / 3, 1 / 3)


def test_length_and_specificity():
    assert log_length("") == 0.0
    assert log_length("The cat sat.") == pytest.approx(math.log1p(2))  # "the" is an article
    # tokens: Paris(start) is in France. In(start) 1889 Gustave Eiffel built it. I agree.
    text = "Paris is in France. In 1889 Gustave Eiffel built it. I agree."
    words = text.split()
    assert specificity(text) == pytest.approx((1 + 3) / len(words))
    assert specificity('He said "Rome" twice.') == pytest.approx(1 / 4)
    assert specificity("") == 0.0


def test_lex_overlap_ignores_question_tokens():
    q = "Where is the Eiffel Tower?"
    assert lex_overlap_max("The Eiffel Tower is in Paris.", q, ["Paris is big.", "No match."]) == 1.0
    assert lex_overlap_max("The Eiffel Tower.", q, ["Paris"]) == 0.0
    assert lex_overlap_max("Paris, France.", q, ["Paris", "France and Paris"]) == 1.0
    assert lex_overlap_max("Paris, France.", q, ["Paris"]) == 0.5


def test_sentence_split():
    assert split_sentences("A b. C d! 3 e? (f) g. h i.") == ["A b.", "C d!", "3 e?", "(f) g. h i."]
    assert split_sentences("") == []


def test_parse_dpr_contents():
    assert parse_dpr_contents('"Aaron"\nAaron is a prophet.') == ("Aaron", "Aaron is a prophet.")
    assert Passage("1", 3.0, "Aaron", "Aaron is a prophet.").formatted() == "Aaron: Aaron is a prophet."


# --- extraction ---


def test_extraction_is_deterministic(tmp_path):
    a = extractor(tmp_path / "a").rows(q_input(), answers(), CONDITIONS)
    b = extractor(tmp_path / "b").rows(q_input(), answers(), CONDITIONS)
    strip = lambda r: {k: v for k, v in r.items() if not k.startswith("time_")}  # noqa: E731
    assert {c: strip(r) for c, r in a.items()} == {c: strip(r) for c, r in b.items()}


def test_all_three_conditions(tmp_path):
    conds = {**CONDITIONS, "gold": list(q_input().gold_passages)}
    rows = extractor(tmp_path).rows(q_input(), answers(), conds)
    assert set(rows) == {"retrieved", "none", "gold"}
    for condition, r in rows.items():
        assert r["condition"] == condition
        for f in feature_columns(condition):
            assert r[f] is not None and math.isfinite(r[f])
    assert rows["retrieved"]["sim_max"] != rows["gold"]["sim_max"]
    assert rows["gold"]["meta_n_passages"] == 1 and rows["retrieved"]["meta_n_passages"] == 2


def test_evidence_free_features_with_no_evidence(tmp_path):
    r = extractor(tmp_path).rows(q_input(), answers(), {"none": None})["none"]
    assert not any(f in r for f in EVIDENCE_FEATURES)
    assert all(math.isfinite(r[f]) for f in EVIDENCE_FREE_FEATURES + RESAMPLING_FEATURES)
    assert r["meta_n_passages"] == 0


def test_single_shot_features_use_only_the_primary_answer(tmp_path):
    base = extractor(tmp_path / "a").rows(q_input(), answers(), CONDITIONS)["retrieved"]
    other = extractor(tmp_path / "b").rows(
        q_input(), answers(resamples=[f"Lyon {i}." for i in range(5)]), CONDITIONS
    )["retrieved"]
    for f in EVIDENCE_FEATURES + EVIDENCE_FREE_FEATURES:
        assert base[f] == other[f], f
    assert any(base[f] != other[f] for f in RESAMPLING_FEATURES)


def test_resampling_uses_exactly_the_cached_resamples(tmp_path):
    ex = extractor(tmp_path)
    resamples = ["*One*.", "Two.", "Three.", "Four.", "Five."]
    ex.rows(q_input(), answers(primary="Paris.", resamples=resamples), {"none": None})
    sc_premises = [p for p, h in ex.nli.calls if h == "Paris."]
    assert sc_premises == ["One.", "Two.", "Three.", "Four.", "Five."]
    assert ex.embedder.calls == ["Paris.", "One.", "Two.", "Three.", "Four.", "Five."]


def test_empty_resamples_are_excluded_and_zero_usable_is_flagged(tmp_path):
    r = extractor(tmp_path).rows(q_input(), answers(resamples=["A.", " ", "**B**", "", "C."]), {"none": None})["none"]
    assert (r["meta_n_resamples_cached"], r["meta_n_resamples_usable"]) == (5, 3)
    r0 = extractor(tmp_path / "z").rows(q_input(), answers(resamples=["", " "] + [""] * 3), {"none": None})["none"]
    assert r0["meta_n_resamples_usable"] == 0
    assert all(r0[f] is None for f in RESAMPLING_FEATURES)


def test_more_than_five_resamples_is_rejected(tmp_path):
    bad = answers()
    bad = Answers("q-1", bad.primary, bad.resamples + (row("Six.", 6),))
    with pytest.raises(ValueError):
        extractor(tmp_path).rows(q_input(), bad, {"none": None})


def test_raw_text_is_preserved_and_cleaned_text_is_used(tmp_path):
    raw = answers(primary="**Paris** is in\n\nFrance.  ")
    before = copy.deepcopy(raw)
    ex = extractor(tmp_path)
    r = ex.rows(q_input(), raw, {"none": None})["none"]
    assert raw == before
    assert r["meta_raw_sha256"] == sha256_text("**Paris** is in\n\nFrance.  ")
    assert "Paris is in France." in ex.embedder.calls
    clean = extractor(tmp_path / "c").rows(q_input(), answers(primary="Paris is in France."), {"none": None})["none"]
    for f in EVIDENCE_FREE_FEATURES + RESAMPLING_FEATURES:
        assert r[f] == clean[f]


def test_fever_verdict_is_metadata_not_a_feature(tmp_path):
    q = QuestionInput("q-1", "fever", "train", "model_b", "A claim.", (), ())
    r = extractor(tmp_path).rows(q, answers(primary="*NOT* ENOUGH INFO. Unclear."), {"none": None})["none"]
    assert r["meta_fever_verdict"] == "NOT ENOUGH INFO"
    assert "meta_fever_verdict" not in feature_columns("none")
    other = extractor(tmp_path / "h").rows(q_input(), answers(), {"none": None})["none"]
    assert other["meta_fever_verdict"] is None


def test_truncated_answers_are_kept_and_flagged(tmp_path):
    res = answers(finish="length")
    res = Answers("q-1", res.primary, res.resamples[:4] + (row("Cut off mid", 5, "length"),))
    r = extractor(tmp_path).rows(q_input(), res, CONDITIONS)
    assert r["retrieved"]["meta_truncated_primary"] is True
    assert r["retrieved"]["meta_n_resamples_truncated"] == 1
    assert all(math.isfinite(r["retrieved"][f]) for f in EVIDENCE_FEATURES + EVIDENCE_FREE_FEATURES)


# --- cache and versions ---


def test_cache_hits_and_invalidation(tmp_path):
    ex = extractor(tmp_path)
    first = ex.rows(q_input(), answers(), CONDITIONS)
    calls = len(ex.nli.calls)
    again = ex.rows(q_input(), answers(), CONDITIONS)
    assert len(ex.nli.calls) == calls and again == first
    assert ex.cache.hits == 2

    ex.rows(q_input(), answers(), {"retrieved": ["A different passage."], "none": None})
    assert len(ex.nli.calls) > calls

    newer = Versions.current(ModelSpec("emb", "rev-e"), ModelSpec("nli", "rev-n2"))
    ex2 = FeatureExtractor(FakeEmbedder(), FakeNLI(), ex.cache, newer)
    r = ex2.rows(q_input(), answers(), CONDITIONS)
    assert ex2.nli.calls, "a new NLI revision must not reuse cached blocks"
    assert r["retrieved"]["nli_model"] == "nli@rev-n2"


def test_rows_record_versions(tmp_path):
    r = extractor(tmp_path).rows(q_input(), answers(), CONDITIONS)["retrieved"]
    assert r["feature_version"] == "features-v1"
    assert r["clean_version"] == "1" and r["lexicon_version"] == "hedges-v1"
    assert r["embedding_model"] == "emb@rev-e" and r["nli_model"] == "nli@rev-n"
    assert {k for k in r if k.startswith("time_")} == {
        "time_hedge_ms", "time_length_ms", "time_self_contra_ms", "time_sc_embed_ms", "time_sc_nli_ms",
        "time_sim_ms", "time_nli_ms", "time_lex_ms",
    }


def test_check_complete_rejects_missing_values():
    good = {"question_id": "q", "meta_n_resamples_usable": 5, **{f: 0.5 for f in feature_columns("retrieved")}}
    check_complete(good, "retrieved")
    with pytest.raises(MissingFeatureValue):
        check_complete({**good, "sim_max": float("nan")}, "retrieved")
    with pytest.raises(MissingFeatureValue):
        check_complete({**good, "sc_embed": None}, "retrieved")
    with pytest.raises(MissingFeatureValue):
        check_complete(good, "none")


# --- recall ---


def test_recall_per_dataset():
    fever = QuestionInput("f", "fever", "test_heldout", "model_b", "Claim.",
                          ("Dan O'Bannon: He died in 2009.",), ("Dan_O'Bannon",))
    halu = QuestionInput("h", "halueval", "val_id", "model_a", "Q?", ("First fact here. Second fact there.",), ())
    tqa = QuestionInput("t", "truthfulqa", "train", "model_a", "Q?", (), ())
    retrieved = {
        "f": [Passage("1", 1.0, "dan o'bannon", "Bio. He died in 2009 of illness.")],
        "h": [Passage("2", 1.0, "X", "Unrelated."), Passage("3", 1.0, "Y", "Yes, second fact there, indeed.")],
        "t": [Passage("4", 1.0, "Z", "Anything.")],
    }
    out = recall([fever, halu, tqa], retrieved)
    assert out["fever"] == {"n": 1, "page_recall@5": 1.0, "sentence_recall@5": 1.0}
    assert out["halueval"] == {"n": 1, "knowledge_sentence_recall@5": 1.0}
    assert out["truthfulqa"] == {"n": 1, "recall": NOT_MEASURABLE}
    assert fever_gold_sentences(fever) == ["He died in 2009."]
    miss = recall([halu], {"h": [Passage("2", 1.0, "X", "Unrelated.")]})
    assert miss["halueval"]["knowledge_sentence_recall@5"] == 0.0
