"""The pinned embedding and NLI models: label mapping, revisions, and determinism.

Marked `models`: needs the pinned weights in the local Hugging Face cache (no network).
"""

from __future__ import annotations

import os

import pytest
import yaml

from tests.conftest import ROOT

pytestmark = pytest.mark.models

FCFG = yaml.safe_load((ROOT / "configs" / "features.yaml").read_text())


@pytest.fixture(scope="module")
def models():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from hdetect.features.models import NLI, Embedder, ModelSpec, configure_torch

    configure_torch(FCFG["runtime"]["torch_threads"])
    emb = ModelSpec(FCFG["embedding"]["model_id"], FCFG["embedding"]["revision"])
    nli = ModelSpec(FCFG["nli"]["model_id"], FCFG["nli"]["revision"])
    try:
        return Embedder(emb, FCFG["embedding"]["max_seq_length"]), NLI(nli, FCFG["nli"]["max_length"])
    except OSError as e:
        pytest.skip(f"pinned models not cached: {e}")


def test_config_pins_preregistered_revisions():
    assert FCFG["embedding"]["revision"] == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert FCFG["nli"]["revision"] == "fa2804872c3b4bd748f38c0185cc85775361e735"
    assert FCFG["nli"]["id2label"] == {0: "contradiction", 1: "entailment", 2: "neutral"}


def test_nli_label_mapping(models):
    _, nli = models
    assert nli.index == {"contradiction": 0, "entailment": 1, "neutral": 2}
    premise = "Paris is the capital of France."
    assert max(nli.probs(premise, "The capital of France is Paris.").items(), key=lambda kv: kv[1])[0] == "entailment"
    assert max(nli.probs(premise, "Berlin is the capital of France.").items(), key=lambda kv: kv[1])[0] == "contradiction"
    assert max(nli.probs(premise, "France produces cheese.").items(), key=lambda kv: kv[1])[0] == "neutral"


def test_wrong_label_mapping_is_refused(models, monkeypatch):
    from hdetect.features.models import NLI, LabelMappingError, ModelSpec

    _, nli = models
    monkeypatch.setattr(nli.model.config, "id2label", {0: "entailment", 1: "contradiction", 2: "neutral"})
    import transformers

    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification, "from_pretrained", lambda *a, **k: nli.model
    )
    with pytest.raises(LabelMappingError):
        NLI(ModelSpec(FCFG["nli"]["model_id"], FCFG["nli"]["revision"]), 512)


def test_real_models_are_deterministic(models):
    from hdetect.features.extract import evidence_block, evidence_free_block

    emb, nli = models
    answer = "Paris is in France. It is not in Germany."
    resamples = ["Paris is in France.", "Paris, France.", "It is in France.", "France.", "Paris is French."]
    passages = ["Paris: Paris is the capital of France.", "Berlin: Berlin is in Germany."]
    a = evidence_free_block(answer, resamples, emb, nli)["features"]
    b = evidence_free_block(answer, resamples, emb, nli)["features"]
    assert a == b
    c = evidence_block(answer, "Where is Paris?", passages, emb, nli)["features"]
    d = evidence_block(answer, "Where is Paris?", passages, emb, nli)["features"]
    assert c == d
    assert 0.0 <= c["nli_entail_max"] <= 1.0 and -1.0 <= c["sim_max"] <= 1.0
